#!/usr/bin/env python3
"""
DDP HR Players Builder -- daily GitHub Actions job
Pulls current-season batted balls from Baseball Savant (batted-ball rows only,
so it's a much lighter pull than fetch_savant.py) and writes players.json:

  hitters : per-batter empirical (EV, LA, spray) samples split vs LHP / vs RHP,
            with BBE counts and BBE/PA contact rates
  pitchers: per-pitcher contact-quality logit shift (scored against the fence
            model grid in fence_model.json) plus BBE/PA contact rate allowed
  pools   : league batted-ball samples vs LHP / vs RHP (shrinkage targets)
  meta    : league BBE/PA, league mean fence logit, shrinkage constant

Standard library only -- no pip installs needed in the Action.
Spray conventions (must match ddp-hr.html's simHitter):
  stored hitter samples & pools: batter-relative, POSITIVE = PULL side
    (the app un-mirrors to field coords via the hitter's stand)
  pools keyed by BATTER STAND (league shrinkage stays spray-coherent)
  pitcher shifts / league_z: field coordinates against the fence grid
"""
import csv
import io
import json
import math
import random
import sys
import time
import urllib.request
from datetime import datetime, date, timedelta, timezone

SEASON = datetime.now().year
SEASON_START = date(SEASON, 3, 15)
TODAY = date.today()
CHUNK_DAYS = 5
SAMPLE_CAP = 120        # max stored (EV, LA, spray) samples per hitter split
POOL_CAP = 450          # league pool size per pitcher hand
SHRINK_K_BBE = 80       # shrinkage constant (client-side + pitcher shifts)
MIN_HITTER_BBE = 20     # total BBE across both splits to be included
MIN_PITCHER_BBE = 25    # BBE allowed to be included
FENCE_MODEL_PATH = "fence_model.json"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0 Safari/537.36",
    "Accept": "text/csv,*/*",
    "Referer": "https://baseballsavant.mlb.com/statcast_search",
}

# Server-side batted-ball filter. If Savant ever ignores it, 5-day chunks stay
# under the row cap anyway and we filter client-side on bb_type.
BBT_FILTER = "&hfBBT=ground%5C.%5C.ball%7Cline%5C.%5C.drive%7Cfly%5C.%5C.ball%7Cpopup%7C"

# events that end a row but are not plate-appearance endings
NON_PA_EVENTS = {
    "caught_stealing_2b", "caught_stealing_3b", "caught_stealing_home",
    "pickoff_1b", "pickoff_2b", "pickoff_3b",
    "pickoff_caught_stealing_2b", "pickoff_caught_stealing_3b",
    "pickoff_caught_stealing_home",
    "stolen_base_2b", "stolen_base_3b", "stolen_base_home",
    "wild_pitch", "passed_ball", "game_advisory", "ejection",
}


def to_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def spray_angle(hc_x, hc_y):
    """Field spray angle in degrees. Negative = LF line, positive = RF line."""
    return math.degrees(math.atan2(hc_x - 125.42, 198.27 - hc_y))


# ---------------------------------------------------------------- fence grid
class FenceGrid:
    """Trilinear interpolation over the league P(HR|EV,LA,spray) grid."""

    def __init__(self, path):
        with open(path) as f:
            fm = json.load(f)
        g = fm["grid"]
        self.ev, self.la, self.spray, self.p = g["ev"], g["la"], g["spray"], g["p"]

    @staticmethod
    def _locate(axis, x):
        x = min(max(x, axis[0]), axis[-1])
        step = axis[1] - axis[0]
        i = min(int((x - axis[0]) / step), len(axis) - 2)
        frac = (x - axis[i]) / step
        return i, frac

    def prob(self, ev, la, spray):
        i, fx = self._locate(self.ev, ev)
        j, fy = self._locate(self.la, la)
        k, fz = self._locate(self.spray, spray)
        p = self.p
        c00 = p[i][j][k] * (1 - fx) + p[i + 1][j][k] * fx
        c01 = p[i][j][k + 1] * (1 - fx) + p[i + 1][j][k + 1] * fx
        c10 = p[i][j + 1][k] * (1 - fx) + p[i + 1][j + 1][k] * fx
        c11 = p[i][j + 1][k + 1] * (1 - fx) + p[i + 1][j + 1][k + 1] * fx
        c0 = c00 * (1 - fy) + c10 * fy
        c1 = c01 * (1 - fy) + c11 * fy
        return c0 * (1 - fz) + c1 * fz

    def logit(self, ev, la, spray):
        p = min(max(self.prob(ev, la, spray), 1e-6), 1 - 1e-6)
        return math.log(p / (1 - p))


# ---------------------------------------------------------------- reservoir
def reservoir_add(sample_list, seen_count, item, cap, rng):
    """Standard reservoir sampling. seen_count is the count BEFORE this item."""
    if len(sample_list) < cap:
        sample_list.append(item)
    else:
        r = rng.randint(0, seen_count)
        if r < cap:
            sample_list[r] = item


# ---------------------------------------------------------------- fetch
def fetch_chunk(d1, d2, bbt_filter=True):
    url = (
        "https://baseballsavant.mlb.com/statcast_search/csv?all=true"
        f"&player_type=batter&type=details&minors=false"
        f"&game_date_gt={d1}&game_date_lt={d2}"
        + (BBT_FILTER if bbt_filter else "")
    )
    for attempt in (1, 2, 3):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=120) as r:
                return r.read().decode("utf-8-sig")
        except Exception as e:
            print(f"  chunk {d1}..{d2} attempt {attempt} failed: {e}", file=sys.stderr)
            if attempt == 2 and bbt_filter:
                # last resort: retry without the batted-ball filter param
                return fetch_chunk(d1, d2, bbt_filter=False)
            time.sleep(3)
    return None


# ---------------------------------------------------------------- main
def process_rows(reader, state, grid, rng):
    """Accumulate one CSV chunk into state. Returns rows processed."""
    hitters = state["hitters"]
    pitchers = state["pitchers"]
    pools = state["pools"]
    n = 0
    for row in reader:
        n += 1
        bid = str(row.get("batter", "")).strip().split(".")[0]
        pid = str(row.get("pitcher", "")).strip().split(".")[0]
        if not (bid.isdigit() and pid.isdigit()):
            continue
        stand = (row.get("stand") or "").strip() or "R"
        throws = (row.get("p_throws") or "").strip() or "R"
        events = (row.get("events") or "").strip()
        bb = (row.get("bb_type") or "").strip()
        is_pa_end = bool(events) and events not in NON_PA_EVENTS

        # hitter split accumulator
        hkey = (bid, throws)
        if hkey not in hitters:
            hitters[hkey] = {"n": 0, "pa": 0, "s": [], "stand": {}}
        h = hitters[hkey]
        # pitcher accumulator
        if pid not in pitchers:
            pitchers[pid] = {"n": 0, "pa": 0, "zsum": 0.0, "zn": 0, "throws": {}}
        p = pitchers[pid]
        p["throws"][throws] = p["throws"].get(throws, 0) + 1

        if is_pa_end:
            h["pa"] += 1
            p["pa"] += 1

        if not bb:
            continue  # not a batted ball

        h["n"] += 1
        h["stand"][stand] = h["stand"].get(stand, 0) + 1
        p["n"] += 1

        ev = to_float(row.get("launch_speed"))
        la = to_float(row.get("launch_angle"))
        hx = to_float(row.get("hc_x"))
        hy = to_float(row.get("hc_y"))
        if ev is None or la is None or hx is None or hy is None:
            continue  # untracked ball: counts toward n, no sample
        sp = spray_angle(hx, hy)          # field coords (neg = LF) for the grid
        sb = -sp if stand == "R" else sp  # batter-relative (pos = pull) for storage
        item = [round(ev, 1), round(la, 1), round(sb, 1)]

        reservoir_add(h["s"], h["n"] - 1, item, SAMPLE_CAP, rng)

        pool = pools[stand]               # pools keyed by BATTER STAND, pull-relative
        reservoir_add(pool["s"], pool["seen"], item, POOL_CAP, rng)
        pool["seen"] += 1

        if grid is not None:
            z = grid.logit(ev, la, sp)
            p["zsum"] += z
            p["zn"] += 1
            state["league_zsum"] += z
            state["league_zn"] += 1

        state["league_bbe"] += 1
    return n


def main():
    rng = random.Random(20260808)
    try:
        grid = FenceGrid(FENCE_MODEL_PATH)
        print("Loaded fence model grid")
    except Exception as e:
        grid = None
        print(f"WARNING: no fence model available ({e}) -- pitcher shifts will be 0",
              file=sys.stderr)

    state = {
        "hitters": {},   # (batter_id, p_throws) -> acc
        "pitchers": {},  # pitcher_id -> acc
        "pools": {"L": {"s": [], "seen": 0}, "R": {"s": [], "seen": 0}},
        "league_zsum": 0.0, "league_zn": 0,
        "league_bbe": 0,
    }

    total = 0
    d = SEASON_START
    while d <= TODAY:
        d2 = min(d + timedelta(days=CHUNK_DAYS - 1), TODAY)
        text = fetch_chunk(d.isoformat(), d2.isoformat())
        if text and "," in text:
            n = process_rows(csv.DictReader(io.StringIO(text)), state, grid, rng)
            total += n
            print(f"[{d2}] +{n} rows (running total {total})")
        d = d2 + timedelta(days=1)
        time.sleep(0.6)

    league_pa = sum(a["pa"] for a in state["hitters"].values())
    league_bbe_pa = round(state["league_bbe"] / league_pa, 3) if league_pa else 0.661
    league_z = (state["league_zsum"] / state["league_zn"]) if state["league_zn"] else -7.657
    league_z = round(league_z, 3)

    # ---- hitters -----------------------------------------------------------
    by_batter = {}
    for (bid, throws), a in state["hitters"].items():
        by_batter.setdefault(bid, {})[throws] = a
    hitters_out = {}
    for bid, splits in by_batter.items():
        if sum(a["n"] for a in splits.values()) < MIN_HITTER_BBE:
            continue
        e = {}
        for throws, a in splits.items():
            if a["n"] == 0 or not a["s"]:
                continue
            stand = max(a["stand"], key=a["stand"].get) if a["stand"] else "R"
            split = {"n": a["n"], "stand": stand, "s": a["s"]}
            if a["pa"] > 0:
                split["bbe_pa"] = round(a["n"] / a["pa"], 3)
            e["vs" + throws] = split
        if e:
            hitters_out[bid] = e

    # ---- pitchers ----------------------------------------------------------
    pitchers_out = {}
    for pid, a in state["pitchers"].items():
        if a["n"] < MIN_PITCHER_BBE:
            continue
        throws = max(a["throws"], key=a["throws"].get) if a["throws"] else "R"
        if a["zn"] > 0 and state["league_zn"] > 0:
            raw = (a["zsum"] / a["zn"]) - (state["league_zsum"] / state["league_zn"])
            shift = raw * a["zn"] / (a["zn"] + SHRINK_K_BBE)
        else:
            shift = 0.0
        entry = {"throws": throws, "shift": round(shift, 4), "n": a["n"]}
        if a["pa"] > 0:
            entry["bbe_pa"] = round(a["n"] / a["pa"], 3)
        pitchers_out[pid] = entry

    out = {
        "meta": {
            "vintage": f"{SEASON} season through {TODAY.isoformat()} (current ball)",
            "updated": datetime.now(timezone.utc).isoformat(),
            "league_bbe_pa": league_bbe_pa,
            "league_z": league_z,
            "shrink_k_bbe": SHRINK_K_BBE,
        },
        "pools": {"L": state["pools"]["L"]["s"], "R": state["pools"]["R"]["s"]},
        "hitters": hitters_out,
        "pitchers": pitchers_out,
    }
    with open("players.json", "w") as f:
        json.dump(out, f, separators=(",", ":"))

    print(f"\nWrote players.json -- {len(hitters_out)} hitters, "
          f"{len(pitchers_out)} pitchers, league_bbe_pa={league_bbe_pa}, "
          f"league_z={league_z}")
    if len(hitters_out) < 100:
        print("WARNING: sparse hitter output -- check fetch errors above",
              file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
