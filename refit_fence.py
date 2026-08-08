#!/usr/bin/env python3
"""
DDP Fence Model Refit -- monthly GitHub Actions job
Pulls batted balls for the current-ball era (2023 -> today) from Baseball
Savant and refits the fence-clearing model, writing fence_model.json in the
exact schema the HR tool consumes:

  grid                : league P(HR | EV, LA, spray) on a 33 x 23 x 19 lattice
                        (EV 85..117 x1, LA 6..50 x2, spray -45..45 x5)
  park_logit_offsets  : per-park, per-sector logit residuals
                        (LF_line / LF_gap / CF / RF_gap / RF_line)
  meta.platt          : walk-forward Platt calibration fitted on the most
                        recent 20% of batted balls (never in-sample)

Spray convention: field coordinates, negative = LF line -- identical to
build_players.py, so grid, park sectors, and player samples always agree.

Era note: 2023+ is one ball era, so the old dead-ball season shift is retired.
season_logit_shift_2022 is kept at 0.0 for HTML compatibility; the Platt
intercept now carries any level correction.

Requires: numpy, scikit-learn (installed by the workflow).
"""
import csv
import io
import json
import math
import sys
import time
import urllib.request
from datetime import datetime, date, timedelta, timezone

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

TRAIN_START_YEAR = 2023          # first current-ball season in the fit window
SEASON = datetime.now().year
TODAY = date.today()
CHUNK_DAYS = 5

EV_AXIS = [float(v) for v in range(85, 118)]           # 33
LA_AXIS = [float(v) for v in range(6, 51, 2)]          # 23
SPRAY_AXIS = [float(v) for v in range(-45, 46, 5)]     # 19
SECTOR_EDGES = [-45.0, -27.0, -9.0, 9.0, 27.0, 45.0]
SECTOR_NAMES = ["LF_line", "LF_gap", "CF", "RF_gap", "RF_line"]
PARK_SHRINK_K = 800              # BBE shrinkage for park-sector offsets
OFFSET_CLAMP = 1.5

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0 Safari/537.36",
    "Accept": "text/csv,*/*",
    "Referer": "https://baseballsavant.mlb.com/statcast_search",
}
BBT_FILTER = "&hfBBT=ground%5C.%5C.ball%7Cline%5C.%5C.drive%7Cfly%5C.%5C.ball%7Cpopup%7C"

# normalize Savant home_team codes to the codes parks.json / the HTML use
TEAM_ALIASES = {
    "AZ": "ARI", "CHW": "CWS", "KCR": "KC", "SDP": "SD",
    "SFG": "SF", "TBR": "TB", "WSN": "WSH",
}


def to_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def spray_angle(hc_x, hc_y):
    return math.degrees(math.atan2(hc_x - 125.42, 198.27 - hc_y))


def logit(p, eps=1e-6):
    p = min(max(p, eps), 1 - eps)
    return math.log(p / (1 - p))


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
            with urllib.request.urlopen(req, timeout=180) as r:
                return r.read().decode("utf-8-sig")
        except Exception as e:
            print(f"  chunk {d1}..{d2} attempt {attempt} failed: {e}", file=sys.stderr)
            if attempt == 2 and bbt_filter:
                return fetch_chunk(d1, d2, bbt_filter=False)
            time.sleep(3)
    return None


def collect_rows():
    """Yield (date_str, ev, la, spray, park, season, is_hr) for tracked BBE."""
    rows = []
    for year in range(TRAIN_START_YEAR, SEASON + 1):
        d = date(year, 3, 15)
        end = min(date(year, 11, 5), TODAY)
        while d <= end:
            d2 = min(d + timedelta(days=CHUNK_DAYS - 1), end)
            text = fetch_chunk(d.isoformat(), d2.isoformat())
            if text and "," in text:
                n = 0
                for row in csv.DictReader(io.StringIO(text)):
                    bb = (row.get("bb_type") or "").strip()
                    if not bb:
                        continue
                    ev = to_float(row.get("launch_speed"))
                    la = to_float(row.get("launch_angle"))
                    hx = to_float(row.get("hc_x"))
                    hy = to_float(row.get("hc_y"))
                    if ev is None or la is None or hx is None or hy is None:
                        continue
                    park = (row.get("home_team") or "").strip()
                    park = TEAM_ALIASES.get(park, park)
                    gd = (row.get("game_date") or "")[:10]
                    is_hr = 1 if (row.get("events") or "").strip() == "home_run" else 0
                    rows.append((gd, ev, la, spray_angle(hx, hy), park, year, is_hr))
                    n += 1
                print(f"[{year} {d2}] +{n} batted balls (total {len(rows)})")
            d = d2 + timedelta(days=1)
            time.sleep(0.5)
    return rows


def sector_index(spray):
    for i in range(5):
        if spray < SECTOR_EDGES[i + 1]:
            return i
    return 4


def main():
    rows = collect_rows()
    if len(rows) < 100_000:
        print(f"ERROR: only {len(rows)} batted balls collected -- refusing to "
              f"overwrite fence_model.json with a thin fit", file=sys.stderr)
        sys.exit(1)

    rows.sort(key=lambda r: r[0])  # chronological
    X = np.array([[r[1], r[2], r[3]] for r in rows], dtype=np.float32)
    y = np.array([r[6] for r in rows], dtype=np.int8)
    parks = [r[4] for r in rows]
    seasons = [r[5] for r in rows]

    split = int(len(rows) * 0.8)  # walk-forward: newest 20% held out
    print(f"\nTrain {split} rows / calibrate {len(rows) - split} rows "
          f"(holdout starts {rows[split][0]})")

    model = HistGradientBoostingClassifier(
        max_iter=400, learning_rate=0.08, max_leaf_nodes=63,
        min_samples_leaf=200, l2_regularization=1.0, random_state=7,
    )
    model.fit(X[:split], y[:split])

    # ---- league grid -------------------------------------------------------
    mesh = np.array(
        [[e, l, s] for e in EV_AXIS for l in LA_AXIS for s in SPRAY_AXIS],
        dtype=np.float32,
    )
    gp = model.predict_proba(mesh)[:, 1].reshape(len(EV_AXIS), len(LA_AXIS), len(SPRAY_AXIS))
    grid_p = [[[round(float(v), 4) for v in spray_row] for spray_row in la_block]
              for la_block in gp.tolist()]

    # ---- park sector offsets ----------------------------------------------
    pred_all = model.predict_proba(X)[:, 1]
    acc = {}
    for i in range(len(rows)):
        park, season = parks[i], seasons[i]
        if not park:
            continue
        if park == "TB" and season == 2025:
            continue  # Rays' 2025 home games were at Steinbrenner Field, not the Trop
        key = (park, sector_index(float(X[i, 2])))
        a = acc.setdefault(key, [0, 0, 0.0])   # n, hr, pred_sum
        a[0] += 1
        a[1] += int(y[i])
        a[2] += float(pred_all[i])
    park_offsets = {}
    for (park, sec), (n, hr, psum) in acc.items():
        if n < 50:
            continue
        raw = logit(hr / n) - logit(psum / n)
        off = max(-OFFSET_CLAMP, min(OFFSET_CLAMP, raw * n / (n + PARK_SHRINK_K)))
        park_offsets.setdefault(park, {})[SECTOR_NAMES[sec]] = round(off, 3)
    # every park gets all five sectors (0.0 where data was thin)
    for park in park_offsets:
        for name in SECTOR_NAMES:
            park_offsets[park].setdefault(name, 0.0)
    park_offsets = {k: {n: v[n] for n in SECTOR_NAMES}
                    for k, v in sorted(park_offsets.items())}

    # ---- walk-forward Platt on holdout, park offsets applied ---------------
    z_hold = []
    for i in range(split, len(rows)):
        z = logit(float(pred_all[i]))
        off = park_offsets.get(parks[i], {}).get(SECTOR_NAMES[sector_index(float(X[i, 2]))], 0.0)
        z_hold.append(z + off)
    lr = LogisticRegression(C=1e6, solver="lbfgs")
    lr.fit(np.array(z_hold).reshape(-1, 1), y[split:])
    platt_slope = float(lr.coef_[0][0])
    platt_intercept = float(lr.intercept_[0])

    # holdout sanity report
    z_arr = np.array(z_hold)
    p_cal = 1.0 / (1.0 + np.exp(-(platt_slope * z_arr + platt_intercept)))
    print(f"Platt: slope={platt_slope:.4f} intercept={platt_intercept:.4f}")
    print(f"Holdout HR rate observed={y[split:].mean():.4f} "
          f"calibrated_pred={p_cal.mean():.4f}")

    out = {
        "meta": {
            "fit_window": f"{TRAIN_START_YEAR}-{SEASON} regular season (current ball)",
            "refit_date": datetime.now(timezone.utc).isoformat(),
            "n_batted_balls": len(rows),
            "note": ("P(HR|EV,LA,spray,park). Trilinear-interpolate league grid, "
                     "add park sector logit offset, then season/temp calibration."),
            "season_logit_shift_2022": 0.0,
            "platt": {
                "platt_slope": round(platt_slope, 6),
                "platt_intercept": round(platt_intercept, 6),
            },
        },
        "grid": {"ev": EV_AXIS, "la": LA_AXIS, "spray": SPRAY_AXIS, "p": grid_p},
        "park_logit_offsets": park_offsets,
    }
    with open("fence_model.json", "w") as f:
        json.dump(out, f, separators=(",", ":"))
    print(f"\nWrote fence_model.json -- {len(rows)} BBE, {len(park_offsets)} parks")


if __name__ == "__main__":
    main()
