#!/usr/bin/env python3
"""
DDP Savant Stats Fetcher — v4 (pitch-level)
Pulls raw pitch-by-pitch data from statcast_search in 5-day chunks and computes
season stats per pitcher using Savant's own methodology. No ambiguous aggregate
columns — every stat is derived from documented pitch-level fields:

  xwOBA   = sum(est_woba or woba_value over PA-ending rows) / count(woba_denom=1)
  whiff%  = swinging strikes / total swings
  GB%     = ground balls / all batted balls
  velo    = mean fastball release_speed (FF/SI/FT), season + last 21 days
"""
import csv
import io
import json
import sys
import time
import urllib.request
from datetime import datetime, date, timedelta, timezone

SEASON = datetime.now().year
SEASON_START = date(SEASON, 3, 15)
TODAY = date.today()
RECENT_CUTOFF = TODAY - timedelta(days=21)
CHUNK_DAYS = 5  # stay under Savant's per-query row cap

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0 Safari/537.36",
    "Accept": "text/csv,*/*",
    "Referer": "https://baseballsavant.mlb.com/statcast_search",
}

SWING_DESCS = {
    "swinging_strike", "swinging_strike_blocked", "foul", "foul_tip",
    "hit_into_play", "foul_bunt", "missed_bunt", "bunt_foul_tip",
}
WHIFF_DESCS = {"swinging_strike", "swinging_strike_blocked", "missed_bunt"}
FASTBALLS = {"FF", "SI", "FT", "FA"}


def to_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def fetch_chunk(d1, d2):
    url = (
        "https://baseballsavant.mlb.com/statcast_search/csv?all=true"
        f"&player_type=pitcher&type=details&minors=false"
        f"&game_date_gt={d1}&game_date_lt={d2}"
    )
    for attempt in (1, 2):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=120) as r:
                return r.read().decode("utf-8")
        except Exception as e:
            print(f"  chunk {d1}..{d2} attempt {attempt} failed: {e}", file=sys.stderr)
            time.sleep(3)
    return None


def main():
    # Per-pitcher accumulators
    acc = {}

    def get(pid):
        if pid not in acc:
            acc[pid] = {
                "woba_num": 0.0, "woba_den": 0,
                "swings": 0, "whiffs": 0,
                "gb": 0, "bip": 0,
                "fb_velo_sum": 0.0, "fb_velo_n": 0,
                "fb_velo_sum_recent": 0.0, "fb_velo_n_recent": 0,
            }
        return acc[pid]

    total_pitches = 0
    d = SEASON_START
    while d <= TODAY:
        d2 = min(d + timedelta(days=CHUNK_DAYS - 1), TODAY)
        text = fetch_chunk(d.isoformat(), d2.isoformat())
        d = d2 + timedelta(days=1)
        if not text or "," not in text:
            continue
        reader = csv.DictReader(io.StringIO(text))
        n = 0
        for row in reader:
            n += 1
            pid = str(row.get("pitcher", "")).strip().split(".")[0]
            if not (pid.isdigit() and int(pid) >= 100000):
                continue
            a = get(pid)

            # xwOBA accumulation (Savant methodology)
            denom = to_float(row.get("woba_denom"))
            if denom and denom > 0:
                est = to_float(row.get("estimated_woba_using_speedangle"))
                actual = to_float(row.get("woba_value"))
                val = est if est is not None else actual
                if val is not None:
                    a["woba_num"] += val
                    a["woba_den"] += 1

            # Whiff tracking
            desc = (row.get("description") or "").strip()
            if desc in SWING_DESCS:
                a["swings"] += 1
                if desc in WHIFF_DESCS:
                    a["whiffs"] += 1

            # Batted ball type
            bb = (row.get("bb_type") or "").strip()
            if bb:
                a["bip"] += 1
                if bb == "ground_ball":
                    a["gb"] += 1

            # Fastball velocity
            pt = (row.get("pitch_type") or "").strip()
            if pt in FASTBALLS:
                sp = to_float(row.get("release_speed"))
                if sp and sp > 60:
                    a["fb_velo_sum"] += sp
                    a["fb_velo_n"] += 1
                    gd = (row.get("game_date") or "")[:10]
                    try:
                        if date.fromisoformat(gd) >= RECENT_CUTOFF:
                            a["fb_velo_sum_recent"] += sp
                            a["fb_velo_n_recent"] += 1
                    except ValueError:
                        pass
        total_pitches += n
        print(f"[{d2}] +{n} pitches (running total {total_pitches})")
        time.sleep(0.6)  # be polite

    # ── Reduce accumulators to final stats ───────────────────────────────────
    pitchers = {}
    for pid, a in acc.items():
        e = {}
        if a["woba_den"] >= 70:  # ~25+ batters faced
            e["xwoba"] = round(a["woba_num"] / a["woba_den"], 3)
        if a["swings"] >= 80:
            e["whiffPct"] = round(a["whiffs"] / a["swings"], 3)
        if a["bip"] >= 40:
            e["gbRate"] = round(a["gb"] / a["bip"], 3)
        if a["fb_velo_n"] >= 50:
            e["avgVelo"] = round(a["fb_velo_sum"] / a["fb_velo_n"], 1)
        if a["fb_velo_n_recent"] >= 20:
            e["recentVelo"] = round(a["fb_velo_sum_recent"] / a["fb_velo_n_recent"], 1)
        if e.get("avgVelo") and e.get("recentVelo"):
            drop = round(e["avgVelo"] - e["recentVelo"], 1)
            e["veloDrop"] = drop
            e["veloTrend"] = "down" if drop >= 1.5 else "up" if drop <= -0.8 else "stable"
        if e:
            pitchers[pid] = e

    out = {
        "updated": datetime.now(timezone.utc).isoformat(),
        "season": SEASON,
        "count": len(pitchers),
        "pitchers": pitchers,
    }
    with open("pitcher_stats.json", "w") as f:
        json.dump(out, f, separators=(",", ":"))

    print(f"\nWrote pitcher_stats.json — {len(pitchers)} pitchers from {total_pitches} pitches")
    for stat in ["xwoba", "whiffPct", "gbRate", "avgVelo", "recentVelo", "veloTrend"]:
        n = sum(1 for e in pitchers.values() if stat in e)
        print(f"  {stat}: {n}")
    if len(pitchers) < 50:
        print("WARNING: sparse output — check chunk fetch errors above", file=sys.stderr)


if __name__ == "__main__":
    main()
