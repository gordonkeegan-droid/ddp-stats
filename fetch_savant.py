#!/usr/bin/env python3
"""
DDP Savant Stats Fetcher — hardened version
Always writes pitcher_stats.json (even if sparse) so the commit step never fails.
"""
import csv
import io
import json
import sys
import urllib.request
from datetime import datetime, timedelta, timezone

SEASON = datetime.now().year
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0 Safari/537.36",
    "Accept": "text/csv,application/csv,text/plain,*/*",
    "Referer": "https://baseballsavant.mlb.com/",
    "Accept-Language": "en-US,en;q=0.9",
}


def fetch_csv(url, label):
    try:
        req = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=45) as r:
            text = r.read().decode("utf-8")
        rows = list(csv.DictReader(io.StringIO(text)))
        print(f"[{label}] OK — {len(rows)} rows")
        if rows:
            print(f"[{label}] columns: {list(rows[0].keys())[:10]}")
        return rows
    except Exception as e:
        print(f"[{label}] FAILED: {e}", file=sys.stderr)
        return []


def find_col(row, candidates):
    lower = {k.lower().strip(): k for k in row.keys()}
    for c in candidates:
        if c in lower:
            return lower[c]
    return None


def to_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def main():
    pitchers = {}

    # 1. Expected statistics — xwOBA
    rows = fetch_csv(
        f"https://baseballsavant.mlb.com/leaderboard/expected_statistics"
        f"?type=pitcher&year={SEASON}&position=&team=&min=25&csv=true",
        "expected_statistics",
    )
    if rows:
        pid_col = find_col(rows[0], ["player_id", "mlbam_id", "pitcher"])
        xw_col = find_col(rows[0], ["est_woba", "xwoba", "expected_woba"])
        if pid_col:
            for row in rows:
                pid = str(row.get(pid_col, "")).strip().split(".")[0]
                if not pid.isdigit():
                    continue
                xw = to_float(row.get(xw_col)) if xw_col else None
                if xw:
                    pitchers.setdefault(pid, {})["xwoba"] = round(xw, 3)

    # 2. Custom leaderboard — GB%, whiff%, velocity
    rows = fetch_csv(
        f"https://baseballsavant.mlb.com/leaderboard/custom"
        f"?year={SEASON}&type=pitcher&filter=&min=25"
        f"&selections=gb_percent%2Cwhiff_percent%2Cfastball_avg_speed"
        f"&chart=false&x=gb_percent&y=gb_percent&r=no&chartType=beeswarm&csv=true",
        "custom_leaderboard",
    )
    if rows:
        pid_col = find_col(rows[0], ["player_id", "mlbam_id", "pitcher"])
        gb_col = find_col(rows[0], ["gb_percent", "gb%", "groundball_percent"])
        wh_col = find_col(rows[0], ["whiff_percent", "whiff%", "swinging_strike_percent"])
        sp_col = find_col(rows[0], ["fastball_avg_speed", "avg_speed", "ff_avg_speed", "avg_fastball_speed"])
        if pid_col:
            for row in rows:
                pid = str(row.get(pid_col, "")).strip().split(".")[0]
                if not pid.isdigit():
                    continue
                entry = pitchers.setdefault(pid, {})
                gb = to_float(row.get(gb_col)) if gb_col else None
                wh = to_float(row.get(wh_col)) if wh_col else None
                sp = to_float(row.get(sp_col)) if sp_col else None
                if gb is not None:
                    entry["gbRate"] = round(gb / 100 if gb > 1 else gb, 3)
                if wh is not None:
                    entry["whiffPct"] = round(wh / 100 if wh > 1 else wh, 3)
                if sp:
                    entry["avgVelo"] = round(sp, 1)

    # 3. Recent velocity — last 21 days bulk, grouped by pitcher
    start = (datetime.now() - timedelta(days=21)).strftime("%Y-%m-%d")
    end = datetime.now().strftime("%Y-%m-%d")
    rows = fetch_csv(
        f"https://baseballsavant.mlb.com/statcast_search/csv?all=true"
        f"&player_type=pitcher&group_by=name"
        f"&game_date_gt={start}&game_date_lt={end}"
        f"&type=details&minors=false",
        "recent_velocity",
    )
    if rows:
        pid_col = find_col(rows[0], ["player_id", "pitcher", "mlbam_id"])
        sp_col = find_col(rows[0], ["release_speed", "avg_speed", "effective_speed"])
        if pid_col:
            for row in rows:
                pid = str(row.get(pid_col, "")).strip().split(".")[0]
                if not pid.isdigit():
                    continue
                sp = to_float(row.get(sp_col)) if sp_col else None
                if sp and sp > 60:
                    pitchers.setdefault(pid, {})["recentVelo"] = round(sp, 1)

    # Velocity trend
    for pid, e in pitchers.items():
        if e.get("avgVelo") and e.get("recentVelo"):
            drop = round(e["avgVelo"] - e["recentVelo"], 1)
            e["veloDrop"] = drop
            e["veloTrend"] = "down" if drop >= 1.5 else "up" if drop <= -0.8 else "stable"

    # ALWAYS write the file — even sparse — so git add never fails
    out = {
        "updated": datetime.now(timezone.utc).isoformat(),
        "season": SEASON,
        "count": len(pitchers),
        "pitchers": pitchers,
    }
    with open("pitcher_stats.json", "w") as f:
        json.dump(out, f, separators=(",", ":"))

    print(f"\nWrote pitcher_stats.json — {len(pitchers)} pitchers")
    for stat in ["xwoba", "gbRate", "whiffPct", "avgVelo", "recentVelo"]:
        n = sum(1 for e in pitchers.values() if stat in e)
        print(f"  {stat}: {n}")

    if len(pitchers) < 50:
        print("WARNING: sparse data — Savant may be blocking or format changed", file=sys.stderr)
        # Do NOT exit(1) — a sparse file is better than a failed workflow


if __name__ == "__main__":
    main()

