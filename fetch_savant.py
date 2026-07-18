#!/usr/bin/env python3
"""
DDP Savant Stats Fetcher — v3
All data now comes from the statcast_search endpoint, which is the only
Savant endpoint that reliably serves GitHub Actions runners (the /leaderboard/*
endpoints block datacenter traffic).

Provides per pitcher: xwoba, whiffPct, avgVelo, recentVelo, veloDrop, veloTrend, gbRate
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
    "Referer": "https://baseballsavant.mlb.com/statcast_search",
    "Accept-Language": "en-US,en;q=0.9",
}
SEASON_START = f"{SEASON}-03-15"
TODAY = datetime.now().strftime("%Y-%m-%d")


def fetch_csv(url, label):
    try:
        req = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=60) as r:
            text = r.read().decode("utf-8")
        rows = list(csv.DictReader(io.StringIO(text)))
        print(f"[{label}] OK — {len(rows)} rows")
        if rows:
            print(f"[{label}] columns: {list(rows[0].keys())[:14]}")
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


def valid_pid(raw):
    """MLBAM player IDs are 6-7 digits. Rejects junk like '2026'."""
    pid = str(raw).strip().split(".")[0]
    return pid if pid.isdigit() and 100000 <= int(pid) <= 9999999 else None


def search_agg(label, extra=""):
    """statcast_search aggregate grouped by pitcher."""
    url = (
        f"https://baseballsavant.mlb.com/statcast_search/csv?all=true"
        f"&player_type=pitcher&group_by=name"
        f"&game_date_gt={SEASON_START}&game_date_lt={TODAY}"
        f"&type=details&minors=false{extra}"
    )
    return fetch_csv(url, label)


def main():
    pitchers = {}

    # ── 1. Season aggregates: xwOBA, whiff%, velocity ────────────────────────
    rows = search_agg("season_aggregate")
    if rows:
        pid_col = find_col(rows[0], ["player_id", "pitcher", "mlbam_id"])
        xw_col = find_col(rows[0], ["xwoba", "estimated_woba_using_speedangle", "est_woba"])
        wf_col = find_col(rows[0], ["whiffs", "whiff"])
        sw_col = find_col(rows[0], ["swings", "swing"])
        sp_col = find_col(rows[0], ["velocity", "release_speed", "avg_speed", "effective_speed"])
        n_col = find_col(rows[0], ["pitches", "total_pitches", "pitch_count"])
        if pid_col:
            for row in rows:
                pid = valid_pid(row.get(pid_col))
                if not pid:
                    continue
                npitch = to_float(row.get(n_col)) if n_col else None
                if npitch is not None and npitch < 150:
                    continue  # skip tiny samples (~under 10 IP)
                e = pitchers.setdefault(pid, {})
                xw = to_float(row.get(xw_col)) if xw_col else None
                if xw and 0.150 <= xw <= 0.500:
                    e["xwoba"] = round(xw, 3)
                wf = to_float(row.get(wf_col)) if wf_col else None
                sw = to_float(row.get(sw_col)) if sw_col else None
                if wf is not None and sw and sw > 50:
                    e["whiffPct"] = round(wf / sw, 3)
                sp = to_float(row.get(sp_col)) if sp_col else None
                if sp and sp > 60:
                    e["avgVelo"] = round(sp, 1)

    # ── 2. Ground ball rate: two filtered queries ────────────────────────────
    # GB count per pitcher / all batted balls per pitcher
    # Savant batted-ball filter encoding: "ground\.\.ball" → ground%5C.%5C.ball
    gb_rows = search_agg("gb_only", "&hfBBT=ground%5C.%5C.ball%7C")
    bip_rows = search_agg("all_bip", "&hfBBT=ground%5C.%5C.ball%7Cline%5C.%5C.drive%7Cfly%5C.%5C.ball%7Cpopup%7C")
    if gb_rows and bip_rows:
        def counts(rows):
            out = {}
            pid_col = find_col(rows[0], ["player_id", "pitcher", "mlbam_id"])
            n_col = find_col(rows[0], ["pitches", "total_pitches", "abs", "pa"])
            if not pid_col or not n_col:
                return out
            for row in rows:
                pid = valid_pid(row.get(pid_col))
                n = to_float(row.get(n_col))
                if pid and n:
                    out[pid] = n
            return out
        gb_map = counts(gb_rows)
        bip_map = counts(bip_rows)
        applied = 0
        for pid, bip in bip_map.items():
            gb = gb_map.get(pid, 0)
            if bip >= 40 and 0 <= gb <= bip:  # sanity: need 40+ batted balls
                rate = gb / bip
                if 0.10 <= rate <= 0.75:  # sanity range for GB%
                    pitchers.setdefault(pid, {})["gbRate"] = round(rate, 3)
                    applied += 1
        print(f"[gb_rate] applied to {applied} pitchers")

    # ── 3. Recent velocity (last 21 days) ────────────────────────────────────
    start = (datetime.now() - timedelta(days=21)).strftime("%Y-%m-%d")
    url = (
        f"https://baseballsavant.mlb.com/statcast_search/csv?all=true"
        f"&player_type=pitcher&group_by=name"
        f"&game_date_gt={start}&game_date_lt={TODAY}"
        f"&type=details&minors=false"
    )
    rows = fetch_csv(url, "recent_velocity")
    if rows:
        pid_col = find_col(rows[0], ["player_id", "pitcher", "mlbam_id"])
        sp_col = find_col(rows[0], ["velocity", "release_speed", "avg_speed", "effective_speed"])
        if pid_col:
            for row in rows:
                pid = valid_pid(row.get(pid_col))
                if not pid:
                    continue
                sp = to_float(row.get(sp_col)) if sp_col else None
                if sp and sp > 60:
                    pitchers.setdefault(pid, {})["recentVelo"] = round(sp, 1)

    # ── 4. Velocity trend ────────────────────────────────────────────────────
    for pid, e in pitchers.items():
        if e.get("avgVelo") and e.get("recentVelo"):
            drop = round(e["avgVelo"] - e["recentVelo"], 1)
            e["veloDrop"] = drop
            e["veloTrend"] = "down" if drop >= 1.5 else "up" if drop <= -0.8 else "stable"

    # ── Write output (always) ────────────────────────────────────────────────
    out = {
        "updated": datetime.now(timezone.utc).isoformat(),
        "season": SEASON,
        "count": len(pitchers),
        "pitchers": pitchers,
    }
    with open("pitcher_stats.json", "w") as f:
        json.dump(out, f, separators=(",", ":"))

    print(f"\nWrote pitcher_stats.json — {len(pitchers)} pitchers")
    for stat in ["xwoba", "whiffPct", "gbRate", "avgVelo", "recentVelo", "veloTrend"]:
        n = sum(1 for e in pitchers.values() if stat in e)
        print(f"  {stat}: {n}")

    if sum(1 for e in pitchers.values() if "xwoba" in e) < 50:
        print("WARNING: xwOBA sparse — check season_aggregate column log above", file=sys.stderr)


if __name__ == "__main__":
    main()
