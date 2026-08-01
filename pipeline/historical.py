"""
Pulls historical NFL team and player game logs via nfl_data_py (nflverse) and
stores them in SQLite. Run to (re)seed the database — safe to re-run mid-season,
each pull deletes and reinserts only the affected seasons.

nfl_data_py wraps nflverse's published parquet releases (github.com/nflverse) —
free, no key. Column names below match the package as of early 2026; if a pull
prints a KeyError/column warning, the upstream schema moved and this needs a
quick column-name fix (same class of maintenance as pybaseball breakage in
mlb-bet).

Regular season + postseason only — preseason games are backup-heavy and not
representative for training (pipeline/schedule.py separately covers live
preseason schedule/injuries for pipeline smoke-testing before Week 1).
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import nfl_data_py as nfl
from utils.db import get_conn, init_db
from utils.dates import today_local

SEASONS = list(range(today_local().year - 6, today_local().year + 1))
_PLAYED_GAME_TYPES = {"REG", "WC", "DIV", "CON", "SB"}


def _season_type(game_type: str) -> str:
    return "REG" if game_type == "REG" else "POST"


def pull_schedules_and_games():
    """games table: one row per game, from nfl.import_schedules()."""
    conn = get_conn()
    print(f"Pulling NFL schedules {SEASONS[0]}-{SEASONS[-1]}...")
    sched = nfl.import_schedules(SEASONS)
    sched = sched[sched["game_type"].isin(_PLAYED_GAME_TYPES)].copy()

    c = conn.cursor()
    for _, r in sched.iterrows():
        c.execute("""
            INSERT OR REPLACE INTO games
            (game_id, season, week, season_type, date, home_team, away_team,
             home_score, away_score, game_time, stadium, roof, surface, div_game)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            r["game_id"], int(r["season"]), int(r["week"]), _season_type(r["game_type"]),
            r.get("gameday"), r["home_team"], r["away_team"],
            None if pd.isna(r.get("home_score")) else int(r["home_score"]),
            None if pd.isna(r.get("away_score")) else int(r["away_score"]),
            f"{r.get('gameday', '')} {r.get('gametime', '')}".strip(),
            r.get("stadium"), r.get("roof"), r.get("surface"),
            int(r.get("div_game") or 0),
        ))
    conn.commit()
    conn.close()
    print(f"  {len(sched)} games")
    return sched


def _team_game_base_rows(sched: pd.DataFrame) -> pd.DataFrame:
    """One row per team per *played* game (both home and away perspective)."""
    played = sched[sched["home_score"].notna() & sched["away_score"].notna()]
    rows = []
    for _, r in played.iterrows():
        season_type = _season_type(r["game_type"])
        hs, aws = int(r["home_score"]), int(r["away_score"])
        rows.append(dict(
            game_id=r["game_id"], season=int(r["season"]), week=int(r["week"]),
            season_type=season_type, date=r.get("gameday"),
            team=r["home_team"], opponent=r["away_team"], is_home=1,
            points_for=hs, points_against=aws,
            result="W" if hs > aws else ("L" if hs < aws else "T"),
            rest_days=r.get("home_rest"),
        ))
        rows.append(dict(
            game_id=r["game_id"], season=int(r["season"]), week=int(r["week"]),
            season_type=season_type, date=r.get("gameday"),
            team=r["away_team"], opponent=r["home_team"], is_home=0,
            points_for=aws, points_against=hs,
            result="W" if aws > hs else ("L" if aws < hs else "T"),
            rest_days=r.get("away_rest"),
        ))
    return pd.DataFrame(rows)


def _epa_by_team_game(season: int) -> pd.DataFrame:
    """Offense EPA/success rate (pass & rush split) and defense EPA/success
    allowed, per team per game, from play-by-play. This is the feature nflverse
    unlocks that a box-score-only pull (like wnba-bet's) can't get: NFL team
    strength is far better captured by EPA/play than by points or yards alone.
    """
    cols = ["game_id", "season", "week", "posteam", "defteam", "epa", "success", "play_type"]
    pbp = nfl.import_pbp_data([season], columns=cols, downcast=True)
    plays = pbp[pbp["play_type"].isin(["pass", "run"]) & pbp["epa"].notna() & pbp["posteam"].notna()].copy()

    off = (plays.groupby(["game_id", "posteam", "play_type"])
                 .agg(epa=("epa", "mean"), success=("success", "mean"), n=("epa", "size"))
                 .reset_index())
    off_pass = off[off["play_type"] == "pass"].drop(columns="play_type").rename(
        columns={"posteam": "team", "epa": "off_epa_pass", "success": "off_success_pass", "n": "plays_pass"})
    off_rush = off[off["play_type"] == "run"].drop(columns="play_type").rename(
        columns={"posteam": "team", "epa": "off_epa_rush", "success": "off_success_rush", "n": "plays_rush"})

    deff = (plays.groupby(["game_id", "defteam", "play_type"])
                  .agg(epa=("epa", "mean"), success=("success", "mean"))
                  .reset_index())
    def_pass = deff[deff["play_type"] == "pass"].drop(columns="play_type").rename(
        columns={"defteam": "team", "epa": "def_epa_pass", "success": "def_success_pass"})
    def_rush = deff[deff["play_type"] == "run"].drop(columns="play_type").rename(
        columns={"defteam": "team", "epa": "def_epa_rush", "success": "def_success_rush"})
    plays_def = (plays.groupby(["game_id", "defteam"]).size()
                       .reset_index(name="plays_defense").rename(columns={"defteam": "team"}))

    merged = (off_pass.merge(off_rush, on=["game_id", "team"], how="outer")
                       .merge(def_pass, on=["game_id", "team"], how="outer")
                       .merge(def_rush, on=["game_id", "team"], how="outer")
                       .merge(plays_def, on=["game_id", "team"], how="left"))
    merged["off_success_rate"] = merged[["off_success_pass", "off_success_rush"]].mean(axis=1)
    merged["def_success_rate"] = merged[["def_success_pass", "def_success_rush"]].mean(axis=1)
    merged["plays_offense"] = merged[["plays_pass", "plays_rush"]].sum(axis=1)

    keep = ["game_id", "team", "off_epa_pass", "off_epa_rush", "def_epa_pass", "def_epa_rush",
            "off_success_rate", "def_success_rate", "plays_offense", "plays_defense"]
    return merged[keep]


def pull_team_game_logs(sched: pd.DataFrame):
    conn = get_conn()
    print("Pulling NFL team game logs + play-by-play EPA (slow — one season at a time)...")
    base = _team_game_base_rows(sched)

    epa_frames = []
    for season in sorted(base["season"].unique()):
        print(f"  EPA for {season}...")
        try:
            epa_frames.append(_epa_by_team_game(int(season)))
        except Exception as e:
            print(f"    Warning: EPA pull failed for {season}: {e}")
    epa = pd.concat(epa_frames, ignore_index=True) if epa_frames else pd.DataFrame()

    merged = base.merge(epa, on=["game_id", "team"], how="left") if len(epa) else base

    seasons = tuple(int(s) for s in merged["season"].unique())
    if seasons:
        conn.execute(f"DELETE FROM team_game_logs WHERE season IN ({','.join('?' * len(seasons))})", seasons)
    conn.commit()

    cols = ["game_id", "season", "week", "season_type", "date", "team", "opponent", "is_home",
            "points_for", "points_against", "result", "off_epa_pass", "off_epa_rush",
            "def_epa_pass", "def_epa_rush", "off_success_rate", "def_success_rate",
            "plays_offense", "plays_defense", "rest_days"]
    merged[cols].to_sql("team_game_logs", conn, if_exists="append", index=False)
    conn.commit()
    conn.close()
    print(f"  {len(merged)} team-game rows")


def pull_team_stats():
    """Season-level aggregates, computed from team_game_logs (already pulled)."""
    conn = get_conn()
    print("Aggregating NFL team season stats...")
    logs = pd.read_sql("SELECT * FROM team_game_logs", conn)
    if logs.empty:
        print("  No team_game_logs yet — run pull_team_game_logs() first.")
        conn.close()
        return

    agg = logs.groupby(["season", "team"]).agg(
        gp=("game_id", "count"),
        w=("result", lambda s: (s == "W").sum()),
        l=("result", lambda s: (s == "L").sum()),
        t=("result", lambda s: (s == "T").sum()),
        pts_pg=("points_for", "mean"),
        opp_pts_pg=("points_against", "mean"),
        off_epa_pass_pg=("off_epa_pass", "mean"),
        off_epa_rush_pg=("off_epa_rush", "mean"),
        def_epa_pass_pg=("def_epa_pass", "mean"),
        def_epa_rush_pg=("def_epa_rush", "mean"),
    ).reset_index()

    seasons = tuple(int(s) for s in agg["season"].unique())
    if seasons:
        conn.execute(f"DELETE FROM team_stats WHERE season IN ({','.join('?' * len(seasons))})", seasons)
    conn.commit()
    agg.to_sql("team_stats", conn, if_exists="append", index=False)
    conn.commit()
    conn.close()
    print(f"  {len(agg)} team-season rows")


def pull_player_game_logs():
    """Pulled one season at a time and skipped on failure — nflverse's
    player_stats release has lagged the schedules/pbp releases by a season
    before (seen 2026-08: 2025 schedules+pbp were live, player_stats wasn't
    republished yet), so a single missing year must not abort the whole pull."""
    conn = get_conn()
    print(f"Pulling NFL player weekly stats {SEASONS[0]}-{SEASONS[-1]}...")
    frames = []
    for season in SEASONS:
        try:
            frames.append(nfl.import_weekly_data([season]))
        except Exception as e:
            print(f"  Warning: player weekly stats unavailable for {season}: {e}")
    if not frames:
        print("  No player weekly stats pulled.")
        conn.close()
        return
    weekly = pd.concat(frames, ignore_index=True)
    weekly = weekly[weekly.get("season_type", "REG") == "REG"] if "season_type" in weekly.columns else weekly

    # nflverse weekly data carries BOTH "player_name" (abbreviated, "P.Mahomes")
    # and "player_display_name" (full, "Patrick Mahomes"). We want the full
    # name — PrizePicks/Underdog send full names, and utils/names.py's
    # matching needs the real surname, not an abbreviation — so the raw
    # "player_name" column is dropped before renaming, or a plain
    # dict-based .rename() would silently collide the two into one column
    # and keep whichever pandas resolves first (it kept the abbreviated one
    # when this shipped without the drop — verify with the smoke block below
    # after any nfl_data_py upgrade).
    if "player_name" in weekly.columns and "player_display_name" in weekly.columns:
        weekly = weekly.drop(columns="player_name")

    rename = {
        "player_id": "player_id", "player_display_name": "player_name",
        "position": "position", "recent_team": "team", "opponent_team": "opponent",
        "season": "season", "week": "week",
        "completions": "completions", "attempts": "attempts",
        "passing_yards": "passing_yards", "passing_tds": "passing_tds",
        "interceptions": "interceptions",
        "rushing_yards": "rushing_yards", "rushing_tds": "rushing_tds", "carries": "carries",
        "receptions": "receptions", "targets": "targets",
        "receiving_yards": "receiving_yards", "receiving_tds": "receiving_tds",
        "fumbles_lost": "fumbles_lost",
        "fantasy_points": "fantasy_points",
    }
    present = {src: dst for src, dst in rename.items() if src in weekly.columns}
    missing = [src for src in rename if src not in weekly.columns]
    if missing:
        print(f"  Note: columns not found in this nfl_data_py version (left blank): {missing}")

    df = weekly.rename(columns=present)
    df["season_type"] = "REG"
    df["game_id"] = None  # weekly data isn't keyed by game_id; joined via season/week/team downstream

    keep = ["season", "week", "season_type", "game_id", "player_id", "player_name", "position",
            "team", "opponent", "passing_yards", "passing_tds", "interceptions", "completions",
            "attempts", "rushing_yards", "rushing_tds", "carries", "receiving_yards",
            "receiving_tds", "receptions", "targets", "fumbles_lost", "fantasy_points"]
    for col in keep:
        if col not in df.columns:
            df[col] = None
    df = df[keep]

    seasons = tuple(int(s) for s in df["season"].dropna().unique())
    if seasons:
        conn.execute(f"DELETE FROM player_game_logs WHERE season IN ({','.join('?' * len(seasons))})", seasons)
    conn.commit()
    df.to_sql("player_game_logs", conn, if_exists="append", index=False)
    conn.commit()
    conn.close()
    print(f"  {len(df)} player-week rows")


def pull_injuries():
    """Historical weekly injury reports (practice + game status) via nfl_data_py.
    Feeds the QB-out-flag training feature in models/train.py. The LIVE
    equivalent for this week's games is pipeline/injuries.py (ESPN scoreboard/
    roster) — different source because nfl_data_py's injuries release lags a
    week or two behind the current slate."""
    conn = get_conn()
    print(f"Pulling NFL injury reports {SEASONS[0]}-{SEASONS[-1]}...")
    frames = []
    for season in SEASONS:
        try:
            frames.append(nfl.import_injuries([season]))
        except Exception as e:
            print(f"  Warning: injury report unavailable for {season}: {e}")
    if not frames:
        print("  No injury reports pulled.")
        conn.close()
        return
    inj = pd.concat(frames, ignore_index=True)

    rename = {
        "season": "season", "week": "week", "team": "team",
        "full_name": "player_name", "position": "position",
        "practice_status": "practice_status", "report_status": "game_status",
    }
    present = {src: dst for src, dst in rename.items() if src in inj.columns}
    df = inj.rename(columns=present)
    df["fetched_at"] = None
    keep = ["fetched_at", "season", "week", "team", "player_name", "position",
            "practice_status", "game_status"]
    for col in keep:
        if col not in df.columns:
            df[col] = None
    df = df[keep]

    seasons = tuple(int(s) for s in df["season"].dropna().unique())
    if seasons:
        conn.execute(f"DELETE FROM injuries WHERE season IN ({','.join('?' * len(seasons))})", seasons)
    conn.commit()
    df.to_sql("injuries", conn, if_exists="append", index=False)
    conn.commit()
    conn.close()
    print(f"  {len(df)} injury-report rows")


if __name__ == "__main__":
    init_db()
    sched = pull_schedules_and_games()
    pull_team_game_logs(sched)
    pull_team_stats()
    pull_player_game_logs()
    pull_injuries()
    print("\nHistorical data pull complete.")
