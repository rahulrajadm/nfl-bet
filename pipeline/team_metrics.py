"""
Team metrics for the props model's opponent-defense and pace adjustments,
computed directly from team_game_logs (which already carries EPA split by
pass/rush and offensive play counts from pipeline/historical.py's play-by-play
pull — no separate advanced-stats endpoint needed, unlike wnba-bet's NBA Stats
API fallback).
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
from utils.db import get_conn
from pipeline.team_names import to_abbr

LEAGUE_AVG_PLAYS = 63.5   # approx NFL offensive plays per team per game
LEAGUE_AVG_SACKS_ALLOWED = 2.3   # approx NFL sacks allowed per team per game
RECENT_N = 8


def _team_rows(team: str, n: int, game_logs_df: pd.DataFrame | None = None) -> pd.DataFrame:
    abbr = to_abbr(team) or team
    if game_logs_df is not None:
        df = game_logs_df[game_logs_df["team"] == abbr]
        return df.sort_values("date", ascending=False).head(n)
    conn = get_conn()
    df = pd.read_sql(
        "SELECT * FROM team_game_logs WHERE team = ? AND season_type = 'REG' "
        "ORDER BY date DESC LIMIT ?",
        conn, params=(abbr, n),
    )
    conn.close()
    return df


def get_opp_def_epa(team: str, split: str, n: int = RECENT_N,
                     game_logs_df: pd.DataFrame | None = None) -> float | None:
    """split: 'pass' or 'rush'. Opponent's def_epa_{split} allowed over their
    last N games. None (not 0.0) when no data, so callers fall back to a
    league-neutral 1.0x multiplier rather than an artificial penalty."""
    col = f"def_epa_{split}"
    rows = _team_rows(team, n, game_logs_df)
    if rows.empty or col not in rows.columns or rows[col].isna().all():
        return None
    return round(float(rows[col].mean()), 4)


def get_def_adj(opp_def_epa: float | None) -> float:
    """Opponent's EPA/play allowed (in one split) → a multiplier on expected
    production. Negative EPA allowed = strong defense = harder environment.
    NFL EPA/play is roughly in [-0.3, +0.3] most weeks; dampened to a ±25%
    swing (same "don't let one signal dominate" reasoning as wnba-bet's
    get_def_rating_adj, tuned for EPA's scale instead of points-allowed)."""
    if opp_def_epa is None:
        return 1.0
    adj = 1.0 + 0.25 * (opp_def_epa / 0.15)
    return float(max(0.75, min(1.25, adj)))


def get_team_plays_pg(team: str, n: int = RECENT_N, game_logs_df: pd.DataFrame | None = None) -> float:
    rows = _team_rows(team, n, game_logs_df)
    if rows.empty or "plays_offense" not in rows.columns or rows["plays_offense"].isna().all():
        return LEAGUE_AVG_PLAYS
    return round(float(rows["plays_offense"].mean()), 2)


def get_game_pace_factor(team: str, opponent: str, game_logs_df: pd.DataFrame | None = None) -> float:
    """Expected play-volume pace for this matchup vs. league average — blends
    the team's own offensive pace with the opponent's (opponents that force
    long/quick drives shift play volume too). >1.0 = more plays = more
    counting-stat opportunity."""
    team_pace = get_team_plays_pg(team, game_logs_df=game_logs_df) / LEAGUE_AVG_PLAYS
    opp_pace = get_team_plays_pg(opponent, game_logs_df=game_logs_df) / LEAGUE_AVG_PLAYS
    return round((team_pace + opp_pace) / 2, 4)


def get_opp_sacks_allowed(opp_team: str, n: int = RECENT_N, game_logs_df: pd.DataFrame | None = None) -> float | None:
    """Opponent's sacks-allowed rate (from their OFFENSE's pass protection) —
    the opponent-adjustment axis for defensive Sacks props, since a pass
    rusher's expected sacks depends on how leaky the opposing line is, not
    how good the rusher's OWN defense is overall."""
    rows = _team_rows(opp_team, n, game_logs_df)
    if rows.empty or "off_sacks_allowed" not in rows.columns or rows["off_sacks_allowed"].isna().all():
        return None
    return round(float(rows["off_sacks_allowed"].mean()), 3)


def get_sacks_def_adj(opp_sacks_allowed: float | None) -> float:
    """Opponent's sacks-allowed rate -> a multiplier on a pass rusher's
    expected sacks this game. Dampened to a +/-30% swing (sacks are high-
    variance and a good pass rusher isn't purely a function of matchup)."""
    if opp_sacks_allowed is None:
        return 1.0
    adj = 1.0 + 0.6 * ((opp_sacks_allowed - LEAGUE_AVG_SACKS_ALLOWED) / LEAGUE_AVG_SACKS_ALLOWED)
    return float(max(0.70, min(1.30, adj)))


def get_rest_days(team: str, game_logs_df: pd.DataFrame | None = None) -> float:
    rows = _team_rows(team, 1, game_logs_df)
    if rows.empty:
        return 7.0
    try:
        from datetime import date
        last = pd.to_datetime(rows["date"].iloc[0]).date()
        return float((date.today() - last).days)
    except Exception:
        return 7.0


if __name__ == "__main__":
    teams = [("KC", "LV"), ("SF", "SEA"), ("BAL", "PIT"), ("DAL", "PHI")]
    print(f"{'Team':<6} {'Opp':<6} {'DefEPA-pass':>12} {'DefEPA-rush':>12} {'Pace':>8}")
    print("-" * 50)
    for team, opp in teams:
        pass_epa = get_opp_def_epa(opp, "pass")
        rush_epa = get_opp_def_epa(opp, "rush")
        pace = get_game_pace_factor(team, opp)
        print(f"{team:<6} {opp:<6} {pass_epa if pass_epa is not None else 'N/A':>12} "
              f"{rush_epa if rush_epa is not None else 'N/A':>12} {pace:>8.4f}")
