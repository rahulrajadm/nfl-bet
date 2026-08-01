"""
NFL game prediction engine.
Loads trained models and generates spread and totals predictions for a
matchup using rolling team EPA/success-rate, rest, division, and QB-out
signals — the same feature set models/train.py built, computed live instead
of from history.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import functools
from datetime import datetime, timezone
import pandas as pd
import joblib
from scipy.stats import norm
from utils.db import get_conn
from pipeline.team_names import to_abbr, is_div_game

MODELS_DIR = os.path.join(os.path.dirname(__file__), "../data/models")
ROLLING_WINDOWS = [4, 8]
STAT_COLS = ["off_epa_pass", "off_epa_rush", "def_epa_pass", "def_epa_rush",
             "off_success_rate", "def_success_rate", "points_for", "points_against"]


@functools.lru_cache(maxsize=1)
def load_models():
    sprd  = joblib.load(os.path.join(MODELS_DIR, "game_spread.pkl"))
    total = joblib.load(os.path.join(MODELS_DIR, "game_totals.pkl"))
    feats = joblib.load(os.path.join(MODELS_DIR, "game_feature_cols.pkl"))
    try:
        calib = joblib.load(os.path.join(MODELS_DIR, "game_calibration.pkl"))
    except Exception:
        calib = {"spread_std": 13.5, "totals_std": 10.0}
    # Single-threaded predict: XGBoost's OpenMP pool spawns a thread per visible
    # CPU on every predict; on Streamlit Cloud's small container, predicts fired
    # from short-lived Streamlit script threads segfault the whole process
    # (same failure mode documented in wnba-bet's CLAUDE.md).
    for m in (sprd, total):
        try:
            m.set_params(n_jobs=1)
        except Exception:
            pass
    return sprd, total, feats, calib


def get_team_rolling_stats(team_abbr: str, n: int, game_logs_df: pd.DataFrame | None = None) -> dict:
    """Last N regular-season games' rolling stats for a team, from SQLite or an
    in-memory DataFrame (cloud path). team_abbr must be the canonical nflverse
    abbreviation (pipeline.team_names.to_abbr) — team_game_logs is keyed by it."""
    if game_logs_df is not None:
        df = game_logs_df[game_logs_df["team"] == team_abbr]
        df = df.sort_values("date", ascending=False).head(n)
    else:
        conn = get_conn()
        df = pd.read_sql(
            "SELECT * FROM team_game_logs WHERE team = ? AND season_type = 'REG' "
            "ORDER BY date DESC LIMIT ?",
            conn, params=(team_abbr, n),
        )
        conn.close()

    if df.empty:
        return {}

    stats = {col: df[col].mean() for col in STAT_COLS}
    stats["win_pct"] = (df["result"] == "W").mean()
    stats["margin"] = (df["points_for"] - df["points_against"]).mean()
    stats["last_game_date"] = df["date"].max()
    stats["games_sample"] = len(df)
    return stats


def get_rest_days(team_abbr: str, upcoming_date: str, game_logs_df: pd.DataFrame | None = None) -> float:
    """Days between a team's most recent game and the upcoming one. Defaults
    to 7 (a normal week) if no prior game is on record — e.g. Week 1."""
    recent = get_team_rolling_stats(team_abbr, n=1, game_logs_df=game_logs_df)
    last_date = recent.get("last_game_date")
    if not last_date:
        return 7.0
    try:
        upcoming = pd.Timestamp(upcoming_date)
        last = pd.Timestamp(last_date)
        return max(float((upcoming - last).days), 1.0)
    except (ValueError, TypeError):
        return 7.0


def get_current_starter(team_abbr: str, player_logs_df: pd.DataFrame | None = None) -> str | None:
    """The QB with the most pass attempts in the most recent season on record
    for this team — a snapshot, not a mid-season-change-aware lookup."""
    if player_logs_df is not None:
        df = player_logs_df[(player_logs_df["team"] == team_abbr) & (player_logs_df["position"] == "QB")]
    else:
        conn = get_conn()
        latest_season = conn.execute(
            "SELECT MAX(season) FROM player_game_logs WHERE team = ?", (team_abbr,)
        ).fetchone()[0]
        if latest_season is None:
            conn.close()
            return None
        df = pd.read_sql(
            "SELECT player_name, attempts FROM player_game_logs "
            "WHERE team = ? AND position = 'QB' AND season = ?",
            conn, params=(team_abbr, latest_season),
        )
        conn.close()

    if df.empty or df["attempts"].isna().all():
        return None
    totals = df.groupby("player_name")["attempts"].sum()
    return totals.idxmax()


def get_qb_out_flag(team_full_name: str, team_abbr: str, injury_flags: dict | None = None,
                     player_logs_df: pd.DataFrame | None = None) -> int:
    """1 if this team's current starting QB shows up in the live injury report
    as Out/Doubtful+, else 0. injury_flags is pipeline.injuries.fetch_injury_flags()'s
    return value ({team_name: [{"name","status",...}]}) — pass it in once per
    call site rather than re-fetching per team (ESPN lookups are the slow part)."""
    starter = get_current_starter(team_abbr, player_logs_df=player_logs_df)
    if not starter or not injury_flags:
        return 0
    from utils.names import normalize_name
    starter_key = normalize_name(starter)
    for flag in injury_flags.get(team_full_name, []):
        status = str(flag.get("status", "")).lower()
        if normalize_name(flag.get("name", "")) == starter_key and \
           any(tag in status for tag in ("out", "doubtful", "ir", "pup")):
            return 1
    return 0


def build_matchup_vector(home_team: str, away_team: str, feat_cols: list[str],
                          game_date: str | None = None, game_logs_df: pd.DataFrame | None = None,
                          player_logs_df: pd.DataFrame | None = None,
                          injury_flags: dict | None = None) -> pd.DataFrame | None:
    """Build a single-row feature vector for a matchup. home_team/away_team
    are full names (as odds/schedule sources use); converted to nflverse
    abbreviations internally since team_game_logs is keyed by those."""
    home_abbr, away_abbr = to_abbr(home_team), to_abbr(away_team)
    game_date = game_date or datetime.now(timezone.utc).date().isoformat()

    home_n8 = get_team_rolling_stats(home_abbr, n=8, game_logs_df=game_logs_df)
    away_n8 = get_team_rolling_stats(away_abbr, n=8, game_logs_df=game_logs_df)
    if not home_n8 or not away_n8:
        return None
    home_n4 = get_team_rolling_stats(home_abbr, n=4, game_logs_df=game_logs_df) or home_n8
    away_n4 = get_team_rolling_stats(away_abbr, n=4, game_logs_df=game_logs_df) or away_n8
    windows = {4: (home_n4, away_n4), 8: (home_n8, away_n8)}

    row = {}
    for col in STAT_COLS:
        for w in ROLLING_WINDOWS:
            h, a = windows[w]
            row[f"home_{col}_last{w}"] = h.get(col, 0)
            row[f"away_{col}_last{w}"] = a.get(col, 0)
    row["home_win_pct_last8"] = home_n8.get("win_pct", 0.5)
    row["away_win_pct_last8"] = away_n8.get("win_pct", 0.5)
    row["home_margin_last4"] = home_n4.get("margin", 0)
    row["away_margin_last4"] = away_n4.get("margin", 0)
    row["home_margin_last8"] = home_n8.get("margin", 0)
    row["away_margin_last8"] = away_n8.get("margin", 0)

    home_rest = get_rest_days(home_abbr, game_date, game_logs_df=game_logs_df)
    away_rest = get_rest_days(away_abbr, game_date, game_logs_df=game_logs_df)
    row["home_rest_days"] = home_rest
    row["away_rest_days"] = away_rest
    row["home_is_short_week"] = int(home_rest <= 5)
    row["away_is_short_week"] = int(away_rest <= 5)
    row["home_is_long_week"] = int(home_rest >= 10)
    row["away_is_long_week"] = int(away_rest >= 10)
    row["rest_advantage"] = home_rest - away_rest
    row["home_field"] = 1
    row["div_game"] = int(is_div_game(home_abbr, away_abbr))
    row["home_qb_out"] = get_qb_out_flag(home_team, home_abbr, injury_flags, player_logs_df)
    row["away_qb_out"] = get_qb_out_flag(away_team, away_abbr, injury_flags, player_logs_df)

    df = pd.DataFrame([row])
    for col in feat_cols:
        if col not in df.columns:
            df[col] = 0
    return df[feat_cols]


def predict_game(home_team: str, away_team: str, game_date: str | None = None,
                  game_logs_df: pd.DataFrame | None = None, player_logs_df: pd.DataFrame | None = None,
                  injury_flags: dict | None = None) -> dict | None:
    """Generate spread and totals predictions for a matchup. Win probability
    is derived from the spread model (norm.cdf(pred_diff / spread_std)) —
    the single source of truth, same reasoning as wnba-bet: a separate win
    classifier can produce a win% that contradicts the cover%."""
    try:
        sprd_model, total_model, feat_cols, calib = load_models()
    except Exception:
        return None

    X = build_matchup_vector(home_team, away_team, feat_cols, game_date=game_date,
                              game_logs_df=game_logs_df, player_logs_df=player_logs_df,
                              injury_flags=injury_flags)
    if X is None:
        return None

    SPREAD_STD = float(calib.get("spread_std", 13.5))
    TOTALS_STD = float(calib.get("totals_std", 10.0))
    pred_diff = float(sprd_model.predict(X)[0])
    home_win_prob = float(norm.cdf(pred_diff / SPREAD_STD))
    away_win_prob = 1.0 - home_win_prob

    pred_total = float(total_model.predict(X)[0])

    return {
        "home_team":  home_team,
        "away_team":  away_team,
        "home_win_prob": round(home_win_prob, 4),
        "away_win_prob": round(away_win_prob, 4),
        "pred_diff":  round(pred_diff, 1),
        "pred_total": round(pred_total, 1),
        "spread_std": SPREAD_STD,
        "totals_std": TOTALS_STD,
        "home_qb_out": bool(X["home_qb_out"].iloc[0]) if "home_qb_out" in X.columns else False,
        "away_qb_out": bool(X["away_qb_out"].iloc[0]) if "away_qb_out" in X.columns else False,
    }


def prob_cover_spread(pred_diff: float, spread_line: float, std: float = 13.5) -> float:
    """P(home team covers spread_line) given predicted point differential.

    Home covers if actual_diff > -spread_line (e.g. home -6.5 needs to win by
    6.5+; home +3.5 covers as long as it doesn't lose by more than 3.5).
    """
    return float(norm.cdf(pred_diff + spread_line, 0, std))


def prob_over_total(pred_total: float, total_line: float, std: float = 10.0) -> float:
    """P(total goes over total_line) given predicted total."""
    return float(1 - norm.cdf(total_line, pred_total, std))


if __name__ == "__main__":
    from pipeline.schedule import get_week_games
    from pipeline.injuries import fetch_injury_flags

    games = get_week_games()
    print(f"\nGame predictions for {len(games)} games:\n")
    teams = sorted({g["home_team"] for g in games} | {g["away_team"] for g in games})
    flags = fetch_injury_flags(teams) if teams else {}

    for g in games:
        result = predict_game(g["home_team"], g["away_team"], game_date=g.get("date"), injury_flags=flags)
        if result:
            print(f"{g['away_team']} @ {g['home_team']}")
            print(f"  Win prob: Home {result['home_win_prob']:.1%} | Away {result['away_win_prob']:.1%}")
            print(f"  Diff:  Home predicted by {result['pred_diff']:+.1f}")
            print(f"  Total: Predicted {result['pred_total']:.1f} pts")
            if result["home_qb_out"] or result["away_qb_out"]:
                print(f"  QB-out: home={result['home_qb_out']} away={result['away_qb_out']}")
        else:
            print(f"{g['away_team']} @ {g['home_team']} — insufficient data")
