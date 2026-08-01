"""
Trains NFL game prediction models (spread, totals — moneyline is derived from
spread at predict time, not trained separately; see models/game.py) from
historical data pulled by pipeline/historical.py.

Features: team rolling EPA/success-rate (pass & rush split, 4- and 8-game
windows), rest days (from nflverse, so byes are handled correctly — not a
naive date-diff), short/long-week flags, division-game flag, and a starting-
QB-out flag (the single largest swing factor in NFL win probability — no
equivalent signal in wnba-bet/mlb-bet).

Weather is deliberately NOT a trained feature here: training would need a
historical weather archive per game (open-meteo's archive API, not yet
integrated — pipeline/weather.py only does live forecasts). It's fetched and
shown for context at predict/UI time, not baked into the point predictions.
See CLAUDE.md's caveats section.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
from utils.db import get_conn

MODELS_DIR = os.path.join(os.path.dirname(__file__), "../data/models")
os.makedirs(MODELS_DIR, exist_ok=True)

ROLLING_WINDOWS = [4, 8]   # ~quarter-season and ~half-season windows (17-game season)
STAT_COLS = ["off_epa_pass", "off_epa_rush", "def_epa_pass", "def_epa_rush",
             "off_success_rate", "def_success_rate", "points_for", "points_against"]

# Game/Doubtful-or-worse statuses treated as "starter unavailable." Nflverse's
# report_status values are inconsistent about casing/wording across seasons,
# so this matches loosely (see _is_out).
_OUT_STATUSES = {"out", "doubtful", "ir", "injured reserve", "pup"}


def load_game_logs() -> pd.DataFrame:
    conn = get_conn()
    df = pd.read_sql(
        "SELECT * FROM team_game_logs WHERE season_type = 'REG' ORDER BY date ASC", conn
    )
    conn.close()
    df["date"] = pd.to_datetime(df["date"])
    return df


def build_team_rolling_features(logs: pd.DataFrame) -> pd.DataFrame:
    """Per team, rolling averages over the last 4 and 8 games — shifted so a
    game's own result never leaks into its own features."""
    dfs = []
    for team, grp in logs.groupby("team"):
        grp = grp.sort_values("date").copy()
        for col in STAT_COLS:
            for w in ROLLING_WINDOWS:
                grp[f"{col}_last{w}"] = grp[col].shift(1).rolling(w, min_periods=1).mean()
        grp["win_pct_last8"] = (grp["result"] == "W").astype(int).shift(1).rolling(8, min_periods=1).mean()
        margin = grp["points_for"] - grp["points_against"]
        grp["margin_last4"] = margin.shift(1).rolling(4, min_periods=1).mean()
        grp["margin_last8"] = margin.shift(1).rolling(8, min_periods=1).mean()
        grp["rest_days"] = grp["rest_days"].fillna(7)
        grp["is_short_week"] = (grp["rest_days"] <= 5).astype(int)
        grp["is_long_week"] = (grp["rest_days"] >= 10).astype(int)
        dfs.append(grp)
    return pd.concat(dfs).sort_values("date")


def _is_out(status: str | None) -> bool:
    if not status:
        return False
    s = str(status).strip().lower()
    return any(tag in s for tag in _OUT_STATUSES)


def build_starter_map(conn) -> dict[tuple[int, str], str]:
    """{(season, team): starting QB player_name} — the QB with the most pass
    attempts that season. A per-season snapshot (not per-week): good enough to
    catch "the guy who normally starts is out," not mid-season QB changes."""
    logs = pd.read_sql(
        "SELECT season, team, player_name, attempts FROM player_game_logs "
        "WHERE position = 'QB' AND attempts IS NOT NULL", conn,
    )
    if logs.empty:
        return {}
    totals = logs.groupby(["season", "team", "player_name"])["attempts"].sum().reset_index()
    idx = totals.groupby(["season", "team"])["attempts"].idxmax()
    starters = totals.loc[idx]
    return {(int(r.season), r.team): r.player_name for r in starters.itertuples()}


def build_qb_out_flags(conn, starter_map: dict) -> dict[tuple[int, int, str], int]:
    """{(season, week, team): 1 if that team's starter was Out/Doubtful+ that week}."""
    inj = pd.read_sql(
        "SELECT season, week, team, player_name, game_status FROM injuries", conn,
    )
    if inj.empty:
        return {}
    flags = {}
    for r in inj.itertuples():
        starter = starter_map.get((int(r.season), r.team))
        if starter and r.player_name == starter and _is_out(r.game_status):
            flags[(int(r.season), int(r.week), r.team)] = 1
    return flags


def build_matchup_features(logs: pd.DataFrame, conn) -> pd.DataFrame:
    """One row per game (home perspective), joining home + away team rolling
    features plus div_game, and the QB-out flag for each side."""
    rolling_cols = [c for c in logs.columns if "_last" in c] + ["rest_days", "is_short_week", "is_long_week"]

    home = logs[logs["is_home"] == 1]
    away = logs[logs["is_home"] == 0]

    home_feat = home[["game_id", "date", "season", "week", "team", "points_for", "points_against"] + rolling_cols].copy()
    home_feat.columns = ["game_id", "date", "season", "week", "home_team", "home_pts", "home_pts_against"] + \
                        [f"home_{c}" for c in rolling_cols]

    away_feat = away[["game_id", "team", "points_for"] + rolling_cols].copy()
    away_feat.columns = ["game_id", "away_team", "away_pts"] + [f"away_{c}" for c in rolling_cols]

    merged = home_feat.merge(away_feat, on="game_id", how="inner")

    merged["total_pts"] = merged["home_pts"] + merged["away_pts"]
    merged["pt_diff"] = merged["home_pts"] - merged["away_pts"]
    merged["rest_advantage"] = merged["home_rest_days"] - merged["away_rest_days"]
    merged["home_field"] = 1

    games = pd.read_sql("SELECT game_id, div_game FROM games", conn)
    merged = merged.merge(games, on="game_id", how="left")
    merged["div_game"] = merged["div_game"].fillna(0).astype(int)

    starter_map = build_starter_map(conn)
    qb_out = build_qb_out_flags(conn, starter_map)
    merged["home_qb_out"] = merged.apply(
        lambda r: qb_out.get((int(r.season), int(r.week), r.home_team), 0), axis=1)
    merged["away_qb_out"] = merged.apply(
        lambda r: qb_out.get((int(r.season), int(r.week), r.away_team), 0), axis=1)

    return merged.sort_values("date")


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    exclude = {"game_id", "date", "season", "week", "home_team", "away_team",
               "home_pts", "away_pts", "home_pts_against", "total_pts", "pt_diff"}
    return [c for c in df.columns if c not in exclude and df[c].dtype in [float, int, "float64", "int64"]]


def train_models(matchups: pd.DataFrame):
    matchups = matchups.dropna(subset=["total_pts", "pt_diff"])
    feat_cols = get_feature_cols(matchups)

    # Chronological split: train on the past, evaluate on the most recent 20%
    # (same reasoning as wnba-bet — a random split lets games from the same
    # week land on both sides and inflates holdout metrics).
    matchups = matchups.sort_values("date")
    cut = int(len(matchups) * 0.8)
    train_df, test_df = matchups.iloc[:cut], matchups.iloc[cut:]
    print(f"  Train: {len(train_df)} games through {train_df['date'].max().date()}")
    print(f"  Test:  {len(test_df)} games from {test_df['date'].min().date()}")

    fill_means = train_df[feat_cols].mean()
    X_tr = train_df[feat_cols].fillna(fill_means)
    X_te = test_df[feat_cols].fillna(fill_means)

    results = {}

    # Point differential regressor — the single source of truth for win
    # probability at predict time (see models/game.py). No separate moneyline
    # classifier is trained, to avoid the two numbers contradicting each other.
    diff_model = XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42)
    diff_model.fit(X_tr, train_df["pt_diff"])
    diff_pred = diff_model.predict(X_te)
    mae = mean_absolute_error(test_df["pt_diff"], diff_pred)
    spread_std = float(np.std(test_df["pt_diff"] - diff_pred))
    print(f"  Point diff MAE (holdout): {mae:.2f} pts | residual std: {spread_std:.2f}")
    results["spread"] = diff_model

    total_model = XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42)
    total_model.fit(X_tr, train_df["total_pts"])
    total_pred = total_model.predict(X_te)
    mae = mean_absolute_error(test_df["total_pts"], total_pred)
    totals_std = float(np.std(test_df["total_pts"] - total_pred))
    print(f"  Totals MAE (holdout): {mae:.2f} pts | residual std: {totals_std:.2f}")
    results["totals"] = total_model

    calibration = {"spread_std": round(spread_std, 2), "totals_std": round(totals_std, 2)}

    # Refit on ALL data so the deployed models see the newest games.
    X_all = matchups[feat_cols].fillna(fill_means)
    results["spread"].fit(X_all, matchups["pt_diff"])
    results["totals"].fit(X_all, matchups["total_pts"])

    return results, feat_cols, calibration


def main():
    print("Loading team game logs...")
    logs = load_game_logs()
    print(f"  {len(logs)} team-game rows across {logs['season'].nunique()} seasons")

    print("Building rolling features...")
    logs_with_rolling = build_team_rolling_features(logs)

    conn = get_conn()
    print("Building matchup features (rest, div game, QB-out)...")
    matchups = build_matchup_features(logs_with_rolling, conn)
    conn.close()
    print(f"  {len(matchups)} matchups")

    print("Training models...")
    models, feat_cols, calibration = train_models(matchups)

    print("Saving models...")
    for name, model in models.items():
        path = os.path.join(MODELS_DIR, f"game_{name}.pkl")
        joblib.dump(model, path)
        print(f"  Saved {path}")

    joblib.dump(feat_cols, os.path.join(MODELS_DIR, "game_feature_cols.pkl"))
    joblib.dump(calibration, os.path.join(MODELS_DIR, "game_calibration.pkl"))
    print(f"  Saved calibration: {calibration}")
    print("\nTraining complete.")


if __name__ == "__main__":
    main()
