"""
NFL player prop prediction engine.
Predicts per-game stat values using season averages + recent-form blend,
adjusted for opponent defense (split pass vs. rush — NFL defenses vary far
more by pass/rush than a single overall rating) and play-volume pace.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import numpy as np
from scipy.stats import poisson, norm
from utils.db import get_conn
from utils.names import normalize_name
from pipeline.team_metrics import get_opp_def_epa, get_def_adj, get_game_pace_factor
from pipeline.team_names import to_full_name
from analysis.ev import breakeven_prob

RECENT_GAMES  = 4
RECENT_WEIGHT = 0.55
SEASON_WEIGHT = 0.45

# Maps platform stat names -> internal stat column. "Rush Yards"/"Pass Yards"/
# "Receiving Yards"/"INTs Thrown"/"Pass TDs"/"Rush + Rec TDs" are confirmed
# against a live Underdog pull (2026-08-01); the rest are plausible platform
# spellings not yet confirmed live — check the [props] unknown_stat_types
# counter after a real run and add any that show up unmapped.
STAT_MAP = {
    "Pass Yards":                  "passing_yards",
    "Passing Yards":               "passing_yards",
    "Pass TDs":                    "passing_tds",
    "Passing TDs":                 "passing_tds",
    "Pass Touchdowns":             "passing_tds",
    "INTs Thrown":                 "interceptions",
    "Interceptions":               "interceptions",
    "Interceptions Thrown":        "interceptions",
    "Completions":                 "completions",
    "Pass Completions":            "completions",
    "Rush Yards":                  "rushing_yards",
    "Rushing Yards":               "rushing_yards",
    "Rush TDs":                    "rushing_tds",
    "Rushing TDs":                 "rushing_tds",
    "Rushing Touchdowns":          "rushing_tds",
    "Receiving Yards":             "receiving_yards",
    "Receptions":                  "receptions",
    "Receiving TDs":               "receiving_tds",
    "Receiving Touchdowns":        "receiving_tds",
    "Rush + Rec TDs":              "rush_rec_tds",
    "Rush + Rec Yards":            "rush_rec_yards",
    "Rushing + Receiving Yards":   "rush_rec_yards",
    "Rushing + Receiving TDs":     "rush_rec_tds",
    "Pass + Rush Yards":           "pass_rush_yards",
    "Pass + Rush + Rec Yards":     "total_yards",
    "Fantasy Score":               "fantasy",
    "Fantasy Points":              "fantasy",
}

# Deliberately unmodeled — add here (not left to fall into unknown_stat_types)
# so a skip reads as "decided," same convention as mlb-bet's UNMODELED_STATS.
UNMODELED_STATS = {
    "Sacks",                                             # defensive player prop; no defensive stats pulled
    "First TD Scorer", "Anytime TD Scorer",               # anytime/first-scorer market, not a numeric line
    "1H Rush + Rec TDs", "1Q Rush + Rec TDs",              # partial-game props need in-game TD-timing, not built
    "1H Pass Yards", "1H Rush Yards", "1H Receiving Yards",
    "Kicking Points", "FG Made", "Extra Points Made",      # separate nflverse "kicking" release, not integrated
    "Tackles", "Tackles + Assists", "Solo Tackles",        # defensive player props
    "Longest Reception", "Longest Rush", "Longest Completion",  # extreme-value stat, not a per-game-rate model
}

POISSON_STATS = {"passing_tds", "interceptions", "rushing_tds", "receiving_tds",
                  "receptions", "completions", "rush_rec_tds"}
NORMAL_STATS  = {"passing_yards", "rushing_yards", "receiving_yards",
                  "rush_rec_yards", "pass_rush_yards", "total_yards", "fantasy"}

PASS_DEFENSE_STATS = {"passing_yards", "passing_tds", "interceptions", "completions",
                       "receiving_yards", "receiving_tds", "receptions"}
RUSH_DEFENSE_STATS = {"rushing_yards", "rushing_tds"}
# Everything else modeled (rush_rec_yards, rush_rec_tds, pass_rush_yards, total_yards,
# fantasy) is a combo of both game phases — blend pass and rush defense adjustment.

PACE_APPLIED_STATS = POISSON_STATS | NORMAL_STATS


def load_player_logs() -> pd.DataFrame:
    conn = get_conn()
    df = pd.read_sql("SELECT * FROM player_game_logs ORDER BY season ASC, week ASC", conn)
    conn.close()
    if df.empty:
        return df
    df["rush_rec_tds"]   = df["rushing_tds"].fillna(0) + df["receiving_tds"].fillna(0)
    df["rush_rec_yards"] = df["rushing_yards"].fillna(0) + df["receiving_yards"].fillna(0)
    df["pass_rush_yards"] = df["passing_yards"].fillna(0) + df["rushing_yards"].fillna(0)
    df["total_yards"] = (df["passing_yards"].fillna(0) + df["rushing_yards"].fillna(0)
                          + df["receiving_yards"].fillna(0))
    # Standard-ish PPR scoring — an approximation, not any one platform's exact
    # ruleset (which vary), same caveat class as wnba-bet's fantasy formula.
    df["fantasy"] = (
        df["passing_yards"].fillna(0) * 0.04 + df["passing_tds"].fillna(0) * 4
        - df["interceptions"].fillna(0) * 2
        + df["rushing_yards"].fillna(0) * 0.1 + df["rushing_tds"].fillna(0) * 6
        + df["receiving_yards"].fillna(0) * 0.1 + df["receiving_tds"].fillna(0) * 6
        + df["receptions"].fillna(0) * 1.0
    )
    df["_name_key"] = df["player_name"].map(normalize_name)
    return df


def _player_rows(player_name: str, logs: pd.DataFrame) -> pd.DataFrame:
    """Exact normalized-name match, with last-name fallback ONLY when
    unambiguous — a bare contains() can silently blend two players' logs."""
    key = normalize_name(player_name)
    rows = logs[logs["_name_key"] == key]
    if not rows.empty:
        return rows
    if not key:
        return rows
    last = key.split()[-1]
    cand = logs[logs["_name_key"].str.split().str[-1] == last]
    if not cand.empty and cand["_name_key"].nunique() == 1:
        return cand
    return cand.iloc[0:0]


def get_player_profile(player_name: str, stat_col: str, logs: pd.DataFrame) -> dict | None:
    """Season/recent rates and empirical std for one stat. Current-season
    games are preferred for the baseline once at least 3 are on record
    (NFL's 17-game season means a strict 5-game threshold like wnba-bet's
    would exclude half the season) — else falls back to all seasons on file."""
    rows = _player_rows(player_name, logs)
    if rows.empty or stat_col not in rows.columns:
        return None
    rows = rows.sort_values(["season", "week"])
    vals = pd.to_numeric(rows[stat_col], errors="coerce")
    rows = rows.assign(_v=vals).dropna(subset=["_v"])
    if rows.empty:
        return None

    latest_season = rows["season"].max()
    cur = rows[rows["season"] == latest_season]
    base = cur if len(cur) >= 3 else rows

    recent = rows.tail(RECENT_GAMES)
    return {
        "season_rate": float(base["_v"].mean()),
        "recent_rate": float(recent["_v"].mean()) if len(recent) >= 2 else None,
        "std":         float(base["_v"].std(ddof=1)) if len(base) >= 3 else None,
        "n_games":     len(base),
    }


def prob_over_line(expected: float, line: float, stat_col: str, std: float | None = None) -> float:
    """P(stat > line).

    Player stats are overdispersed, so Poisson is overconfident for
    high-volume stats. Prefer a normal with the player's own empirical std;
    keep Poisson only for low-mean discrete stats (TDs, INTs) where normal is
    a poor fit — same dispatch rule as wnba-bet's prob_over_line.
    """
    use_normal = std is not None and std > 0 and (stat_col in NORMAL_STATS or expected >= 3.0)
    if use_normal:
        return float(1 - norm.cdf(line, loc=expected, scale=max(std, 1.0)))
    if stat_col in POISSON_STATS:
        if expected <= 0:
            return 0.0
        threshold = int(np.floor(line)) + 1
        return float(1.0 - poisson.cdf(threshold - 1, mu=expected))
    return float(1 - norm.cdf(line, loc=expected, scale=max(expected * 0.35, 2.0)))


def _stat_def_adjustment(stat_col: str, opp_team: str, game_logs_df=None):
    """Returns (multiplier, epa_for_explain, split_label)."""
    if stat_col in PASS_DEFENSE_STATS:
        epa = get_opp_def_epa(opp_team, "pass", game_logs_df=game_logs_df)
        return get_def_adj(epa), epa, "pass"
    if stat_col in RUSH_DEFENSE_STATS:
        epa = get_opp_def_epa(opp_team, "rush", game_logs_df=game_logs_df)
        return get_def_adj(epa), epa, "rush"
    pass_epa = get_opp_def_epa(opp_team, "pass", game_logs_df=game_logs_df)
    rush_epa = get_opp_def_epa(opp_team, "rush", game_logs_df=game_logs_df)
    blended = (get_def_adj(pass_epa) + get_def_adj(rush_epa)) / 2
    return blended, {"pass": pass_epa, "rush": rush_epa}, "blend"


def _build_team_opponent_map(games: list[dict]) -> dict[str, str]:
    opp_map = {}
    for g in games:
        opp_map[g["home_team"]] = g["away_team"]
        opp_map[g["away_team"]] = g["home_team"]
    return opp_map


def predict_props(
    lines_data: list[dict] | None = None,
    games: list[dict] | None = None,
    player_logs_df=None,
    team_logs_df=None,
) -> list[dict]:
    logs = player_logs_df if player_logs_df is not None else load_player_logs()
    if logs is None or (hasattr(logs, "empty") and logs.empty):
        return []
    if "_name_key" not in logs.columns:
        logs = logs.assign(_name_key=logs["player_name"].map(normalize_name))

    if games is None:
        from pipeline.schedule import get_week_games
        games = get_week_games()
    opp_map = _build_team_opponent_map(games)

    if lines_data is not None:
        lines = pd.DataFrame(lines_data)
    else:
        conn = get_conn()
        lines = pd.read_sql(
            "SELECT * FROM prop_lines WHERE DATE(fetched_at) >= DATE('now', '-1 day')", conn
        )
        conn.close()

    if lines.empty:
        return []

    if "odds_type" not in lines.columns:
        lines["odds_type"] = "standard"
    if "allowed_direction" not in lines.columns:
        lines["allowed_direction"] = None

    # Prefer the two-way standard line over one-way alternates (Underdog sends
    # both under odds_type="standard") — same dedupe reasoning as wnba-bet.
    lines = lines.assign(_is_alt=lines["allowed_direction"].notna())
    lines = (
        lines.sort_values(["_is_alt", "fetched_at"], ascending=[True, False])
        .drop_duplicates(subset=["platform", "player_name", "stat_type", "odds_type"])
        .drop(columns="_is_alt")
    )

    _dbg = {"total": len(lines), "no_stat": 0, "unmodeled": 0, "combo": 0, "no_rate": 0,
            "zero_blend": 0, "2x": 0, "no_edge": 0, "direction": 0, "passed": 0, "_unk": {}}

    predictions = []
    for _, row in lines.iterrows():
        stat_type = str(row.get("stat_type", "?"))
        if stat_type in UNMODELED_STATS:
            _dbg["unmodeled"] += 1
            continue

        stat_col = STAT_MAP.get(stat_type)
        if stat_col is None or row["line"] is None:
            _dbg["no_stat"] += 1
            _dbg["_unk"][stat_type] = _dbg["_unk"].get(stat_type, 0) + 1
            continue

        player_name = row["player_name"]
        line = float(row["line"])

        if " + " in player_name or " & " in player_name:
            _dbg["combo"] += 1
            continue

        prof = get_player_profile(player_name, stat_col, logs)
        if prof is None or prof["season_rate"] < 0:
            _dbg["no_rate"] += 1
            continue
        season_rate = prof["season_rate"]
        recent_rate = prof["recent_rate"]

        if recent_rate is not None:
            blended = RECENT_WEIGHT * recent_rate + SEASON_WEIGHT * season_rate
            form = "blended"
        else:
            blended = season_rate
            form = "season_only"
        base_rate = blended

        player_team = to_full_name(row.get("player_team", "")) or row.get("player_team", "")
        opp_team = opp_map.get(player_team, "")

        def_adj = epa_for_explain = def_split = None
        if opp_team:
            def_adj, epa_for_explain, def_split = _stat_def_adjustment(stat_col, opp_team, team_logs_df)
            blended = max(blended * def_adj, 0.0)

        pace_factor = 1.0
        if opp_team and player_team and stat_col in PACE_APPLIED_STATS:
            pace_factor = get_game_pace_factor(player_team, opp_team, game_logs_df=team_logs_df)
            blended = max(blended * pace_factor, 0.0)

        if blended <= 0:
            _dbg["zero_blend"] += 1
            continue

        # Demon/elevated lines above 2x expectation make "Less" near-certain —
        # skip rather than surface a manufactured-looking edge.
        if line > blended * 2.0:
            _dbg["2x"] += 1
            continue

        p_more = prob_over_line(blended, line, stat_col, std=prof["std"])
        p_less = 1.0 - p_more

        breakeven = breakeven_prob(row.get("platform", ""), slip_size=2)
        if p_more >= p_less:
            direction, model_prob = "More", p_more
        else:
            direction, model_prob = "Less", p_less
        edge = model_prob - breakeven

        if edge <= 0:
            _dbg["no_edge"] += 1
            continue

        allowed = row.get("allowed_direction")
        if pd.notna(allowed) and direction != allowed:
            _dbg["direction"] += 1
            continue

        _dbg["passed"] += 1

        std = prof["std"]
        if std is not None and std > 0 and (stat_col in NORMAL_STATS or blended >= 3.0):
            dist = "normal"
        elif stat_col in POISSON_STATS:
            dist = "poisson"
        else:
            dist = "normal_heuristic"

        predictions.append({
            "platform":      row["platform"],
            "player_name":   player_name,
            "player_team":   player_team,
            "stat_type":     stat_type,
            "line":          line,
            "direction":     direction,
            "model_prob":    round(model_prob, 4),
            "implied_prob":  round(breakeven, 4),
            "edge":          round(edge, 4),
            "expected_rate": round(blended, 3),
            "season_rate":   round(season_rate, 3),
            "recent_rate":   round(recent_rate, 3) if recent_rate is not None else None,
            "form_source":   form,
            "game_id":       row.get("game_id", ""),
            "odds_type":     row.get("odds_type", "standard"),
            "explain": {
                "form_source": form,
                "n_games":     prof["n_games"],
                "base_rate":   round(base_rate, 3),
                "opponent":    opp_team,
                "def_split":   def_split,
                "def_epa":     epa_for_explain,
                "def_adj":     round(def_adj, 4) if def_adj is not None else None,
                "pace_factor": pace_factor,
                "expected":    round(blended, 3),
                "std":         round(std, 3) if std is not None else None,
                "dist":        dist,
                "p_more":      round(p_more, 4),
                "breakeven":   round(breakeven, 4),
            },
        })

    top_unk = sorted(_dbg["_unk"].items(), key=lambda x: -x[1])[:10]
    print(
        f"[props] lines={_dbg['total']} no_stat={_dbg['no_stat']} unmodeled={_dbg['unmodeled']} "
        f"combo={_dbg['combo']} no_rate={_dbg['no_rate']} zero_blend={_dbg['zero_blend']} "
        f"2x_filter={_dbg['2x']} no_edge={_dbg['no_edge']} direction={_dbg['direction']} "
        f"passed={_dbg['passed']}",
        file=sys.stderr,
    )
    print(f"[props] unknown_stat_types: {top_unk}", file=sys.stderr)
    return predictions


if __name__ == "__main__":
    preds = predict_props()
    preds.sort(key=lambda x: x["edge"], reverse=True)
    hi = [p for p in preds if 0.55 <= p["model_prob"] <= 0.85]
    print(f"Total prop predictions: {len(preds)}")
    print(f"High-interest (55-85% model prob): {len(hi)}")
    print(f"\n{'Player':<22} {'Stat':<20} {'Line':>5} {'Dir':>5} {'Model%':>7} {'Edge':>6}  {'Platform'}")
    print("-" * 85)
    for p in hi[:15]:
        print(f"{p['player_name']:<22} {p['stat_type']:<20} {p['line']:>5} "
              f"{p['direction']:>5} {p['model_prob']:>6.1%} {p['edge']:>+6.1%}  {p['platform']}")
