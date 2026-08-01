"""
In-memory (no-SQLite) fetchers backing ui/app_cloud.py's refresh flow and
analysis/tracking.py's grading.

Two different data sources for two different jobs:
  * Completed-game results/box scores (grading, "what actually happened")
    come from ESPN's unofficial API — same as wnba-bet's cloud_data.py.
  * Team/player game logs for LIVE PREDICTION (fetch_team_game_logs,
    fetch_player_game_logs) come from nfl_data_py instead, unlike wnba-bet's
    all-ESPN cloud path: the game model needs EPA, which requires real
    play-by-play — ESPN's public API has no equivalent, only nflverse
    publishes it. GitHub's CDN (nfl_data_py's data source) isn't IP-blocked
    the way stats.nba.com or PrizePicks' DataDome are, so this works the same
    on Streamlit Cloud as it does locally.

Category label mapping below is confirmed against a real completed 2025-season
game (ESPN event 401772636, 2025-11-09) — not guessed:
  passing:   C/ATT, YDS, AVG, TD, INT, SACKS, QBR, RTG
  rushing:   CAR, YDS, AVG, TD, LONG
  receiving: REC, YDS, AVG, TD, LONG, TGTS
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import requests
from datetime import date, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd

TIMEOUT = 10
SCOREBOARD_URL = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"
SUMMARY_URL = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/summary"

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json",
}

# ESPN box-score category -> {label: our player_game_logs column}
_CATEGORY_MAP = {
    "passing":   {"YDS": "passing_yards", "TD": "passing_tds", "INT": "interceptions"},
    "rushing":   {"YDS": "rushing_yards", "TD": "rushing_tds"},
    "receiving": {"YDS": "receiving_yards", "TD": "receiving_tds", "REC": "receptions"},
}


def _get(url: str, params: dict | None = None) -> dict:
    r = requests.get(url, headers=_HEADERS, params=params or {}, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()


def _completed_game_ids(lookback: int = 10) -> list[str]:
    """ESPN event ids for completed NFL games in the last `lookback` days."""
    ids = []
    today = date.today()
    for offset in range(lookback + 1):
        d = (today - timedelta(days=offset)).strftime("%Y%m%d")
        try:
            data = _get(SCOREBOARD_URL, {"dates": d})
        except Exception:
            continue
        for event in data.get("events", []):
            if event.get("status", {}).get("type", {}).get("completed", False):
                ids.append(str(event["id"]))
    return ids


def _box_score_team_rows(game_id: str) -> pd.DataFrame:
    """Final team scores for one completed game."""
    try:
        data = _get(SUMMARY_URL, {"event": game_id})
    except Exception:
        return pd.DataFrame()
    comp = data.get("header", {}).get("competitions", [{}])[0]
    game_date = comp.get("date", "")
    rows = []
    for c in comp.get("competitors", []):
        rows.append({
            "game_id": game_id,
            "game_date": game_date,
            "team_name": c.get("team", {}).get("displayName", ""),
            "pts": float(c.get("score", 0) or 0),
        })
    return pd.DataFrame(rows)


def _box_score_player_rows(game_id: str) -> pd.DataFrame:
    """Per-player final passing/rushing/receiving stats for one completed game,
    one row per player (categories merged) matching player_game_logs' columns."""
    try:
        data = _get(SUMMARY_URL, {"event": game_id})
    except Exception:
        return pd.DataFrame()

    comp = data.get("header", {}).get("competitions", [{}])[0]
    game_date = comp.get("date", "")

    per_player: dict[str, dict] = {}
    for team_block in data.get("boxscore", {}).get("players", []):
        for cat in team_block.get("statistics", []):
            cat_name = cat.get("name", "")
            col_map = _CATEGORY_MAP.get(cat_name)
            if not col_map:
                continue
            labels = cat.get("labels", [])
            for athlete in cat.get("athletes", []):
                name = athlete.get("athlete", {}).get("displayName", "")
                stats = athlete.get("stats", [])
                if not name:
                    continue
                row = per_player.setdefault(
                    name, {"player_name": name, "game_id": game_id, "game_date": game_date})
                for label, val in zip(labels, stats):
                    if cat_name == "passing" and label == "C/ATT":
                        try:
                            comp_n, _att = str(val).split("/")
                            row["completions"] = float(comp_n)
                        except (ValueError, AttributeError):
                            pass
                        continue
                    dest = col_map.get(label)
                    if dest:
                        try:
                            row[dest] = float(val)
                        except (ValueError, TypeError):
                            pass

    return pd.DataFrame(list(per_player.values()))


def _fetch_all(game_ids: list[str], fetch_fn) -> pd.DataFrame:
    if not game_ids:
        return pd.DataFrame()
    frames = []
    with ThreadPoolExecutor(max_workers=min(8, len(game_ids))) as ex:
        futures = [ex.submit(fetch_fn, gid) for gid in game_ids]
        for fut in as_completed(futures):
            df = fut.result()
            if df is not None and not df.empty:
                frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# -- Live prediction data (nfl_data_py, not ESPN) -------------------------------

_last_fetch_failures = 0


# How many seasons back to probe if the most recent ones aren't published yet
# (seen 2026-08: nflverse's player_stats release lagged schedules/pbp by a
# full season). Keep the most recent 2 SEASONS THAT ACTUALLY HAVE DATA, not
# the 2 most recent calendar years — those can both be empty in the offseason.
_MAX_SEASONS_BACK = 4
_KEEP_SEASONS = 2


def fetch_team_game_logs() -> pd.DataFrame:
    """In-memory team_game_logs-equivalent for the cloud app: the most recent
    available seasons via nfl_data_py, same shape (columns, EPA split
    pass/rush) as the SQLite table pipeline/historical.py populates locally,
    so it drops straight into models/game.py's game_logs_df /
    pipeline/team_metrics.py's game_logs_df parameters."""
    global _last_fetch_failures
    import nfl_data_py as nfl
    from utils.dates import today_local
    from pipeline.historical import _team_game_base_rows, _epa_by_team_game, _PLAYED_GAME_TYPES

    this_year = today_local().year
    frames = []
    failures = 0
    for season in range(this_year, this_year - _MAX_SEASONS_BACK, -1):
        if len(frames) >= _KEEP_SEASONS:
            break
        try:
            sched = nfl.import_schedules([season])
            sched = sched[sched["game_type"].isin(_PLAYED_GAME_TYPES)]
            if sched["home_score"].notna().sum() == 0:
                continue
            base = _team_game_base_rows(sched)
            epa = _epa_by_team_game(season)
            frames.append(base.merge(epa, on=["game_id", "team"], how="left"))
        except Exception as e:
            print(f"  cloud team logs: {season} unavailable ({e})")
            failures += 1
    _last_fetch_failures = failures
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def fetch_player_game_logs() -> pd.DataFrame:
    """In-memory player_game_logs-equivalent for the cloud app: the most
    recent available seasons via nfl_data_py, same shape as the SQLite table."""
    global _last_fetch_failures
    import nfl_data_py as nfl
    from utils.dates import today_local
    from pipeline.historical import normalize_weekly_player_df

    this_year = today_local().year
    frames = []
    failures = 0
    for season in range(this_year, this_year - _MAX_SEASONS_BACK, -1):
        if len(frames) >= _KEEP_SEASONS:
            break
        try:
            weekly = nfl.import_weekly_data([season])
            if not weekly.empty:
                frames.append(weekly)
        except Exception as e:
            print(f"  cloud player logs: {season} unavailable ({e})")
            failures += 1
    _last_fetch_failures = failures
    if not frames:
        return pd.DataFrame()
    return normalize_weekly_player_df(pd.concat(frames, ignore_index=True))


if __name__ == "__main__":
    ids = _completed_game_ids(lookback=14)
    print(f"{len(ids)} completed games in the last 14 days")
    team_df = _fetch_all(ids[:3], _box_score_team_rows)
    player_df = _fetch_all(ids[:3], _box_score_player_rows)
    print(team_df)
    print(player_df.head(10))
