"""
In-memory (no-SQLite) fetchers backing ui/app_cloud.py's refresh flow and
analysis/tracking.py's grading — completed-game results and box scores from
ESPN's unofficial API, since neither the deployed app nor the grading loop
has a local database to draw from for "what actually happened."

Category label mapping below is confirmed against a real completed 2025-season
game (ESPN event 401772636, 2025-11-09) — not guessed:
  passing:   C/ATT, YDS, AVG, TD, INT, SACKS, QBR, RTG
  rushing:   CAR, YDS, AVG, TD, LONG
  receiving: REC, YDS, AVG, TD, LONG, TGTS
"""
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


if __name__ == "__main__":
    ids = _completed_game_ids(lookback=14)
    print(f"{len(ids)} completed games in the last 14 days")
    team_df = _fetch_all(ids[:3], _box_score_team_rows)
    player_df = _fetch_all(ids[:3], _box_score_player_rows)
    print(team_df)
    print(player_df.head(10))
