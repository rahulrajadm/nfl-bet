"""
Fetches this week's NFL schedule (Thu/Sun/Mon slate + buffer, not just "today"
— NFL is weekly, unlike wnba-bet/mlb-bet's daily cadence).

Primary: derived from The Odds API (already fetched for game odds, so this is
free if odds were just pulled). Falls back to ESPN's unofficial scoreboard API
when Odds API has no NFL markets yet — common in early preseason, when many
books haven't posted lines — or when no ODDS_API_KEY is configured. The ESPN
path is also cloud-safe (same role as wnba-bet's cloud_data.py fallback: no
datacenter-IP blocking issue).
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import requests
from datetime import date, timedelta
from utils.db import get_conn, set_meta
from utils.dates import today_local

ODDS_SPORT = "americanfootball_nfl"
ESPN_SCOREBOARD_URL = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"
WEEK_WINDOW_DAYS = 8  # covers a full Thu-Mon slate plus a day of buffer either side


def fetch_week_games_from_odds() -> list[dict]:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), "../.env"))
    api_key = os.getenv("ODDS_API_KEY")
    if not api_key:
        return []

    url = f"https://api.the-odds-api.com/v4/sports/{ODDS_SPORT}/odds"
    params = {"apiKey": api_key, "regions": "us", "markets": "h2h", "oddsFormat": "american"}
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"  Warning fetching schedule from Odds API: {e}")
        return []

    remaining = resp.headers.get("x-requests-remaining")
    if remaining is not None:
        set_meta("odds_api_remaining", remaining)

    today = today_local()
    window_end = today + timedelta(days=WEEK_WINDOW_DAYS)

    games = []
    for g in data:
        game_dt = g.get("commence_time", "")[:10]
        try:
            gd = date.fromisoformat(game_dt)
        except ValueError:
            continue
        if not (today <= gd <= window_end):
            continue
        games.append({
            "game_id":   g["id"],
            "date":      game_dt,
            "home_team": g["home_team"],
            "away_team": g["away_team"],
            "game_time": g.get("commence_time", ""),
        })
    return games


def fetch_week_games_from_espn() -> list[dict]:
    """Cloud-safe fallback that spends no API credits — used in early preseason
    and as the smoke-test path for pipeline/injuries.py's team-id lookup."""
    games = []
    today = today_local()
    for offset in range(WEEK_WINDOW_DAYS + 1):
        d = today + timedelta(days=offset)
        try:
            resp = requests.get(ESPN_SCOREBOARD_URL, params={"dates": d.strftime("%Y%m%d")}, timeout=10)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            print(f"  Warning fetching ESPN scoreboard for {d}: {e}")
            continue
        for event in data.get("events", []):
            comp = event.get("competitions", [{}])[0]
            competitors = comp.get("competitors", [])
            home = next((c for c in competitors if c.get("homeAway") == "home"), {})
            away = next((c for c in competitors if c.get("homeAway") == "away"), {})
            games.append({
                "game_id":   f"espn_{event.get('id')}",
                "date":      d.isoformat(),
                "home_team": home.get("team", {}).get("displayName", ""),
                "away_team": away.get("team", {}).get("displayName", ""),
                "game_time": event.get("date", ""),
            })
    return games


def fetch_week_games() -> list[dict]:
    games = fetch_week_games_from_odds()
    if games:
        return games
    print("  No games from Odds API — falling back to ESPN scoreboard.")
    return fetch_week_games_from_espn()


def save_schedule(games: list[dict]):
    conn = get_conn()
    c = conn.cursor()
    for g in games:
        c.execute("""
            INSERT OR REPLACE INTO games (game_id, date, home_team, away_team, game_time)
            VALUES (:game_id, :date, :home_team, :away_team, :game_time)
        """, g)
    conn.commit()
    conn.close()


def load_saved_games(day: date | None = None) -> list[dict]:
    """Read this week's games from SQLite without touching the Odds API — UI
    code that reloads periodically should use this and leave live fetches to
    the explicit Refresh action (or start.sh)."""
    day = day or today_local()
    window_end = day + timedelta(days=WEEK_WINDOW_DAYS)

    conn = get_conn()
    rows = conn.execute(
        """SELECT game_id, date, home_team, away_team, game_time
           FROM games WHERE date BETWEEN ? AND ? ORDER BY game_time""",
        (day.isoformat(), window_end.isoformat()),
    ).fetchall()
    conn.close()
    cols = ["game_id", "date", "home_team", "away_team", "game_time"]
    return [dict(zip(cols, r)) for r in rows]


def get_week_games() -> list[dict]:
    games = fetch_week_games()
    if games:
        save_schedule(games)
    return games


if __name__ == "__main__":
    games = get_week_games()
    print(f"Found {len(games)} NFL games in the next {WEEK_WINDOW_DAYS} days:")
    for g in games:
        print(f"  {g['away_team']} @ {g['home_team']}  |  {g['game_time']}")
