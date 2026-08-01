"""
Wind/precipitation forecast for outdoor NFL stadiums (pipeline/stadiums.py),
via open-meteo's free forecast API (no key, no rate-limit auth). Fixed-dome
games are skipped entirely — weather can't affect them.

Feeds both the totals model (wind suppresses scoring) and passing/kicking
props (wind suppresses passing yards, FG accuracy) — no equivalent in
wnba-bet (indoor) or mlb-bet (park factors are a different, non-weather concept).
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from datetime import datetime, timezone
import requests
from utils.db import get_conn
from pipeline.stadiums import get_stadium, needs_weather

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
TIMEOUT = 10

_WEATHER_CODES = {
    0: "clear", 1: "mostly clear", 2: "partly cloudy", 3: "overcast",
    45: "fog", 48: "fog", 51: "light drizzle", 53: "drizzle", 55: "heavy drizzle",
    61: "light rain", 63: "rain", 65: "heavy rain",
    71: "light snow", 73: "snow", 75: "heavy snow",
    80: "rain showers", 81: "rain showers", 82: "violent rain showers",
    95: "thunderstorm",
}


def fetch_game_weather(home_team_abbr: str, game_time_iso: str) -> dict | None:
    """home_team_abbr: nflverse team abbreviation (the stadium is theirs).
    game_time_iso: kickoff time, ISO 8601. Returns None for dome games (still
    a meaningful "no weather effect" result, not a failure) or on fetch failure."""
    if not needs_weather(home_team_abbr):
        return {"team": home_team_abbr, "is_dome": True,
                "temp_f": None, "wind_mph": None, "precip_pct": None, "condition": "dome"}

    stadium = get_stadium(home_team_abbr)
    if not stadium or not game_time_iso:
        return None

    try:
        game_dt = datetime.fromisoformat(game_time_iso.replace("Z", "+00:00"))
    except ValueError:
        return None

    params = {
        "latitude": stadium["lat"], "longitude": stadium["lon"],
        "hourly": "temperature_2m,precipitation_probability,wind_speed_10m,weather_code",
        "temperature_unit": "fahrenheit", "wind_speed_unit": "mph",
        "timezone": "UTC", "forecast_days": 16,
    }
    try:
        resp = requests.get(FORECAST_URL, params=params, timeout=TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"  Warning fetching weather for {home_team_abbr}: {e}")
        return None

    hourly = data.get("hourly", {})
    times = hourly.get("time", [])
    if not times:
        return None

    target = game_dt.strftime("%Y-%m-%dT%H:00")
    if target not in times:
        # Kickoff is beyond the forecast window (>16 days out) — use the
        # nearest available hour instead of failing outright.
        naive_target = game_dt.replace(tzinfo=None)
        target = min(times, key=lambda t: abs(datetime.fromisoformat(t) - naive_target))
    idx = times.index(target)

    def _at(key):
        vals = hourly.get(key, [])
        return vals[idx] if idx < len(vals) else None

    return {
        "team": home_team_abbr,
        "is_dome": False,
        "temp_f": _at("temperature_2m"),
        "wind_mph": _at("wind_speed_10m"),
        "precip_pct": _at("precipitation_probability"),
        "condition": _WEATHER_CODES.get(_at("weather_code"), "unknown"),
    }


def save_weather(game_id: str, weather: dict):
    conn = get_conn()
    conn.execute("""
        INSERT INTO weather (fetched_at, game_id, temp_f, wind_mph, precip_pct, condition, is_dome)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        datetime.now(timezone.utc).isoformat(), game_id,
        weather.get("temp_f"), weather.get("wind_mph"), weather.get("precip_pct"),
        weather.get("condition"), int(weather.get("is_dome", False)),
    ))
    conn.commit()
    conn.close()


def get_week_weather(games: list[dict], home_abbr_lookup) -> dict[str, dict]:
    """games: schedule rows with game_id/home_team/game_time (pipeline/schedule.py
    shape). home_abbr_lookup: full-team-name -> nflverse abbreviation function
    (pipeline.team_names.to_abbr), since stadiums.py is keyed by abbreviation."""
    result = {}
    for g in games:
        abbr = home_abbr_lookup(g["home_team"])
        w = fetch_game_weather(abbr, g.get("game_time", ""))
        if w:
            save_weather(g["game_id"], w)
            result[g["game_id"]] = w
    return result


if __name__ == "__main__":
    from pipeline.schedule import fetch_week_games
    from pipeline.team_names import to_abbr

    games = fetch_week_games()
    print(f"Fetching weather for {len(games)} games this week...")
    weather = get_week_weather(games, to_abbr)
    for gid, w in weather.items():
        if w.get("is_dome"):
            print(f"  {gid}: dome")
        else:
            print(f"  {gid}: {w['temp_f']}F, wind {w['wind_mph']}mph, "
                  f"precip {w['precip_pct']}%, {w['condition']}")
