"""
Fetches live NFL player prop lines from Underdog Fantasy.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import requests
from datetime import datetime, timezone
from utils.db import get_conn, ensure_schema

UNDERDOG_URL = "https://api.underdogfantasy.com/beta/v5/over_under_lines"
HEADERS      = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}

# Underdog's sport_id string for NFL games — unverified against a live call
# (wnba-bet/mlb-bet only had to confirm "WNBA"/"MLB", the presumed same
# pattern). The __main__ smoke block below prints every distinct sport_id it
# sees if this filter yields zero games, so a wrong guess is easy to spot.
NFL_SPORT_ID = "NFL"


def fetch_nfl_lines() -> list[dict]:
    resp = requests.get(UNDERDOG_URL, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    data = resp.json()

    games       = {g["id"]: g for g in data.get("games", [])}
    appearances = {a["id"]: a for a in data.get("appearances", [])}
    players     = {p["id"]: p for p in data.get("players", [])}

    nfl_match_ids = {gid for gid, g in games.items() if g.get("sport_id") == NFL_SPORT_ID}
    if not nfl_match_ids:
        seen = sorted({g.get("sport_id") for g in games.values()})
        print(f"  Warning: no games matched sport_id={NFL_SPORT_ID!r}. "
              f"Sport ids seen in this response: {seen}")

    props = []
    for line in data.get("over_under_lines", []):
        app_id     = line.get("over_under", {}).get("appearance_stat", {}).get("appearance_id")
        appearance = appearances.get(app_id, {})
        match_id   = appearance.get("match_id")

        if match_id not in nfl_match_ids:
            continue

        player_id = appearance.get("player_id")
        player    = players.get(player_id, {})
        stat      = line.get("over_under", {}).get("appearance_stat", {}).get("display_stat", "")
        ou_line   = line.get("stat_value")

        # Resolve UUID → full team name using the game's title field
        game      = games.get(match_id, {})
        team_uuid = appearance.get("team_id", "")
        title     = game.get("full_team_names_title", "")
        if " @ " in title:
            away_name, home_name = [t.strip() for t in title.split(" @ ", 1)]
            if team_uuid == game.get("home_team_id", ""):
                player_team = home_name
            elif team_uuid == game.get("away_team_id", ""):
                player_team = away_name
            else:
                player_team = team_uuid
        else:
            player_team = team_uuid

        # "balanced" lines have both higher and lower options.
        # "alternate" lines have only one direction — read it from options[].choice.
        line_type = line.get("line_type", "balanced")
        if line_type == "alternate":
            choices = [o.get("choice", "") for o in line.get("options", [])]
            if choices == ["higher"]:
                allowed = "More"
            elif choices == ["lower"]:
                allowed = "Less"
            else:
                allowed = None
        else:
            allowed = None   # balanced: both directions available

        props.append({
            "platform":          "underdog",
            "fetched_at":        datetime.now(timezone.utc).isoformat(),
            "player_name":       f"{player.get('first_name', '')} {player.get('last_name', '')}".strip(),
            "player_team":       player_team,
            "stat_type":         stat,
            "line":              float(ou_line) if ou_line is not None else None,
            "game_id":           str(match_id),
            "allowed_direction": allowed,
            "odds_type":         "standard",
            "more_odds":         None,
            "less_odds":         None,
        })
    return props


def save_lines(props: list[dict]):
    conn = get_conn()
    ensure_schema(conn)
    c    = conn.cursor()
    for p in props:
        c.execute("""
            INSERT INTO prop_lines
            (fetched_at, platform, game_id, player_name, player_team, stat_type, line, more_odds, less_odds,
             odds_type, allowed_direction)
            VALUES (:fetched_at, :platform, :game_id, :player_name, :player_team, :stat_type, :line, :more_odds, :less_odds,
                    :odds_type, :allowed_direction)
        """, p)
    conn.commit()
    conn.close()


def get_underdog_lines() -> list[dict]:
    props = fetch_nfl_lines()
    save_lines(props)
    return props


if __name__ == "__main__":
    props = get_underdog_lines()
    print(f"Fetched {len(props)} Underdog NFL props:")
    for p in props[:8]:
        print(f"  {p['player_name']} | {p['stat_type']} | {p['line']}")
