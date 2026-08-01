"""
Fetch NFL player injury/availability status from ESPN for this week's games.
No API key required. In-memory only (not persisted) — same as wnba-bet's
injuries.py, which this is adapted from. For historical week-by-week practice
reports used at training time, see pipeline/historical.py:pull_injuries()
(nfl_data_py's injuries release, which lags a week or two behind the current
slate and so isn't usable for live picks).

ESPN exposes two shapes here:
  * a per-team "injuries" endpoint with game status (Questionable/Doubtful/Out)
    and, when available, practice-report detail — tried first.
  * the roster endpoint's per-athlete "status" field (coarser: Active/Out/IR) —
    fallback if the injuries endpoint's response shape has drifted.
Run the __main__ smoke block after any ESPN response-shape change and check
which path actually returned rows.
"""
import requests
from datetime import date, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

TIMEOUT = 10

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json",
}

_ACTIVE = {"active"}
_WEEK_WINDOW_DAYS = 8


def _get(url: str, params: dict | None = None) -> dict:
    r = requests.get(url, headers=_HEADERS, params=params or {}, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()


def _week_team_ids() -> dict[str, str]:
    """{ESPN displayName: team_id} for every team with a game in the next
    _WEEK_WINDOW_DAYS days."""
    team_map = {}
    for offset in range(_WEEK_WINDOW_DAYS + 1):
        d = (date.today() + timedelta(days=offset)).strftime("%Y%m%d")
        try:
            data = _get(
                "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard",
                {"dates": d},
            )
        except Exception:
            continue
        for event in data.get("events", []):
            comp = event.get("competitions", [{}])[0]
            for c in comp.get("competitors", []):
                team = c.get("team", {})
                name, tid = team.get("displayName", ""), str(team.get("id", ""))
                if name and tid:
                    team_map[name] = tid
    return team_map


def _team_injuries(team_id: str) -> list[dict]:
    """Preferred source: dedicated injuries endpoint (game status + practice detail)."""
    try:
        data = _get(f"https://site.api.espn.com/apis/site/v2/sports/football/nfl/teams/{team_id}/injuries")
    except Exception:
        return []

    flags = []
    for item in data.get("injuries", []):
        athlete = item.get("athlete", {}) or item
        name = athlete.get("displayName") or athlete.get("fullName") or ""
        status = item.get("status") or item.get("type", {}).get("description", "")
        detail = (item.get("details") or {}).get("detail", "") or item.get("shortComment", "")
        if name and status:
            flags.append({"name": name, "status": status, "detail": detail})
    return flags


def _roster_flags(team_id: str) -> list[dict]:
    """Fallback: roster endpoint's per-athlete status (coarser than the
    injuries endpoint, but a reliable source). NFL rosters are grouped by
    position group, so athletes may be nested under "items"."""
    try:
        data = _get(f"https://site.api.espn.com/apis/site/v2/sports/football/nfl/teams/{team_id}/roster")
    except Exception:
        return []

    flags = []
    for group in data.get("athletes", []):
        for a in group.get("items", [group]):
            s = a.get("status", {})
            stype = s.get("type", "active").lower()
            sname = s.get("name", "Active")
            if stype not in _ACTIVE:
                flags.append({"name": a.get("fullName", "Unknown"), "status": sname, "detail": ""})
    return flags


def _team_injuries_or_roster(team_id: str) -> list[dict]:
    flags = _team_injuries(team_id)
    return flags if flags else _roster_flags(team_id)


def fetch_injury_flags(team_names: list[str]) -> dict[str, list[dict]]:
    """
    Return {team_name: [{"name": player, "status": "Questionable"/"Out"/...,
    "detail": ...}]} for teams that have at least one flagged player this week.

    team_names: list of team names as they appear in the Odds API schedule.
    """
    try:
        week_map = _week_team_ids()
    except Exception:
        return {}

    # Build name → team_id, fuzzy-matching Odds API names to ESPN display names
    resolved: dict[str, str] = {}
    for name in team_names:
        tid = week_map.get(name)
        if not tid:
            last = name.strip().split()[-1].lower()
            for espn_name, tid2 in week_map.items():
                if last in espn_name.lower():
                    tid = tid2
                    break
        if tid:
            resolved[name] = tid

    if not resolved:
        return {}

    result: dict[str, list[dict]] = {}
    with ThreadPoolExecutor(max_workers=len(resolved)) as ex:
        futures = {ex.submit(_team_injuries_or_roster, tid): name for name, tid in resolved.items()}
        for fut in as_completed(futures):
            name = futures[fut]
            flags = fut.result()
            if flags:
                result[name] = flags

    return result


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from pipeline.schedule import fetch_week_games

    games = fetch_week_games()
    teams = sorted({g["home_team"] for g in games} | {g["away_team"] for g in games})
    print(f"Checking injuries for {len(teams)} teams with games this week...")
    flags = fetch_injury_flags(teams)
    if not flags:
        print("  No flagged players found (or ESPN lookup failed).")
    for team, players in flags.items():
        print(f"  {team}:")
        for p in players:
            suffix = f" ({p['detail']})" if p.get("detail") else ""
            print(f"    {p['name']} — {p['status']}{suffix}")
