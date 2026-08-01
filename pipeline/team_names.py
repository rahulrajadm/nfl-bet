"""
Canonical NFL team identity mapping.

nfl_data_py/nflverse uses short codes ("KC", "SF"); ESPN and the Odds API use
full names ("Kansas City Chiefs"); PrizePicks/Underdog send their own
abbreviations, which occasionally differ from nflverse's (e.g. "JAC" vs.
nflverse's "JAX", "WSH" vs. "WAS"). Relocated franchises show up under old
codes in historical nflverse data (OAK→LV, SD→LAC, STL→LA). Everything
downstream (opponent, stadium, game grouping) matches on the full name via
to_full_name() — exact dict mapping, no substring matching (NFL city names
like "Bay" collide across teams).
"""

TEAM_ABBR_TO_NAME = {
    "ARI": "Arizona Cardinals",
    "ATL": "Atlanta Falcons",
    "BAL": "Baltimore Ravens",
    "BUF": "Buffalo Bills",
    "CAR": "Carolina Panthers",
    "CHI": "Chicago Bears",
    "CIN": "Cincinnati Bengals",
    "CLE": "Cleveland Browns",
    "DAL": "Dallas Cowboys",
    "DEN": "Denver Broncos",
    "DET": "Detroit Lions",
    "GB":  "Green Bay Packers",   "GNB": "Green Bay Packers",
    "HOU": "Houston Texans",
    "IND": "Indianapolis Colts",
    "JAX": "Jacksonville Jaguars", "JAC": "Jacksonville Jaguars",
    "KC":  "Kansas City Chiefs",  "KAN": "Kansas City Chiefs",
    "LA":  "Los Angeles Rams",    "LAR": "Los Angeles Rams",    "STL": "Los Angeles Rams",
    "LAC": "Los Angeles Chargers", "SD":  "Los Angeles Chargers", "SDG": "Los Angeles Chargers",
    "LV":  "Las Vegas Raiders",   "OAK": "Las Vegas Raiders",
    "MIA": "Miami Dolphins",
    "MIN": "Minnesota Vikings",
    "NE":  "New England Patriots", "NWE": "New England Patriots",
    "NO":  "New Orleans Saints",  "NOR": "New Orleans Saints",
    "NYG": "New York Giants",
    "NYJ": "New York Jets",
    "PHI": "Philadelphia Eagles",
    "PIT": "Pittsburgh Steelers",
    "SF":  "San Francisco 49ers", "SFO": "San Francisco 49ers",
    "SEA": "Seattle Seahawks",
    "TB":  "Tampa Bay Buccaneers", "TAM": "Tampa Bay Buccaneers",
    "TEN": "Tennessee Titans",
    "WAS": "Washington Commanders", "WSH": "Washington Commanders",
}

_NAME_TO_ABBR = {}
for abbr, name in TEAM_ABBR_TO_NAME.items():
    _NAME_TO_ABBR.setdefault(name, abbr)  # first (canonical) abbr wins


def to_full_name(team: str) -> str:
    """Abbreviation or full name → full name. '' for unknown/combo teams."""
    if not team:
        return ""
    team = team.strip()
    if "/" in team:  # multi-team combo prop label
        return ""
    if team in _NAME_TO_ABBR:  # already a full name
        return team
    return TEAM_ABBR_TO_NAME.get(team.upper(), "")


def to_abbr(team: str) -> str:
    """Full name or abbreviation → canonical (nflverse) abbreviation. Input echoed if unknown."""
    if not team:
        return ""
    team = team.strip()
    if team in _NAME_TO_ABBR:
        return _NAME_TO_ABBR[team]
    if team.upper() in TEAM_ABBR_TO_NAME:
        return TEAM_ABBR_TO_NAME[team.upper()] and _NAME_TO_ABBR[TEAM_ABBR_TO_NAME[team.upper()]]
    return team
