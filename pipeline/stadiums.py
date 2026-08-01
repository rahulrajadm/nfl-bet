"""
NFL stadium metadata: location (for weather lookups) and roof type (whether
weather applies at all).

roof_type: "outdoor" (fully exposed), "dome" (fixed roof, weather irrelevant),
"retractable" (usually closed for game-time comfort/rain but can be open —
treated as outdoor-risk unless a closed-roof feed is added later).

Caveat (same class of staleness as MLB's park_factors.py venue-name caveat):
stadium sponsorship names change often (Browns/FirstEnergy/Huntington Bank,
Commanders/FedEx, Jaguars/TIAA Bank/EverBank) and Tennessee's franchise has a
new stadium in development — re-verify names/coordinates/roof status each
season rather than trusting this table blindly.
"""

# keyed by nflverse team abbreviation (pipeline/team_names.py canonical form)
STADIUMS = {
    "ARI": {"name": "State Farm Stadium",         "lat": 33.5276,  "lon": -112.2626, "roof_type": "retractable"},
    "ATL": {"name": "Mercedes-Benz Stadium",       "lat": 33.7554,  "lon": -84.4008,  "roof_type": "retractable"},
    "BAL": {"name": "M&T Bank Stadium",            "lat": 39.2780,  "lon": -76.6227,  "roof_type": "outdoor"},
    "BUF": {"name": "Highmark Stadium",            "lat": 42.7738,  "lon": -78.7870,  "roof_type": "outdoor"},
    "CAR": {"name": "Bank of America Stadium",     "lat": 35.2258,  "lon": -80.8528,  "roof_type": "outdoor"},
    "CHI": {"name": "Soldier Field",               "lat": 41.8623,  "lon": -87.6167,  "roof_type": "outdoor"},
    "CIN": {"name": "Paycor Stadium",              "lat": 39.0954,  "lon": -84.5160,  "roof_type": "outdoor"},
    "CLE": {"name": "Huntington Bank Field",       "lat": 41.5061,  "lon": -81.6995,  "roof_type": "outdoor"},
    "DAL": {"name": "AT&T Stadium",                "lat": 32.7473,  "lon": -97.0945,  "roof_type": "retractable"},
    "DEN": {"name": "Empower Field at Mile High",  "lat": 39.7439,  "lon": -105.0201, "roof_type": "outdoor"},
    "DET": {"name": "Ford Field",                  "lat": 42.3400,  "lon": -83.0456,  "roof_type": "dome"},
    "GB":  {"name": "Lambeau Field",               "lat": 44.5013,  "lon": -88.0622,  "roof_type": "outdoor"},
    "HOU": {"name": "NRG Stadium",                 "lat": 29.6847,  "lon": -95.4107,  "roof_type": "retractable"},
    "IND": {"name": "Lucas Oil Stadium",           "lat": 39.7601,  "lon": -86.1639,  "roof_type": "retractable"},
    "JAX": {"name": "EverBank Stadium",            "lat": 30.3239,  "lon": -81.6373,  "roof_type": "outdoor"},
    "KC":  {"name": "GEHA Field at Arrowhead Stadium", "lat": 39.0489, "lon": -94.4839, "roof_type": "outdoor"},
    "LA":  {"name": "SoFi Stadium",                "lat": 33.9535,  "lon": -118.3392, "roof_type": "dome"},
    "LAC": {"name": "SoFi Stadium",                "lat": 33.9535,  "lon": -118.3392, "roof_type": "dome"},
    "LV":  {"name": "Allegiant Stadium",           "lat": 36.0909,  "lon": -115.1833, "roof_type": "dome"},
    "MIA": {"name": "Hard Rock Stadium",           "lat": 25.9580,  "lon": -80.2389,  "roof_type": "outdoor"},
    "MIN": {"name": "U.S. Bank Stadium",           "lat": 44.9738,  "lon": -93.2577,  "roof_type": "dome"},
    "NE":  {"name": "Gillette Stadium",            "lat": 42.0909,  "lon": -71.2643,  "roof_type": "outdoor"},
    "NO":  {"name": "Caesars Superdome",           "lat": 29.9511,  "lon": -90.0812,  "roof_type": "dome"},
    "NYG": {"name": "MetLife Stadium",             "lat": 40.8135,  "lon": -74.0745,  "roof_type": "outdoor"},
    "NYJ": {"name": "MetLife Stadium",             "lat": 40.8135,  "lon": -74.0745,  "roof_type": "outdoor"},
    "PHI": {"name": "Lincoln Financial Field",     "lat": 39.9008,  "lon": -75.1675,  "roof_type": "outdoor"},
    "PIT": {"name": "Acrisure Stadium",            "lat": 40.4468,  "lon": -80.0158,  "roof_type": "outdoor"},
    "SF":  {"name": "Levi's Stadium",              "lat": 37.4032,  "lon": -121.9698, "roof_type": "outdoor"},
    "SEA": {"name": "Lumen Field",                 "lat": 47.5952,  "lon": -122.3316, "roof_type": "outdoor"},
    "TB":  {"name": "Raymond James Stadium",       "lat": 27.9759,  "lon": -82.5033,  "roof_type": "outdoor"},
    "TEN": {"name": "Nissan Stadium",              "lat": 36.1665,  "lon": -86.7713,  "roof_type": "outdoor"},
    "WAS": {"name": "Commanders Field",            "lat": 38.9077,  "lon": -76.8645,  "roof_type": "outdoor"},
}


def needs_weather(team_abbr: str) -> bool:
    """True unless the team's home is a fixed dome (weather can't affect the game)."""
    info = STADIUMS.get(team_abbr)
    return info is None or info["roof_type"] != "dome"


def get_stadium(team_abbr: str) -> dict | None:
    return STADIUMS.get(team_abbr)
