"""Risk profile assignment for NFL picks."""

# TD/turnover-type props are low-count, boom-or-bust (0 or 1 most games) —
# the highest-variance prop class, same role as wnba-bet's Steals/Blocks.
HIGH_VARIANCE_STATS = {
    "Pass TDs", "Passing TDs", "Pass Touchdowns",
    "Rush TDs", "Rushing TDs", "Rushing Touchdowns",
    "Receiving TDs", "Receiving Touchdowns",
    "Rush + Rec TDs", "Rushing + Receiving TDs",
    "INTs Thrown", "Interceptions", "Interceptions Thrown",
}
MEDIUM_VARIANCE_STATS = {
    "Receptions", "Completions", "Pass Completions",
}
LOW_VARIANCE_STATS = {
    "Pass Yards", "Passing Yards",
    "Rush Yards", "Rushing Yards",
    "Receiving Yards",
    "Rush + Rec Yards", "Rushing + Receiving Yards",
    "Pass + Rush Yards", "Pass + Rush + Rec Yards",
    "Fantasy Score", "Fantasy Points",
}

# Game pick risk profiles
GAME_RISK = {
    "moneyline": "MEDIUM",
    "spread":    "MEDIUM",
    "totals":    "LOW",
}


def get_risk_profile(stat_type: str, edge: float = 0.0) -> str:
    if stat_type in HIGH_VARIANCE_STATS:
        return "HIGH"
    if stat_type in MEDIUM_VARIANCE_STATS:
        return "MEDIUM"
    if stat_type in LOW_VARIANCE_STATS:
        return "LOW" if edge >= 0.10 else "MEDIUM"
    if stat_type in GAME_RISK:
        return GAME_RISK[stat_type]
    return "MEDIUM"


RISK_COLORS = {
    "LOW":    "#16a34a",
    "MEDIUM": "#c2410c",
    "HIGH":   "#dc2626",
}
