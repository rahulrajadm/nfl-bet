"""
App-wide "today" is US Central time — the audience timezone and the anchor
for an NFL week. Naive date.today() breaks in the evening: SQLite's
DATE('now') is UTC and Streamlit Cloud runs in UTC, so Sunday/Monday Night
Football (kickoff ~7-8pm CT, running past midnight UTC) rolls to "tomorrow"
while the game is still being played.
"""
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

APP_TZ = ZoneInfo("America/Chicago")


def today_local() -> date:
    return datetime.now(APP_TZ).date()


def today_str() -> str:
    return today_local().isoformat()


def local_day_utc_bounds(day: date | None = None) -> tuple[str, str]:
    """UTC ISO bounds [start, end) of a Central calendar day, for filtering
    UTC fetched_at timestamps."""
    d = day or today_local()
    start = datetime(d.year, d.month, d.day, tzinfo=APP_TZ)
    end = start + timedelta(days=1)
    return (start.astimezone(timezone.utc).isoformat(),
            end.astimezone(timezone.utc).isoformat())
