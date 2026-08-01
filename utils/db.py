import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "../data/nfl_bet.db")


def get_conn():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    return sqlite3.connect(DB_PATH)


# Columns added after the original schema shipped. Existing local DBs predate
# them, so writers call ensure_schema() before INSERTing these columns.
_PROP_LINE_MIGRATIONS = (
    ("odds_type",         "TEXT DEFAULT 'standard'"),
    ("allowed_direction", "TEXT"),
)


def ensure_schema(conn):
    """Additive, idempotent migrations for DBs created before newer columns existed."""
    for col, typ in _PROP_LINE_MIGRATIONS:
        try:
            conn.execute(f"ALTER TABLE prop_lines ADD COLUMN {col} {typ}")
        except sqlite3.OperationalError:
            pass  # column already exists
    conn.execute("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT)")
    conn.commit()


def set_meta(key: str, value: str):
    conn = get_conn()
    ensure_schema(conn)
    conn.execute("INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)", (key, str(value)))
    conn.commit()
    conn.close()


def get_meta(key: str) -> str | None:
    conn = get_conn()
    ensure_schema(conn)
    row = conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
    conn.close()
    return row[0] if row else None


def init_db():
    conn = get_conn()
    c = conn.cursor()

    c.executescript("""
        CREATE TABLE IF NOT EXISTS games (
            game_id TEXT PRIMARY KEY,
            season INTEGER,
            week INTEGER,
            season_type TEXT,
            date TEXT,
            home_team TEXT,
            away_team TEXT,
            home_score INTEGER,
            away_score INTEGER,
            game_time TEXT,
            stadium TEXT,
            roof TEXT,
            surface TEXT,
            div_game INTEGER
        );

        -- Two rows per game (one per team's perspective), the shape every
        -- rolling-average / feature-engineering pass over team history wants.
        CREATE TABLE IF NOT EXISTS team_game_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            game_id TEXT,
            season INTEGER,
            week INTEGER,
            season_type TEXT,
            date TEXT,
            team TEXT,
            opponent TEXT,
            is_home INTEGER,
            points_for INTEGER,
            points_against INTEGER,
            result TEXT,
            off_epa_pass REAL,
            off_epa_rush REAL,
            def_epa_pass REAL,
            def_epa_rush REAL,
            off_success_rate REAL,
            def_success_rate REAL,
            plays_offense INTEGER,
            plays_defense INTEGER,
            rest_days INTEGER
        );

        CREATE TABLE IF NOT EXISTS team_stats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            season INTEGER,
            team TEXT,
            gp INTEGER,
            w INTEGER,
            l INTEGER,
            t INTEGER,
            pts_pg REAL,
            opp_pts_pg REAL,
            off_epa_pass_pg REAL,
            off_epa_rush_pg REAL,
            def_epa_pass_pg REAL,
            def_epa_rush_pg REAL
        );

        CREATE TABLE IF NOT EXISTS player_game_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            season INTEGER,
            week INTEGER,
            season_type TEXT,
            game_id TEXT,
            player_id TEXT,
            player_name TEXT,
            position TEXT,
            team TEXT,
            opponent TEXT,
            passing_yards REAL,
            passing_tds REAL,
            interceptions REAL,
            completions REAL,
            attempts REAL,
            rushing_yards REAL,
            rushing_tds REAL,
            carries REAL,
            receiving_yards REAL,
            receiving_tds REAL,
            receptions REAL,
            targets REAL,
            fumbles_lost REAL,
            fantasy_points REAL
        );

        CREATE TABLE IF NOT EXISTS injuries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            fetched_at TEXT,
            season INTEGER,
            week INTEGER,
            team TEXT,
            player_name TEXT,
            position TEXT,
            practice_status TEXT,
            game_status TEXT
        );

        CREATE TABLE IF NOT EXISTS weather (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            fetched_at TEXT,
            game_id TEXT,
            temp_f REAL,
            wind_mph REAL,
            precip_pct REAL,
            condition TEXT,
            is_dome INTEGER
        );

        CREATE TABLE IF NOT EXISTS game_odds (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            fetched_at TEXT,
            platform TEXT,
            game_id TEXT,
            home_team TEXT,
            away_team TEXT,
            market TEXT,
            home_odds REAL,
            away_odds REAL,
            home_spread REAL,
            away_spread REAL,
            over_odds REAL,
            under_odds REAL,
            total_line REAL
        );

        CREATE TABLE IF NOT EXISTS prop_lines (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            fetched_at TEXT,
            platform TEXT,
            game_id TEXT,
            player_name TEXT,
            player_team TEXT,
            stat_type TEXT,
            line REAL,
            more_odds REAL,
            less_odds REAL,
            odds_type TEXT DEFAULT 'standard',
            allowed_direction TEXT
        );

        CREATE TABLE IF NOT EXISTS picks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            generated_at TEXT,
            pick_type TEXT,
            selection TEXT,
            best_platform TEXT,
            model_prob REAL,
            implied_prob REAL,
            edge REAL,
            ev_per_100 REAL,
            confidence_tier TEXT,
            risk_profile TEXT,
            kelly_pct REAL,
            units REAL,
            details TEXT
        );

        CREATE TABLE IF NOT EXISTS meta (
            key TEXT PRIMARY KEY,
            value TEXT
        );
    """)

    ensure_schema(conn)
    conn.commit()
    conn.close()
    print("NFL database initialized.")


if __name__ == "__main__":
    init_db()
