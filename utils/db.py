import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "../data/nfl_bet.db")


def get_conn():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    ensure_schema(conn)
    return conn


# Columns added after the original schema shipped. Existing local DBs predate
# them, so writers call ensure_schema() before INSERTing these columns.
_PROP_LINE_MIGRATIONS = (
    ("odds_type",         "TEXT DEFAULT 'standard'"),
    ("allowed_direction", "TEXT"),
)

# Kicking/defensive player props + the sacks-allowed opponent-adjustment
# metric, added after the original schema shipped.
_TEAM_GAME_LOG_MIGRATIONS = (
    ("off_sacks_allowed", "INTEGER"),
)
_TEAM_STATS_MIGRATIONS = (
    ("off_sacks_allowed_pg", "REAL"),
)
_PLAYER_GAME_LOG_MIGRATIONS = (
    ("fg_made", "REAL"), ("fg_att", "REAL"), ("fg_long", "REAL"),
    ("pat_made", "REAL"), ("pat_att", "REAL"),
    ("def_sacks", "REAL"), ("def_tackles_solo", "REAL"), ("def_tackle_assists", "REAL"),
    ("def_tackles_for_loss", "REAL"), ("def_qb_hits", "REAL"),
    ("def_interceptions", "REAL"), ("def_tds", "REAL"),
)


def ensure_schema(conn):
    """Additive, idempotent migrations for DBs created before newer columns existed."""
    for table, migrations in (
        ("prop_lines", _PROP_LINE_MIGRATIONS),
        ("team_game_logs", _TEAM_GAME_LOG_MIGRATIONS),
        ("team_stats", _TEAM_STATS_MIGRATIONS),
        ("player_game_logs", _PLAYER_GAME_LOG_MIGRATIONS),
    ):
        for col, typ in migrations:
            try:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {typ}")
            except sqlite3.OperationalError:
                pass  # column already exists (or table doesn't exist yet — init_db() creates it)
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
            rest_days INTEGER,
            off_sacks_allowed INTEGER
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
            def_epa_rush_pg REAL,
            off_sacks_allowed_pg REAL
        );

        -- One row per player per week. Offense/kicking/defense are three
        -- different nflverse source files unioned into one wide table (each
        -- row populated by whichever source it came from, other columns
        -- NULL) — simpler than three tables plus a merge at predict time,
        -- and get_player_profile()/predict_props() are already generic over
        -- "any stat column in this DataFrame."
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
            fantasy_points REAL,
            fg_made REAL,
            fg_att REAL,
            fg_long REAL,
            pat_made REAL,
            pat_att REAL,
            def_sacks REAL,
            def_tackles_solo REAL,
            def_tackle_assists REAL,
            def_tackles_for_loss REAL,
            def_qb_hits REAL,
            def_interceptions REAL,
            def_tds REAL
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
