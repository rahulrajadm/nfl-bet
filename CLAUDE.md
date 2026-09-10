# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

ML-driven NFL betting tool: XGBoost game models (spread/totals, moneyline derived from spread) + a hybrid normal/Poisson player-props model, compared against live odds and pick'em lines to surface +EV picks with confidence tiers and Kelly stakes. Sibling project to `wnba-bet` (same architecture — game-first, hybrid props distribution) and `mlb-bet` (props-first, pure Poisson); a few utilities (`utils/names.py`, `utils/dates.py`, `utils/team_names.py`) are ported from `mlb-bet` instead, since NFL player names carry suffixes and NFL games span timezones the way MLB's late games do. See `plan.md` for the full rationale and build order.

**Status: build complete (all 6 phases), live on Streamlit Community Cloud at [bet-nfl.streamlit.app](https://bet-nfl.streamlit.app).** Everything below reflects real, browser-tested behavior, not aspiration. See README.md's "Deployed to Streamlit Community Cloud" section for the setup steps that were used.

## Commands

```bash
# One-time local setup
pip install -r requirements.txt
cp .env.example .env              # add ODDS_API_KEY
python pipeline/historical.py     # pull historical seasons into data/nfl_bet.db via nfl_data_py
python models/train.py            # train + save models to data/models/

# Weekly local run (refreshes SQLite, grades last week's picks, opens local dashboard)
./start.sh

# Run the local dashboard directly
streamlit run ui/app.py

# Smoke-test modules directly (no test suite, same convention as wnba-bet/mlb-bet)
python models/props.py
python picks/engine.py
python pipeline/prizepicks.py
```

## Two entry points, two data paths

Same convention as wnba-bet/mlb-bet:
- **`ui/app_cloud.py`** — the deployed app (Streamlit Community Cloud). In-memory data, passcode-gated refresh (`REFRESH_CODE` in `st.secrets`).
- **`ui/app.py`** — local dashboard backed by SQLite (`data/nfl_bet.db`).

Model/metrics functions should take an optional DataFrame and fall back to SQLite when `None` — **preserve this convention when changing signatures**, a change that only handles one path silently breaks the other app.

## NFL-specific conventions (no analog in WNBA/MLB)

- **Weekly cadence, not daily.** Injury reports firm up Wed→Fri (DNP/Limited/Full → Questionable/Doubtful/Out), inactives post ~90 minutes before kickoff. Don't assume fresh games every day.
- **QB-out is a first-class feature.** No other sport in this family has a single position swing win probability this much — treat starting-QB availability as a required signal in the game model, not an afterthought.
- **Bye weeks must be excluded from rolling averages**, not treated as a missed/zero game.
- **Weather (`pipeline/weather.py`, open-meteo) is fetched live for display/context only — it is NOT a trained model feature.** Training would need a historical weather archive per game (open-meteo has a separate `archive-api` endpoint for this, not yet integrated); baking in an untrained wind/precip adjustment would be an uncalibrated guess. A high-wind outdoor game's model total may not reflect real wind risk until this is built.
- **Opponent defense is split by pass vs. rush**, not a single overall rating — NFL defenses vary far more by pass/rush than WNBA's single defensive-rating approach.
- **Game features are EPA-based** (`off_epa_pass/rush`, `def_epa_pass/rush`, `off/def_success_rate` from play-by-play via `pipeline/historical.py:_epa_by_team_game`), not box-score stats — richer signal than wnba-bet's points/rebounds rolling averages, but means `models/train.py` needs a full play-by-play pull (`nfl.import_pbp_data`), which is the slow part of `pipeline/historical.py`.
- **`nfl_data_py`'s release freshness varies by dataset** — as of this build, `import_schedules`/`import_pbp_data`/`import_injuries` were current through the 2025 season, but `import_weekly_data` (player weekly stats) still only went through 2024. `pipeline/historical.py` pulls every dataset one season at a time and skips a missing year with a warning rather than aborting the whole run — expect this gap to be present for a while each offseason.
- **`import_weekly_data` ships both an abbreviated `player_name` ("P.Mahomes") and a full `player_display_name` ("Patrick Mahomes")** — `pipeline/historical.py:pull_player_game_logs` explicitly drops the abbreviated column before renaming, because a naive `.rename()` collides the two into one column and silently keeps the wrong one. If a name-matching bug ever looks like "half the props don't match," check this first.
- **QB-out flag (`models/train.py:build_qb_out_flags`, `models/game.py:get_qb_out_flag`) is a per-season starter snapshot** (whoever has the most pass attempts that season), not mid-season-QB-change-aware. It also only checks the *current* starter's live injury status — a backup who becomes the starter mid-season won't be tracked as "the starter" until next season's aggregate updates.

## Known modeling caveats (deliberate, deferred)

- **No moneyline classifier is trained** — win probability is derived from the spread regressor (`norm.cdf(pred_diff / spread_std)`), same reasoning as wnba-bet: a separately-trained classifier produced win% that contradicted cover%.
- **Division-game flag (`pipeline/team_names.py:NFL_DIVISIONS`) is a hardcoded static mapping**, not sourced from live schedule data — division realignment last happened in 2002 (plus relocations, already reflected in `TEAM_ABBR_TO_NAME`), so this only needs revisiting if the NFL realigns again.
- **Rest-day handling relies on nflverse's `home_rest`/`away_rest` at training time** (correctly bye-week-aware) but is recomputed from `team_game_logs` date deltas at live predict time (`models/game.py:get_rest_days`) — the two paths should agree, but haven't been cross-checked against each other for a team coming off an actual bye during live prediction yet (no bye weeks have occurred in-season since this was built pre-season).
- Holdout numbers on the initial 2020-2025 training run: point-diff MAE ~10.5 pts (residual std ~13.3), totals MAE ~11.3 pts (residual std ~14.0) — in the same range as published Vegas-line accuracy, but this is a first pass, not a tuned model.
- **Props edges skew larger than wnba-bet/mlb-bet's, especially early in a season.** Validated against real live Underdog Week 1 (2026) lines: median edge ~10%, but a handful of props cleared +30% — traced one (Brian Thomas Jr. receiving yards) end to end and confirmed it's not a bug: his actual 2024 game log has a genuine hot closing stretch (100+ yards in 4 of his last 5 games) that the 55/45 recent/season blend picks up, compared against a Week-1-just-opened line that likely isn't sharp yet. `def_adj`/`pace_factor` were both sane and correctly bounded in that trace. Treat isolated huge-edge props as "the market hasn't caught up," not as a model defect — but if the *median* edge (not just the outliers) starts running consistently above ~15%, that would be a real signal something's off and worth re-checking.
- **Props' recent-form window is 4 games** (`RECENT_GAMES` in `models/props.py`), not wnba-bet's 10 — NFL's 17-game season makes a 10-game window nearly season-long, which would defeat the point of a "recent form" signal.
- **No per-minute/per-snap projection** (unlike wnba-bet's `per_min_season`/`per_min_recent`) — NFL snap counts aren't pulled into `player_game_logs` yet, so props use simple per-game rate blending only, same simplification mlb-bet makes.
- **Fantasy Score uses an approximate PPR-ish formula** (`models/props.py:load_player_logs`), not any one platform's exact scoring rules, which vary — same caveat class as wnba-bet's fantasy formula.
- **Deliberately unmodeled props** (`UNMODELED_STATS` in `models/props.py`): First TD Scorer (needs drive-order/game-flow modeling — genuinely different from a per-game-rate problem, unlike Anytime TD Scorer which is just `P(rush_rec_tds >= 1)`), 1H/1Q partial-game props (need in-game TD-timing, not built), Longest Reception/Rush/Completion (extreme-value stat, not a per-game-rate model). Add a stat here rather than letting it fall into `unknown_stat_types` if the skip is intentional.
- **Kicking and defensive-player props (added after initial ship) come from two separate nflverse files**, not the main `player_stats_{year}` release: `player_stats_kicking.parquet` and `player_stats_def.parquet` — confirmed live, and notably NOT split by year (one file, full history) unlike the main release, and neither carries an `opponent` column, so it's looked up via `pipeline/historical.py:team_week_opponent_map_from_df` against `team_game_logs`/schedules instead. `player_game_logs` is a wide table with all three sources unioned in (offense/kicking/defense columns all live in one table, each row populated by whichever source it came from) — simpler than three tables plus a merge at predict time, since `get_player_profile()`/`predict_props()` are already generic over "any stat column in this DataFrame."
- **Sacks get their own opponent-adjustment axis** (`pipeline/team_metrics.py:get_opp_sacks_allowed`/`get_sacks_def_adj`), not the pass/rush EPA split everything else uses — a pass rusher's sack total depends on the *opposing offense's* pass-block quality (a new `off_sacks_allowed` column on `team_game_logs`, aggregated from play-by-play's `sack` column), not their own team's defensive rating. **Tackles and kicking props get no opponent adjustment at all** (deliberate) — tackle volume already tracks the existing pace factor (more plays run against you = more tackle chances), and a kicker's volume is driven by their own offense's red-zone efficiency, not the opponent's defense.
- **Kicking Points uses flat scoring (3/FG, 1/PAT)** (`models/props.py:derive_prop_columns`) — some platforms weight FGs by distance (longer = more points); this is a simplification, same caveat class as the Fantasy Score formula below.
- Validated live: Aidan Hutchinson (elite pass rusher) correctly projected above-average expected sacks (1.14) vs. teammates/peers projected below 0.5; Justin Tucker's kicking-points rate (7.4/game) and Micah Parsons' low-tackle/high-sack profile (3.1 tackles, 0.92 sacks/game — matches his real role as an edge rusher, not an off-ball linebacker) all came out realistic.
- **Game picks systematically lean Under/underdog, same as wnba-bet.** Live-tested `picks/engine.py:build_picks` against real Week 1 (2026) odds across 40 games: 15/22 totals picks (68%) were Under, median game-pick edge ~9%. This is the same "raw model margins compress toward the mean" effect wnba-bet's CLAUDE.md documents — treat a lopsided Under/dog lean as expected model behavior, not a bug, unless the median edge itself drifts outside a sane single-digit-to-teens range.
- **`pipeline/cloud_data.py`'s player box-score category mapping is confirmed against one real completed game** (ESPN event 401772636, Colts 31 – Falcons 25, 2025-11-09) — passing/rushing/receiving label sets (`C/ATT`, `YDS`, `TD`, `INT`, `REC`, `TGTS`, etc.) matched exactly and `analysis/tracking.py`'s grading math was verified end to end against it (found the real final score and Bijan Robinson's real rushing/receiving lines correctly). Re-verify if ESPN changes its box-score JSON shape.
- **No per-minute what-if in `analysis/explain.py`** (unlike wnba-bet's `whatif_minutes`) — dropped rather than faked, since NFL snap counts aren't modeled (see above). `whatif_line` and `flip_direction` still work.

## UI (Phase 5) — architecture notes and what's been browser-tested

- **The cloud app can't get EPA from ESPN the way wnba-bet's cloud path gets box scores** — EPA requires real play-by-play, which only nflverse publishes; ESPN's public API has no equivalent. So unlike wnba-bet (all-ESPN in-memory fetch on refresh), `ui/app_cloud.py` calls `pipeline/cloud_data.py:fetch_team_game_logs`/`fetch_player_game_logs`, which hit `nfl_data_py` directly (GitHub's CDN, not IP-blocked the way stats.nba.com or PrizePicks' DataDome are) for the most recent seasons that actually have data — walks back up to `_MAX_SEASONS_BACK` years since nflverse's release freshness varies by dataset (see above).
- **`models/props.py:derive_prop_columns` is the shared combo-stat/name-key derivation** for both `load_player_logs()` (SQLite) and the cloud path — `predict_props()` also applies it defensively to any `player_logs_df` passed in, so a caller can't forget to call it first (this was a real gap during the build: passing a raw in-memory DataFrame silently produced zero combo-prop predictions until fixed).
- **Both `ui/app.py` and `ui/app_cloud.py` were launched and driven with a real browser (Playwright) against real live data**, not just import-checked: all six tabs clicked through on both apps, a real cloud refresh executed (nflverse + Odds API + PrizePicks + Underdog together — 1140 team-log rows, 10745 player-log rows, 59 total picks, 21 STRONG), the Game Predictions tab showed a real injury alert and a real +15.7%-edge spread pick, and the Ask Why chat was exercised with a real question + a real follow-up (both resolved correctly against actual computed picks). Zero browser console errors across either app. This was one preseason game with almost no market coverage yet (Odds API had no NFL lines beyond that one), so it validates the pipeline plumbing, not model accuracy — same caveat as everywhere else in this file.
- **QB-out is now surfaced in the UI, not just the model** — both apps show a "QB OUT ALERT" badge on a game's expander header and a warning line inside it when `predict_game`'s `home_qb_out`/`away_qb_out` flags are set, sourced from the same `injury_flags` passed into `build_game_picks`.
- **Local app's staleness threshold is 48h, not wnba-bet's 6h** (`ui/app.py:timestamp_bar`) — NFL is weekly, not daily; a 6h threshold would flag data as "stale" every Tuesday/Wednesday right after a Monday refresh.
