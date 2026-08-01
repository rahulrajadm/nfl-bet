# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

ML-driven NFL betting tool: XGBoost game models (spread/totals, moneyline derived from spread) + a hybrid normal/Poisson player-props model, compared against live odds and pick'em lines to surface +EV picks with confidence tiers and Kelly stakes. Sibling project to `wnba-bet` (same architecture — game-first, hybrid props distribution) and `mlb-bet` (props-first, pure Poisson); a few utilities (`utils/names.py`, `utils/dates.py`, `utils/team_names.py`) are ported from `mlb-bet` instead, since NFL player names carry suffixes and NFL games span timezones the way MLB's late games do. See `plan.md` for the full rationale and build order.

**Status: under active build.** This file will be filled in with real commands, caveats, and conventions as each phase lands — treat sections below as scaffold/placeholder until the corresponding phase is checked off in `plan.md`.

## Commands (target, once built)

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
- **Weather (`pipeline/weather.py`, open-meteo) feeds both the totals model and passing/kicking props** for outdoor stadiums only — indoor/retractable-roof games should skip it.
- **Opponent defense is split by pass vs. rush**, not a single overall rating — NFL defenses vary far more by pass/rush than WNBA's single defensive-rating approach.

## Known modeling caveats (deliberate, deferred)

To be filled in as the build progresses and real caveats are discovered (mirroring wnba-bet's and mlb-bet's CLAUDE.md — e.g., pick'em multiplier assumptions, per-leg Kelly sizing, unmodeled prop types).
