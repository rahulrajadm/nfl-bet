# NFL Bet: AI-Powered NFL Betting Decision Tool

A local + cloud tool that predicts NFL game outcomes (moneyline, spread, totals) and player performance, compares model probabilities against live odds and pick'em lines, and surfaces +EV picks with confidence scores, risk profiles, and Kelly-sized stakes.

**Primary focus: game predictions (ML, spread, totals)**
**Secondary focus: player props (Passing/Rushing/Receiving Yards, TDs, Receptions, INTs, etc.)**

Scaffolded from `wnba-bet` (game-first architecture, hybrid normal+Poisson props model) with a few pieces pulled from `mlb-bet` (`utils/names.py` suffix-safe matching, `utils/dates.py` timezone-safe day boundary, exact-dict `team_names.py`). See the top-level build plan for the full rationale.

---

## Texas Legal Platforms

Same as wnba-bet / mlb-bet — all Texas-legal:

| Platform | Type | NFL coverage |
|---|---|---|
| **PrizePicks** | Pick'em DFS | ✅ High-volume during season |
| **Underdog Fantasy** | Pick'em DFS | ✅ Active |
| **Fliff** | Sweepstakes | ✅ ML, spread, totals |
| **Polymarket** | Prediction market | ✅ Game outcomes |
| **DraftKings Pick6** | Pick'em DFS | ✅ Active |
| **Chalkboard** | Pick'em DFS | ✅ Active |
| **Sleeper** | Pick'em / Prediction | ✅ Active |

---

## Output Per Pick

### Game predictions (ML, spread, totals)
| Field | Description |
|---|---|
| **Selection** | "Chiefs ML", "Chiefs -3.5", "Over 47.5" |
| **Best platform** | Which platform has the best line/odds for this pick |
| **Model probability** | Model's estimated win/cover/hit probability |
| **Implied probability** | What the odds imply (de-vigged) |
| **Edge** | Model prob − implied prob |
| **EV per $100** | Expected value |
| **Confidence tier** | STRONG / HIGH / MEDIUM / LOW |
| **Risk profile** | LOW / MEDIUM / HIGH |
| **Units** | Quarter-Kelly stake in units |

### Player props (pick'em)
Model prediction vs line, edge vs pick'em break-even, confidence + risk + units.

---

## Game Prediction Model

Trained ML (XGBoost), same pattern as wnba-bet — game markets are core, not an afterthought.

### Features
| Category | Features |
|---|---|
| Offense/Defense | EPA per play (pass/rush split), success rate, points per game — from `nfl_data_py` play-by-play |
| Recent form | Rolling 4-game and 8-game averages |
| Rest | Days since last game; short-week (Thursday) and long-week (post-bye/Monday) flags; **bye weeks excluded from rolling averages, not treated as a 0** |
| Home/away | Home field win% splits |
| Head-to-head | Season/recent H2H record and scoring margin |
| Weather | Wind/precipitation for outdoor stadiums (open-meteo forecast); dome/retractable-roof flag |
| QB availability | Starting-QB-out flag — the single largest swing factor in NFL win probability, with no analog in WNBA/MLB |

### Models
- **Moneyline**: derived from the spread regressor via `norm.cdf`, same as wnba-bet (the spread model is the single source of truth for win probability; a separate classifier is not used)
- **Spread**: XGBoost regressor → predicted point differential → P(cover), calibrated on holdout residuals
- **Totals**: XGBoost regressor → predicted total points → P(over), same calibration approach

### EV calculation (game picks)
- Odds API provides American odds → implied probability (de-vigged, `analysis/ev.py:remove_vig`)
- Edge = model prob − true implied prob
- Predictions are shrunk toward the market line before probabilities are computed (`MODEL_WEIGHT` anchoring in `picks/engine.py`), same convention as wnba-bet

---

## Player Props Model

Hybrid distribution model (ported from wnba-bet's `prob_over_line`, not MLB's pure Poisson):
- **Normal distribution** for continuous/high-volume stats (passing/rushing/receiving yards)
- **Poisson distribution** for low-count discrete stats (passing/rushing/receiving TDs, INTs, receptions)
- Season averages + recent-form blend (last 4 games weighted vs. season)
- Opponent defense adjustment, split by **pass defense vs. rush defense** (NFL defenses vary far more by pass/rush than a single overall rating)
- Plays-per-game / pace adjustment

### Prop types covered
Passing Yards, Passing TDs, Interceptions, Completions, Rushing Yards, Rushing TDs, Receptions, Receiving Yards, Receiving TDs, Rush+Rec Yards, Pass+Rush+Rec Yards, Fantasy Score, Kicking Points

---

## Data Sources

| Source | Data | Access |
|---|---|---|
| `nfl_data_py` (nflverse) | Historical team/player game logs, play-by-play EPA, schedules, injuries, rosters | Free Python package, no key — verify current package name at build time (nflverse ecosystem has shifted before) |
| ESPN unofficial API | Live schedule, box scores, injury/practice reports | Free, no key; cloud-safe (no datacenter-IP block issue) |
| The Odds API | NFL moneyline, spread, totals across all books (`americanfootball_nfl`) | Free tier (shared key with wnba-bet/mlb-bet) |
| PrizePicks API | Live NFL player prop lines | Free unofficial endpoint |
| Underdog API | Live NFL player prop lines | Free unofficial endpoint |
| open-meteo | Wind/precipitation forecast for outdoor stadiums | Free, no key |

---

## Stack

| Layer | Tool |
|---|---|
| Historical stats | `nfl_data_py` |
| Live schedule/injuries | ESPN unofficial API |
| Live odds | The Odds API (`americanfootball_nfl`) |
| Live props | PrizePicks · Underdog |
| Weather | open-meteo |
| ML models | XGBoost + scikit-learn |
| Storage (local) | SQLite |
| Dashboard | Streamlit |

---

## Project Structure

```
nfl-bet-app/
├── plan.md
├── .env.example              # ODDS_API_KEY (shared with wnba-bet/mlb-bet)
├── requirements.txt
├── data/
│   └── nfl_bet.db            # SQLite: game logs, team stats, player stats, odds, prop lines, picks
├── pipeline/
│   ├── historical.py         # nfl_data_py: multi-season team/player game logs + play-by-play EPA
│   ├── schedule.py           # ESPN API: this week's schedule
│   ├── odds.py                # Odds API: h2h, spreads, totals for americanfootball_nfl
│   ├── prizepicks.py          # PrizePicks NFL prop lines
│   ├── underdog.py            # Underdog NFL prop lines
│   ├── injuries.py            # ESPN injury/practice reports (Wed/Thu/Fri → Q/D/Out → inactives)
│   └── weather.py             # open-meteo wind/precip for outdoor stadiums
├── models/
│   ├── train.py               # Feature engineering + train all models
│   ├── game.py                # Spread, totals game prediction models (moneyline derived from spread)
│   └── props.py               # Player prop prediction engine (hybrid normal/Poisson)
├── analysis/
│   ├── ev.py                  # EV calculation (traditional odds + pick'em)
│   ├── confidence.py          # Confidence tier assignment
│   ├── risk.py                # Risk profile assignment
│   ├── kelly.py               # Fractional Kelly sizing
│   ├── tracking.py            # Pick grading
│   ├── backtest.py            # Historical backtest
│   └── explain.py             # Ask-Why explain payloads
├── picks/
│   └── engine.py               # Assembles game + prop picks, ranks by EV
├── ui/
│   ├── app.py                  # Local Streamlit dashboard (SQLite)
│   └── app_cloud.py             # Cloud Streamlit dashboard (in-memory)
└── utils/
    ├── db.py                   # SQLite schema + helpers
    ├── names.py                 # Suffix-safe player name matching (ported from mlb-bet)
    ├── dates.py                  # Timezone-safe "today" boundary (ported from mlb-bet)
    └── team_names.py              # Exact-dict team identity mapping (mlb-bet pattern)
```

---

## Build Order

### Phase 0 — Scaffold
- [x] Repo structure, `.gitignore`, `LICENSE`, `.streamlit/`, `.github/workflows/keep-alive.yml`, `requirements.txt`/`requirements-local.txt`

### Phase 1 — Data Pipeline
- [ ] `utils/db.py` — SQLite schema: games, team_stats, player_stats, odds, prop_lines, picks (with `week`, `season_type`)
- [ ] `utils/names.py`, `utils/dates.py`, `utils/team_names.py`
- [ ] `pipeline/historical.py` — multi-season NFL game logs + player stats + play-by-play EPA via `nfl_data_py`
- [ ] `pipeline/schedule.py` — this week's NFL schedule from ESPN API
- [ ] `pipeline/injuries.py` — practice reports + inactives
- [ ] `pipeline/weather.py` — outdoor stadium wind/precip
- [ ] `pipeline/odds.py` — live ML, spread, totals from Odds API
- [ ] `pipeline/prizepicks.py` — live NFL prop lines
- [ ] `pipeline/underdog.py` — live NFL prop lines

### Phase 2 — Game Models (primary focus)
- [ ] `models/train.py` — feature engineering: EPA-based team ratings, rest, weather, QB-out flag, H2H
- [ ] `models/game.py` — train spread, totals XGBoost models with holdout calibration; serialize to disk

### Phase 3 — Props Model
- [ ] `models/props.py` — per-player season + recent-form rates; hybrid normal/Poisson probability vs. line
- [ ] Opponent pass/rush defense adjustment
- [ ] Pace/plays-per-game adjustment

### Phase 4 — Analysis Engine
- [ ] `analysis/ev.py`, `confidence.py`, `risk.py`, `kelly.py` — ported from wnba-bet
- [ ] `analysis/tracking.py`, `backtest.py`, `explain.py` — ported from wnba-bet
- [ ] `picks/engine.py` — assemble + rank all picks; market anchoring; tag best platform

### Phase 5 — UI
- [ ] `ui/app.py` — local Streamlit: Game Predictions, Player Props, Top Picks, Bankroll, Platform Comparison
- [ ] `ui/app_cloud.py` — cloud version with passcode refresh + timestamp

### Phase 6 — Deploy
- [ ] GitHub repo `nfl-bet` (public)
- [ ] Streamlit Cloud deployment
- [ ] Live smoke test through preseason weeks

---

## Key Differences from WNBA Bet / MLB Bet

| | MLB Bet | WNBA Bet | NFL Bet |
|---|---|---|---|
| Primary focus | Player props | Game predictions | **Game predictions** |
| Game model | Heuristic (no ML) | XGBoost | **XGBoost** |
| Props distribution | Poisson only | Normal + Poisson | **Normal + Poisson** |
| Cadence | Daily | Daily | **Weekly** |
| Weather | No | No (indoor) | **Yes — wind/precip via open-meteo** |
| Dominant single-position effect | No | No | **Yes — starting QB availability** |
| Opponent defense | Via K-rate | Single defensive rating | **Split by pass/rush defense** |
| Bye weeks | N/A | N/A | **Yes — must exclude from rolling averages** |

---

## Setup (after build)

```bash
cd nfl-bet-app
pip install -r requirements.txt
cp .env.example .env
python pipeline/historical.py    # one-time: pull historical data
python models/train.py           # one-time: train models
streamlit run ui/app.py          # weekly: run dashboard
```

---

## Built by

Rahul Raja Durai Murugan
BS Biomedical Engineering, UT Austin · MS Engineering Data Science & AI, University of Houston (incoming)
