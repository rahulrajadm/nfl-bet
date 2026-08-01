# NFL Bet: AI-Powered NFL Betting Decision Tool

An ML-driven tool that predicts NFL game outcomes (moneyline, spread, totals) and player performance, comparing model probabilities against live odds and pick'em lines to surface +EV picks with confidence scores, risk profiles, and Kelly-sized stakes.

**Primary focus: game predictions (ML, spread, totals)**
**Secondary focus: player props (Passing/Rushing/Receiving Yards, TDs, Receptions, INTs, etc.)**

Sibling project to [wnba-bet](https://github.com/rahulrajadm/wnba-bet) and [mlb-bet](https://github.com/rahulrajadm/mlb-bet) — same architecture, adapted for NFL. See `plan.md` for the full design doc and build order.

---

## Live Demo

🚀 **[nfl-bet.streamlit.app](https://nfl-bet.streamlit.app)** *(pending initial deploy)*

---

## What it does

1. **Fetches live data** — NFL schedule, PrizePicks + Underdog props, Odds API game lines, weather for outdoor games
2. **Predicts game outcomes** using XGBoost models trained on multiple seasons of NFL play-by-play (EPA-based features)
3. **Predicts player props** using per-player stat models with:
   - Season averages + recent-form blend
   - Opponent defense adjustment (pass/rush split)
   - Plays-per-game / pace adjustment
   - Hybrid normal (yardage) + Poisson (TDs/INTs/receptions) distributions
4. **Computes edge** — model probability vs. bookmaker implied probability (de-vigged) or pick'em break-even
5. **Ranks picks** by confidence tier (STRONG / HIGH / MEDIUM / LOW) and risk profile
6. **Sizes stakes** using fractional Kelly criterion (0.25×) in units

---

## Status

Build in progress — see `plan.md` for phase-by-phase status.

---

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env              # add ODDS_API_KEY
python pipeline/historical.py     # pull historical seasons into data/nfl_bet.db
python models/train.py            # train + save models to data/models/

# Weekly local run
./start.sh

# Run the local dashboard directly
streamlit run ui/app.py
```

---

## Built by

Rahul Raja Durai Murugan
BS Biomedical Engineering, UT Austin · MS Engineering Data Science & AI, University of Houston (incoming)
