# NFL Bet: AI-Powered NFL Betting Decision Tool

An ML-driven tool that predicts NFL game outcomes (moneyline, spread, totals) and player performance, comparing model probabilities against live odds and pick'em lines to surface +EV picks with confidence scores, risk profiles, and Kelly-sized stakes.

**Primary focus: game predictions (ML, spread, totals)**
**Secondary focus: player props (Passing/Rushing/Receiving Yards, TDs, Receptions, INTs, etc.)**

Sibling project to [wnba-bet](https://github.com/rahulrajadm/wnba-bet) and [mlb-bet](https://github.com/rahulrajadm/mlb-bet) — same architecture, adapted for NFL. See `plan.md` for the full design doc and build order.

---

## Live Demo

🚀 **[bet-nfl.streamlit.app](https://bet-nfl.streamlit.app)**

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

All 6 build phases complete (see `plan.md`) and validated against real live data — schedule, odds, PrizePicks/Underdog props, injuries, weather, a full nflverse historical pull, a trained game model, and both Streamlit apps driven end-to-end in a real browser (zero console errors). **Live on Streamlit Community Cloud** — see [Live Demo](#live-demo) above.

---

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env              # add ODDS_API_KEY
python pipeline/historical.py     # pull historical seasons into data/nfl_bet.db (~5-10 min, mostly play-by-play)
python models/train.py            # train + save models to data/models/ (already committed — this retrains)

# Weekly local run
./start.sh

# Run the local dashboard directly
streamlit run ui/app.py
```

---

## Deployed to Streamlit Community Cloud

Already live at [bet-nfl.streamlit.app](https://bet-nfl.streamlit.app) (repo `rahulrajadm/nfl-bet`, main file `ui/app_cloud.py`). For reference, these were the one-time manual setup steps:

1. [share.streamlit.io](https://share.streamlit.io) → sign in with the GitHub account that owns the repo.
2. **New app** → repo `rahulrajadm/nfl-bet`, branch `main`, main file path `ui/app_cloud.py`.
3. Under **Advanced settings → Secrets**:
   ```toml
   REFRESH_CODE = "pick a 4-digit code"
   ```
   (`ODDS_API_KEY` is optional there too — without it the cloud app's game markets show model-only projections, same as local without a key.)
4. On first load, enter the passcode and hit **Refresh All Data** in the sidebar (costs ~3 Odds API credits) to populate data.
5. If the app name ever moves off `bet-nfl.streamlit.app`, update the URL in `.github/workflows/keep-alive.yml` to match.

---

## Built by

Rahul Raja Durai Murugan
BS Biomedical Engineering, UT Austin · MS Engineering Data Science & AI, University of Houston (incoming)
