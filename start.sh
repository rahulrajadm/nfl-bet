#!/bin/bash
# NFL Bet -- weekly launcher
# Refreshes all data then opens the dashboard

set -e
cd "$(dirname "$0")"

echo "NFL Bet starting..."
echo ""

echo "Fetching this week's schedule..."
python pipeline/schedule.py

echo "Checking injuries..."
python pipeline/injuries.py

echo "Fetching weather for outdoor games..."
python pipeline/weather.py

echo "Fetching PrizePicks lines..."
python pipeline/prizepicks.py

echo "Fetching Underdog lines..."
python pipeline/underdog.py

echo "Fetching odds (ML, spread, totals)..."
python pipeline/odds.py

echo "Grading last week's picks..."
python analysis/tracking.py grade

echo ""
echo "Data ready. Opening dashboard..."
echo ""

streamlit run ui/app.py
