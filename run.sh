#!/usr/bin/env bash
# GapFinder launcher for macOS/Linux: creates a private Python environment the first time, then starts the app.
cd "$(dirname "$0")"
if [ ! -d .venv ]; then
  echo "First run: creating the environment and installing packages (a few minutes)..."
  (python3.12 -m venv .venv 2>/dev/null || python3 -m venv .venv)
fi
source .venv/bin/activate
python -m pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt || { echo "Installation failed. Try Python 3.12 and delete the .venv folder."; exit 1; }
streamlit run app.py
