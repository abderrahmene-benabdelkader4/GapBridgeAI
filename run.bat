@echo off
REM GapFinder launcher for Windows: creates a private Python environment the first time, then starts the app.
cd /d "%~dp0"
if not exist .venv (
  echo First run: creating the environment and installing packages (a few minutes)...
  py -3.12 -m venv .venv 2>nul || py -3 -m venv .venv 2>nul || python -m venv .venv
)
call .venv\Scripts\activate.bat
python -m pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt || (echo Installation failed. If you use a very new Python, install Python 3.12 and delete the .venv folder, then run this again. & pause & exit /b 1)
streamlit run app.py
