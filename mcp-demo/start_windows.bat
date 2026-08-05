@echo off
setlocal
cd /d "%~dp0"

if not exist .venv (
  py -m venv .venv
)

call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt

ollama list >nul 2>&1
if errorlevel 1 (
  echo Ollama wurde nicht gefunden oder laeuft nicht.
  echo Installiere/starte Ollama und fuehre danach diese Datei erneut aus.
  pause
  exit /b 1
)

ollama pull qwen3:4b
set OLLAMA_MODEL=qwen3:4b
python app.py
pause
