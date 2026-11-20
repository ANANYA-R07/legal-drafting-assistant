@echo off
cd /d "%~dp0"
echo Starting Legal Drafting Assistant...
call .venv\Scripts\python.exe -m app.cli %*
pause
