@echo off
python -m venv .venv
call .venv\Scripts\activate
pip install -r requirements.txt

@REM Windows – CMD/PowerShell (đứng ở thư mục gốc repo diabetes-analysis-app):

cd cd path\to\diabetes-analysis-app
set PYTHONPATH=%CD%
streamlit run app\main.py

@REM PowerShell (nếu dùng PS):

cd path\to\diabetes-analysis-app
$env:PYTHONPATH = (Get-Location).Path
streamlit run app\main.py


@REM macOS/Linux:

cd /path/to/diabetes-analysis-app
export PYTHONPATH=$(pwd)
streamlit run app/main.py
