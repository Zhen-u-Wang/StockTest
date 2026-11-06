@echo off
chcp 65001 >nul
REM 本機一鍵執行（Windows）：雙擊即可。之後每天雙擊一次就會更新建議。
cd /d "%~dp0"

where python >nul 2>nul || (echo 請先安裝 Python 3.11+ ^(https://www.python.org/downloads/^)，安裝時勾選 "Add python.exe to PATH" & pause & exit /b 1)
if not exist .venv python -m venv .venv
call .venv\Scripts\activate.bat
pip install -q -r requirements.txt

if exist .env for /f "usebackq tokens=1,* delims==" %%a in (".env") do set %%a=%%b
if "%FINMIND_TOKEN%"=="" (
  echo （選填）FinMind token 可提高抓資料配額，免費註冊：https://finmindtrade.com
  set /p tok=貼上 token 後按 Enter，或直接 Enter 略過:
  if not "%tok%"=="" (echo FINMIND_TOKEN=%tok%>> .env & set FINMIND_TOKEN=%tok%)
)

echo == 抓資料（第一次約 10~20 分鐘）==
python -m pipeline.fetch || goto :err
if not exist models\model_latest.txt (
  echo == 訓練與回測 ==
  python -m pipeline.train || goto :err
)
echo == 產生建議 ==
python -m pipeline.predict || goto :err
echo == 回填結果 ==
python -m pipeline.score || goto :err
python -m pipeline.export_json || goto :err

echo.
echo 完成。回測報表：output\backtest_report.md
echo 看板：http://localhost:8765  （關閉此視窗即結束）
start "" http://localhost:8765
python -m http.server 8765 --directory docs
exit /b 0

:err
echo.
echo 執行失敗，請把上面的錯誤訊息貼給 Claude。
pause
