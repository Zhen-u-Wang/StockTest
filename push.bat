@echo off
chcp 65001 >nul
REM 雙擊此檔：把整個資料夾推上 GitHub（Zhen-u-Wang/StockTest）。
REM 第一次會跳出 GitHub 登入視窗（瀏覽器），登入後自動繼續。
cd /d "%~dp0"

where git >nul 2>nul
if errorlevel 1 (
  echo 找不到 git。請先安裝 Git for Windows（全部用預設選項即可），裝好後再雙擊一次這個檔案。
  start https://git-scm.com/download/win
  pause
  exit /b 1
)

if not exist .git git init >nul
git config user.name >nul 2>nul || git config user.name "Zhen-u-Wang"
git config user.email >nul 2>nul || git config user.email "Zhen-u-Wang@users.noreply.github.com"
git branch -M main
git add -A
git commit -m "init: twstock-signal" >nul 2>nul || echo （沒有新的變更需要 commit）
git remote remove origin >nul 2>nul
git remote add origin https://github.com/Zhen-u-Wang/StockTest.git

echo == 推送到 GitHub（可能會跳出登入視窗）==
git push -u origin main
if errorlevel 1 (
  echo.
  echo 推送失敗。常見原因：
  echo   1. 登入視窗被關掉了 → 再雙擊一次
  echo   2. GitHub 上的 repo 已經有檔案（例如網頁上建了 README）→ 把錯誤訊息貼給 Claude
  pause
  exit /b 1
)

echo.
echo 推送完成！接下來到 GitHub 做兩件事：
echo   1. Settings → Pages → Branch 選 main、資料夾選 /docs → Save
echo   2. Actions → daily-signal → Run workflow（第一次約 20~40 分鐘）
start https://github.com/Zhen-u-Wang/StockTest/settings/pages
pause
