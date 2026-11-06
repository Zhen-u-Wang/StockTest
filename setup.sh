#!/usr/bin/env bash
# 本機一鍵執行（macOS / Linux）：建虛擬環境 → 裝套件 → 抓資料 → 訓練 → 產生建議 → 開啟看板
# 用法：bash setup.sh            （之後每天只要 bash setup.sh daily）
set -e
cd "$(dirname "$0")"

if ! command -v python3 >/dev/null; then echo "請先安裝 Python 3.11+ (https://www.python.org/downloads/)"; exit 1; fi
[ -d .venv ] || python3 -m venv .venv
source .venv/bin/activate
pip install -q -r requirements.txt

if [ -f .env ]; then set -a; source .env; set +a; fi
if [ -z "$FINMIND_TOKEN" ]; then
  echo "（選填）FinMind token 可提高抓資料配額，免費註冊：https://finmindtrade.com"
  read -r -p "貼上 token 後按 Enter，或直接 Enter 略過: " tok
  if [ -n "$tok" ]; then echo "FINMIND_TOKEN=$tok" >> .env; export FINMIND_TOKEN="$tok"; fi
fi

mode="${1:-full}"
echo "== 抓資料（第一次約 10~20 分鐘）=="
python -m pipeline.fetch
if [ "$mode" = "full" ] || [ ! -f models/model_latest.txt ]; then
  echo "== 訓練與回測 =="; python -m pipeline.train
fi
echo "== 產生建議 =="; python -m pipeline.predict
echo "== 回填結果 =="; python -m pipeline.score
python -m pipeline.export_json

echo
echo "完成。回測報表：output/backtest_report.md"
echo "看板：http://localhost:8765  （Ctrl+C 結束）"
( sleep 1; command -v open >/dev/null && open http://localhost:8765 || command -v xdg-open >/dev/null && xdg-open http://localhost:8765 ) >/dev/null 2>&1 &
python -m http.server 8765 --directory docs
