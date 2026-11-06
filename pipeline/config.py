"""
全域設定。環境變數優先，沒有就用預設值。
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"          # 行情與籌碼歷史（parquet）
MODEL_DIR = ROOT / "models"       # 訓練好的模型
OUT_DIR = ROOT / "output"         # 回測報表、本機預測日誌
UNIVERSE_FILE = ROOT / "universe.csv"

for d in (DATA_DIR, MODEL_DIR, OUT_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ---- 資料 ----
START_DATE = os.getenv("START_DATE", "2017-01-01")
INDEX_ID = "TAIEX"                # FinMind 加權指數代碼
FINMIND_TOKEN = os.getenv("FINMIND_TOKEN", "")
FETCH_SLEEP = float(os.getenv("FETCH_SLEEP", "0.4"))   # 每次 API 呼叫間隔秒數
QUOTA_WAIT_SEC = int(os.getenv("QUOTA_WAIT_SEC", "600"))  # 配額用盡時等待秒數
QUOTA_MAX_WAITS = int(os.getenv("QUOTA_MAX_WAITS", "6"))  # 最多等幾次（6 × 10 分鐘 = 1 小時）

# ---- 標籤 ----
HORIZON = 5            # 持有 N 個交易日
LABEL_TOP = 0.30       # 超額報酬前 30% 標 1
LABEL_BOTTOM = 0.30    # 後 30% 標 0，中間捨棄
COST = 0.006           # 來回成本：手續費 0.1425%×2 + 證交稅 0.3%
SLIPPAGE = 0.002       # 來回滑價假設（單邊 0.1%），依實際成交再調整
NET_COST = COST + SLIPPAGE  # 回測與績效統一扣這個數字

# ---- 訓練 ----
FIRST_TEST_YEAR = int(os.getenv("FIRST_TEST_YEAR", "2021"))
SAMPLE_EVERY = HORIZON  # 每 N 個交易日取一個樣本，避免標籤重疊
LGB_PARAMS = dict(
    objective="binary",
    learning_rate=0.03,
    num_leaves=31,
    min_child_samples=100,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=1,
    lambda_l2=1.0,
    verbose=-1,
    seed=42,
)
LGB_ROUNDS = 400

# ---- 訊號 ----
TOP_N = 10             # 每日最多幾檔建議
SCORE_STRONG = 0.65    # 強烈買進門檻
SCORE_BUY = 0.55       # 買進門檻
MARKET_MA = 60         # 大盤需站上 N 日均線
MARKET_MOM = 20        # 且 N 日報酬為正

# ---- 策略開關（貝氏）----
GATE_MIN_N = 30        # 至少累積幾個「不重疊持有期」才評估（約 30 週以上）
# 啟用條件：期數足夠、每期扣成本後超額報酬的 95% 信賴下界 > 0、且回測通過

# ---- Supabase ----
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY", "")

MODEL_VERSION = os.getenv("MODEL_VERSION", "lgb-v2")
