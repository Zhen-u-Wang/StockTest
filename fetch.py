"""
從 FinMind 抓取行情、三大法人、融資券、月營收、除權息結果、加權指數，存成 parquet。
支援增量更新：已有資料的股票只抓最後一天之後的部分。

用法：
    python -m pipeline.fetch            # 增量更新
    python -m pipeline.fetch --full     # 全部重抓
"""
import sys
import time
import datetime as dt
import pandas as pd
from FinMind.data import DataLoader

from . import config as C

DATASETS = {
    # 名稱: (DataLoader 方法名, 存檔名)
    "price":   ("taiwan_stock_daily", "price.parquet"),
    "inst":    ("taiwan_stock_institutional_investors", "inst.parquet"),
    "margin":  ("taiwan_stock_margin_purchase_short_sale", "margin.parquet"),
    "revenue": ("taiwan_stock_month_revenue", "revenue.parquet"),
    # 除權除息結果表：用來還原股價（見 adjust.py）。放最後，配額不夠時先保住其他資料
    "dividend": ("taiwan_stock_dividend_result", "dividend.parquet"),
}


def _api() -> DataLoader:
    api = DataLoader()
    if C.FINMIND_TOKEN:
        try:
            api.login_by_token(api_token=C.FINMIND_TOKEN)
            print("FinMind token 登入成功")
        except Exception as e:
            print(f"FinMind token 登入失敗（改用匿名額度）: {e}")
    else:
        print("未設定 FINMIND_TOKEN，使用匿名額度（較低，第一次全量抓可能需要等待配額重置）")
    return api


def _is_quota_error(e: Exception) -> bool:
    s = str(e).lower()
    return "402" in s or "quota" in s or "limit" in s or "too many" in s or "429" in s


def _load(path):
    if path.exists():
        df = pd.read_parquet(path)
        df["date"] = pd.to_datetime(df["date"])
        return df
    return pd.DataFrame()


def _call(api, method, stock_id, start, end, retries=3):
    fn = getattr(api, method)
    quota_waits = 0
    i = 0
    while i < retries:
        try:
            df = fn(stock_id=stock_id, start_date=start, end_date=end)
            time.sleep(C.FETCH_SLEEP)
            return df
        except Exception as e:
            if _is_quota_error(e) and quota_waits < C.QUOTA_MAX_WAITS:
                # 配額用盡：等它重置，不算進重試次數
                quota_waits += 1
                print(f"  [{method} {stock_id}] 配額用盡，等待 {C.QUOTA_WAIT_SEC // 60} 分鐘後繼續（第 {quota_waits} 次）")
                time.sleep(C.QUOTA_WAIT_SEC)
                continue
            i += 1
            wait = 10 * i
            print(f"  [{method} {stock_id}] 失敗: {e} -> {wait}s 後重試")
            time.sleep(wait)
    return pd.DataFrame()


def fetch_dataset(api, name, universe, full=False):
    method, fname = DATASETS[name]
    path = C.DATA_DIR / fname
    if not hasattr(api, method):
        print(f"[{name}] 這個版本的 FinMind 沒有 {method}，略過")
        return
    old = pd.DataFrame() if full else _load(path)
    today = dt.date.today().isoformat()
    last_by_stock = {}
    if not old.empty:
        last_by_stock = old.groupby("stock_id")["date"].max().to_dict()

    frames = [old] if not old.empty else []
    for sid in universe:
        start = C.START_DATE
        if sid in last_by_stock:
            start = (last_by_stock[sid] + pd.Timedelta(days=1)).date().isoformat()
            if start > today:
                continue
        df = _call(api, method, sid, start, today)
        if df is None or df.empty:
            continue
        df["date"] = pd.to_datetime(df["date"])
        df["stock_id"] = df["stock_id"].astype(str)
        frames.append(df)
        print(f"  {name} {sid}: +{len(df)} 列 (從 {start})")

    if not frames:
        return
    out = pd.concat(frames, ignore_index=True)
    key = ["date", "stock_id"] + (["name"] if name == "inst" else [])
    out = out.drop_duplicates(subset=key, keep="last").sort_values(key)
    out.to_parquet(path, index=False)
    print(f"[{name}] 共 {len(out)} 列 -> {path.name}")


def fetch_index(api, full=False):
    """加權指數。FinMind 的 TaiwanStockPrice 以 TAIEX 取得。"""
    path = C.DATA_DIR / "index.parquet"
    old = pd.DataFrame() if full else _load(path)
    start = C.START_DATE
    if not old.empty:
        start = (old["date"].max() + pd.Timedelta(days=1)).date().isoformat()
    today = dt.date.today().isoformat()
    if start > today:
        return
    df = _call(api, "taiwan_stock_daily", C.INDEX_ID, start, today)
    if df is None or df.empty:
        print("[index] 無新資料（或 TAIEX 不可用，features 會改用等權市場指數）")
        if old.empty:
            return
        df = old
    else:
        df["date"] = pd.to_datetime(df["date"])
        df = pd.concat([old, df], ignore_index=True) if not old.empty else df
    df = df.drop_duplicates(subset=["date"], keep="last").sort_values("date")
    df.to_parquet(path, index=False)
    print(f"[index] 共 {len(df)} 列")


def main(full=False):
    universe = pd.read_csv(C.UNIVERSE_FILE, dtype=str)["stock_id"].tolist()
    api = _api()
    print(f"股票池 {len(universe)} 檔，{'全量' if full else '增量'}抓取")
    for name in DATASETS:
        fetch_dataset(api, name, universe, full=full)
    fetch_index(api, full=full)


if __name__ == "__main__":
    main(full="--full" in sys.argv)
