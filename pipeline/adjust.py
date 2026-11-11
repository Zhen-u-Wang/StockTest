"""
還原股價（回溯調整除權息）。

FinMind 免費版的 taiwan_stock_daily 是原始價：除息當天股價會憑空掉一段，
動能特徵與標籤都會把它當成真的下跌。這裡用「除權除息結果表」
（taiwan_stock_dividend_result）的 參考價 / 除權息前收盤價 當調整係數，
把除權息日「之前」的 open / max / min / close 全部乘上之後所有事件的係數連乘。

- 只影響報酬的計算；最新一天之後沒有事件，所以最新價不變。
- close_raw 保留原始收盤價，給看板顯示用。
- 沒有除權息資料時不調整（行為與舊版相同），只印警告。
- 尚未處理：股票分割、變更面額、減資。股票池內若有這類事件，該檔在事件前後仍會失真。
"""
import numpy as np
import pandas as pd

PRICE_COLS = ["open", "max", "min", "close"]


def dividend_factors(div: pd.DataFrame) -> pd.DataFrame:
    """每個除權息事件一列：stock_id, date, cum（該事件與之後所有事件的係數連乘）。"""
    cols = ["stock_id", "date", "cum"]
    need = {"date", "stock_id", "before_price", "reference_price"}
    if div is None or div.empty or not need.issubset(div.columns):
        return pd.DataFrame(columns=cols)
    d = div[list(need)].copy()
    d["date"] = pd.to_datetime(d["date"]).astype("datetime64[ns]")
    d["stock_id"] = d["stock_id"].astype(str)
    before = pd.to_numeric(d["before_price"], errors="coerce")
    ref = pd.to_numeric(d["reference_price"], errors="coerce")
    d["f"] = ref / before.where(before > 0)
    # 合理範圍外的係數（資料缺漏或異常）直接丟掉，不調整
    d = d[np.isfinite(d["f"]) & (d["f"] > 0.2) & (d["f"] <= 1.0)]
    d = d.drop_duplicates(subset=["stock_id", "date"], keep="last")
    d = d.sort_values(["stock_id", "date"]).reset_index(drop=True)
    if d.empty:
        return pd.DataFrame(columns=cols)
    # 由後往前連乘：cum[i] = f[i] * f[i+1] * ...
    d["cum"] = d.iloc[::-1].groupby("stock_id")["f"].cumprod().iloc[::-1]
    return d[cols]


def adjust_prices(price: pd.DataFrame, div: pd.DataFrame) -> pd.DataFrame:
    p = price.copy()
    p["date"] = pd.to_datetime(p["date"])
    p["stock_id"] = p["stock_id"].astype(str)
    p["close_raw"] = p["close"]
    fac = dividend_factors(div)
    if fac.empty:
        print("[adjust] 沒有除權息資料，股價未還原")
        p["adj_factor"] = 1.0
        return p
    p = p.sort_values("date").reset_index(drop=True)
    fac = fac.rename(columns={"date": "ex_date"}).sort_values("ex_date")
    fac["ex_date"] = fac["ex_date"].astype(p["date"].dtype)  # merge_asof 要求兩邊日期型別完全相同
    # 每一列找「日期之後」最近的一個除權息事件，取它的連乘係數
    p = pd.merge_asof(p, fac, left_on="date", right_on="ex_date", by="stock_id",
                      direction="forward", allow_exact_matches=False)
    p["adj_factor"] = p["cum"].fillna(1.0)
    for c in PRICE_COLS:
        if c in p.columns:
            p[c] = p[c] * p["adj_factor"]
    n_ev, n_st = len(fac), fac["stock_id"].nunique()
    print(f"[adjust] 已還原 {n_st} 檔、{n_ev} 個除權息事件")
    return p.drop(columns=["ex_date", "cum"])
