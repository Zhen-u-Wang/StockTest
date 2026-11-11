"""
特徵工程：
1. 每檔股票算動能、波動、量能、籌碼、營收特徵
2. 同一天跨股票做百分位標準化（0~1）
3. 另外算大盤環境（只給大盤濾網與看板用，不進模型）

股價先用除權息結果表還原（adjust.py），報酬才不會被除息缺口污染。

所有特徵只用「當天收盤後已公布」的資料：
- 法人買賣超、融資券：當日盤後公布，可用
- 月營收：以「公布月份 1 日 + 10 天」為可用日，保守避免偷看
"""
import numpy as np
import pandas as pd

from . import config as C
from .adjust import adjust_prices

# 會做跨股票百分位標準化的特徵
STOCK_FEATURES = [
    "ret_5", "ret_20", "ret_60", "vol_20", "vol_ratio",
    "dist_ma20", "dist_ma60", "dist_hi240",
    "foreign_1", "foreign_5", "foreign_20",
    "trust_1", "trust_5", "trust_20",
    "dealer_5",
    "trust_streak", "foreign_streak",
    "margin_chg_5", "short_chg_5",
    "rev_yoy", "rev_mom",
]
# 大盤環境（所有股票同一天相同）。標籤是「同一天跨股票排名」，大盤數值本身分不出個股高下，
# 放進模型只會讓樹去記特定時期的雜訊，所以不當特徵，只留給大盤濾網與看板。
MARKET_FEATURES = ["idx_ret_5", "idx_ret_20", "idx_vol_20", "idx_dist_ma60"]
FEATURE_COLS = [f + "_rk" for f in STOCK_FEATURES]


def _load(name):
    path = C.DATA_DIR / f"{name}.parquet"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    if "stock_id" in df.columns:
        df["stock_id"] = df["stock_id"].astype(str)
    return df


def _streak(s: pd.Series) -> pd.Series:
    out, k = [], 0
    for v in s.to_numpy():
        k = k + 1 if v > 0 else 0
        out.append(k)
    return pd.Series(out, index=s.index)


def _roll(g, col, n, fn="mean", min_periods=None):
    mp = min_periods or n
    return g[col].transform(lambda s: getattr(s.rolling(n, min_periods=mp), fn)())


def price_features(price: pd.DataFrame) -> pd.DataFrame:
    keep = ["date", "stock_id", "open", "close", "Trading_Volume", "Trading_money"]
    keep += [c for c in ("close_raw", "adj_factor") if c in price.columns]
    p = price[keep].copy()
    p = p[p["close"] > 0].sort_values(["stock_id", "date"]).reset_index(drop=True)
    g = p.groupby("stock_id")

    p["ret_1"] = g["close"].pct_change(1)
    for n in (5, 20, 60):
        p[f"ret_{n}"] = g["close"].pct_change(n)
    g = p.groupby("stock_id")
    p["vol_20"] = _roll(g, "ret_1", 20, "std")
    p["ma20"] = _roll(g, "close", 20)
    p["ma60"] = _roll(g, "close", 60)
    p["dist_ma20"] = p["close"] / p["ma20"] - 1
    p["dist_ma60"] = p["close"] / p["ma60"] - 1
    p["hi240"] = _roll(g, "close", 240, "max", min_periods=120)
    p["dist_hi240"] = p["close"] / p["hi240"] - 1
    p["avgvol_20"] = _roll(g, "Trading_Volume", 20)
    p["avgvol_60"] = _roll(g, "Trading_Volume", 60)
    p["vol_ratio"] = _roll(g, "Trading_Volume", 5) / p["avgvol_60"]
    return p


def inst_features(p: pd.DataFrame, inst: pd.DataFrame) -> pd.DataFrame:
    if inst.empty:
        for c in ["foreign", "trust", "dealer"]:
            p[c] = 0.0
    else:
        i = inst.copy()
        i["net"] = i["buy"] - i["sell"]
        piv = i.pivot_table(index=["date", "stock_id"], columns="name",
                            values="net", aggfunc="sum").reset_index()
        piv.columns.name = None
        def col(n):
            return piv[n] if n in piv.columns else 0.0
        piv["foreign"] = col("Foreign_Investor") + col("Foreign_Dealer_Self")
        piv["trust"] = col("Investment_Trust")
        piv["dealer"] = col("Dealer_self") + col("Dealer_Hedging")
        p = p.merge(piv[["date", "stock_id", "foreign", "trust", "dealer"]],
                    on=["date", "stock_id"], how="left")
        p[["foreign", "trust", "dealer"]] = p[["foreign", "trust", "dealer"]].fillna(0.0)

    p = p.sort_values(["stock_id", "date"]).reset_index(drop=True)
    g = p.groupby("stock_id")
    denom = p["avgvol_20"].replace(0, np.nan)
    p["foreign_1"] = p["foreign"] / denom
    p["trust_1"] = p["trust"] / denom
    for n in (5, 20):
        p[f"foreign_{n}"] = _roll(g, "foreign", n, "sum") / denom
        p[f"trust_{n}"] = _roll(g, "trust", n, "sum") / denom
    p["dealer_5"] = _roll(g, "dealer", 5, "sum") / denom
    p["trust_streak"] = g["trust"].transform(_streak)
    p["foreign_streak"] = g["foreign"].transform(_streak)
    return p


def margin_features(p: pd.DataFrame, margin: pd.DataFrame) -> pd.DataFrame:
    if margin.empty:
        p["margin_chg_5"] = 0.0
        p["short_chg_5"] = 0.0
        return p
    m = margin[["date", "stock_id", "MarginPurchaseTodayBalance", "ShortSaleTodayBalance"]]
    m = m.rename(columns={"MarginPurchaseTodayBalance": "margin_bal",
                          "ShortSaleTodayBalance": "short_bal"})
    p = p.merge(m, on=["date", "stock_id"], how="left")
    p = p.sort_values(["stock_id", "date"]).reset_index(drop=True)
    g = p.groupby("stock_id")
    p["margin_bal"] = g["margin_bal"].ffill()
    p["short_bal"] = g["short_bal"].ffill()
    g = p.groupby("stock_id")
    p["margin_chg_5"] = g["margin_bal"].pct_change(5).replace([np.inf, -np.inf], np.nan)
    p["short_chg_5"] = g["short_bal"].pct_change(5).replace([np.inf, -np.inf], np.nan)
    return p


def revenue_features(p: pd.DataFrame, revenue: pd.DataFrame) -> pd.DataFrame:
    if revenue.empty:
        p["rev_yoy"] = np.nan
        p["rev_mom"] = np.nan
        return p
    r = revenue[["date", "stock_id", "revenue", "revenue_year", "revenue_month"]].copy()
    r = r.sort_values(["stock_id", "revenue_year", "revenue_month"])
    r["key"] = r["revenue_year"] * 100 + r["revenue_month"]
    # 去年同月
    prev = r[["stock_id", "key", "revenue"]].copy()
    prev["key"] = prev["key"] + 100
    prev = prev.rename(columns={"revenue": "rev_ly"})
    r = r.merge(prev, on=["stock_id", "key"], how="left")
    r["rev_yoy"] = r["revenue"] / r["rev_ly"].replace(0, np.nan) - 1
    r["rev_mom"] = r.groupby("stock_id")["revenue"].pct_change(1)
    # 公布月 1 日 + 10 天才可用
    r["avail"] = pd.to_datetime(r["date"]) + pd.to_timedelta(10, unit="D")
    r = r[["stock_id", "avail", "rev_yoy", "rev_mom"]].sort_values("avail")

    p = p.sort_values("date").reset_index(drop=True)
    p = pd.merge_asof(p, r, left_on="date", right_on="avail", by="stock_id",
                      direction="backward")
    p = p.drop(columns=["avail"])
    return p


def market_features(p: pd.DataFrame, index: pd.DataFrame) -> pd.DataFrame:
    if index.empty or "close" not in index.columns:
        # 沒有指數資料時，用股票池等權報酬合成一個指數
        idx = (p.groupby("date")["ret_1"].mean().fillna(0).add(1).cumprod()
               .rename("close").reset_index())
        idx["source"] = "equal_weight"
    else:
        idx = index[["date", "close"] + (["open"] if "open" in index.columns else [])].copy()
        idx["source"] = C.INDEX_ID
    if "open" not in idx.columns:
        idx["open"] = np.nan
    idx = idx.sort_values("date").reset_index(drop=True)
    idx["idx_ret_1"] = idx["close"].pct_change(1)
    idx["idx_ret_5"] = idx["close"].pct_change(5)
    idx["idx_ret_20"] = idx["close"].pct_change(C.MARKET_MOM)
    idx["idx_vol_20"] = idx["idx_ret_1"].rolling(20).std()
    idx["idx_ma60"] = idx["close"].rolling(C.MARKET_MA).mean()
    idx["idx_dist_ma60"] = idx["close"] / idx["idx_ma60"] - 1
    idx["market_on"] = (idx["close"] > idx["idx_ma60"]) & (idx["idx_ret_20"] > 0)
    idx = idx.rename(columns={"close": "idx_close", "open": "idx_open"})
    cols = ["date", "idx_close", "idx_ret_1", "idx_ret_5", "idx_ret_20",
            "idx_vol_20", "idx_ma60", "idx_dist_ma60", "market_on"]
    return p.merge(idx[cols], on="date", how="left"), idx


def rank_cross_section(p: pd.DataFrame) -> pd.DataFrame:
    g = p.groupby("date")
    for f in STOCK_FEATURES:
        p[f + "_rk"] = g[f].rank(pct=True)
    return p


def build(save=True):
    price, inst, margin, revenue, index, dividend = (
        _load(n) for n in ("price", "inst", "margin", "revenue", "index", "dividend"))
    if price.empty:
        raise SystemExit("沒有行情資料，請先執行 python -m pipeline.fetch")
    universe = pd.read_csv(C.UNIVERSE_FILE, dtype=str)
    price = price[price["stock_id"].isin(universe["stock_id"])]
    price = adjust_prices(price, dividend)

    p = price_features(price)
    p = inst_features(p, inst)
    p = margin_features(p, margin)
    p = revenue_features(p, revenue)
    p, idx = market_features(p, index)
    p = rank_cross_section(p)
    p = p.merge(universe.rename(columns={"name": "stock_name"}), on="stock_id", how="left")
    p = p.sort_values(["date", "stock_id"]).reset_index(drop=True)

    if save:
        p.to_parquet(C.DATA_DIR / "features.parquet", index=False)
        idx.to_parquet(C.DATA_DIR / "index_features.parquet", index=False)
        print(f"[features] {len(p)} 列, {p['date'].min().date()} ~ {p['date'].max().date()}, "
              f"{p['stock_id'].nunique()} 檔")
    return p, idx


if __name__ == "__main__":
    build()
