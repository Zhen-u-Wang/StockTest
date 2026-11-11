"""
產生與 FinMind 欄位相同的合成資料，用來離線測試 pipeline（不需要網路）。
會刻意讓「投信連買」與未來報酬有微弱正相關，讓模型有東西可學。
每檔每年 7 月放一次除息（原始價在除息日掉一段），並產生對應的除權息結果表，用來測試還原股價。

用法：python scripts/make_synthetic.py [--end 2026-11-05] [--n 60]
"""
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)


def main(end: str, n_stocks: int, seed: int = 7):
    rng = np.random.default_rng(seed)
    uni = pd.read_csv(ROOT / "universe.csv", dtype=str).head(n_stocks)
    dates = pd.bdate_range("2017-01-01", end)
    T = len(dates)

    # 市場因子 + 個股殘差
    mkt = rng.normal(0.0003, 0.011, T)
    idx_close = 10000 * np.cumprod(1 + mkt)
    pd.DataFrame({"date": dates, "stock_id": "TAIEX", "close": idx_close,
                  "open": idx_close, "max": idx_close, "min": idx_close,
                  "Trading_Volume": 0, "Trading_money": 0, "spread": 0,
                  "Trading_turnover": 0}).to_parquet(DATA / "index.parquet", index=False)

    price, inst, margin, rev, div = [], [], [], [], []
    for sid in uni["stock_id"]:
        beta = rng.uniform(0.6, 1.4)
        # 投信淨買超（股），有自相關
        trust = np.zeros(T)
        for t in range(1, T):
            trust[t] = 0.7 * trust[t - 1] + rng.normal(0, 1)
        foreign = rng.normal(0, 1, T) + 0.3 * trust
        signal = np.zeros(T)
        signal[5:] = 0.0015 * np.clip(trust[:-5], -3, 3)  # 5 天後才反映
        r = beta * mkt + rng.normal(0, 0.018, T) + signal
        close = 100 * rng.uniform(0.3, 5) * np.cumprod(1 + r)
        # 除息：close 是「還原後」的真實走勢；原始價 = 還原價 / 之後所有除息係數的連乘
        ex_pos = [int(np.searchsorted(dates, pd.Timestamp(year=y, month=7, day=int(rng.integers(5, 25)))))
                  for y in sorted(set(dates.year))]
        ex_pos = [k for k in ex_pos if 0 < k < T]
        fac = np.ones(T)
        for k in ex_pos:
            f = 1 - rng.uniform(0.02, 0.06)
            fac[:k] *= f
            div.append((dates[k], sid, f, k))
        close = close / fac
        for j in range(len(div) - len(ex_pos), len(div)):
            d, s_, f, k = div[j]
            div[j] = dict(date=d, stock_id=s_, before_price=close[k - 1],
                          reference_price=close[k - 1] * f, stock_or_cache_dividend="息")
        vol = (rng.lognormal(14, 0.5, T)).astype(int)
        price.append(pd.DataFrame({
            "date": dates, "stock_id": sid, "Trading_Volume": vol,
            "Trading_money": vol * close, "open": close * (1 + rng.normal(0, 0.003, T)),
            "max": close * 1.01, "min": close * 0.99, "close": close,
            "spread": np.r_[0, np.diff(close)], "Trading_turnover": rng.integers(1000, 9000, T)}))
        base = vol.mean() * 0.05
        for name, s in (("Foreign_Investor", foreign), ("Investment_Trust", trust),
                        ("Dealer_self", rng.normal(0, 0.5, T))):
            buy = np.clip(base * (1 + s), 0, None)
            sell = np.clip(base * (1 - s), 0, None)
            inst.append(pd.DataFrame({"date": dates, "stock_id": sid, "name": name,
                                      "buy": buy.astype(int), "sell": sell.astype(int)}))
        mb = np.cumsum(rng.normal(0, 50, T)) + 10000
        sb = np.cumsum(rng.normal(0, 10, T)) + 1000
        margin.append(pd.DataFrame({"date": dates, "stock_id": sid,
                                    "MarginPurchaseTodayBalance": np.clip(mb, 100, None).astype(int),
                                    "ShortSaleTodayBalance": np.clip(sb, 10, None).astype(int)}))
        months = pd.period_range(dates[0], dates[-1], freq="M")
        base_rev = rng.uniform(1e8, 1e10)
        rv = base_rev * np.cumprod(1 + rng.normal(0.005, 0.1, len(months)))
        rev.append(pd.DataFrame({
            "date": [(m + 1).to_timestamp() for m in months],  # 次月 1 日公布
            "stock_id": sid, "country": "Taiwan", "revenue": rv.astype(int),
            "revenue_month": [m.month for m in months], "revenue_year": [m.year for m in months]}))

    pd.concat(price).to_parquet(DATA / "price.parquet", index=False)
    pd.concat(inst).to_parquet(DATA / "inst.parquet", index=False)
    pd.concat(margin).to_parquet(DATA / "margin.parquet", index=False)
    pd.concat(rev).to_parquet(DATA / "revenue.parquet", index=False)
    pd.DataFrame(div).to_parquet(DATA / "dividend.parquet", index=False)
    print(f"合成資料：{n_stocks} 檔 × {T} 天，至 {end}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--end", default="2026-11-05")
    ap.add_argument("--n", type=int, default=60)
    a = ap.parse_args()
    main(a.end, a.n)
