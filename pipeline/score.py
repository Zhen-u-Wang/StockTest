"""
回填結果：預測隔日開盤進場、第 HORIZON 個交易日收盤出場，算實際報酬、相對大盤超額報酬、是否命中。
然後更新統計。策略開關看「不重疊持有期」，因為同一期的股票會一起漲跌，不能當成獨立樣本。

命中定義：
- hit      超額報酬 > 0（跑贏大盤）
- hit_abs  絕對報酬 > 來回成本加滑價（真的有賺）
"""
import json
import datetime as dt
import numpy as np
import pandas as pd
from scipy.stats import beta, t as t_dist

from . import config as C
from . import supabase_io as sb
from .predict import LOG_FILE, STATS_FILE, PRED_COLS

BUY_SET = {"買進", "強烈買進"}


def _price_index():
    price = pd.read_parquet(C.DATA_DIR / "price.parquet")
    price["date"] = pd.to_datetime(price["date"])
    price["stock_id"] = price["stock_id"].astype(str)
    ipath = C.DATA_DIR / "index_features.parquet"
    if not ipath.exists():
        from .features import build
        build(save=True)
    idx = pd.read_parquet(ipath)
    idx["date"] = pd.to_datetime(idx["date"])
    return price, idx[["date", "idx_open", "idx_close"]].sort_values("date").reset_index(drop=True)


def resolve(log: pd.DataFrame, price: pd.DataFrame, idx: pd.DataFrame) -> pd.DataFrame:
    log = log.copy()
    log["pred_date"] = log["pred_date"].astype(str)
    todo = log.index[log["resolved_date"].isna()]
    idx_dates = idx["date"].to_numpy()
    idx_close = idx["idx_close"].to_numpy()
    idx_open = idx["idx_open"].to_numpy()
    n_done = 0
    for i in todo:
        sid, d = log.at[i, "stock_id"], pd.Timestamp(log.at[i, "pred_date"])
        s = price[(price["stock_id"] == sid) & (price["close"] > 0)].sort_values("date")
        dates = s["date"].to_numpy()
        pos = np.searchsorted(dates, np.datetime64(d))
        if pos >= len(dates) or dates[pos] != np.datetime64(d) or pos + C.HORIZON >= len(dates):
            continue
        c0, c1 = s["open"].iloc[pos + 1], s["close"].iloc[pos + C.HORIZON]
        if not c0 > 0:
            continue
        end = dates[pos + C.HORIZON]
        ip = np.searchsorted(idx_dates, np.datetime64(d))
        ie = np.searchsorted(idx_dates, end)
        if ip + 1 >= len(idx_dates) or ie >= len(idx_dates):
            continue
        # 指數沒有開盤價時，退回用預測日收盤當起點
        i0 = idx_open[ip + 1] if idx_open[ip + 1] > 0 else idx_close[ip]
        r = c1 / c0 - 1
        ir = idx_close[ie] / i0 - 1
        log.at[i, "resolved_date"] = pd.Timestamp(end).date().isoformat()
        log.at[i, "actual_return"] = round(float(r), 5)
        log.at[i, "index_return"] = round(float(ir), 5)
        log.at[i, "excess_return"] = round(float(r - ir), 5)
        log.at[i, "hit"] = bool(r - ir > 0)
        log.at[i, "hit_abs"] = bool(r > C.NET_COST)
        n_done += 1
    print(f"[score] 本次回填 {n_done} 筆，尚未到期 {log['resolved_date'].isna().sum()} 筆")
    return log


def _ci(h, n):
    if n == 0:
        return None, None
    lo, hi = beta.ppf([0.025, 0.975], 1 + h, 1 + n - h)
    return round(float(lo), 4), round(float(hi), 4)


def independent_periods(act: pd.DataFrame, trading_dates) -> pd.DataFrame:
    """把每日訊號收斂成不重疊的持有期：選一期後，隔 HORIZON 個交易日才選下一期。"""
    cols = ["pred_date", "ret", "index_ret", "net_ret", "net_excess"]
    if act.empty:
        return pd.DataFrame(columns=cols)
    days = pd.DatetimeIndex(trading_dates).sort_values()
    per = (act.groupby("pred_date")
           .agg(ret=("actual_return", "mean"), index_ret=("index_return", "first"),
                excess=("excess_return", "mean")).reset_index().sort_values("pred_date"))
    keep, next_pos = [], 0
    for d in per["pred_date"]:
        pos = int(days.searchsorted(pd.Timestamp(d)))
        keep.append(pos >= next_pos)
        if keep[-1]:
            next_pos = pos + C.HORIZON
    per = per[keep].copy()
    per["net_ret"] = per["ret"] - C.NET_COST
    per["net_excess"] = per["excess"] - C.NET_COST
    return per[cols]


def _backtest_summary() -> dict:
    path = C.OUT_DIR / "backtest_summary.json"
    return json.loads(path.read_text()) if path.exists() else {}


def compute_stats(log: pd.DataFrame, trading_dates) -> dict:
    done = log[log["resolved_date"].notna()].copy()
    done["hit"] = done["hit"].astype(bool)
    done["hit_abs"] = done["hit_abs"].astype(bool)
    done["market_on"] = done["market_on"].astype(str).str.lower().eq("true")
    sig = done[done["raw_signal"].isin(BUY_SET)]
    act = sig[sig["market_on"]]
    strong = sig[sig["raw_signal"] == "強烈買進"]

    def block(d):
        n, h = len(d), int(d["hit"].sum())
        lo, hi = _ci(h, n)
        last = d.sort_values(["resolved_date", "stock_id"]).tail(20)
        return dict(n=n, hits=h, hit_rate=round(h / n, 4) if n else None,
                    hit_rate_abs=round(float(d["hit_abs"].mean()), 4) if n else None,
                    hit_rate_last20=round(float(last["hit"].mean()), 4) if len(last) else None,
                    ci_lower=lo, ci_upper=hi,
                    avg_excess=round(float(d["excess_return"].mean()), 5) if n else None,
                    avg_return=round(float(d["actual_return"].mean()), 5) if n else None)

    a = block(act)
    per = independent_periods(act, trading_dates)
    x = per["net_excess"].astype(float)
    n_per = len(x)
    p_mean = round(float(x.mean()), 5) if n_per else None
    p_lo = p_hi = None
    if n_per >= 2:
        half = float(t_dist.ppf(0.975, n_per - 1) * x.std(ddof=1) / np.sqrt(n_per))
        p_lo, p_hi = round(float(x.mean()) - half, 5), round(float(x.mean()) + half, 5)
    bt = _backtest_summary()
    bt_passed = bool(bt.get("passed", False))
    enabled = bool(bt_passed and n_per >= C.GATE_MIN_N and p_lo is not None and p_lo > 0)
    return dict(
        id=1,
        updated_at=dt.datetime.now().isoformat(timespec="seconds"),
        n_pending=int(log["resolved_date"].isna().sum()),
        enabled=enabled,
        gate_min_n=C.GATE_MIN_N,
        # 不重疊持有期（開關依據）
        n_periods=n_per,
        period_hit_rate=round(float((x > 0).mean()), 4) if n_per else None,
        period_net_excess=p_mean, period_ci_lower=p_lo, period_ci_upper=p_hi,
        net_cost=C.NET_COST,
        backtest_passed=bt_passed,
        backtest_net_excess=bt.get("net_excess_per_period"),
        backtest_net_excess_ex_best=bt.get("net_excess_ex_best_year"),
        # 實際出手（買進訊號且大盤濾網開）
        n=a["n"], hits=a["hits"], hit_rate=a["hit_rate"], hit_rate_abs=a["hit_rate_abs"],
        hit_rate_last20=a["hit_rate_last20"], ci_lower=a["ci_lower"], ci_upper=a["ci_upper"],
        avg_excess=a["avg_excess"], avg_return=a["avg_return"],
        # 不看大盤濾網的所有買進訊號
        all_n=len(sig), all_hit_rate=block(sig)["hit_rate"],
        strong_n=len(strong), strong_hit_rate=block(strong)["hit_rate"],
    )


def main():
    if not LOG_FILE.exists():
        raise SystemExit("還沒有預測日誌，請先執行 python -m pipeline.predict")
    log = pd.read_csv(LOG_FILE, dtype={"stock_id": str})
    for c in PRED_COLS:
        if c not in log.columns:
            log[c] = np.nan
    for c in ("resolved_date", "hit", "hit_abs"):
        log[c] = log[c].astype(object)
    price, idx = _price_index()
    before = log["resolved_date"].notna()
    log = resolve(log, price, idx)
    log.to_csv(LOG_FILE, index=False)

    newly = log[log["resolved_date"].notna() & ~before]
    if len(newly):
        sb.upsert("predictions", newly[PRED_COLS], on_conflict="pred_date,stock_id")

    stats = compute_stats(log, idx["date"])
    STATS_FILE.write_text(json.dumps(stats, ensure_ascii=False, indent=2))
    sb.upsert("stats", pd.DataFrame([stats]), on_conflict="id")
    print(f"[score] 已結算 {stats['n']} 筆實際出手訊號（{stats['n_periods']} 個不重疊持有期），"
          f"每期扣成本超額 {stats['period_net_excess']}，95% 下界 {stats['period_ci_lower']}，"
          f"回測{'通過' if stats['backtest_passed'] else '未通過'}，模型開關：{'啟用' if stats['enabled'] else '觀察期'}")


if __name__ == "__main__":
    main()
