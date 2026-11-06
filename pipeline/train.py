"""
Walk-forward 訓練與回測。

每年為一個測試期：只用「標籤在該年第一個決策日前已結算」的資料訓練
（最後 15% 時間當 early-stopping 驗證，同樣做隔離），在該年測試。
測試時每 HORIZON 個交易日取一個決策日，避免標籤重疊。
進場價為隔日開盤；股票池固定，倖存者偏差尚未修正。

輸出：
- output/backtest_report.csv / .md   每年績效
- output/equity_curve.csv            策略 vs 大盤權益曲線
- output/feature_importance.csv
- models/model_latest.txt + meta.json  用全部資料訓練的最終模型
"""
import json
import datetime as dt
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.stats import t as t_dist
from sklearn.metrics import roc_auc_score

from . import config as C
from .features import FEATURE_COLS, build
from .labels import add_labels


def sample_dates(dates: pd.Series) -> np.ndarray:
    d = np.sort(dates.unique())
    return d[:: C.SAMPLE_EVERY]


def fit(train_df: pd.DataFrame) -> lgb.Booster:
    dates = np.sort(train_df["date"].unique())
    cut = dates[int(len(dates) * 0.85)]
    tr = train_df[train_df["exit_date"] < cut]
    va = train_df[train_df["date"] >= cut]
    dtr = lgb.Dataset(tr[FEATURE_COLS], tr["label"])
    dva = lgb.Dataset(va[FEATURE_COLS], va["label"], reference=dtr)
    return lgb.train(C.LGB_PARAMS, dtr, C.LGB_ROUNDS, valid_sets=[dva],
                     callbacks=[lgb.early_stopping(50, verbose=False)])


def evaluate(model: lgb.Booster, test: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """test: 測試期、已抽樣日期、excess 不為 NaN 的所有列（含未標籤的中間 40%）。"""
    t = test.copy()
    t["score"] = model.predict(t[FEATURE_COLS], num_iteration=model.best_iteration)
    lab = t[t["label"].notna()]
    auc = roc_auc_score(lab["label"], lab["score"]) if lab["label"].nunique() == 2 else np.nan

    top = (t.sort_values(["date", "score"], ascending=[True, False])
             .groupby("date").head(C.TOP_N))
    sig = top[top["score"] >= C.SCORE_BUY]
    strong = top[top["score"] >= C.SCORE_STRONG]
    act = sig[sig["market_on"] == True]  # noqa: E712  實際會出手的訊號

    # 週期報酬：有訊號且大盤濾網開才進場，等權，扣來回成本
    per = (act.groupby("date")["fwd_ret"].mean() - C.NET_COST).rename("strategy")
    net_excess = (act.groupby("date")["excess"].mean() - C.NET_COST).rename("net_excess")
    dates = pd.Index(np.sort(t["date"].unique()), name="date")
    curve = pd.DataFrame(index=dates)
    curve["strategy"] = per.reindex(dates).fillna(0.0)
    curve["benchmark"] = t.groupby("date")["idx_fwd"].first().reindex(dates).fillna(0.0)
    curve["net_excess"] = net_excess.reindex(dates)  # 沒出手的期為 NaN
    curve["n_signals"] = act.groupby("date").size().reindex(dates).fillna(0).astype(int)

    m = dict(
        n_dates=len(dates),
        auc=round(float(auc), 4) if pd.notna(auc) else None,
        prec_top10=round(float((top["excess"] > 0).mean()), 4),
        avg_excess_top10=round(float(top["excess"].mean()), 4),
        n_signals=int(len(sig)),
        prec_signal=round(float((sig["excess"] > 0).mean()), 4) if len(sig) else None,
        avg_ret_signal_net=round(float((sig["fwd_ret"] - C.NET_COST).mean()), 4) if len(sig) else None,
        n_strong=int(len(strong)),
        prec_strong=round(float((strong["excess"] > 0).mean()), 4) if len(strong) else None,
        n_acted=int(len(act)),
        prec_acted=round(float((act["excess"] > 0).mean()), 4) if len(act) else None,
        n_periods=int(net_excess.notna().sum()),
        net_excess_per_period=round(float(net_excess.mean()), 5) if len(net_excess) else None,
        strategy_return=round(float((1 + curve["strategy"]).prod() - 1), 4),
        benchmark_return=round(float((1 + curve["benchmark"]).prod() - 1), 4),
        best_iteration=int(model.best_iteration),
    )
    return m, curve


def summarize(eq: pd.DataFrame) -> dict:
    """回測門檻：每期扣成本後超額報酬平均為正，而且拿掉最好的一年仍為正。"""
    x = eq["net_excess"].dropna()
    by_year = x.groupby(pd.DatetimeIndex(x.index).year).sum()
    best = int(by_year.idxmax()) if len(by_year) else None
    rest = x[pd.DatetimeIndex(x.index).year != best]
    mean = float(x.mean()) if len(x) else None
    mean_ex = float(rest.mean()) if len(rest) else None
    lo = None
    if len(x) >= 2:
        lo = float(x.mean() - t_dist.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)))
    return dict(
        n_periods=int(len(x)),
        net_excess_per_period=round(mean, 5) if mean is not None else None,
        ci_lower=round(lo, 5) if lo is not None else None,
        period_hit_rate=round(float((x > 0).mean()), 4) if len(x) else None,
        best_year=best,
        net_excess_ex_best_year=round(mean_ex, 5) if mean_ex is not None else None,
        positive_years=int((by_year > 0).sum()),
        n_years=int(len(by_year)),
        net_cost=C.NET_COST,
        passed=bool(mean is not None and mean_ex is not None and mean > 0 and mean_ex > 0),
    )


def main():
    feat, idx = build(save=True)
    ds = add_labels(feat, idx)
    ds = ds.dropna(subset=FEATURE_COLS, how="all")
    ds = ds[ds["excess"].notna()]
    keep = sample_dates(ds["date"])
    ds = ds[ds["date"].isin(keep)].reset_index(drop=True)
    ds["year"] = ds["date"].dt.year
    print(f"[train] 樣本 {len(ds)} 列（已抽樣），{ds['date'].min().date()} ~ {ds['date'].max().date()}")

    years = [y for y in sorted(ds["year"].unique()) if y >= C.FIRST_TEST_YEAR]
    rows, curves = [], []
    for y in years:
        test = ds[ds["year"] == y]
        if test.empty:
            continue
        # 隔離：標籤必須在測試期第一個決策日之前就結算，否則等於偷看測試期股價
        train = ds[(ds["exit_date"] < test["date"].min()) & ds["label"].notna()]
        if len(train) < 2000 or len(test) < 50:
            print(f"  {y}: 資料不足，跳過")
            continue
        model = fit(train)
        m, curve = evaluate(model, test)
        m["year"] = y
        m["n_train"] = len(train)
        rows.append(m)
        curves.append(curve)
        print(f"  {y}: AUC={m['auc']} 前10精準={m['prec_top10']} 訊號數={m['n_signals']} "
              f"訊號精準={m['prec_signal']} 策略={m['strategy_return']:+.1%} 大盤={m['benchmark_return']:+.1%}")

    if not rows:
        raise SystemExit("沒有任何測試年份有足夠資料")

    report = pd.DataFrame(rows).set_index("year")
    report.to_csv(C.OUT_DIR / "backtest_report.csv")
    with open(C.OUT_DIR / "backtest_report.md", "w", encoding="utf-8") as f:
        f.write("# Walk-forward 回測報告\n\n")
        f.write(f"產生時間：{dt.datetime.now():%Y-%m-%d %H:%M}\n\n")
        f.write(f"欄位說明：prec_* = 超額報酬 > 0 的比例；avg_ret_signal_net 已扣來回成本與滑價 {C.NET_COST:.1%}；"
                "進場價為隔日開盤；strategy_return 只在大盤濾網開啟時進場。"
                "股票池固定，倖存者偏差尚未修正，結果仍可能偏樂觀。\n\n")
        f.write(report.to_markdown())

    eq = pd.concat(curves).sort_index()
    eq["strategy_equity"] = (1 + eq["strategy"]).cumprod()
    eq["benchmark_equity"] = (1 + eq["benchmark"]).cumprod()
    eq.to_csv(C.OUT_DIR / "equity_curve.csv")
    summary = summarize(eq)
    (C.OUT_DIR / "backtest_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    with open(C.OUT_DIR / "backtest_report.md", "a", encoding="utf-8") as f:
        f.write("\n\n## 回測門檻\n\n```json\n" + json.dumps(summary, ensure_ascii=False, indent=2) + "\n```\n")

    # 最終模型：用全部有標籤資料
    final = fit(ds[ds["label"].notna()])
    final.save_model(str(C.MODEL_DIR / "model_latest.txt"))
    imp = pd.DataFrame({"feature": FEATURE_COLS,
                        "gain": final.feature_importance("gain")}).sort_values("gain", ascending=False)
    imp.to_csv(C.OUT_DIR / "feature_importance.csv", index=False)
    meta = dict(model_version=C.MODEL_VERSION, trained_at=dt.datetime.now().isoformat(timespec="seconds"),
                trained_until=str(ds["date"].max().date()), best_iteration=int(final.best_iteration),
                features=FEATURE_COLS, horizon=C.HORIZON)
    (C.MODEL_DIR / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))

    overall = pd.DataFrame(rows)
    print("\n[train] 所有測試年份平均："
          f"前10精準 {overall['prec_top10'].mean():.3f}，訊號精準 {overall['prec_signal'].mean():.3f}，"
          f"策略累積 {eq['strategy_equity'].iloc[-1]-1:+.1%} vs 大盤 {eq['benchmark_equity'].iloc[-1]-1:+.1%}")
    print(f"[train] 回測門檻：{'通過' if summary['passed'] else '未通過'} {summary}")
    print(f"[train] 模型已存：{C.MODEL_DIR / 'model_latest.txt'}；報表：{C.OUT_DIR / 'backtest_report.md'}")


if __name__ == "__main__":
    main()
