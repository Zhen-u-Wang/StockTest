"""
Walk-forward 訓練與回測。

每年為一個測試期：用該年之前的所有資料訓練（最後 15% 時間當 early-stopping 驗證），
在該年測試。測試時每 HORIZON 個交易日取一個決策日，避免標籤重疊。

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
    tr = train_df[train_df["date"] < cut]
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
    per = (act.groupby("date")["fwd_ret"].mean() - C.COST).rename("strategy")
    dates = pd.Index(np.sort(t["date"].unique()), name="date")
    curve = pd.DataFrame(index=dates)
    curve["strategy"] = per.reindex(dates).fillna(0.0)
    curve["benchmark"] = t.groupby("date")["idx_fwd"].first().reindex(dates).fillna(0.0)
    curve["n_signals"] = act.groupby("date").size().reindex(dates).fillna(0).astype(int)

    m = dict(
        n_dates=len(dates),
        auc=round(float(auc), 4) if pd.notna(auc) else None,
        prec_top10=round(float((top["excess"] > 0).mean()), 4),
        avg_excess_top10=round(float(top["excess"].mean()), 4),
        n_signals=int(len(sig)),
        prec_signal=round(float((sig["excess"] > 0).mean()), 4) if len(sig) else None,
        avg_ret_signal_net=round(float((sig["fwd_ret"] - C.COST).mean()), 4) if len(sig) else None,
        n_strong=int(len(strong)),
        prec_strong=round(float((strong["excess"] > 0).mean()), 4) if len(strong) else None,
        n_acted=int(len(act)),
        prec_acted=round(float((act["excess"] > 0).mean()), 4) if len(act) else None,
        strategy_return=round(float((1 + curve["strategy"]).prod() - 1), 4),
        benchmark_return=round(float((1 + curve["benchmark"]).prod() - 1), 4),
        best_iteration=int(model.best_iteration),
    )
    return m, curve


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
        train = ds[(ds["year"] < y) & ds["label"].notna()]
        test = ds[ds["year"] == y]
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
        f.write("欄位說明：prec_* = 超額報酬 > 0 的比例；avg_ret_signal_net 已扣來回成本 0.6%；"
                "strategy_return 只在大盤濾網開啟時進場。\n\n")
        f.write(report.to_markdown())

    eq = pd.concat(curves).sort_index()
    eq["strategy_equity"] = (1 + eq["strategy"]).cumprod()
    eq["benchmark_equity"] = (1 + eq["benchmark"]).cumprod()
    eq.to_csv(C.OUT_DIR / "equity_curve.csv")

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
    print(f"[train] 模型已存：{C.MODEL_DIR / 'model_latest.txt'}；報表：{C.OUT_DIR / 'backtest_report.md'}")


if __name__ == "__main__":
    main()
