"""
每日盤後執行：算最新特徵 → 模型打分 → 產生建議 → 寫入本機日誌與 Supabase。

寫入三張表：
- predictions    當日前 TOP_N 檔與建議（不可竄改：已存在的列不覆寫）
- daily_snapshot 股票池全部分數與主要特徵（可覆寫）
- market_status  大盤濾網與模型開關狀態
"""
import json
import datetime as dt
import numpy as np
import pandas as pd
import lightgbm as lgb

from . import config as C
from .features import FEATURE_COLS, build
from . import supabase_io as sb

LOG_FILE = C.OUT_DIR / "predictions_log.csv"
STATS_FILE = C.OUT_DIR / "stats.json"
PRED_COLS = ["pred_date", "stock_id", "stock_name", "score", "rank", "raw_signal", "signal",
             "market_on", "close_at_pred", "model_version",
             "resolved_date", "actual_return", "index_return", "excess_return", "hit", "hit_abs"]
SNAPSHOT_FEATS = ["ret_5", "ret_20", "dist_ma60", "vol_ratio", "foreign_5", "trust_5",
                  "trust_streak", "foreign_streak", "margin_chg_5", "rev_yoy"]


def load_model():
    path = C.MODEL_DIR / "model_latest.txt"
    if not path.exists():
        raise SystemExit("找不到模型，請先執行 python -m pipeline.train")
    meta = json.loads((C.MODEL_DIR / "meta.json").read_text())
    return lgb.Booster(model_file=str(path)), meta


def gate_enabled() -> bool:
    if STATS_FILE.exists():
        return bool(json.loads(STATS_FILE.read_text()).get("enabled", False))
    return False


def make_signals(latest: pd.DataFrame, model: lgb.Booster, meta: dict) -> pd.DataFrame:
    df = latest.copy()
    # 用模型訓練時的特徵清單，程式的特徵清單改了但模型還沒重訓時也不會錯位
    cols = meta.get("features") or FEATURE_COLS
    df["score"] = model.predict(df[cols], num_iteration=meta.get("best_iteration"))
    df = df.sort_values("score", ascending=False).reset_index(drop=True)
    df["rank"] = np.arange(1, len(df) + 1)
    df["raw_signal"] = np.where(df["score"] >= C.SCORE_STRONG, "強烈買進",
                       np.where(df["score"] >= C.SCORE_BUY, "買進", "觀望"))
    on = bool(df["market_on"].iloc[0])
    df["signal"] = df["raw_signal"]
    if not on:
        df.loc[df["raw_signal"] != "觀望", "signal"] = "觀望（大盤濾網）"
    return df


def append_local_log(pred: pd.DataFrame):
    old = pd.read_csv(LOG_FILE, dtype={"stock_id": str}) if LOG_FILE.exists() else pd.DataFrame(columns=PRED_COLS)
    new = pred[[c for c in PRED_COLS if c in pred.columns]].copy()
    for c in PRED_COLS:
        if c not in new.columns:
            new[c] = np.nan
    key = set(zip(old["pred_date"].astype(str), old["stock_id"].astype(str)))
    new = new[[ (d, s) not in key for d, s in zip(new["pred_date"].astype(str), new["stock_id"].astype(str))]]
    out = new[PRED_COLS] if old.empty else pd.concat([old, new[PRED_COLS]], ignore_index=True)
    out.to_csv(LOG_FILE, index=False)
    print(f"[predict] 本機日誌新增 {len(new)} 列（已存在者不覆寫）-> {LOG_FILE.name}")


def main():
    feat, idx = build(save=True)
    model, meta = load_model()
    last_date = feat["date"].max()
    latest = feat[feat["date"] == last_date].dropna(subset=FEATURE_COLS, how="all")
    df = make_signals(latest, model, meta)
    df["pred_date"] = last_date.date().isoformat()
    if "close_raw" in df.columns:
        df["close"] = df["close_raw"]  # 看板與日誌顯示原始收盤價，不顯示還原價
    df["close_at_pred"] = df["close"]
    df["model_version"] = meta.get("model_version", C.MODEL_VERSION)
    enabled = gate_enabled()

    top = df.head(C.TOP_N)
    pred = top[["pred_date", "stock_id", "stock_name", "score", "rank", "raw_signal", "signal",
                "market_on", "close_at_pred", "model_version"]].copy()
    append_local_log(pred)
    sb.upsert("predictions", pred, on_conflict="pred_date,stock_id", ignore_duplicates=True)

    snap = df[["pred_date", "stock_id", "stock_name", "score", "rank", "raw_signal", "close", "market_on"]
              + SNAPSHOT_FEATS].copy()
    sb.upsert("daily_snapshot", snap, on_conflict="pred_date,stock_id")
    snap.to_csv(C.OUT_DIR / "daily_snapshot_latest.csv", index=False)

    status = pd.DataFrame([dict(
        date=last_date.date().isoformat(),
        idx_close=float(df["idx_close"].iloc[0]),
        idx_ma60=float(df["idx_ma60"].iloc[0]),
        idx_ret_20=float(df["idx_ret_20"].iloc[0]),
        market_on=bool(df["market_on"].iloc[0]),
        gate_enabled=enabled,
        n_signals=int((top["raw_signal"] != "觀望").sum()),
        model_version=meta.get("model_version", C.MODEL_VERSION),
        updated_at=dt.datetime.now().isoformat(timespec="seconds"),
    )])
    sb.upsert("market_status", status, on_conflict="date")
    status.to_csv(C.OUT_DIR / "market_status_latest.csv", index=False)

    print(f"\n=== {last_date.date()} 建議 ===  大盤濾網：{'開' if df['market_on'].iloc[0] else '關'}  "
          f"模型開關：{'啟用' if enabled else '觀察期'}")
    for _, r in top.iterrows():
        print(f"  {r['rank']:>2}. {r['stock_id']} {r['stock_name']:<8} 分數 {r['score']:.3f}  {r['signal']}")


if __name__ == "__main__":
    main()
