"""
把本機輸出匯出成前端可直接讀的 JSON（docs/data/），給「只用 GitHub」模式用：
GitHub Actions 跑完 commit 這些檔案，GitHub Pages 的 index.html 直接 fetch。

產出：
  docs/data/status.json       大盤濾網 + 模型統計 + 產生時間
  docs/data/predictions.json  最新一天的建議
  docs/data/history.json      最近 200 筆已結算訊號
  docs/data/equity.json       策略 vs 大盤權益曲線（由已結算訊號計算）
  docs/data/snapshot.json     股票池前 30 名
  docs/data/backtest.json     walk-forward 回測報表（若有）
"""
import json
import datetime as dt
import math
import numpy as np
import pandas as pd

from . import config as C
from .predict import LOG_FILE, STATS_FILE
from .score import BUY_SET, independent_periods, _price_index

DOCS_DATA = C.ROOT / "docs" / "data"


def _clean(v):
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        return None if (v is None or math.isnan(v) or math.isinf(v)) else round(float(v), 6)
    if isinstance(v, pd.Timestamp):
        return v.date().isoformat()
    if v is pd.NaT:
        return None
    return v


def _records(df: pd.DataFrame) -> list[dict]:
    return [{k: _clean(v) for k, v in r.items()} for r in df.to_dict("records")]


def _dump(name: str, obj):
    DOCS_DATA.mkdir(parents=True, exist_ok=True)
    (DOCS_DATA / name).write_text(json.dumps(obj, ensure_ascii=False, separators=(",", ":")))


def equity_from_log(log: pd.DataFrame) -> list[dict]:
    done = log[log["resolved_date"].notna()].copy()
    if done.empty:
        return []
    done["market_on"] = done["market_on"].astype(str).str.lower().eq("true")
    act = done[done["raw_signal"].isin(BUY_SET) & done["market_on"]]
    # 只用不重疊持有期複利；每天都有的 5 日報酬直接連乘會重複計算
    _, idx = _price_index()
    per = independent_periods(act, idx["date"])
    if per.empty:
        return []
    per = per.sort_values("pred_date")
    strategy = (1 + per["net_ret"].astype(float)).cumprod() - 1
    benchmark = (1 + per["index_ret"].astype(float)).cumprod() - 1
    return [dict(date=str(d), strategy=round(float(s), 5), benchmark=round(float(b), 5))
            for d, s, b in zip(per["pred_date"], strategy, benchmark)]


def main():
    if not LOG_FILE.exists():
        raise SystemExit("還沒有預測日誌，請先執行 python -m pipeline.predict")
    log = pd.read_csv(LOG_FILE, dtype={"stock_id": str})
    log["pred_date"] = log["pred_date"].astype(str)

    stats = json.loads(STATS_FILE.read_text()) if STATS_FILE.exists() else {}
    status_path = C.OUT_DIR / "market_status_latest.csv"
    status = pd.read_csv(status_path).iloc[0].to_dict() if status_path.exists() else {}
    status = {k: _clean(v) for k, v in status.items()}
    _dump("status.json", dict(status=status, stats=stats,
                              generated_at=dt.datetime.now().isoformat(timespec="seconds")))

    last = log["pred_date"].max()
    today = log[log["pred_date"] == last].sort_values("rank")
    _dump("predictions.json", _records(today))

    hist = log[log["resolved_date"].notna()].sort_values(["resolved_date", "rank"], ascending=[False, True]).head(200)
    _dump("history.json", _records(hist))

    _dump("equity.json", equity_from_log(log))

    snap_path = C.OUT_DIR / "daily_snapshot_latest.csv"
    snap = pd.read_csv(snap_path, dtype={"stock_id": str}).head(30) if snap_path.exists() else pd.DataFrame()
    _dump("snapshot.json", _records(snap))

    bt = C.OUT_DIR / "backtest_report.csv"
    _dump("backtest.json", _records(pd.read_csv(bt)) if bt.exists() else [])

    print(f"[export] docs/data/*.json 已更新：建議 {len(today)} 筆、歷史 {len(hist)} 筆、曲線 {len(equity_from_log(log))} 點")


if __name__ == "__main__":
    main()
