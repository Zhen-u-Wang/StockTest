"""
標籤：未來 HORIZON 個交易日的報酬，減去大盤同期報酬 = 超額報酬。
同一天跨股票排名，前 LABEL_TOP 標 1、後 LABEL_BOTTOM 標 0、中間捨棄（NaN）。
"""
import numpy as np
import pandas as pd

from . import config as C


def add_labels(feat: pd.DataFrame, idx: pd.DataFrame) -> pd.DataFrame:
    p = feat.sort_values(["stock_id", "date"]).copy()
    g = p.groupby("stock_id")
    p["fwd_ret"] = g["close"].shift(-C.HORIZON) / p["close"] - 1

    i = idx[["date", "idx_close"]].sort_values("date").copy()
    i["idx_fwd"] = i["idx_close"].shift(-C.HORIZON) / i["idx_close"] - 1
    p = p.merge(i[["date", "idx_fwd"]], on="date", how="left")

    p["excess"] = p["fwd_ret"] - p["idx_fwd"]
    pct = p.groupby("date")["excess"].rank(pct=True)
    p["label"] = np.where(pct >= 1 - C.LABEL_TOP, 1.0,
                 np.where(pct <= C.LABEL_BOTTOM, 0.0, np.nan))
    p.loc[p["excess"].isna(), "label"] = np.nan
    return p.sort_values(["date", "stock_id"]).reset_index(drop=True)


if __name__ == "__main__":
    feat = pd.read_parquet(C.DATA_DIR / "features.parquet")
    idx = pd.read_parquet(C.DATA_DIR / "index_features.parquet")
    out = add_labels(feat, idx)
    out.to_parquet(C.DATA_DIR / "dataset.parquet", index=False)
    print(f"[labels] 有標籤 {out['label'].notna().sum()} 列 / 全部 {len(out)} 列")
