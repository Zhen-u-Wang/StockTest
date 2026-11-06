"""
標籤：隔日開盤買進、持有到第 HORIZON 個交易日收盤的報酬，減去大盤同期報酬 = 超額報酬。
訊號在收盤後才產生，所以不能用當日收盤價當進場價。
同一天跨股票排名，前 LABEL_TOP 標 1、後 LABEL_BOTTOM 標 0、中間捨棄（NaN）。
"""
import numpy as np
import pandas as pd

from . import config as C


def add_labels(feat: pd.DataFrame, idx: pd.DataFrame) -> pd.DataFrame:
    p = feat.sort_values(["stock_id", "date"]).copy()
    g = p.groupby("stock_id")
    entry = g["open"].shift(-1)
    p["fwd_ret"] = g["close"].shift(-C.HORIZON) / entry.where(entry > 0) - 1
    p["exit_date"] = g["date"].shift(-C.HORIZON)  # 標籤要到這天才知道，訓練切分時用來隔離
    i = idx[["date", "idx_open", "idx_close"]].sort_values("date").copy()
    # 指數沒有開盤價時，退回用當日收盤當起點
    idx_entry = i["idx_open"].shift(-1)
    idx_entry = idx_entry.where(idx_entry > 0, i["idx_close"])
    i["idx_fwd"] = i["idx_close"].shift(-C.HORIZON) / idx_entry - 1
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
