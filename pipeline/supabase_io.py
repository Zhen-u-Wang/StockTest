"""
Supabase 讀寫。沒有設定 SUPABASE_URL / SUPABASE_SERVICE_KEY 時所有函式直接略過，
pipeline 仍會把結果寫到本機 output/，方便離線測試。
"""
import math
import numpy as np
import pandas as pd

from . import config as C

_client = None


def client():
    global _client
    if _client is None and C.SUPABASE_URL and C.SUPABASE_SERVICE_KEY:
        from supabase import create_client
        _client = create_client(C.SUPABASE_URL, C.SUPABASE_SERVICE_KEY)
    return _client


def _clean(v):
    """pandas / numpy 型別轉成 JSON 可接受的值。"""
    if v is None:
        return None
    if isinstance(v, (pd.Timestamp,)):
        return v.date().isoformat()
    if isinstance(v, (np.bool_, bool)):
        return bool(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        return None if (math.isnan(v) or math.isinf(v)) else float(v)
    if isinstance(v, float) and math.isnan(v):
        return None
    return v


def records(df: pd.DataFrame) -> list[dict]:
    return [{k: _clean(v) for k, v in row.items()} for row in df.to_dict("records")]


def upsert(table: str, df: pd.DataFrame, on_conflict: str, chunk=500, ignore_duplicates=False):
    """ignore_duplicates=True：已存在的列不覆寫（預測日誌用，保持不可竄改）。"""
    c = client()
    if c is None or df.empty:
        return False
    rows = records(df)
    for i in range(0, len(rows), chunk):
        c.table(table).upsert(rows[i:i + chunk], on_conflict=on_conflict,
                              ignore_duplicates=ignore_duplicates).execute()
    print(f"[supabase] {table}: upsert {len(rows)} 列")
    return True


def fetch_all(table: str, select="*", order=None, limit=10000) -> pd.DataFrame:
    c = client()
    if c is None:
        return pd.DataFrame()
    q = c.table(table).select(select)
    if order:
        q = q.order(order)
    res = q.limit(limit).execute()
    return pd.DataFrame(res.data)
