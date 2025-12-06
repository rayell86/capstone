# this builds out/tables/events_canonical.parquet from JSONL/Parquet event files i complied earlier.
from pathlib import Path
import json
import yaml
import pandas as pd
import numpy as np

#load config
CFG_PATH = Path("config.yml")
assert CFG_PATH.exists(), "config.yml not found in current directory."
with CFG_PATH.open("r", encoding="utf-8") as f:
    CFG = yaml.safe_load(f)

SUBS = [s.lower() for s in CFG.get("subreddits", [])]  
DR   = CFG.get("date_range", {}) or {}
START = pd.to_datetime(DR.get("start", "1900-01-01"), utc=True, errors="coerce")
END   = pd.to_datetime(DR.get("end",   "2100-01-01"), utc=True, errors="coerce")

P = CFG.get("paths", {}) or {}

scan_roots = [Path(p) for p in P.get("scan_roots", [])]
if not scan_roots:
    scan_roots = [Path(P.get("processed", "data/processed")),
                  Path(P.get("raw", "data/raw"))]
scan_roots = [r for r in scan_roots if r and Path(r).exists()]
assert scan_roots, "No valid scan roots. Add paths.scan_roots to config.yml (full Windows paths)."

OUT_EVENTS = Path(P.get("events_canonical", "out/tables/events_canonical.parquet"))
OUT_EVENTS.parent.mkdir(parents=True, exist_ok=True)

# there are helpers
TS_CANDIDATES = ("created_utc", "created", "created_ts", "timestamp", "time", "date")

def norm_sub(s: str | None) -> str | None:
    if s is None: return None
    return str(s).lower().replace("r/", "").strip()

def parse_time_val(v):
    """Return UTC-aware pandas Timestamp or NaT. Supports epoch sec/ms or ISO strings."""
    if v is None:
        return pd.NaT
    
    if isinstance(v, (int, float, np.integer, np.floating)) or (isinstance(v, str) and v.strip().isdigit()):
        try:
            fv = float(v)
            unit = "ms" if fv > 1e12 else "s"   
            return pd.to_datetime(fv, unit=unit, utc=True, errors="coerce")
        except Exception:
            pass
    
    try:
        return pd.to_datetime(v, utc=True, errors="coerce")
    except Exception:
        return pd.NaT

def extract_ts(d: dict):
    for k in TS_CANDIDATES:
        if k in d:
            dt = parse_time_val(d.get(k))
            if pd.notna(dt):
                return dt
    return pd.NaT

def sniff_source_type(path_str: str, rec: dict) -> str:
    """Return 'post' or 'comment' using hints from record or file path."""
    v = (rec.get("source_type") or rec.get("kind") or "").lower()
    if v in ("post", "submission"): return "post"
    if v in ("comment", "comments"): return "comment"
    ps = path_str.lower()
    if "comments" in ps: return "comment"
    if "posts" in ps or "submission" in ps: return "post"
    if "title" in rec: return "post"
    if "parent_id" in rec or "link_id" in rec: return "comment"
    return "post"

def pick_id(rec: dict):
    # this captures prefered stable reddit ids if present
    return rec.get("id") or rec.get("name") or rec.get("link_id") or None

# this scans and ingests
rows = []
jsonl_files = parquet_files = 0

for root in scan_roots:
    for p in Path(root).rglob("*.jsonl"):
        jsonl_files += 1
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line: continue
                try:
                    d = json.loads(line)
                except Exception:
                    continue
                sub = norm_sub(d.get("subreddit") or d.get("sub"))
                if not sub: 
                    continue
                if SUBS and sub not in [s.lower() for s in SUBS]:
                    continue
                dt = extract_ts(d)
                if pd.isna(dt): 
                    continue
                st = sniff_source_type(str(p), d)
                rid = pick_id(d)
                rows.append((rid, sub, dt, st))

    
    for p in Path(root).rglob("*.parquet"):
        try:
            df = pd.read_parquet(p)
        except Exception:
            continue
        cols_lower = {c.lower(): c for c in df.columns}
        if "subreddit" not in cols_lower:
            continue
        tcol = next((cols_lower[c] for c in TS_CANDIDATES if c in cols_lower), None)
        if not tcol:
            continue
        parquet_files += 1

        subcol = cols_lower["subreddit"]
        idcol  = cols_lower.get("id") or cols_lower.get("name") or None
        kcol   = cols_lower.get("source_type") or cols_lower.get("kind") or None

        use_cols = [c for c in [idcol, subcol, tcol, kcol] if c is not None]
        df = df[use_cols].copy()

        # i normalize subreddits here
        df[subcol] = df[subcol].map(norm_sub)
        if SUBS:
            df = df[df[subcol].isin([s.lower() for s in SUBS])]
        if df.empty:
            continue

        dt_series = pd.to_datetime(df[tcol], utc=True, errors="coerce")
        bad = dt_series.isna()
        if bad.any():
            s_num = pd.to_numeric(df.loc[bad, tcol], errors="coerce")
            med = float(s_num.dropna().median()) if s_num.notna().any() else np.nan
            unit = "ms" if pd.notna(med) and med > 1e12 else "s"
            dt_series.loc[bad] = pd.to_datetime(s_num, unit=unit, utc=True, errors="coerce")
        df["__ts"] = dt_series
        df = df.dropna(subset=["__ts"])
        if df.empty:
            continue


        if kcol:
            st = df[kcol].astype(str).str.lower().str.replace(r"s$", "", regex=True)
            st = st.where(st.isin(["post", "comment"]), "post")
        else:
            st = pd.Series(["post"] * len(df), index=df.index)

        rid = df[idcol] if idcol else pd.Series([None] * len(df), index=df.index)

        rows.extend(zip(rid.tolist(), df[subcol].tolist(), df["__ts"].tolist(), st.tolist()))

print(f"[scan] jsonl files: {jsonl_files}  | parquet files: {parquet_files}")
print(f"[scan] collected rows: {len(rows):,}")
assert rows, "No events found. Check paths.scan_roots or add your folders in config.yml."


events = pd.DataFrame(rows, columns=["id", "subreddit", "created_dt", "source_type"]).dropna(subset=["subreddit", "created_dt"])
events["subreddit"] = events["subreddit"].astype(str)
events["source_type"] = events["source_type"].astype(str).str.lower().str.replace(r"s$", "", regex=True)
events["source_type"] = events["source_type"].where(events["source_type"].isin(["post", "comment"]), "post")

# date range filter
events = events[(events["created_dt"] >= START) & (events["created_dt"] <= END)]

# dedupe
if events["id"].notna().any():
    events = events.sort_values("created_dt").drop_duplicates(subset=["id"], keep="first")
else:
    events = (events.sort_values("created_dt")
                    .drop_duplicates(subset=["subreddit", "created_dt", "source_type"], keep="first"))

# check my work here
by_sub = events.groupby("subreddit")["created_dt"].agg(["min", "max", "count"]).sort_index()
print("\n[summary by subreddit]")
print(by_sub)

events.to_parquet(OUT_EVENTS, index=False)
print(f"\n✅ Wrote {OUT_EVENTS.resolve()}  rows={len(events):,}")
