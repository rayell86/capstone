from pathlib import Path
import yaml, pandas as pd
import numpy as np
import pyarrow.parquet as pq

with open("config.yml","r",encoding="utf-8") as f:
    CFG = yaml.safe_load(f) or {}
P = CFG.get("paths", {}) or {}
EVENTS = Path(P.get("events_canonical","out/tables/events_canonical.parquet"))

if not EVENTS.exists():
    print(f"not found: {EVENTS.resolve()}")
else:
    pf = pq.ParquetFile(EVENTS)
    cols_in_file = set(pf.schema.names)
    print("[schema] columns in file:", sorted(cols_in_file))

    # i grab columns that actually exist
    want = ["subreddit","source_type","created_dt","title","selftext","body","text","id"]
    cols_to_read = [c for c in want if c in cols_in_file]
    print("[read] columns_to_read:", cols_to_read)

    df = pd.read_parquet(EVENTS, columns=cols_to_read)
    print(f"{EVENTS} rows={len(df):,}")

    # i normalized subreddits and created_dt
    df["subreddit"] = (df["subreddit"].astype(str)
                       .str.lower().str.replace(r"^\s*r/","", regex=True).str.strip())
    if "created_dt" in df.columns:
        dt = pd.to_datetime(df["created_dt"], errors="coerce", utc=True)
        try:
            dt = dt.dt.tz_convert(None)
        except Exception:
            dt = dt.dt.tz_localize(None)
        print("created_dt range:", dt.min(), "→", dt.max())

    # i spilted posts and comments
    if "source_type" in df.columns:
        print("\nsource_type counts:\n", df["source_type"].astype(str).value_counts(dropna=False))
        print("\nby subreddit & type (top 10):\n",
              df.groupby(["subreddit","source_type"]).size().sort_values(ascending=False).head(10))
    else:
        
        has_title = "title" in df.columns
        if has_title:
            ht = np.where(df["title"].astype(str).str.len()>0, "post", "comment")
            print("\n(no source_type) heuristic counts:\n", pd.Series(ht).value_counts())
            print("\nby subreddit (heuristic, top 10):\n",
                  pd.DataFrame({"subreddit": df["subreddit"], "ht": ht})
                    .groupby(["subreddit","ht"]).size()
                    .sort_values(ascending=False).head(10))
        else:
            print("\n(no source_type and no text/title columns present)")

    print("\ncolumns sample:", sorted(df.columns)[:20])
