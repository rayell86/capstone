from pathlib import Path
import pandas as pd

TEST = Path("out/tables/_parquet_sanity_test.parquet")
TEST.parent.mkdir(parents=True, exist_ok=True)

# this writes a tiny parquet
pd.DataFrame({"a":[1,2], "b":["x","y"]}).to_parquet(TEST, engine="pyarrow")

# this verifies if header is readable
with open(TEST, "rb") as f:
    head = f.read(4)
    f.seek(-4, 2); tail = f.read(4)
print("magic:", head, tail)

df_chk = pd.read_parquet(TEST, engine="pyarrow")
print("rows, cols:", df_chk.shape)
print(df_chk.head(2))


#########################step 2
from pathlib import Path
import shutil, os
import pandas as pd
import pyarrow as pa, pyarrow.parquet as pq, pyarrow.dataset as ds

ROOT = Path("out/tables/docs_canonical_ds")
if ROOT.exists():
    shutil.rmtree(ROOT)
ROOT.mkdir(parents=True, exist_ok=True)

# minimal sample with target columns
pdf = pd.DataFrame({
    "subreddit":   ["antiwork","remotework"],
    "source_type": ["post","post"],
    "title":       ["t1","t2"],
    "selftext":    ["s1","s2"],
    "body":        ["",""],
    "created_dt":  pd.to_datetime(["2024-01-01","2024-01-02"], utc=True),
})

schema = pa.schema([
    ("subreddit",   pa.string()),
    ("source_type", pa.string()),
    ("title",       pa.string()),
    ("selftext",    pa.string()),
    ("body",        pa.string()),
    ("created_dt",  pa.timestamp("us", tz="UTC")),
])

tbl  = pa.Table.from_pandas(pdf, schema=schema, preserve_index=False, safe=False).cast(schema, safe=False)
part = ROOT / "part-00000.parquet"
pq.write_table(tbl, part, compression="zstd", use_dictionary=True, data_page_version="2.0")

# quick test to see if everything checks out till now
with open(part, "rb") as f:
    head=f.read(4); f.seek(-4, os.SEEK_END); tail=f.read(4)
print("magic:", head, tail, "| file:", part.name)

# dataset check
dataset = ds.dataset(ROOT, format="parquet")
t = dataset.to_table(columns=list(pdf.columns))
print("rows, cols:", t.num_rows, len(t.column_names))
print(t.column_names)


##################step3
from pathlib import Path
import pandas as pd

DIR = Path(r"C:\Users\raypo\Capstone\data\raw")
files = sorted(DIR.glob("*.jsonl"))
assert files, f"No .jsonl files found in {DIR}"

priority = [f for f in files if "post" in f.name.lower()] or \
           [f for f in files if "comment" in f.name.lower()] or files
RAW = priority[0]
print("Using file:", RAW)

# I wanted to read a chunk here to see if everything checks out
df = next(pd.read_json(RAW, lines=True, chunksize=100_000))
print("df shape:", df.shape)

TARGET = ["subreddit","source_type","title","selftext","body","created_dt"]
present = [c for c in TARGET if c in df.columns]
print("present cols:", present)

if "created_dt" not in df.columns:
    time_cands = [c for c in df.columns if c.lower() in ("created_utc","created","created_at","created_ts")]
    print("time candidates:", time_cands[:3])

# check the sample of relevant columns
show = [c for c in ["subreddit","title","selftext","body"] if c in df.columns]
print(df[show].head(3).to_string(index=False) if show else df.head(3).to_string(index=False))

#################################step 4
import pandas as pd

TARGET = ["subreddit","source_type","title","selftext","body","created_dt"]

def normalize_posts(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["subreddit"]   = df["subreddit"].astype("string")
    out["title"]       = df["title"].astype("string")
    out["selftext"]    = df["selftext"].astype("string")
    out["body"]        = pd.Series(pd.NA, index=df.index, dtype="string")
    out["source_type"] = pd.Series("post", index=df.index, dtype="string")
    created = (pd.to_datetime(df["created_dt"], errors="coerce", utc=True)
               if "created_dt" in df.columns
               else pd.to_datetime(df["created_utc"], errors="coerce", unit="s", utc=True))
    out["created_dt"]  = created
    return out[TARGET]

# build and check
pdf = normalize_posts(df)
print(pdf.head(3).to_string(index=False))
print("dtypes:\n", pdf.dtypes)
print("nulls:\n", pdf.isna().sum())

##############################validate step 4
import pyarrow as pa, pyarrow.parquet as pq, pyarrow.dataset as ds
from pathlib import Path

SCHEMA = pa.schema([
    ("subreddit",   pa.string()),
    ("source_type", pa.string()),
    ("title",       pa.string()),
    ("selftext",    pa.string()),
    ("body",        pa.string()),
    ("created_dt",  pa.timestamp("us", tz="UTC")),
])

ROOT = Path("out/tables/docs_canonical_ds"); ROOT.mkdir(parents=True, exist_ok=True)
tbl  = pa.Table.from_pandas(pdf, schema=SCHEMA, preserve_index=False, safe=False).cast(SCHEMA, safe=False)
part = ROOT / "part-00001.parquet"
pq.write_table(tbl, part, compression="zstd", use_dictionary=True, data_page_version="2.0")

with open(part, "rb") as f:
    head=f.read(4); f.seek(-4, 2); tail=f.read(4)
print("magic:", head, tail, "| rows:", tbl.num_rows)

import os, pyarrow.dataset as ds

with open(part, "rb") as f:
    head = f.read(4)
    f.seek(-4, os.SEEK_END)
    tail = f.read(4)
print("magic:", head, tail) 

dataset = ds.dataset(ROOT, format="parquet")
t = dataset.to_table(columns=list(SCHEMA.names))
print("dataset rows:", t.num_rows)
print(t.slice(0, 2).to_pandas())

###################################step 5

import pandas as pd, pyarrow as pa, pyarrow.parquet as pq, pyarrow.dataset as ds
from pathlib import Path

it = pd.read_json(RAW, lines=True, chunksize=100_000)
next(it)                    
df2 = next(it)                

pdf2 = normalize_posts(df2)
tbl2 = pa.Table.from_pandas(pdf2, schema=SCHEMA, preserve_index=False, safe=False).cast(SCHEMA, safe=False)

part2 = Path("out/tables/docs_canonical_ds") / "part-00002.parquet"
pq.write_table(tbl2, part2, compression="zstd", use_dictionary=True, data_page_version="2.0")

with open(part2, "rb") as f:
    head=f.read(4); f.seek(-4, 2); tail=f.read(4)
print("magic:", head, tail, "| rows:", tbl2.num_rows)

t = ds.dataset("out/tables/docs_canonical_ds", format="parquet").to_table(columns=list(SCHEMA.names))
print("dataset rows:", t.num_rows)

ROOT = Path("out/tables/docs_canonical_ds")

def next_part_index(root: Path) -> int:
    nums = []
    for p in root.glob("part-*.parquet"):
        m = re.search(r"part-(\d+)\.parquet", p.name)
        if m: nums.append(int(m.group(1)))
    return (max(nums) if nums else -1) + 1

n = next_part_index(ROOT) 

for chunk in it:
    pdf_i  = normalize_posts(chunk)
    tbl_i  = pa.Table.from_pandas(pdf_i, schema=SCHEMA, preserve_index=False, safe=False).cast(SCHEMA, safe=False)
    part_i = ROOT / f"part-{n:05d}.parquet"; n += 1
    pq.write_table(tbl_i, part_i, compression="zstd", use_dictionary=True, data_page_version="2.0")
    with open(part_i, "rb") as f:
        head = f.read(4); _ = f.seek(-4, os.SEEK_END); tail = f.read(4)
    print(f"OK {part_i.name} rows={tbl_i.num_rows} magic={(head, tail)}")

#############################step6    
# ---- config
DIR  = Path(r"C:\Users\raypo\Capstone\data\raw")
ROOT = Path("out/tables/docs_canonical_ds")
CHUNK = 100_000

SCHEMA = pa.schema([
    ("subreddit",   pa.string()),
    ("source_type", pa.string()),
    ("title",       pa.string()),
    ("selftext",    pa.string()),
    ("body",        pa.string()),
    ("created_dt",  pa.timestamp("us", tz="UTC")),
])

TARGET = ["subreddit","source_type","title","selftext","body","created_dt"]

def _dt(df, col):
    return (pd.to_datetime(df[col], errors="coerce", utc=True)
            if col in ("created_dt","created_at")
            else pd.to_datetime(df[col], errors="coerce", unit="s", utc=True))

def normalize_posts(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["subreddit"]   = df["subreddit"].astype("string")
    out["title"]       = df["title"].astype("string")
    out["selftext"]    = df["selftext"].astype("string")
    out["body"]        = pd.Series(pd.NA, index=df.index, dtype="string")
    out["source_type"] = pd.Series("post", index=df.index, dtype="string")
    tcol = "created_dt" if "created_dt" in df.columns else "created_utc"
    out["created_dt"]  = _dt(df, tcol)
    return out[TARGET]

def normalize_comments(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    sub  = next((c for c in ["subreddit","subreddit_name","subreddit_display_name"] if c in df.columns), None)
    body = next((c for c in ["body","comment_body","text","content","message","message_body"] if c in df.columns), None)
    tcol = next((c for c in ["created_dt","created_at","created_utc","created","created_ts"] if c in df.columns), None)
    out["subreddit"]   = (df[sub].astype("string") if sub else pd.Series(pd.NA, index=df.index, dtype="string"))
    out["title"]       = pd.Series(pd.NA, index=df.index, dtype="string")
    out["selftext"]    = pd.Series(pd.NA, index=df.index, dtype="string")
    out["source_type"] = pd.Series("comment", index=df.index, dtype="string")
    out["body"]        = (df[body].astype("string") if body else pd.Series(pd.NA, index=df.index, dtype="string"))
    out["created_dt"]  = (_dt(df, tcol) if tcol else pd.Series(pd.NaT, index=df.index))
    return out[TARGET]

def normalize_auto(df: pd.DataFrame) -> pd.DataFrame:
    is_post = any(c in df.columns for c in ("title","selftext"))
    return normalize_posts(df) if is_post else normalize_comments(df)

def next_part_index(root: Path) -> int:
    nums=[int(m.group(1)) for p in root.glob("part-*.parquet")
          for m in [re.search(r"part-(\d+)\.parquet", p.name)] if m]
    return (max(nums) if nums else -1) + 1

files = sorted(DIR.glob("*.jsonl"))
assert files, f"No *.jsonl in {DIR}"
n = next_part_index(ROOT)
ROOT.mkdir(parents=True, exist_ok=True)

for fp in files:
    it = pd.read_json(fp, lines=True, chunksize=CHUNK)
    for j, chunk in enumerate(it, start=1):
        pdf = normalize_auto(chunk)
        if pdf.empty: 
            continue
        tbl  = pa.Table.from_pandas(pdf, schema=SCHEMA, preserve_index=False, safe=False).cast(SCHEMA, safe=False)
        part = ROOT / f"part-{n:05d}.parquet"; n += 1
        pq.write_table(tbl, part, compression="zstd", use_dictionary=True, data_page_version="2.0")
        with open(part, "rb") as f:
            head=f.read(4); _=f.seek(-4, os.SEEK_END); tail=f.read(4)
        print(f"[{fp.name} #{j}] OK {part.name} rows={tbl.num_rows} {head, tail}")

ROOT = "out/tables/docs_canonical_ds"
d = ds.dataset(ROOT, format="parquet")

print("total rows:", d.count_rows())
print("posts:",     d.count_rows(filter=field("source_type") == "post"))
print("comments:",  d.count_rows(filter=field("source_type") == "comment"))


#######step7
import collections, pyarrow.dataset as ds
from pyarrow.dataset import field
d = ds.dataset("out/tables/docs_canonical_ds", format="parquet")

sub = collections.Counter()
for batch in d.to_batches(columns=["subreddit","source_type"], batch_size=1_000_000):
    df = batch.to_pandas()
    sub.update(zip(df["subreddit"], df["source_type"]))
print(sub.most_common(10))

import pandas as pd
mins, maxs = [], []
for batch in d.to_batches(columns=["created_dt","source_type"], batch_size=1_000_000):
    s = batch.to_pandas()
    mins.append(s["created_dt"].min()); maxs.append(s["created_dt"].max())
print("date range:", pd.Series(mins).min(), "→", pd.Series(maxs).max())


import pyarrow as pa, pyarrow.parquet as pq
from pathlib import Path
TARGET = ["subreddit","source_type","title","selftext","body","created_dt"]
out = Path("out/tables/docs_canonical.parquet")
tmp = out.with_suffix(".parquet.tmp")

writer = None
for batch in d.to_batches(columns=TARGET, batch_size=1_000_000):
    if writer is None:
        writer = pq.ParquetWriter(tmp, batch.schema, compression="zstd", use_dictionary=True, data_page_version="2.0")
    writer.write_table(pa.Table.from_batches([batch]))
writer.close(); tmp.replace(out)


#####################step 7.5 test payload

out = Path("out/tables/docs_canonical.parquet")
with open(out, "rb") as f:
    head=f.read(4); f.seek(-4, os.SEEK_END); tail=f.read(4)
print("magic:", head, tail)                       

pf = pq.ParquetFile(out)
print("row_groups:", pf.metadata.num_row_groups)
print("rows_in_file:", pf.metadata.num_rows)     

out = Path("out/tables/docs_canonical.parquet")

tbl = pq.read_table(out, columns=["subreddit","source_type","title","selftext","body","created_dt"])
tbl_head = tbl.slice(0, 1000)

print(tbl_head.schema)
print(tbl_head.to_pandas().head(5))

d = ds.dataset("out/tables/docs_canonical.parquet", format="parquet")
print("posts body non-null:",    d.count_rows(filter=(field("source_type")=="post")    & field("body").is_valid()))
print("comments title non-null:",d.count_rows(filter=(field("source_type")=="comment") & field("title").is_valid()))


###################step8
from collections import Counter
import pandas as pd, pyarrow as pa, pyarrow.parquet as pq, pyarrow.dataset as ds

SRC = "out/tables/docs_canonical_ds"
d = ds.dataset(SRC, format="parquet")
ctr = Counter()

for b in d.to_batches(columns=["created_dt","subreddit","source_type"], batch_size=1_000_000):
    df = b.to_pandas()
    df["week_start"] = df["created_dt"].dt.tz_convert("UTC").dt.to_period("W-MON").dt.start_time
    s = df.groupby(["week_start","subreddit","source_type"]).size()
    ctr.update({k:int(v) for k,v in s.items()})

rows = [{"week_start":k[0], "subreddit":k[1], "source_type":k[2], "n":v} for k,v in ctr.items()]
evt = pd.DataFrame(rows).sort_values(["week_start","subreddit","source_type"])
pq.write_table(pa.Table.from_pandas(evt, preserve_index=False),
               "out/tables/events_canonical.parquet",
               compression="zstd", use_dictionary=True, data_page_version="2.0")


############step 9
SRC = "out/tables/docs_canonical_ds" 
d = ds.dataset(SRC, format="parquet")
ctr = Counter()

for b in d.to_batches(columns=["created_dt","subreddit","source_type"], batch_size=1_000_000):
    df = b.to_pandas()
    t = df["created_dt"].dt.tz_convert("UTC").dt.tz_localize(None)
    wk = t.dt.to_period("W-SUN").dt.start_time
    df["week_start"] = pd.to_datetime(wk, utc=True)
    s = df.groupby(["week_start","subreddit","source_type"]).size()
    ctr.update(s.to_dict())

rows = [
    {"week_start": k[0], "subreddit": k[1], "source_type": k[2], "n": v}
    for k, v in ctr.items()
]
evt = pd.DataFrame(rows).sort_values(["week_start","subreddit","source_type"])
pq.write_table(pa.Table.from_pandas(evt, preserve_index=False),
               "out/tables/events_canonical.parquet",
               compression="zstd", use_dictionary=True, data_page_version="2.0")


d_evt = ds.dataset("out/tables/events_canonical.parquet", format="parquet")
total_docs = ds.dataset(SRC, format="parquet").count_rows()
sum_n = sum(b.to_pandas()["n"].sum() for b in d_evt.to_batches(columns=["n"], batch_size=1_000_000))
print("sum(n):", sum_n, "| total_docs:", total_docs)

#######################step 10

EVT = "out/tables/events_canonical.parquet"
evt = pq.read_table(EVT).to_pandas()
evt = evt.sort_values("week_start")

# weekly totals (all types)
wk_all = (evt.groupby(["week_start","subreddit"])["n"]
            .sum().reset_index()
            .pivot(index="week_start", columns="subreddit", values="n")
            .fillna(0).sort_index())

# comments-only / posts-only
wk_com = (evt[evt.source_type=="comment"]
          .groupby(["week_start","subreddit"])["n"].sum().unstack(fill_value=0))
wk_post = (evt[evt.source_type=="post"]
           .groupby(["week_start","subreddit"])["n"].sum().unstack(fill_value=0))

# simple rolling z-score spikes
def zspike(s, win=13):
    mu = s.rolling(win, center=True, min_periods=win//2).mean()
    sd = s.rolling(win, center=True, min_periods=win//2).std(ddof=0)
    return (s - mu) / sd

z_all = wk_all.apply(zspike)
spikes_all = (z_all.abs() > 3)

# this saves artifacts i created
out_dir = "out/stl_weekly"; Path(out_dir).mkdir(parents=True, exist_ok=True)
pq.write_table(pa.Table.from_pandas(wk_all.reset_index()), f"{out_dir}/weekly_totals_all.parquet", compression="zstd")
pq.write_table(pa.Table.from_pandas(wk_com.reset_index()), f"{out_dir}/weekly_totals_comments.parquet", compression="zstd")
pq.write_table(pa.Table.from_pandas(wk_post.reset_index()), f"{out_dir}/weekly_totals_posts.parquet", compression="zstd")
pq.write_table(pa.Table.from_pandas(z_all.reset_index()),  f"{out_dir}/weekly_z_all.parquet", compression="zstd")
pq.write_table(pa.Table.from_pandas(spikes_all.reset_index()), f"{out_dir}/weekly_spikes_all.parquet", compression="zstd")


###########################################################step 11 stl
import pandas as pd, pyarrow.parquet as pq, pyarrow as pa
from statsmodels.tsa.seasonal import STL
import matplotlib.pyplot as plt
from pathlib import Path

wk_all = pq.read_table("out/stl_weekly/weekly_totals_all.parquet").to_pandas()
wk_all["week_start"] = pd.to_datetime(wk_all["week_start"], utc=True)
wk_all = wk_all.sort_values("week_start").set_index("week_start")
wk_all = wk_all.reindex(pd.date_range(wk_all.index.min(), wk_all.index.max(), freq="W-MON", tz="UTC")).fillna(0)

out_dir = Path("out/stl_weekly/stl"); out_dir.mkdir(parents=True, exist_ok=True)
rows=[]; period=52
for sub in wk_all.columns:
    s = wk_all[sub].astype(float)
    if s.notna().sum() < period*2: continue
    res = STL(s, period=period, robust=True).fit()
    rows.append(pd.DataFrame({"week_start": s.index, "subreddit": sub,
                              "y": s.values, "trend": res.trend, "seasonal": res.seasonal, "resid": res.resid}))
    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(s.index, s.values, label="y", linewidth=1)
    ax.plot(s.index, res.trend, label="trend", linewidth=1)
    ax.legend(); fig.tight_layout(); fig.savefig(out_dir/f"{sub}_stl.png", dpi=150); plt.close(fig)

if rows:
    stl_df = pd.concat(rows, ignore_index=True)
    pq.write_table(pa.Table.from_pandas(stl_df), "out/stl_weekly/stl_components.parquet",
                   compression="zstd", use_dictionary=True, data_page_version="2.0")
print("STL done.")


############################step 12
from pathlib import Path; import pyarrow.parquet as pq

print(sorted(p.name for p in Path("out/stl_weekly/stl").glob("*.png"))[:5])
stl = pq.read_table("out/stl_weekly/stl_components.parquet").to_pandas()
print(stl.head(3))

stl = pq.read_table("out/stl_weekly/stl_components.parquet").to_pandas()
stl = stl.sort_values(["subreddit","week_start"])
stl["resid_z"] = stl.groupby("subreddit")["resid"].transform(lambda s: (s - s.mean())/s.std(ddof=0))
spikes = (stl.assign(abs_z=lambda x: x["resid_z"].abs())
              .sort_values(["subreddit","abs_z"], ascending=[True, False])
              .groupby("subreddit").head(10)
              [["week_start","subreddit","y","trend","seasonal","resid","resid_z"]])
spikes.to_csv("out/stl_weekly/stl_spikes_top10_per_sub.csv", index=False)
print(spikes.head(12))

def slope_last_26(g):
    s = g.tail(26)["trend"].astype(float)
    x = np.arange(len(s))
    return float(np.polyfit(x, s.values, 1)[0])

trend_26w = (stl.groupby("subreddit")
               .apply(slope_last_26)
               .rename("trend_slope_26w")
               .reset_index())
trend_26w.to_csv("out/stl_weekly/stl_trend_slope_26w.csv", index=False)
print(trend_26w)

#################step 13

# weekly totals
wk = pq.read_table("out/stl_weekly/weekly_totals_all.parquet").to_pandas()
wk["week_start"] = pd.to_datetime(wk["week_start"], utc=True)
wk = wk.sort_values("week_start").set_index("week_start")

# artifacts
trend = pd.read_csv("out/stl_weekly/stl_trend_slope_26w.csv")
spk   = pd.read_csv("out/stl_weekly/stl_spikes_top10_per_sub.csv", parse_dates=["week_start"])

last   = wk.index.max()
last_s = wk.loc[last]
avg4   = wk.tail(4).mean()

spk_counts = spk.groupby("subreddit").size().rename("spikes_gt3")
spk_top = (spk.assign(abs_z=spk["resid_z"].abs())
             .sort_values(["subreddit","abs_z"], ascending=[True, False])
             .groupby("subreddit").head(1)
             .set_index("subreddit"))

summary = (pd.DataFrame({"last_week": last, "last_week_total": last_s, "avg4wk": avg4})
           .merge(trend.set_index("subreddit"), left_index=True, right_index=True, how="left")
           .merge(spk_counts, left_index=True, right_index=True, how="left")
           .merge(spk_top[["week_start","resid_z","y"]],
                  left_index=True, right_index=True, how="left")
           .rename(columns={"week_start":"topspike_week","resid_z":"topspike_z","y":"topspike_n"})
           .reset_index().rename(columns={"index":"subreddit"}))

summary.to_csv("out/stl_weekly/summary_insights.csv", index=False)
print(summary)

if not Path("out/stl_weekly/weekly_z_all.parquet").exists():
    wk = pq.read_table("out/stl_weekly/weekly_totals_all.parquet").to_pandas()
    wk["week_start"] = pd.to_datetime(wk["week_start"], utc=True)
    wk = wk.sort_values("week_start").set_index("week_start")

    def zspike(s, win=13):
        mu = s.rolling(win, center=True, min_periods=win//2).mean()
        sd = s.rolling(win, center=True, min_periods=win//2).std(ddof=0)
        return (s - mu) / sd

    z_all = wk.apply(zspike)
    pq.write_table(pa.Table.from_pandas(z_all.reset_index()),
                   "out/stl_weekly/weekly_z_all.parquet",
                   compression="zstd", use_dictionary=True, data_page_version="2.0")


#####################################step14

z_all = pq.read_table("out/stl_weekly/weekly_z_all.parquet").to_pandas()
z_all["week_start"] = pd.to_datetime(z_all["week_start"], utc=True)
z = z_all.set_index("week_start")
spike_counts = z.abs().gt(3).sum().rename("spikes_gt3_real").rename_axis("subreddit").reset_index()

summary = pd.read_csv("out/stl_weekly/summary_insights.csv")
summary = summary.drop(columns=["spikes_gt3"], errors="ignore").merge(spike_counts, on="subreddit", how="left")
summary.to_csv("out/stl_weekly/summary_insights.csv", index=False)
print(summary)

# pip install vaderSentiment  (once)
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pyarrow as pa, pyarrow.parquet as pq, pyarrow.dataset as ds
import pandas as pd

SRC = "out/tables/docs_canonical_ds"
d   = ds.dataset(SRC, format="parquet")
an  = SentimentIntensityAnalyzer()

acc = {}
for bi, b in enumerate(d.to_batches(columns=["created_dt","subreddit","source_type","title","selftext"],
                                    batch_size=1_000_000), 1):
    df = b.to_pandas()
    df = df[(df["source_type"]=="post") & df["created_dt"].notna()]
    if df.empty:
        if bi % 5 == 0: print(f"[batch {bi}] (no posts)"); 
        continue

    txt = (df["title"].fillna("") + " " + df["selftext"].fillna("")).astype(str)
    comp = [an.polarity_scores(t)["compound"] for t in txt]

    t  = df["created_dt"].dt.tz_convert("UTC").dt.tz_localize(None)
    wk = t.dt.to_period("W-MON").dt.start_time
    df = pd.DataFrame({"week_start": pd.to_datetime(wk, utc=True),
                       "subreddit": df["subreddit"].astype("string"),
                       "compound": comp})

    g = df.groupby(["week_start","subreddit"])["compound"].agg(["sum","size"]).reset_index()
    for _, r in g.iterrows():
        key = (r["week_start"], r["subreddit"])
        s, n = acc.get(key, (0.0, 0))
        acc[key] = (s + float(r["sum"]), n + int(r["size"]))

    if bi % 5 == 0:
        print(f"[batch {bi}] agg_keys={len(acc)}")

# finalize
rows = [{"week_start":k[0], "subreddit":k[1], "sent_mean": (s/n if n else 0.0), "n_posts": int(n)}
        for k,(s,n) in acc.items()]
sent = pd.DataFrame(rows).sort_values(["week_start","subreddit"])
pq.write_table(pa.Table.from_pandas(sent, preserve_index=False),
               "out/stl_weekly/weekly_post_sentiment.parquet",
               compression="zstd", use_dictionary=True, data_page_version="2.0")
print(sent.tail(5))


# i loaded and aligned week labels to monday to match STL artifacts
sent = pq.read_table("out/stl_weekly/weekly_post_sentiment.parquet").to_pandas()
sent["week_start"] = pd.to_datetime(sent["week_start"], utc=True)
sent["week_start"] = (sent["week_start"].dt.normalize()
                      - pd.to_timedelta(sent["week_start"].dt.weekday, unit="D"))

# i built a wide matrix and sentiment features
S = sent.pivot(index="week_start", columns="subreddit", values="sent_mean").sort_index()
sent_last = S.iloc[-1].rename("sent_last")
sent_ma4  = S.rolling(4, min_periods=2).mean().iloc[-1].rename("sent_ma4")
mu = S.rolling(13, center=True, min_periods=6).mean()
sd = S.rolling(13, center=True, min_periods=6).std(ddof=0)
sent_z13  = ((S - mu) / sd).iloc[-1].rename("sent_z13")

# merged everything into summary
summary = pd.read_csv("out/stl_weekly/summary_insights.csv").set_index("subreddit")
summary = summary.join([sent_last, sent_ma4, sent_z13], how="left").reset_index()
summary.to_csv("out/stl_weekly/summary_insights.csv", index=False)

print(summary[["subreddit","sent_last","sent_ma4","sent_z13"]])


last = pd.to_datetime(pd.read_csv("out/stl_weekly/summary_insights.csv")["last_week"][0], utc=True)
print("S last week:", S.index.max(), "| summary last_week:", last)
print(S.iloc[-1])
