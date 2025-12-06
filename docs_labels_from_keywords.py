from pathlib import Path
import re, yaml
import pandas as pd
import pyarrow.parquet as pq


def load_cfg(path: str = "config.yml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def build_concept_regexes(kw_df: pd.DataFrame) -> dict:
 
# i grouped patterns by concept to build one alternation regex per concept

    byc = (
        kw_df.dropna()
            .assign(pattern=lambda d: d["pattern"].astype(str).str.strip())
            .groupby("concept")["pattern"].apply(list)
    )
    rx = {}
    for concept, pats in byc.items():
        pats = [re.escape(p) for p in pats if p]
        if not pats:
            continue
        rx[concept] = re.compile(rf"\b(?:{'|'.join(pats)})\b", re.I)
    return rx


# mapping from low level concepts to high level constructs
CONSTRUCT_MAP = {
    "remote_work": ["remote", "wfh", "hybrid", "in_office"],
    "rto": ["rto"],
    "commute": ["commute"],
    "policy": ["policy"],
    "employer_org": ["employer_org"],
}


def make_flags(text: pd.Series,
               concept_rx: dict,
               construct_map: dict = CONSTRUCT_MAP) -> pd.DataFrame:

    flags = {}
    for construct, concepts in construct_map.items():
        m = pd.Series(False, index=text.index)
        for c in concepts:
            rx = concept_rx.get(c)
            if rx is None:
                continue
            m |= text.str.contains(rx, regex=True)
        flags[construct] = m.astype("int8")
    return pd.DataFrame(flags)


def main():
    cfg = load_cfg()
    paths = cfg.get("paths", {}) or {}

    docs_fp = Path(paths.get("docs_canonical", "out/tables/docs_canonical.parquet"))
    kw_fp   = Path(paths.get("keywords", "ref/keywords.csv"))
    out_fp  = Path("out/tables/docs_labels.parquet")

    print(f"[paths] docs_canonical = {docs_fp.resolve()}")
    print(f"[paths] keywords       = {kw_fp.resolve()}")
    out_fp.parent.mkdir(parents=True, exist_ok=True)

    # i loaded keyword dictionary and built concept regexes
    kw = pd.read_csv(kw_fp, dtype=str)
    concept_rx = build_concept_regexes(kw)
    print(f"[keywords] concepts: {sorted(concept_rx.keys())}")

    pf = pq.ParquetFile(docs_fp)
    print(f"[docs] row groups: {pf.num_row_groups}")

    out_parts = []

    for rg in range(pf.num_row_groups):
        batch = pf.read_row_group(
            rg,
            columns=["subreddit", "source_type", "title", "selftext", "body", "created_dt"],
        )
        df = batch.to_pandas()

        full_text = (
            df["title"].fillna("")
            + "\n"
            + df["selftext"].fillna("")
            + "\n"
            + df["body"].fillna("")
        ).astype(str)

        labels = make_flags(full_text, concept_rx)

        out_chunk = pd.concat(
            [
                df[["subreddit", "source_type", "created_dt"]].reset_index(drop=True),
                labels.reset_index(drop=True),
            ],
            axis=1,
        )
        out_parts.append(out_chunk)

        print(f"processed row-group {rg+1}/{pf.num_row_groups} "
              f"(rows={len(df):,})")

    out = pd.concat(out_parts, ignore_index=True)
    out.to_parquet(out_fp, index=False)
    print(f"wrote {out_fp.resolve()} with {len(out):,} rows")


if __name__ == "__main__":
    main()

with open("config.yml", "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f) or {}

paths = cfg.get("paths", {}) or {}
LABELS_FP = Path("out/tables/docs_labels.parquet")
print("labels path:", LABELS_FP.resolve(), "exists:", LABELS_FP.exists())

df = pd.read_parquet(LABELS_FP)

# this bit here parses time and normalizes subreddits
df["created_dt"] = pd.to_datetime(df["created_dt"], utc=True, errors="coerce")
df = df.dropna(subset=["created_dt"])

df["subreddit_norm"] = (
    df["subreddit"].astype(str)
      .str.lower()
      .str.replace(r"^\s*r/","", regex=True)
      .str.strip()
)

df = df[df["subreddit_norm"].isin(["antiwork","remotework","workfromhome"])].copy()

df["week_start"] = df["created_dt"].dt.to_period("W-MON").dt.start_time

constructs = ["remote_work", "rto", "commute", "policy", "employer_org"]

group_cols = ["subreddit_norm", "week_start"]

agg = (
    df.groupby(group_cols)[constructs]
      .sum()
      .reset_index()
)

N_docs = (
    df.groupby(group_cols)
      .size()
      .rename("N_docs")
      .reset_index()
)

weekly_constructs = agg.merge(N_docs, on=group_cols, how="left")

for c in constructs:
    weekly_constructs[f"{c}_share"] = weekly_constructs[c] / weekly_constructs["N_docs"]

OUT_FP = Path("out/tables/weekly_constructs.parquet")
weekly_constructs.to_parquet(OUT_FP, index=False)
print("wrote", OUT_FP.resolve(), "rows:", len(weekly_constructs))
print(weekly_constructs.head())


fp = Path("out/tables/weekly_constructs.parquet")
print("reading:", fp.resolve())

df = pd.read_parquet(fp)

csv_fp = fp.with_suffix(".csv")
df.to_csv(csv_fp, index=False)

print("wrote:", csv_fp.resolve(), "rows:", len(df))
print(df.head())

ts = pd.read_parquet("out/tables/concept_timeseries_norm_z.parquet")
remote_concepts = ["remote", "wfh", "hybrid", "in_office"]

remote_ts = (ts[ts["concept"].isin(remote_concepts)]
             .groupby(["subreddit","month"], as_index=False)
             .agg(remote_cnt=("cnt","sum"),
                  remote_rate_per_1k=("rate_per_1k","sum"),
                  remote_z=("z_rate","mean")))

print(remote_ts.head())
