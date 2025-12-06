from pathlib import Path
import re, yaml
import pandas as pd
import pyarrow.dataset as ds
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
from statsmodels.nonparametric.smoothers_lowess import lowess


def load_cfg(p="config.yml"):
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def build_concept_regexes(kw_df: pd.DataFrame) -> dict:
    byc = (kw_df.dropna()
                  .assign(pattern=lambda d: d["pattern"].astype(str).str.strip())
                  .groupby("concept")["pattern"].apply(list))

    rx = {}
    for concept, pats in byc.items():
        pats = [re.escape(p) for p in pats if p]
        if not pats:
            continue
        rx[concept] = re.compile(rf"\b(?:{'|'.join(pats)})\b", flags=re.I)
    return rx

def main():
    cfg = load_cfg("config.yml")
    kw_path = cfg["paths"]["keywords"]
    kw_df   = pd.read_csv(kw_path, dtype=str)

    # i built regex per the concept labels that i already defined.
    concept_rx = build_concept_regexes(kw_df)
    concepts   = list(concept_rx.keys())

    SRC = "out/tables/docs_canonical_ds"
    d   = ds.dataset(SRC, format="parquet")

    an = SentimentIntensityAnalyzer()

    frames = []
    for b in d.to_batches(columns=["subreddit", "source_type",
                                   "title", "selftext", "body",
                                   "created_dt"],
                          batch_size=500_000):
        df = b.to_pandas()

        # i wanted to keep posts and comments to see if i want to adjust later if i want posts only at some point
        df = df[df["created_dt"].notna()].copy()
        df["created_dt"] = pd.to_datetime(df["created_dt"], utc=True)
        df["week_start"] = df["created_dt"].dt.to_period("W").dt.start_time

        text = (
            df["title"].fillna("").astype(str) + " " +
            df["selftext"].fillna("").astype(str) + " " +
            df["body"].fillna("").astype(str)
        ).str.replace(r"\s+", " ", regex=True).str.strip()

        df["vader_compound"] = [an.polarity_scores(t)["compound"] for t in text]

        # this catches concept hits based on word list labels i defined earlier
        for c, rx in concept_rx.items():
            df[c] = text.str.contains(rx, regex=True)

        long = df.melt(
            id_vars=["subreddit", "week_start", "vader_compound"],
            value_vars=concepts,
            var_name="concept",
            value_name="flag",
        )
        long = long[long["flag"]].drop(columns="flag")

        frames.append(long)

    if not frames:
        print("no concept hits found – check keywords/paths")
        return

    hits = pd.concat(frames, ignore_index=True)

    agg = (
        hits.groupby(["subreddit", "concept", "week_start"])
            .agg(
                n_posts   = ("vader_compound", "size"),
                sent_mean = ("vader_compound", "mean"),
                pos_share = ("vader_compound",
                             lambda s: (s >=  0.05).mean()),
                neg_share = ("vader_compound",
                             lambda s: (s <= -0.05).mean()),
            )
            .reset_index()
            .sort_values(["subreddit", "concept", "week_start"])
    )

    agg["sent_mean_z"] = (
        agg.groupby(["subreddit", "concept"])["sent_mean"]
           .transform(lambda s: (s - s.mean()) / s.std(ddof=0))
    )

    out_dir = Path("out/analysis")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_fp = out_dir / "rq1_concept_sentiment_timeseries.csv"
    agg.to_csv(out_fp, index=False, float_format="%.4f")

    print(f"wrote {out_fp} | rows={len(agg)}")
    print(agg.head())

if __name__ == "__main__":
    main()

############
############
#PLOTS
############
############
CSV_PATH = Path("out/analysis/rq1_concept_sentiment_timeseries.csv")
VALUE_COL = "sent_mean_z"
MIN_NONMISSING_POINTS = 5

def main():
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Could not find {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)

    df["week_start"] = pd.to_datetime(df["week_start"])

    figs_dir = CSV_PATH.parent / "figs_concept_sentiment"
    figs_dir.mkdir(parents=True, exist_ok=True)

    for subreddit, g_sr in df.groupby("subreddit"):
        g_sr = g_sr.sort_values("week_start")

        ts = (
            g_sr.pivot(index="week_start",
                       columns="concept",
                       values=VALUE_COL)
               .sort_index()
        )

        if ts.empty:
            continue

        plt.figure(figsize=(11, 6))

        for concept in ts.columns:
            series = ts[concept]
            if series.notna().sum() < MIN_NONMISSING_POINTS:
                continue 

            plt.plot(series.index, series.values, label=concept)

        plt.axhline(0.0, linestyle="--", linewidth=0.8)

        if VALUE_COL == "sent_mean_z":
            ylabel = "VADER sentiment (z-score within concept)"
        else:
            ylabel = "VADER sentiment (−1 to +1)"

        plt.title(f"{subreddit} — concept-level sentiment over time")
        plt.xlabel("Week start")
        plt.ylabel(ylabel)
        plt.grid(alpha=0.3)
        plt.legend(loc="upper right", ncol=2, fontsize=8)
        plt.tight_layout()

        out_png = figs_dir / f"concept_sentiment_timeseries_{subreddit}_{VALUE_COL}.png"
        plt.savefig(out_png, dpi=150)
        plt.close()

        print(f"Wrote {out_png}")

if __name__ == "__main__":
    main()




CSV_PATH = Path("out/analysis/rq1_concept_sentiment_timeseries.csv")
SUBSET_CONCEPTS = ["remote", "wfh", "hybrid", "rto", "commute"]
MIN_N_POSTS = 10
LOESS_FRAC = 0.25

def main():
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Could not find {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)
    df["week_start"] = pd.to_datetime(df["week_start"])

    df = df[df["concept"].isin(SUBSET_CONCEPTS)]

    figs_dir = CSV_PATH.parent / "figs_concept_sentiment_loess"
    figs_dir.mkdir(parents=True, exist_ok=True)

    for subreddit, g_sr in df.groupby("subreddit"):
        g_sr = g_sr.sort_values(["concept", "week_start"])

        plt.figure(figsize=(11, 6))

        for concept, g_con in g_sr.groupby("concept"):

            g_con = g_con[g_con["n_posts"] >= MIN_N_POSTS].copy()
            if len(g_con) < 10:
                continue 

            x = g_con["week_start"].map(pd.Timestamp.toordinal).to_numpy()
            y = g_con["sent_mean"].to_numpy()

            y_smooth = lowess(
                endog=y,
                exog=x,
                frac=LOESS_FRAC,
                it=0,
                return_sorted=False,
            )

            plt.plot(g_con["week_start"], y_smooth, label=concept)

        plt.axhline(0.0, linestyle="--", linewidth=0.8)

        plt.title(f"{subreddit} — LOESS-smoothed concept sentiment")
        plt.xlabel("Week start")
        plt.ylabel("Mean VADER sentiment (LOESS)")
        plt.grid(alpha=0.3)
        plt.legend(loc="upper right", ncol=2, fontsize=8)
        plt.tight_layout()

        out_png = figs_dir / f"concept_sentiment_loess_{subreddit}.png"
        plt.savefig(out_png, dpi=150)
        plt.close()

        print(f"Wrote {out_png}")

if __name__ == "__main__":
    main()

csv_path = Path("out/analysis/rq1_concept_sentiment_timeseries.csv")
df = pd.read_csv(csv_path)
df["week_start"] = pd.to_datetime(df["week_start"])

bins = [
    pd.Timestamp("2019-01-01"),
    pd.Timestamp("2021-12-31"),
    pd.Timestamp("2023-12-31"),
    pd.Timestamp("2025-12-31"),
]
labels = ["2019–2021", "2022–2023", "2024–2025"]
df["period"] = pd.cut(df["week_start"], bins=bins, labels=labels, include_lowest=True)

agg = (
    df.dropna(subset=["period"])
      .groupby(["subreddit", "concept", "period"])
      .apply(lambda g: pd.Series({
          "N_posts": g["n_posts"].sum(),
          "mean_sent": (g["sent_mean"] * g["n_posts"]).sum() / g["n_posts"].sum()
      }))
      .reset_index()
)

main_concepts = ["remote", "wfh", "hybrid", "rto", "commute"]
for sr in ["antiwork", "remotework", "workfromhome"]:
    sub = (agg[(agg["subreddit"] == sr) &
               (agg["concept"].isin(main_concepts))]
           .pivot(index="concept", columns="period", values="mean_sent")
           .round(3))
    print(f"\n--- {sr} ---")
    print(sub)

csv_path = Path("out/analysis/rq1_concept_sentiment_timeseries.csv")

df = pd.read_csv(csv_path)
df["week_start"] = pd.to_datetime(df["week_start"])

agg_overall = (
    df.groupby(["subreddit", "concept"])
      .apply(lambda g: pd.Series({
          "N_posts_total": g["n_posts"].sum(),
          "mean_sent": (g["sent_mean"] * g["n_posts"]).sum() / g["n_posts"].sum(),
          "mean_pos_share": (g["pos_share"] * g["n_posts"]).sum() / g["n_posts"].sum(),
          "mean_neg_share": (g["neg_share"] * g["n_posts"]).sum() / g["n_posts"].sum(),
      }))
      .reset_index()
)

out_dir = csv_path.parent
table3_path = out_dir / "table3_concept_sentiment_overall.xlsx"
agg_overall.to_excel(table3_path, sheet_name="overall_concepts", index=False)
print(f"Wrote Table 3 workbook to {table3_path}")

bins = [
    pd.Timestamp("2019-01-01"),
    pd.Timestamp("2021-12-31"),
    pd.Timestamp("2023-12-31"),
    pd.Timestamp("2025-12-31"),
]
labels = ["2019–2021", "2022–2023", "2024–2025"]
df["period"] = pd.cut(df["week_start"], bins=bins, labels=labels, include_lowest=True)

agg_period = (
    df.dropna(subset=["period"])
      .groupby(["subreddit", "concept", "period"])
      .apply(lambda g: pd.Series({
          "N_posts": g["n_posts"].sum(),
          "mean_sent": (g["sent_mean"] * g["n_posts"]).sum() / g["n_posts"].sum(),
      }))
      .reset_index()
)

main_concepts = ["remote", "wfh", "hybrid", "in_office",
                 "rto", "commute", "policy", "employer_org"]

subreddits = ["antiwork", "remotework", "workfromhome"]

period_xlsx = out_dir / "concept_sentiment_by_period.xlsx"
with pd.ExcelWriter(period_xlsx) as writer:
    for sr in subreddits:
        sub = agg_period[
            (agg_period["subreddit"] == sr)
            & (agg_period["concept"].isin(main_concepts))
        ]
        table = (
            sub.pivot(index="concept", columns="period", values="mean_sent")
               .reindex(index=main_concepts, columns=labels)
               .round(3)
        )
        table.to_excel(writer, sheet_name=sr)

print(f"Wrote period tables workbook to {period_xlsx}")
