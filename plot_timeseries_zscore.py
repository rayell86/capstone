from pathlib import Path
import re
import pandas as pd
import matplotlib.pyplot as plt

TSZ_FP  = Path("out/tables/concept_timeseries_norm_z.parquet")
FIG_DIR = Path("out/z_figs"); FIG_DIR.mkdir(parents=True, exist_ok=True)

# knobs i designed for timeseries analysis
TOP_K            = 6             # this lets me how many concepts  i wanna feature
MIN_DOCS         = 200           # this  ignores months with fewer docs than this
SMOOTH_METHOD    = "ewm"         # ewm or medmean
SMOOTH_SPAN      = 6             # edwa span in months
SMOOTH_WIN       = 5             # this calibrates the window for med or mean smoothing
CLIP_QUANTILES   = (0.01, 0.99)  # this cuts extreme z outliers before smoothing
DRAW_RAW_FAINT   = True          # this plots the raw z line faint behind smooth

def _safe(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", s.strip().lower())

def _smooth(series: pd.Series, method="ewm", span=6, win=5) -> pd.Series:
    if method == "ewm":
        # exponential moving average
        return series.ewm(span=span, adjust=False, min_periods=max(1, span // 2)).mean()
    elif method == "medmean":
        # robust: rolling median to rolling mean
        med = series.rolling(win, center=True, min_periods=max(1, win // 2)).median()
        return med.rolling(win, center=True, min_periods=1).mean()
    return series

def _top_concepts(tsz: pd.DataFrame, k=6) -> list[str]:
    # this rank by total counts to focus on high signal concepts
    return (tsz.groupby("concept")["cnt"].sum()
              .sort_values(ascending=False)
              .head(k).index.tolist())

def _prep_view(tsz: pd.DataFrame, source_type: str | None) -> pd.DataFrame:
    view = tsz if source_type is None else tsz[tsz["source_type"] == source_type]
    # this here makes month tz-naive for matplotlib
    view = view.copy()
    view["month"] = pd.to_datetime(view["month"]).dt.tz_localize(None)
    return view

def plot_concept(tsz: pd.DataFrame, concept: str, source_type: str | None, out_name: str):
    view = _prep_view(tsz, source_type)
    df = view[view["concept"] == concept]

    plt.figure()
    for sr, g in df.groupby("subreddit"):
        g = g.sort_values("month").copy()

        g = g[g["N_docs"] >= MIN_DOCS]
        if g.empty:
            continue

        lo, hi = g["z_rate"].quantile(list(CLIP_QUANTILES))
        g["z_clip"] = g["z_rate"].clip(lo, hi)

        # i apply smoothing here
        g["z_smooth"] = _smooth(
            g["z_clip"],
            method=SMOOTH_METHOD,
            span=SMOOTH_SPAN,
            win=SMOOTH_WIN,
        )

        if DRAW_RAW_FAINT:
            plt.plot(g["month"], g["z_rate"], alpha=0.25)
        plt.plot(g["month"], g["z_smooth"], label=sr, linewidth=2)

    st_lab = f" • {source_type}" if source_type else ""
    plt.title(f"{concept} (normalized){st_lab}")
    plt.xlabel("Month")
    plt.ylabel("z-score")
    plt.axhline(0, color="k", lw=1)
    plt.axhline(2, color="k", lw=1, ls="--")
    plt.legend()
    plt.tight_layout()
    out = FIG_DIR / out_name
    plt.savefig(out, dpi=150); plt.close()
    print(f"wrote {out}")

def plot_grid(tsz: pd.DataFrame, source_type: str | None, out_name: str, k=6):
    view = _prep_view(tsz, source_type)
    concepts = _top_concepts(view, k=k)
    n = len(concepts)
    rows, cols = 3, 2

    fig, axes = plt.subplots(rows, cols, figsize=(12, 10), sharex=True)
    axes = axes.flatten()

    for ax, concept in zip(axes, concepts):
        df = view[view["concept"] == concept]
        for sr, g in df.groupby("subreddit"):
            g = g.sort_values("month").copy()
            g = g[g["N_docs"] >= MIN_DOCS]
            if g.empty:
                continue
            lo, hi = g["z_rate"].quantile(list(CLIP_QUANTILES))
            z = g["z_rate"].clip(lo, hi)
            z = _smooth(z, method=SMOOTH_METHOD, span=SMOOTH_SPAN, win=SMOOTH_WIN)
            ax.plot(g["month"], z, label=sr, linewidth=2)
        ax.set_title(concept)
        ax.axhline(0, color="k", lw=1)
        ax.axhline(2, color="k", lw=1, ls="--")
        ax.set_ylabel("z-score")

    axes[min(n, rows*cols)-1].legend(loc="best")
    for ax in axes[n:]:
        ax.axis("off")

    st_lab = "all" if source_type is None else source_type
    fig.suptitle(f"Top {min(k, n)} concepts — {st_lab}", y=0.98)
    plt.tight_layout()
    out = FIG_DIR / out_name
    plt.savefig(out, dpi=150); plt.close()
    print(f"wrote {out}")

def main():
    tsz = pd.read_parquet(TSZ_FP)

    concepts = _top_concepts(tsz, k=TOP_K)

    for c in concepts:
        plot_concept(tsz, c, None,            f"{_safe(c)}_z_all.png")
        plot_concept(tsz, c, "comment",       f"{_safe(c)}_z_comments.png")
        plot_concept(tsz, c, "post",          f"{_safe(c)}_z_posts.png")

    # optional grids
    plot_grid(tsz, None,      "top6_grid_z_all.png",      k=TOP_K)
    plot_grid(tsz, "comment", "top6_grid_z_comments.png", k=TOP_K)
    plot_grid(tsz, "post",    "top6_grid_z_posts.png",    k=TOP_K)

if __name__ == "__main__":
    main()
