import yaml
from pathlib import Path
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import acf

print("CWD:", Path.cwd())


CFG = {}
cfg_path = Path("config.yml")
if cfg_path.exists():
    with cfg_path.open("r", encoding="utf-8") as f:
        CFG = yaml.safe_load(f) or {}

paths   = (CFG.get("paths") or {})
analysis = (CFG.get("analysis") or {}).get("stl_weekly", {})
params   = analysis.get("params", {}) or {}

DOCS_FP      = Path(paths.get("docs_canonical", "out/tables/docs_canonical.parquet"))
OUT_DIR      = Path(paths.get("stl_weekly_figs_dir", "out/stl_weekly"))
SUMMARY_FP   = Path(paths.get("stl_weekly_summary", "out/stl_weekly/stl_weekly_summary.csv"))
COMPONENTS_FP= Path(paths.get("stl_weekly_components", "out/stl_weekly/stl_weekly_components.parquet"))

OUT_DIR.mkdir(parents=True, exist_ok=True)

SUFFIX   = str(params.get("suffix", "")) 
PERIOD   = int(params.get("period", 52))  
SEASONAL = int(params.get("seasonal", 13))
TREND    = int(params.get("trend", 53))
ROBUST   = bool(params.get("robust", True))
MIN_POINTS = max(PERIOD + SEASONAL + 5, 30)

print(f"[paths] DOCS_FP={DOCS_FP.resolve()}")
print(f"[paths] OUT_DIR={OUT_DIR.resolve()}")

if not DOCS_FP.exists():
    raise FileNotFoundError(f"docs_canonical not found at {DOCS_FP.resolve()}")

def stl_decompose(series, period=PERIOD, seasonal=SEASONAL, trend=TREND, robust=ROBUST):
    return STL(series, period=period, robust=robust,
               seasonal=seasonal, trend=trend).fit()

def strength(trend, seasonal, resid):
    var = np.nanvar
    Ft = max(0.0, 1.0 - var(resid) / max(var(trend + resid), 1e-12))
    Fs = max(0.0, 1.0 - var(resid) / max(var(seasonal + resid), 1e-12))
    return Ft, Fs

def quick_acf(x, nlags=60):
    x = pd.Series(x).dropna()
    if len(x) < 10:
        return {}
    vals = acf(x, nlags=min(nlags, len(x) - 1), fft=True)
    out = {}
    for L in (26, 52):
        if len(vals) > L:
            out[f"acf_lag{L}"] = float(vals[L])
    return out

def plot_stl(ts, res, title, out_png: Path):
    fig, axes = plt.subplots(4, 1, figsize=(11, 7), sharex=True)
    axes[0].plot(ts.index, ts.values)
    axes[0].set_title(f"{title} — observed (weekly)")
    axes[1].plot(ts.index, res.trend)
    axes[1].set_title("trend")
    axes[2].plot(ts.index, res.seasonal)
    axes[2].set_title(f"seasonal (≈{PERIOD}w)")
    axes[3].plot(ts.index, res.resid)
    axes[3].set_title("residual")
    for ax in axes:
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"wrote {out_png}")

docs = pd.read_parquet(DOCS_FP, columns=["subreddit", "created_dt"])
print(f"[docs] rows={len(docs):,}  cols={list(docs.columns)}")

docs["created_dt"] = pd.to_datetime(docs["created_dt"], utc=True, errors="coerce")
docs = docs.dropna(subset=["created_dt"])

docs["subreddit"] = (
    docs["subreddit"].astype(str)
        .str.lower()
        .str.replace(r"^\s*r/","", regex=True)
        .str.strip()
)

docs["week_start"] = docs["created_dt"].dt.to_period("W-MON").dt.start_time

weekly_vol = (
    docs.groupby(["subreddit", "week_start"])
        .size()
        .rename("N_docs_week")
        .reset_index()
)

print(f"[weekly_vol] rows={len(weekly_vol):,}; subs={weekly_vol['subreddit'].nunique()}")
if weekly_vol.empty:
    raise RuntimeError("weekly_vol is empty after grouping. Check docs_canonical content.")

global_wk = weekly_vol["week_start"].sort_values().unique()

print(f"[span] {global_wk.min().date()} → {global_wk.max().date()} (weeks={len(global_wk)})")

summary_rows = []
components_rows = []
skipped = []

for sr in sorted(weekly_vol["subreddit"].unique()):
    s = (
        weekly_vol[weekly_vol["subreddit"] == sr]
        .set_index("week_start")["N_docs_week"]
        .reindex(global_wk)
        .fillna(0.0)
    )
    print(f"{sr}: min={s.min()}, max={s.max()}, first10={s.head(10).tolist()}")

    if s.shape[0] < MIN_POINTS:
        skipped.append((sr, f"too_short:{s.shape[0]}<{MIN_POINTS}"))
        continue

    try:
        res = stl_decompose(s + 1e-9)
    except Exception as e:
        skipped.append((sr, f"stl_fail:{e.__class__.__name__}"))
        continue

    Ft, Fs = strength(res.trend, res.seasonal, res.resid)
    acf_hint = quick_acf(res.resid)

    plot_stl(s, res, f"r/{sr} — total volume", OUT_DIR / f"stl_weekly_volume_{sr}{SUFFIX}.png")

    components_rows.append(pd.DataFrame({
        "subreddit": sr,
        "series": "weekly_volume",
        "week_start": s.index,
        "observed": s.values,
        "trend": res.trend,
        "seasonal": res.seasonal,
        "resid": res.resid,
    }))

    summary_rows.append({
        "subreddit": sr,
        "series": "weekly_volume",
        "weeks_expected": len(global_wk),
        "weeks_observed": int((s > 0).sum()),
        "coverage_rate": round(float((s > 0).sum()) / len(global_wk), 3),
        "Ft_trend_strength": round(Ft, 3),
        "Fs_season_strength": round(Fs, 3),
        **{f"resid_{k}": round(v, 3) for k, v in acf_hint.items()},
    })

print(f"[loop] components for {len(components_rows)} subreddits; skipped={len(skipped)} {skipped[:5]}")

if summary_rows:
    summary = pd.DataFrame(summary_rows)
    SUMMARY_FP.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(SUMMARY_FP, index=False)
    print(f"summary -> {SUMMARY_FP.resolve()} (rows={len(summary)})")
else:
    print("all series too short or failed")

if components_rows:
    comps = pd.concat(components_rows, ignore_index=True)
    COMPONENTS_FP.parent.mkdir(parents=True, exist_ok=True)
    comps.to_parquet(COMPONENTS_FP, index=False)
    print(f"components -> {COMPONENTS_FP.resolve()} (rows={len(comps)})")
else:
    print("nothing to concatenate")

for col in ["observed", "trend", "seasonal", "resid"]:
    out_csv = COMPONENTS_FP.with_name(f"{COMPONENTS_FP.stem}_{col}.csv")
    (
        comps[["subreddit", "week_start", col]]
        .rename(columns={col: "value"})
        .to_csv(out_csv, index=False)
    )
    print(f"{col} CSV -> {out_csv.resolve()}")

    for col in ["observed", "trend", "seasonal", "resid"]:
        zcol = col + "_z"

        comps[zcol] = comps.groupby("subreddit")[col].transform(
            lambda x: (x - x.mean()) / x.std(ddof=0)
        )

        out_csv = COMPONENTS_FP.with_name(f"{COMPONENTS_FP.stem}_{col}_z.csv")
        (
            comps[["subreddit", "week_start", zcol]]
            .rename(columns={zcol: "z"})
            .to_csv(out_csv, index=False)
        )
        print(f"{col} z-score CSV -> {out_csv.resolve()}")

        for sr, g in comps.groupby("subreddit"):
            fig, ax = plt.subplots(figsize=(11, 4))
            ax.plot(g["week_start"], g[zcol])
            ax.axhline(0, linewidth=0.8)
            ax.set_title(f"r/{sr} — {col} (z-score)")
            ax.grid(True, alpha=0.3)
            fig.autofmt_xdate()
            fig.tight_layout()
            png_path = OUT_DIR / f"stl_weekly_{col}_z_{sr}{SUFFIX}.png"
            fig.savefig(png_path, dpi=150)
            plt.close(fig)
            print(f"{col} z-score PNG -> {png_path.resolve()}")
