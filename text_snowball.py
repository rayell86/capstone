import json, re, collections, csv
from pathlib import Path

INPUT  = r"C:\UZ\Capstone\text reference\merged_text\r_all_merged_posts.jsonl"
OUTPUT = r"C:\UZ\Capstone\text reference\subset_all_texts.jsonl"
SEEDS  = ["return to office", "RTO", "hybrid policy", "work from home", "remote work"]
BASE_DIR = ["C:\UZ\Capstone\text reference\merged_text"]
GLOB     = "r_all_merged_posts.jsonl"
PROGRESS_EVERY = 5000

SEEDS = [
    "return to office", "RTO", "back to office", "work from home", "work-from-home",
    "in-office", "onsite", "on-site", "remote", "telework", "hybrid",
    "hybrid policy", "hybrid schedule", "mandatory days", "commute time",
    "three days office", "go back office"
]

STOPWORDS = set("""
a an the and or but if then with without to from for of on in at as by up out over under
is are was were be been being do does did doing have has had having it its it's this that these those
i you he she we they them me my your our their his her not no yes
""".split())

TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?")

def filter_file(input_path: Path, output_path: Path, seeds, progress_every=PROGRESS_EVERY):
    pat = re.compile("|".join(re.escape(s) for s in seeds), re.IGNORECASE)
    matched = 0
    total = 0
    print(f"\n[Filter] Scanning: {input_path}")
    with open(input_path, "r", encoding="utf-8", errors="ignore") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:
        for i, line in enumerate(fin, 1):
            total += 1
            try:
                obj = json.loads(line)
            except Exception:
                continue
            body = obj.get("body", "")
            if not body or body.lower() in ("[deleted]", "[removed]"):
                continue
            if pat.search(body):
                fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
                matched += 1
            if i % progress_every == 0:
                print(f"  … {i:,} lines; {matched:,} matches")
    print(f"[Filter] Done. Lines: {total:,}; Matches: {matched:,}")
    print(f"[Filter] Output: {output_path}")

def toks(text):
    return [t for t in TOKEN_RE.findall(text.lower())
            if t not in STOPWORDS and len(t) > 1 and not t.isdigit()]

def make_ngrams(tokens, n):
    return [" ".join(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

def ngrams_for_subset(subset_path: Path, csv_out: Path, progress_every=PROGRESS_EVERY):
    bigrams = collections.Counter()
    trigrams = collections.Counter()
    rows = 0
    print(f"[Ngrams] Reading subset: {subset_path}")
    with open(subset_path, "r", encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f, 1):
            try:
                body = json.loads(line).get("body", "")
            except Exception:
                continue
            ts = toks(body)
            if not ts:
                continue
            bigrams.update(make_ngrams(ts, 2))
            trigrams.update(make_ngrams(ts, 3))
            rows += 1
            if i % progress_every == 0:
                print(f"  … {i:,} rows processed")
    # kept >=5 occurrences and dropped stopword ngrams
    def edge_ok(ng):
        parts = ng.split()
        return parts and parts[0] not in STOPWORDS and parts[-1] not in STOPWORDS
    top_bi = [(ng, c) for ng, c in bigrams.items() if c >= 5 and edge_ok(ng)]
    top_tri = [(ng, c) for ng, c in trigrams.items() if c >= 5 and edge_ok(ng)]
    top_bi.sort(key=lambda x: x[1], reverse=True)
    top_tri.sort(key=lambda x: x[1], reverse=True)

    with open(csv_out, "w", newline="", encoding="utf-8") as w:
        cw = csv.writer(w)
        cw.writerow(["ngram", "count", "n"])
        for ng, c in top_bi[:5000]:
            cw.writerow([ng, c, 2])
        for ng, c in top_tri[:5000]:
            cw.writerow([ng, c, 3])

    print(f"[Ngrams] Done. Wrote → {csv_out}")

def main():
    files = sorted(BASE_DIR.glob(GLOB))
    if not files:
        print(f"No files found matching {GLOB} in {BASE_DIR}")
        return
    print(f"Found {len(files)} file(s):")
    for p in files: print(" -", p.name)

    for path in files:
        year_hint = "".join(ch for ch in path.name if ch.isdigit())[:4] or "all"
        subset = BASE_DIR / f"subset_{year_hint}.jsonl"
        ngram_csv = BASE_DIR / f"top_ngrams_{year_hint}.csv"

        filter_file(path, subset, SEEDS, PROGRESS_EVERY)
        ngrams_for_subset(subset, ngram_csv, PROGRESS_EVERY)

if __name__ == "__main__":
    print("snowballing across all dumps")
    main()
