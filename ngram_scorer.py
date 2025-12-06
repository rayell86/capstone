import json, re, collections, math, sys, traceback
from pathlib import Path
import csv

MERGED = Path(r"C:\UZ\Capstone\text reference\merged_text\r_all_merged_posts.jsonl")
OUTCSV = Path(r"C:\UZ\Capstone\text reference\merged_text\r_top_ngrams_pmi.csv")

TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?")

def toks(s: str):
    return [t for t in TOKEN_RE.findall(s.lower()) if not t.isdigit()]

def main():
    if not MERGED.exists():
        print(f"file not found {MERGED}")
        sys.exit(1)

    uni = collections.Counter()
    bi  = collections.Counter()
    tri = collections.Counter()
    docs = 0

    with open(MERGED, "r", encoding="utf-8", errors="ignore") as f:
        for lineno, line in enumerate(f, 1):
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if not isinstance(obj, dict):
                continue
            text = obj.get("body")
            if not text:
                text = (obj.get("title", "") + " " + obj.get("selftext", "")).strip()
            if not text or text.strip().lower() in ("[deleted]", "[removed]"):
                continue
            ts = toks(text)
            if not ts:
                continue
            docs += 1
            uni.update(ts)
            if len(ts) >= 2:
                bi.update(" ".join(ts[i:i+2]) for i in range(len(ts)-1))
            if len(ts) >= 3:
                tri.update(" ".join(ts[i:i+3]) for i in range(len(ts)-2))

            if docs % 100000 == 0:
                print(f"… {docs:,} docs processed")
    if not uni:
        print("no tokens collected")
        sys.exit(1)

    N_uni = sum(uni.values())

    def pmi(count_ng, parts):
        if count_ng < 5:
            return None
        p_ng = count_ng / max(1, N_uni)
        denom = 1.0
        for w in parts:
            uw = uni.get(w, 0)
            if uw == 0:
                return None
            denom *= (uw / N_uni)
        if denom == 0:
            return None
        return math.log2(p_ng / denom)

    def rank(counter, k):
        scored = []
        for ng, c in counter.items():
            parts = ng.split()
            try:
                s = pmi(c, parts)
            except Exception:
                continue
            if s is None:
                continue
            scored.append((ng, c, len(parts), s, c * max(s, 0)))
        scored.sort(key=lambda x: (x[4], x[1]), reverse=True)
        return scored[:k]
    top = rank(bi, 100) + rank(tri, 100)
  

    OUTCSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTCSV, "w", newline="", encoding="utf-8") as w:
        cw = csv.writer(w)
        cw.writerow(["ngram", "count", "n", "PMI", "Freq_x_PMI"])
        for ng, c, n, p, fp in top:
            cw.writerow([ng, c, n, round(p, 6), round(fp, 3)])

    print(f"docs: {docs:,}  Unigrams: {len(uni):,}  Bigrams: {len(bi):,}  Trigrams: {len(tri):,}")
    print(f"wrote → {OUTCSV}")

if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()

