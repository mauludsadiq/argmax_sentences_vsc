import argparse
import json
import re
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

def tokenize(s: str):
    return re.findall(r"[a-z]+", s.lower())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", required=True)
    ap.add_argument("--window", type=int, default=20)
    ap.add_argument("--min_pairs", type=int, default=50)
    ap.add_argument("--out", default="artifacts/learned_Wb.npz")
    args = ap.parse_args()

    VOCAB = [
        "big", "cold", "city", "busy",
        "pizza", "subway", "finance", "art",
        "ai", "startup", "election", "senate",
    ]
    KW = [
        "new", "york", "paris", "wall", "street", "senate", "election",
        "ai", "startup", "pizza", "subway", "art", "cold", "winter",
    ]

    tok2y = {t:i for i,t in enumerate(VOCAB)}
    kw2x = {t:i for i,t in enumerate(KW)}

    raw = Path(args.text).read_text(encoding="utf-8", errors="ignore")
    toks = tokenize(raw)

    X = []
    y = []

    for i in range(len(toks) - 1):
        nxt = toks[i + 1]
        if nxt not in tok2y:
            continue

        start = max(0, i - args.window + 1)
        ctx = toks[start:i + 1]

        x = np.zeros(len(KW), dtype=float)
        for t in ctx:
            if t in kw2x:
                x[kw2x[t]] += 1.0

        X.append(x)
        y.append(tok2y[nxt])

    X = np.stack(X, axis=0) if X else np.zeros((0, len(KW)), dtype=float)
    y = np.array(y, dtype=int)

    if X.shape[0] < args.min_pairs:
        raise SystemExit(f"not enough (context,next) pairs with next in VOCAB: got {X.shape[0]}")

    clf = LogisticRegression(
        multi_class="multinomial",
        solver="lbfgs",
        max_iter=500,
        n_jobs=1,
    )
    clf.fit(X, y)

    W = clf.coef_.astype(float)
    b = clf.intercept_.astype(float)

    Path("artifacts").mkdir(parents=True, exist_ok=True)
    np.savez(args.out, W=W, b=b, vocab=np.array(VOCAB, dtype=object), kw=np.array(KW, dtype=object))

    report = {
        "schema": "argmax-sentences/train-report/v0.1",
        "text_path": args.text,
        "window": args.window,
        "num_pairs": int(X.shape[0]),
        "classes": VOCAB,
        "features": KW,
        "out_npz": args.out,
    }
    Path("artifacts/train_report.json").write_text(json.dumps(report, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))

if __name__ == "__main__":
    main()
