import argparse, json, re, hashlib
from pathlib import Path

import numpy as np

RX = re.compile(r"[A-Za-z0-9']+|[.!?]")

def tokenize(text: str):
    return [t.lower() for t in RX.findall(text)]

def hmod(s: str, mod: int) -> int:
    d = hashlib.sha256(s.encode("utf-8")).digest()
    return int.from_bytes(d[:8], "big") % mod

def load_contexts(path: str):
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(obj, dict) and "contexts" in obj and isinstance(obj["contexts"], list):
        return [str(x) for x in obj["contexts"]]
    raise SystemExit('contexts.json must be {"contexts":[...]}')

def softmax(z, temp=1.0):
    z = np.asarray(z, dtype=float)
    t = float(temp) if float(temp) > 0 else 1e-9
    u = z / t
    u = u - np.max(u)
    ex = np.exp(u)
    return ex / np.sum(ex)

def entropy(p):
    p = np.asarray(p, dtype=float)
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())

def sample_from_probs(p, rng):
    p = np.asarray(p, dtype=float)
    c = np.cumsum(p)
    r = rng.random()
    return int(np.searchsorted(c, r, side='right'))

def topk_indices(z, k: int):
    z = np.asarray(z)
    n = int(z.size)
    k = int(k)
    if k <= 0 or n == 0:
        return np.array([], dtype=int)
    if k >= n:
        idx = np.argsort(-z)
        return idx
    idx = np.argpartition(-z, k - 1)[:k]
    idx = idx[np.argsort(-z[idx])]
    return idx

def top2_margin(z) -> float:
    z = np.asarray(z)
    if z.size <= 1:
        return float('inf')
    t2 = np.partition(z, -2)[-2:]
    return float(t2.max() - t2.min())

def violates_a_an(prev_tok: str, cand_tok: str) -> bool:
    vowels = set("aeiou")
    if prev_tok not in ("a", "an"):
        return False
    if cand_tok in ("<EOS>", ".", "!", "?"):
        return True
    if not cand_tok:
        return True
    ch = cand_tok[0].lower()
    if not ch.isalpha():
        return True
    if prev_tok == "an":
        return ch not in vowels
    return ch in vowels

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--contexts", default="contexts.json")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--stop_punct", type=int, default=1)
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--min_steps", type=int, default=0)
    ap.add_argument("--ban", type=str, default="")
    ap.add_argument("--no_repeat", type=int, default=0)
    ap.add_argument("--no_repeat_window", type=int, default=0)
    ap.add_argument("--mode", type=str, default="greedy", choices=["greedy","sample"])
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--trace_out", type=str, default="artifacts/trace.jsonl")
    ap.add_argument("--csv_out", type=str, default="artifacts/curve.csv")
    ap.add_argument("--det_a_an", type=int, default=0)
    ap.add_argument("--topic_lock", type=int, default=0)
    ap.add_argument("--topic_keep", type=str, default="is,a,an,of,in,and,has,are,the")
    ap.add_argument("--topic_topn", type=int, default=24)
    ap.add_argument("--no_dangling", type=int, default=1)

    args = ap.parse_args()

    Path('artifacts').mkdir(parents=True, exist_ok=True)
    trace_path = Path(getattr(args, 'trace_out', 'artifacts/trace.jsonl'))
    csv_path = Path(getattr(args, 'csv_out', 'artifacts/curve.csv'))
    trace_f = trace_path.open('w', encoding='utf-8')
    csv_f = csv_path.open('w', encoding='utf-8')
    csv_f.write('context_start,step,picked,mode,temp,margin,p_top1,entropy\n')

    data = np.load(args.weights, allow_pickle=False)
    W = data["W"]                    # (C, D)
    b = data["b"]                    # (C,)
    classes = data["classes"]        # (C,)
    in_vocab = data["in_vocab"].tolist()
    out_vocab = data["out_vocab"].tolist()
    classes_list = [int(x) for x in classes.tolist()]
    tok_to_j = {}
    for j, y_idx in enumerate(classes_list):
        tok_to_j[out_vocab[y_idx]] = int(j)

    d2 = int(data["d2"])
    d3 = int(data["d3"])

    BOS, UNK = "<BOS>", "<UNK>"
    in_id = {t: i for i, t in enumerate(in_vocab)}
    D1 = len(in_vocab)

    def norm_in(t):
        return t if t in in_id else UNK

    def mask_tok(zvec, tok):
        jj = tok_to_j.get(tok)
        if jj is not None:
            zvec[int(jj)] = -1e30

    stop_set = {t for t in (".", "!", "?") if t in tok_to_j}
    rng = np.random.default_rng(int(args.seed))

    eff_no_repeat_window = int(args.no_repeat_window) if int(args.no_repeat_window) > 0 else (1 if int(args.no_repeat) else 0)

    contexts = load_contexts(args.contexts)

    for ctx0 in contexts:
        print("CONTEXT_START:", ctx0)
        toks0 = [norm_in(t) for t in tokenize(ctx0)]
        state = [BOS, BOS, BOS] + toks0
        out = []

        # topic control: keep = function words; content = prompt tokens excluding keep
        keep = {t.strip().lower() for t in getattr(args, 'topic_keep', '').split(',') if t.strip()}
        content = {t for t in toks0 if t not in keep and t not in (BOS, UNK)}
        allowed_base = set(keep) | set(content) | set(stop_set) | {"<EOS>"}

        allowed = None
        if int(getattr(args, "topic_lock", 0)):
            keep = {t.strip().lower() for t in getattr(args, "topic_keep", "").split(",") if t.strip()}
            allowed = set(toks0) | keep | set(stop_set) | {"<EOS>"}

        for step in range(args.steps):
            t1 = state[-1]
            t2 = state[-2]
            t3 = state[-3]

            c1 = in_id.get(t1, in_id[UNK])
            c2 = D1 + hmod(f"{t2}|{t1}", d2)
            c3 = D1 + d2 + hmod(f"{t3}|{t2}|{t1}", d3)

            z = b + W[:, c1] + W[:, c2] + W[:, c3]

            if step < int(args.min_steps):
                for tban in stop_set:
                    mask_tok(z, tban)
                mask_tok(z, "<EOS>")

            if args.ban:
                for raw in args.ban.split(","):
                    tban = raw.strip().lower()
                    if tban:
                        mask_tok(z, tban)

            if eff_no_repeat_window > 0:
                # Only ban repeats of *generated* tokens (don't ban prompt/context tokens)
                recent = out[-eff_no_repeat_window:] if eff_no_repeat_window > 0 else []
                for rt in recent:
                    mask_tok(z, rt)

            if int(args.det_a_an) and t1 in ("a", "an"):
                for cand_tok, jj in tok_to_j.items():
                    if violates_a_an(t1, cand_tok):
                        z[int(jj)] = -1e30

            if allowed is not None:
                for tok, jj in tok_to_j.items():
                    if tok not in allowed:
                        z[int(jj)] = -1e30

            # deterministic no-dangling: don't end after function words / prepositions
            if int(getattr(args, 'no_dangling', 1)):
                if t1 in ("a", "an", "the", "of", "in", "and", "is", "are", "has"):
                    for tban in stop_set:
                        mask_tok(z, tban)
                    mask_tok(z, "<EOS>")

            # band-pass topic lock: (keep+prompt-content+punct+EOS) U (top-N by z)
            if int(getattr(args, 'topic_lock', 0)):
                allowed = set(allowed_base)
                topn = int(getattr(args, 'topic_topn', 0))
                if topn > 0:
                    idx_topn = np.argpartition(-z, topn - 1)[:topn]
                    for j2 in idx_topn:
                        tok2 = out_vocab[int(classes_list[int(j2)])]
                        allowed.add(tok2)
                for tok, jj in tok_to_j.items():
                    if tok not in allowed:
                        z[int(jj)] = -1e30

            # fast path: if topic_lock, restrict ALL work to top-N candidates
            if int(getattr(args, 'topic_lock', 0)):
                topn = int(getattr(args, 'topic_topn', 24))
                cand = topk_indices(z, topn)
                zc = z[cand]
                pc = softmax(zc, float(args.temp))
                if args.mode == 'greedy':
                    j = int(cand[int(np.argmax(zc))])
                else:
                    j = int(cand[int(sample_from_probs(pc, rng))])
            else:
                p = softmax(z, float(args.temp))
                if args.mode == 'greedy':
                    j = int(np.argmax(z))
                else:
                    j = sample_from_probs(p, rng)
            y_idx = int(classes_list[j])
            y_tok = out_vocab[y_idx]

            top = topk_indices(z, int(args.topk))
            top_list = [(out_vocab[int(classes_list[k])], float(z[k])) for k in top]

            print("STEP", step, "TOP:", [(t, round(v, 4)) for (t, v) in top_list], "PICK:", y_tok)
            margin = top2_margin(z)
            p = softmax(z, float(args.temp))
            p_top1 = float(np.max(p))
            H = entropy(p)
            trace_rec = {
                'context_start': ctx0,
                'step': int(step),
                'picked': y_tok,
                'mode': getattr(args, 'mode', 'greedy'),
                'temp': float(getattr(args, 'temp', 1.0)),
                'margin': margin,
                'p_top1': p_top1,
                'entropy': H,
                'top': [(t, float(v)) for (t, v) in top_list],
            }
            trace_f.write(json.dumps(trace_rec, ensure_ascii=False) + '\n')
            csv_f.write(f"{json.dumps(ctx0, ensure_ascii=False)},{step},{y_tok},{getattr(args,'mode','greedy')},{float(getattr(args,'temp',1.0))},{margin},{p_top1},{H}\n")

            if y_tok == "<EOS>":
                break

            out.append(y_tok)
            state.append(norm_in(y_tok))

            if args.stop_punct and y_tok in stop_set:
                break

        print("GENERATED:", " ".join(out).strip())
        print("FULL:", (ctx0 + " " + " ".join(out)).strip())
        print("-" * 72)

    trace_f.close()
    csv_f.close()
    print('WROTE:', str(trace_path))
    print('WROTE:', str(csv_path))

if __name__ == "__main__":
    main()
