import argparse
import json
import math
import os
import re
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

def tokenize(s: str):
    return re.findall(r"[a-z]+", s.lower())

def stable_softmax(z: np.ndarray, temp: float):
    if temp <= 0:
        raise ValueError("temp must be > 0")
    zz = z / temp
    zz = zz - np.max(zz)
    e = np.exp(zz)
    return e / np.sum(e)

def top2_margin(z: np.ndarray) -> float:
    zz = np.sort(z)
    return float(zz[-1] - zz[-2])

def entropy(p: np.ndarray) -> float:
    p = np.asarray(p, dtype=float)
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())

def load_contexts(path: str):
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    ctx = obj.get("contexts", [])
    if not isinstance(ctx, list) or not all(isinstance(x, str) for x in ctx):
        raise SystemExit('contexts.json must contain {"contexts": ["..."]}')
    return ctx

def build_model_fixed():
    VOCAB = [
        "big", "cold", "city", "busy",
        "pizza", "subway", "finance", "art",
        "ai", "startup", "election", "senate",
    ]
    KW = [
        "new", "york", "paris", "wall", "street", "senate", "election",
        "ai", "startup", "pizza", "subway", "art", "cold", "winter",
    ]

    V = len(VOCAB)
    K = len(KW)
    tok2idx = {t:i for i,t in enumerate(VOCAB)}
    kw2idx = {t:i for i,t in enumerate(KW)}

    def phi(context: str) -> np.ndarray:
        x = np.zeros(K, dtype=float)
        for t in tokenize(context):
            if t in kw2idx:
                x[kw2idx[t]] += 1.0
        return x

    W = np.zeros((V, K), dtype=float)
    b = np.zeros(V, dtype=float)

    def add_rule(next_token: str, keyword: str, weight: float):
        W[tok2idx[next_token], kw2idx[keyword]] += weight

    for kw in ["new", "york"]:
        add_rule("city", kw, 1.2)
        add_rule("busy", kw, 0.8)
        add_rule("subway", kw, 1.0)
        add_rule("pizza", kw, 0.7)
        add_rule("finance", kw, 0.3)
        add_rule("art", kw, 0.2)

    for kw in ["paris"]:
        add_rule("art", kw, 1.5)
        add_rule("city", kw, 0.9)

    for kw in ["wall", "street"]:
        add_rule("finance", kw, 1.7)

    for kw in ["election"]:
        add_rule("election", kw, 1.9)
        add_rule("senate", kw, 0.7)

    for kw in ["senate"]:
        add_rule("senate", kw, 1.9)
        add_rule("election", kw, 0.4)

    for kw in ["ai"]:
        add_rule("ai", kw, 2.0)
        add_rule("startup", kw, 0.6)

    for kw in ["startup"]:
        add_rule("startup", kw, 2.0)
        add_rule("ai", kw, 0.5)

    for kw in ["cold", "winter"]:
        add_rule("cold", kw, 2.0)

    b[tok2idx["city"]] = 0.10
    b[tok2idx["ai"]] = 0.05
    b[tok2idx["art"]] = 0.02

    return VOCAB, KW, phi, W, b

def load_weights_npz(path: str, VOCAB: list, KW: list):
    data = np.load(path, allow_pickle=True)
    W = data["W"].astype(float)
    b = data["b"].astype(float)
    vocab = data["vocab"].tolist()
    kw = data["kw"].tolist()
    if vocab != VOCAB or kw != KW:
        raise SystemExit("weights vocab/kw do not match this runner's VOCAB/KW")
    return W, b

def plot_heatmap(Z_steps: np.ndarray, vocab: list, out_path: str):
    plt.figure(figsize=(12, 5))
    plt.imshow(Z_steps, aspect="auto", origin="lower")
    plt.yticks(range(len(vocab)), vocab)
    plt.xlabel("step")
    plt.ylabel("token")
    plt.title("logit evolution over generation steps")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--contexts", default="contexts.json")
    ap.add_argument("--mode", choices=["greedy", "sample"], default="greedy")
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--freeze_features", type=int, default=0)
    ap.add_argument("--weights", default="")
    args = ap.parse_args()

    np.random.seed(args.seed)
    os.makedirs("artifacts", exist_ok=True)

    VOCAB, KW, phi, W, b = build_model_fixed()
    if args.weights:
        W, b = load_weights_npz(args.weights, VOCAB, KW)

    contexts = load_contexts(args.contexts)

    all_traces = []
    for ctx0 in contexts:
        ctx = ctx0
        step_logits = []
        step_tokens = []

        for t in range(args.steps):
            if args.freeze_features:
                x = phi(ctx0)
            else:
                x = phi(ctx)
            z = (W @ x) + b
            step_logits.append(z.copy())

            if args.mode == "greedy":
                idx = int(np.argmax(z))
            else:
                p = stable_softmax(z, args.temp)
                idx = int(np.random.choice(len(VOCAB), p=p))

            tok = VOCAB[idx]
            step_tokens.append(tok)
            ctx = ctx + " " + tok

        Z = np.stack(step_logits, axis=1)
        margin_last = top2_margin(step_logits[-1])
        z_last = step_logits[-1]
        i_star = int(np.argmax(z_last))
        p_last = stable_softmax(np.asarray(z_last, dtype=float), args.temp)
        p_top1 = float(p_last[i_star])
        H_last = entropy(p_last)

        trace = {
            "context_start": ctx0,
            "mode": args.mode,
            "temp": args.temp,
            "steps": args.steps,
            "generated_tokens": step_tokens,
            "context_final": ctx,
            "last_step_margin": float(margin_last),
        }
        all_traces.append(trace)

        print("CONTEXT_START:", ctx0)
        print("MODE:", args.mode, "TEMP:", args.temp, "STEPS:", args.steps)
        print("GENERATED:", " ".join(step_tokens))
        print("CONTEXT_FINAL:", ctx)
        print("LAST_P_TOP1:", round(p_top1, 6))
        print("LAST_ENTROPY:", round(H_last, 6))
        print("LAST_MARGIN:", round(margin_last, 6))
        print("-" * 72)

    trace_path = Path("artifacts/trace.json")
    trace_path.write_text(json.dumps(all_traces, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n", encoding="utf-8")

    Z_first = None
    if contexts:
        ctx = contexts[0]
        step_logits = []
        for t in range(args.steps):
            x = phi(ctx)
            z = (W @ x) + b
            step_logits.append(z.copy())
            idx = int(np.argmax(z))
            ctx = ctx + " " + VOCAB[idx]
        Z_first = np.stack(step_logits, axis=1)

    if Z_first is not None:
        heat_path = "artifacts/logits_heatmap.png"
        plot_heatmap(Z_first, VOCAB, heat_path)
        print("WROTE:", "artifacts/trace.json")
        print("WROTE:", heat_path)

if __name__ == "__main__":
    main()
