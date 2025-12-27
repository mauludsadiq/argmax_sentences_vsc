\
import re
import numpy as np

# ------------------------------------------------------------
# Argmax Next-Token Demo (NO softmax, NO training)
# ------------------------------------------------------------
# This demonstrates the *terminal selector* behavior:
#   context -> logits over a tiny vocabulary -> argmax picks next token
#
# The "model" is deliberately tiny:
#   x = φ(context) is a hand-crafted feature vector
#   z = W x + b are logits for each candidate next token
#   next_token = argmax(z)
#
# Tie-break rule: np.argmax (first maximal index).
# ------------------------------------------------------------

# Candidate next tokens (the "vocabulary" for this demo)
VOCAB = [
    "big", "cold", "city", "busy",
    "pizza", "subway", "finance", "art",
    "ai", "startup", "election", "senate",
]
V = len(VOCAB)
tok2idx = {t:i for i,t in enumerate(VOCAB)}

# Context keywords used to build features (toy "understanding")
KW = [
    "new", "york", "paris", "wall", "street", "senate", "election",
    "ai", "startup", "pizza", "subway", "art", "cold", "winter",
]
K = len(KW)
kw2idx = {t:i for i,t in enumerate(KW)}

def tokenize(s: str):
    return re.findall(r"[a-z]+", s.lower())

def phi(context: str) -> np.ndarray:
    """
    Feature map φ(context) -> x in R^K:
      x[j] = count of keyword KW[j] in the context.
    """
    x = np.zeros(K, dtype=float)
    for t in tokenize(context):
        if t in kw2idx:
            x[kw2idx[t]] += 1.0
    return x

# ------------------------------------------------------------
# Linear scorer: z = W x + b, where z in R^V
# ------------------------------------------------------------
W = np.zeros((V, K), dtype=float)
b = np.zeros(V, dtype=float)

def add_rule(next_token: str, keyword: str, weight: float):
    W[tok2idx[next_token], kw2idx[keyword]] += weight

# "New York" priors
for kw in ["new", "york"]:
    add_rule("city", kw, 1.2)
    add_rule("busy", kw, 0.8)
    add_rule("subway", kw, 1.0)
    add_rule("pizza", kw, 0.7)
    add_rule("finance", kw, 0.3)
    add_rule("art", kw, 0.2)

# "Paris" priors
for kw in ["paris"]:
    add_rule("art", kw, 1.5)
    add_rule("city", kw, 0.9)

# Finance / politics cues
for kw in ["wall", "street"]:
    add_rule("finance", kw, 1.7)

for kw in ["election"]:
    add_rule("election", kw, 1.9)
    add_rule("senate", kw, 0.7)

for kw in ["senate"]:
    add_rule("senate", kw, 1.9)
    add_rule("election", kw, 0.4)

# Tech cues
for kw in ["ai"]:
    add_rule("ai", kw, 2.0)
    add_rule("startup", kw, 0.6)

for kw in ["startup"]:
    add_rule("startup", kw, 2.0)
    add_rule("ai", kw, 0.5)

# Weather cue
for kw in ["cold", "winter"]:
    add_rule("cold", kw, 2.0)

# Small biases to break some ties (didactic)
b[tok2idx["city"]] = 0.10
b[tok2idx["ai"]] = 0.05
b[tok2idx["art"]] = 0.02

def logits(context: str) -> np.ndarray:
    x = phi(context)
    return W @ x + b

def argmax_first(z: np.ndarray) -> int:
    return int(np.argmax(z))

def top2_margin(z: np.ndarray) -> float:
    zz = np.sort(z)
    return float(zz[-1] - zz[-2])

def explain(context: str, topk: int = 6):
    x = phi(context)
    z = logits(context)
    y = argmax_first(z)
    m = top2_margin(z)

    # Show nonzero features
    nz = [(KW[i], int(x[i])) for i in range(K) if x[i] > 0]
    nz_str = nz if nz else "(none)"

    # Top-k logits
    idx = np.argsort(-z)[:topk]
    top = [(VOCAB[i], float(np.round(z[i], 3))) for i in idx]

    print("CONTEXT:", context)
    print("tokens:", tokenize(context))
    print("features φ(context):", nz_str)
    print("top logits:", top)
    print("argmax next token:", VOCAB[y], f"(index={y})", "margin=", round(m, 6))
    print("-" * 72)

def main():
    import json
    from pathlib import Path

    contexts_path = Path("contexts.json")
    obj = json.loads(contexts_path.read_text(encoding="utf-8"))
    contexts = obj.get("contexts", [])
    if not isinstance(contexts, list) or not all(isinstance(x, str) for x in contexts):
        raise SystemExit('contexts.json must contain {"contexts": ["..."]}')

    for c in contexts:
        explain(c)

if __name__ == "__main__":
    main()
main()
