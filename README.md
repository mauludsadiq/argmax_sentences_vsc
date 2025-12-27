# argmax_sentences_vsc

A minimal, inspectable **decoder sandbox** for next‑token generation.

It is *not* a transformer. Instead, it builds a **logit vector** from a small set of discrete context features and then runs
LLM‑style decoding (greedy argmax or stochastic sampling) with common constraint masks (ban tokens, stop punctuation, no‑repeat window, etc.).
The point of the repo is to make *argmax behavior* and *decoder pathologies* visible with full traces.

---

## What’s in here (high level)

- `demo_positional_argmax.py`  
  Loads a saved linear logit model (`.npz`) and runs decoding over multiple prompt contexts, printing:
  - the top‑K logits each step
  - the chosen token
  - per‑step metrics (margin, top‑1 prob, entropy)
  - plus JSONL traces and CSV curves in `artifacts/`

- `train_positional_next_token.py` (if present in your repo)  
  Trains the `.npz` weights used by the demo.

- `contexts.json`  
  A small set of prompt “context starts” to decode from.

- `artifacts/positional_Wb.npz`  
  Saved weights and vocab tables.

---

## The math (what the demo is actually computing)

### 1) Logit construction = additive feature model

At each step, the demo constructs a logit vector `z ∈ R^K` as a **sum of learned columns** plus a bias:

```python
z = b + W[:, c1] + W[:, c2] + W[:, c3]
```

Interpretation:

- `W` is a matrix of shape `(K, D)` (K = “class slots”, D = feature dictionary size).
- `b` is a bias vector of shape `(K,)`.
- `c1,c2,c3` are integer feature indices derived from the current state (previous tokens / positional slots).
- This is a linear “energy model” over next-token classes; it is *much simpler than a transformer*,
  but the **decoder dynamics** (argmax vs sampling, masking) are the same type of operations used in LLM inference.

### 2) Softmax distribution (temperature)

The probabilities are computed from logits via temperature‑scaled softmax:

```python
p = softmax(z, float(args.temp))
```

Mathematically:

\[
p_i = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)} \quad\text{with } T = \texttt{args.temp}.
\]

### 3) Decoding rule = greedy argmax or sampling

Greedy (argmax):

```python
j = int(np.argmax(z))
```

Sampling:

```python
j = sample_from_probs(p, rng)
```

This is exactly the same choice point an LLM decoder has: **pick the maximum‑logit token** or **sample** from the distribution.

### 4) Trace metrics = “argmax cliff” visibility

The run prints and logs:

- **margin** (top‑1 minus top‑2 logit gap):  
  big margin ⇒ stable argmax ⇒ “cliff” behavior (hard to escape attractors)
- **p_top1** and **entropy**:  
  low entropy + high p_top1 ⇒ near‑deterministic behavior

You can see that directly in your run logs, e.g.:

```
STEP 1 TOP: [('art', 6.0126), ('.', -6.8776), ...] PICK: art
STEP 2 TOP: [('.', 5.8976), ('is', -6.3929), ...] PICK: .
```

The logit gap makes the pick effectively locked.

---

## How close this is to what “argmax” does in real LLMs

What matches real LLM decoding:

- **You have logits `z` over tokens**, then apply:
  - temperature softmax
  - greedy argmax vs sampling
  - constraint masks (ban/stop/no‑repeat/etc.)
- The *pathologies* match: repetition loops, premature stop tokens, topic drift when the distribution is flat, etc.

What does **not** match a real LLM:

- The “model” here is a small additive linear map with a tiny state, not self‑attention over long context.
- There is no learned continuous embedding space, no layers, no attention heads.
- So this repo is not a language model; it is a **decoder laboratory**.

The key lesson: **decoder policy can dominate perceived behavior** even with a simple logit generator.

---

## What this tells us about LLMs (decoder‑centric statement)

Your traces show that once a system reaches a high‑margin top‑1 token, greedy decoding becomes a
deterministic dynamical system with attractors (“argmax cliffs”).
Sampling + temperature (and masks like no‑repeat windows) are not cosmetic; they are control inputs that
change which attractor you fall into or whether you escape one.

---

## Get the repo (Terminal)

```bash
cd ~/Downloads
git clone https://github.com/mauludsadiq/argmax_sentences_vsc.git
cd argmax_sentences_vsc
git status
git log -1 --oneline
```

### Open in VS Code

If you have the `code` CLI:

```bash
code .
```

Or in VS Code: **File → Open Folder…** and select `argmax_sentences_vsc/`.

---

## Setup (venv)

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -V
```

Install deps (the demo typically only needs NumPy):

```bash
python3 -m pip install -U pip
python3 -m pip install numpy
```

---

## Run the demo

From the repo root (verify `pwd` and that the file exists):

```bash
pwd
ls -la demo_positional_argmax.py
ls -la artifacts/positional_Wb.npz
ls -la contexts.json
```

Then run:

```bash
python3 demo_positional_argmax.py \
  --weights artifacts/positional_Wb.npz \
  --contexts contexts.json \
  --steps 30 \
  --stop_punct 1 \
  --min_steps 1 \
  --no_repeat_window 5 \
  --mode sample \
  --temp 1.3 \
  --seed 0 \
  | tee artifacts/run_sample_t1p3.txt
```

Outputs written (by default):

- `artifacts/trace.jsonl`  (per‑step JSON records)
- `artifacts/curve.csv`    (one line per step; useful for plotting)

---

## Troubleshooting

### “can’t open file ... demo_positional_argmax.py: [Errno 2] No such file or directory”

This means you are **not in the repo root that contains the file**, or your checkout is stale.

Run:

```bash
pwd
ls -la
git status
git branch --show-current
git pull --ff-only
ls -la demo_positional_argmax.py
```

If it’s still missing, you are likely in a different folder than the clone you think you’re in.
Search for the file:

```bash
cd ~
python3 - <<'PY'
import os
hits=[]
for root,dirs,files in os.walk(os.path.expanduser("~")):
    if "demo_positional_argmax.py" in files:
        hits.append(os.path.join(root,"demo_positional_argmax.py"))
        if len(hits) >= 20:
            break
print("\n".join(hits) if hits else "no hits")
PY
```

### “argparse.ArgumentError: ... conflicting option string”

That indicates duplicate `ap.add_argument("--flag", ...)` lines in the script.
Search for duplicates:

```bash
grep -n 'add_argument("--topic_lock"' demo_positional_argmax.py || true
grep -n 'add_argument("--topic_keep"' demo_positional_argmax.py || true
```

Then delete the duplicated definitions and re‑compile:

```bash
python3 -m py_compile demo_positional_argmax.py
```

### “UnboundLocalError: local variable 'p' referenced before assignment”

That means there is a decode path that selects `j` without defining `p`.
The fix is: ensure `p = softmax(z, temp)` exists on *every* path before metrics like `np.max(p)`.

---

## Repo hygiene

The repo typically generates lots of run outputs under `artifacts/`.
A `.gitignore` should ignore those by default; only commit code + small configs.

---

## License

Add a license file if you intend to distribute publicly.
