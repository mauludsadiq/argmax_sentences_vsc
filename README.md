# Argmax with Sentences (VSC Demo)

This is a minimal, self-contained demonstration of **argmax** operating on **sentence-derived logits**.

Pipeline:

1. `context` (a string) is tokenized
2. A tiny feature map φ(context) produces a numeric feature vector
3. Logits are computed: `z = W x + b`
4. **Argmax** selects the next token: `next = argmax(z)`

There is **no** softmax, attention, backpropagation, RLHF, or PPO here—only a deterministic argmax selector on a score vector.

## Run

```bash
python demo_next_token_argmax.py
```

Outputs:
- `artifacts/demo_output.txt` (captured stdout from the demo run)
- `vsc_manifest.json` (SHA-256 digests + run artifact digests)

## Notes

- Tie-break rule: `np.argmax` (first maximal index).
- This is a didactic toy. Real LLMs compute logits via deep networks; here logits come from a small linear scorer.
