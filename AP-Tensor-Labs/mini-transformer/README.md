# AP Tensor Lab mini transformer: training code

This is the code that produced the model embedded in `index.html`.

| File | What it does |
|---|---|
| `build_corpus.py` | Writes the training text (every sentence the model knows) and the held-out test prompts |
| `tokenizer.py` | Byte-pair-encoding tokenizer, written from scratch |
| `train.py` | The model (GPT-style decoder), the training loop, the accuracy report and the export |
| `corpus.json` | The generated training text |
| `out/model.json`, `out/weights.b64` | The exported tokenizer, config and weights used by the page |

## Run it

```
pip install jax optax numpy
python3 build_corpus.py corpus.json
STEPS=5000 BATCH=64 python3 train.py corpus.json out
```

About 13 minutes on a 2-core CPU. Result of the embedded run: 267,648 parameters, 100% recall on 408 trained
facts, 30 of 30 held-out phrasings correct.

## Put your own model in the page

The page reads two `<script>` blocks placed just above `<!-- JAVASCRIPT LOGIC -->`:

- `ap-model-meta`: JSON with `config` (vocab, d_model, n_layer, n_head, block, d_mlp), `vocab`, `merges`,
  `tensors` (name, shape, offset in float16 values) and `stats`
- `ap-model-weights`: all tensors as little-endian float16, concatenated, base64-encoded

Tensor names: `wte`, `wpe`, `lnf_g`, `lnf_b`, and per layer `l{i}_ln1_g/b`, `l{i}_wq/bq`, `l{i}_wk/bk`,
`l{i}_wv/bv`, `l{i}_wo/bo`, `l{i}_ln2_g/b`, `l{i}_w1/b1`, `l{i}_w2/b2`. Linear weights are stored as
`[inputs, outputs]` (the forward pass computes `x @ W + b`). The output layer reuses `wte` (weight tying), GELU
uses the tanh approximation, and LayerNorm uses eps 1e-5. A model you write yourself in PyTorch will load as long
as it exports these names and shapes; PyTorch `nn.Linear` weights need transposing.
