"""Trains the AP Tensor Lab mini transformer (a small GPT-style decoder) with JAX on a CPU.

Usage:  python3 train.py corpus.json out_dir

Architecture (the same shape as GPT, just very small):
  token embedding + position embedding
  N x [ LayerNorm -> causal self-attention -> add ]  [ LayerNorm -> feed-forward (GELU) -> add ]
  final LayerNorm -> multiply by the token-embedding matrix again (weight tying) -> one score per token
"""
import json, sys, time, math, base64, os
import numpy as np
import jax, jax.numpy as jnp, optax
from tokenizer import Tokenizer, train_bpe, WORD

CFG = dict(vocab=1024, d_model=64, n_layer=4, n_head=4, block=32, d_mlp=256)
STEPS, BATCH = int(os.environ.get("STEPS", 6000)), int(os.environ.get("BATCH", 128))
PEAK_LR, DROPOUT, SEED = 3e-3, 0.1, 0


# ------------------------------------------------------------------ model
def init_params(key, c):
    d, m = c["d_model"], c["d_mlp"]
    k = iter(jax.random.split(key, 8 + 8 * c["n_layer"]))
    n = lambda shape, s=0.02: jax.random.normal(next(k), shape) * s
    p = {"wte": n((c["vocab"], d)), "wpe": n((c["block"], d)), "lnf_g": jnp.ones(d), "lnf_b": jnp.zeros(d)}
    out_scale = 0.02 / math.sqrt(2 * c["n_layer"])
    for i in range(c["n_layer"]):
        p.update({
            f"l{i}_ln1_g": jnp.ones(d), f"l{i}_ln1_b": jnp.zeros(d),
            f"l{i}_wq": n((d, d)), f"l{i}_bq": jnp.zeros(d),
            f"l{i}_wk": n((d, d)), f"l{i}_bk": jnp.zeros(d),
            f"l{i}_wv": n((d, d)), f"l{i}_bv": jnp.zeros(d),
            f"l{i}_wo": n((d, d), out_scale), f"l{i}_bo": jnp.zeros(d),
            f"l{i}_ln2_g": jnp.ones(d), f"l{i}_ln2_b": jnp.zeros(d),
            f"l{i}_w1": n((d, m)), f"l{i}_b1": jnp.zeros(m),
            f"l{i}_w2": n((m, d), out_scale), f"l{i}_b2": jnp.zeros(d),
        })
    return p


def layer_norm(x, g, b):
    mu = x.mean(-1, keepdims=True)
    var = ((x - mu) ** 2).mean(-1, keepdims=True)
    return (x - mu) / jnp.sqrt(var + 1e-5) * g + b


def dropout(x, key, rate):
    if key is None or rate == 0.0:
        return x
    keep = jax.random.bernoulli(key, 1.0 - rate, x.shape)
    return jnp.where(keep, x / (1.0 - rate), 0.0)


def forward(p, ids, c, key=None, rate=0.0):
    """ids: [T] token ids of one sequence. Returns logits [T, vocab]."""
    T, d, H = ids.shape[0], c["d_model"], c["n_head"]
    dh = d // H
    keys = iter(jax.random.split(key, 1 + 2 * c["n_layer"])) if key is not None else None
    nk = lambda: next(keys) if keys is not None else None
    x = dropout(p["wte"][ids] + p["wpe"][:T], nk(), rate)
    causal = jnp.tril(jnp.ones((T, T), dtype=bool))                       # token i may only look at tokens 0..i
    for i in range(c["n_layer"]):
        L = lambda name: p[f"l{i}_{name}"]
        h = layer_norm(x, L("ln1_g"), L("ln1_b"))
        split = lambda t: t.reshape(T, H, dh).transpose(1, 0, 2)          # [T, d] -> [H, T, dh]
        q, k, v = split(h @ L("wq") + L("bq")), split(h @ L("wk") + L("bk")), split(h @ L("wv") + L("bv"))
        scores = q @ k.transpose(0, 2, 1) / math.sqrt(dh)                 # [H, T, T]
        att = jax.nn.softmax(jnp.where(causal, scores, -1e9), axis=-1)
        mixed = (att @ v).transpose(1, 0, 2).reshape(T, d)
        x = x + dropout(mixed @ L("wo") + L("bo"), nk(), rate)
        h = layer_norm(x, L("ln2_g"), L("ln2_b"))
        x = x + dropout(jax.nn.gelu(h @ L("w1") + L("b1"), approximate=True) @ L("w2") + L("b2"), nk(), rate)
    return layer_norm(x, p["lnf_g"], p["lnf_b"]) @ p["wte"].T


# ------------------------------------------------------------------ data
def main():
    corpus_path, out = sys.argv[1], sys.argv[2]
    os.makedirs(out, exist_ok=True)
    data = json.load(open(corpus_path))
    c = dict(CFG)
    vocab, merges = train_bpe(data["train"], c["vocab"])
    c["vocab"] = len(vocab)
    tok = Tokenizer(vocab, merges)
    T = c["block"]

    seqs = [tok.encode(s) + [tok.end_id] for s in data["train"]]
    assert max(map(len, seqs)) <= T + 1, max(map(len, seqs))
    X = np.full((len(seqs), T), tok.pad_id, np.int32)
    Y = np.full((len(seqs), T), -1, np.int32)                             # -1 = ignore in the loss
    for r, s in enumerate(seqs):
        X[r, :len(s) - 1], Y[r, :len(s) - 1] = s[:-1], s[1:]
    print(f"vocab {len(vocab)}  sequences {len(seqs)}  tokens {sum(map(len, seqs))}")

    key = jax.random.PRNGKey(SEED)
    params = init_params(key, c)
    n_params = sum(int(np.prod(v.shape)) for v in params.values())
    print("parameters:", n_params)

    sched = optax.warmup_cosine_decay_schedule(0.0, PEAK_LR, 200, STEPS, PEAK_LR * 0.02)
    decay_mask = {k: v.ndim == 2 for k, v in params.items()}             # weight decay on matrices only
    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(sched, weight_decay=0.05, mask=decay_mask))
    state = opt.init(params)

    def loss_fn(p, x, y, key):
        keys = jax.random.split(key, x.shape[0])
        logits = jax.vmap(lambda ids, k: forward(p, ids, c, k, DROPOUT))(x, keys)
        logp = jax.nn.log_softmax(logits, -1)
        mask = y >= 0
        picked = jnp.take_along_axis(logp, jnp.maximum(y, 0)[..., None], -1)[..., 0]
        return -(picked * mask).sum() / mask.sum()

    @jax.jit
    def step(p, s, x, y, key):
        loss, g = jax.value_and_grad(loss_fn)(p, x, y, key)
        upd, s = opt.update(g, s, p)
        return optax.apply_updates(p, upd), s, loss

    rng = np.random.default_rng(SEED)
    t0 = time.time()
    for it in range(1, STEPS + 1):
        idx = rng.integers(0, len(X), BATCH)
        key, sub = jax.random.split(key)
        params, state, loss = step(params, state, X[idx], Y[idx], sub)
        if it % 100 == 0 or it == 1:
            print(f"step {it:5d}  loss {float(loss):.3f}  {time.time() - t0:.0f}s", flush=True)

    # ------------------------------------------------------------------ evaluation
    fwd = jax.jit(lambda ids: forward(params, ids, c))

    def next_logits(ids):
        padded = np.full(T, tok.pad_id, np.int32)
        padded[:len(ids)] = ids
        return np.asarray(fwd(jnp.asarray(padded)))[len(ids) - 1]         # causal: padding after the end is never seen

    def complete(prompt, max_new=12, whole=False):
        ids, new = tok.encode(prompt), []
        while len(new) < max_new and len(ids) < T:
            nxt = int(next_logits(ids).argmax())
            if nxt == tok.end_id or (not whole and new and vocab[nxt].startswith(WORD)):
                break
            ids.append(nxt)
            new.append(nxt)
        return tok.decode(new)

    # 1) every trained sentence: is each next token the model's top choice?
    uniq = sorted(set(data["train"]))
    hit = tot = full = 0
    for s in uniq:
        ids = tok.encode(s) + [tok.end_id]
        pred = np.asarray(fwd(jnp.asarray(np.pad(ids[:-1], (0, T - len(ids) + 1), constant_values=tok.pad_id))))
        ok = [int(pred[i].argmax()) == ids[i + 1] for i in range(len(ids) - 1)]
        hit, tot, full = hit + sum(ok), tot + len(ok), full + all(ok[2:])
    print(f"trained sentences: next-token top-1 {hit / tot:.1%} over {tot} positions")

    # 2) fact recall: prompt up to the answer, check the answer word
    import re
    facts = []
    for s in uniq:
        m = re.match(r"(the (?:capital|opposite) of \w+ is|\w+ plus \w+ (?:is|equals)|\w+ minus \w+ equals) (\w+)", s)
        if m:
            facts.append((m.group(1), m.group(2)))
    good = sum(complete(pr) == ans for pr, ans in facts)
    print(f"fact recall on trained phrasings: {good}/{len(facts)} = {good / len(facts):.1%}")

    # 3) held-out phrasings never seen in training
    by = {}
    for pr, ans, fam in data["heldout"]:
        got = complete(pr)
        by.setdefault(fam, []).append((pr, ans, got))
    for fam, rows in by.items():
        print(f"held-out {fam}: {sum(a == g for _, a, g in rows)}/{len(rows)}")
        for pr, a, g in rows:
            if a != g:
                print(f"     miss: {pr!r} -> {g!r} (expected {a!r})")
    held_ok = sum(a == g for rows in by.values() for _, a, g in rows)

    for pr in ["the capital of france is", "the capital of malaysia is", "two plus two equals", "the opposite of hot is",
               "a tensor is", "attention lets", "ap stands for", "what is a perceptron ?", "the axon", "gpu stands for",
               "a transformer is", "the capital of mars is", "my name is"]:
        print(f"  {pr!r:38} -> {complete(pr, 16, whole=True)!r}")

    # ------------------------------------------------------------------ export for the browser
    names = sorted(params)
    blob, manifest, off = [], [], 0
    for nme in names:
        a = np.asarray(params[nme], np.float16)
        manifest.append({"name": nme, "shape": list(a.shape), "offset": off})
        blob.append(a.tobytes())
        off += a.size
    raw = b"".join(blob)
    meta = {
        "config": c, "vocab": vocab, "merges": merges, "tensors": manifest, "dtype": "float16",
        "stats": {"parameters": n_params, "unique_sentences": len(uniq), "trained_top1": round(hit / tot, 4),
                  "fact_recall": [good, len(facts)], "heldout": [held_ok, len(data["heldout"])], "steps": STEPS},
    }
    json.dump(meta, open(f"{out}/model.json", "w"), ensure_ascii=False)
    open(f"{out}/weights.b64", "w").write(base64.b64encode(raw).decode())
    np.savez(f"{out}/params.npz", **{k: np.asarray(v) for k, v in params.items()})
    print(f"exported: weights {len(raw) / 1024:.0f} KB (float16), base64 {len(raw) * 4 / 3 / 1024:.0f} KB")


if __name__ == "__main__":
    main()
