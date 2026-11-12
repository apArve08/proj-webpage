"""A small byte-pair-encoding (BPE) tokenizer, written from scratch.

How it works:
  1. Text is lowercased and split into words. Each word starts with the marker "▁" (it stands for
     the space before the word), so "the cat" becomes ["▁the", "▁cat"].
  2. Every word starts as single characters: ▁ c a t
  3. Training repeatedly finds the most common neighbouring pair and glues it into one new token.
     After enough merges, common words are a single token and rare words are a few pieces.
  4. Encoding a new word replays those merges in the same order.
"""
import re
from collections import Counter

PAD, END = "<pad>", "<end>"
WORD = "▁"


def normalize(text):
    """Lowercase, keep only a-z and . , ? and put spaces around punctuation."""
    text = re.sub(r"[^a-z.,? ]", " ", text.lower())
    text = re.sub(r"([.,?])", r" \1 ", text)
    return text.split()


def train_bpe(sentences, vocab_size):
    word_freq = Counter(w for s in sentences for w in normalize(s))
    words = {w: [WORD] + list(w) for w in word_freq}
    chars = sorted({c for syms in words.values() for c in syms} | set("abcdefghijklmnopqrstuvwxyz.,?"))
    vocab = [PAD, END] + chars
    merges = []
    while len(vocab) < vocab_size:
        pairs = Counter()
        for w, syms in words.items():
            for a, b in zip(syms, syms[1:]):
                pairs[(a, b)] += word_freq[w]
        if not pairs:
            break
        (a, b), n = max(pairs.items(), key=lambda kv: (kv[1], kv[0]))   # most frequent; ties broken alphabetically
        if n < 2:
            break
        merges.append((a, b))
        vocab.append(a + b)
        for w, syms in words.items():
            words[w] = merge_pair(syms, a, b)
    return vocab, merges


def merge_pair(syms, a, b):
    out, i = [], 0
    while i < len(syms):
        if i + 1 < len(syms) and syms[i] == a and syms[i + 1] == b:
            out.append(a + b)
            i += 2
        else:
            out.append(syms[i])
            i += 1
    return out


class Tokenizer:
    def __init__(self, vocab, merges):
        self.vocab = vocab
        self.id = {t: i for i, t in enumerate(vocab)}
        self.rank = {tuple(m): r for r, m in enumerate(merges)}
        self.pad_id, self.end_id = self.id[PAD], self.id[END]

    def encode_word(self, word):
        syms = [WORD] + list(word)
        while len(syms) > 1:
            ranked = [(self.rank[p], p) for p in zip(syms, syms[1:]) if p in self.rank]
            if not ranked:
                break
            _, (a, b) = min(ranked)                                     # replay the earliest-learned merge first
            syms = merge_pair(syms, a, b)
        return syms

    def encode(self, text):
        return [self.id[s] for w in normalize(text) for s in self.encode_word(w)]

    def decode(self, ids):
        text = "".join(self.vocab[i] for i in ids if i not in (self.pad_id, self.end_id))
        return re.sub(r" ([.,?])", r"\1", text.replace(WORD, " ").strip())
