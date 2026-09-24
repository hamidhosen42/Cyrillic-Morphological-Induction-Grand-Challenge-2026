"""Data preparation for context-aware morphological inflection.

Each example: target (lemma, feats, dialect, yat, freq) + context = other known forms
of the same lemma (from train/dev/wug seeds). Encoder input is a token sequence:
  [DIAL] [POS] [F:...]* [YAT] [FQ] <lemma chars> [SEP] ([CDIAL] [CF:...]* <form chars> [SEP])*
Decoder output: form chars.
"""
import random, math
import numpy as np
import pandas as pd

ACC = '́'
PAD, BOS, EOS, SEP, CTX = '<pad>', '<bos>', '<eos>', '<sep>', '<ctx>'
MAX_CTX = 6


def freq_bucket(f):
    return f"[FQ{int(min(6, max(1, math.floor(math.log10(f)))))}]"


class Vocab:
    def __init__(self, chars, tag_tokens):
        self.itos = [PAD, BOS, EOS, SEP, CTX] + sorted(tag_tokens) + sorted(chars)
        self.stoi = {t: i for i, t in enumerate(self.itos)}

    def __len__(self):
        return len(self.itos)

    def enc(self, toks):
        return [self.stoi[t] for t in toks]

    def dec(self, ids):
        out = []
        for i in ids:
            t = self.itos[i]
            if t == EOS:
                break
            if t in (PAD, BOS):
                continue
            out.append(t)
        return ''.join(out)


def build_vocab(dfs):
    chars = set()
    for d in dfs:
        for s in d.lemma_ru:
            chars.update(s)
        if 'form_vvz' in d:
            for s in d.form_vvz.dropna():
                chars.update(s)
    tags = set()
    for d in dfs:
        for f in d.feats.unique():
            for p in f.split(';'):
                tags.add('[F:' + p + ']')
                tags.add('[CF:' + p + ']')
        for dl in d.dialect.unique():
            tags.add('[D:' + dl + ']'); tags.add('[CD:' + dl + ']')
    for y in (0, 1):
        tags.add(f'[YAT{y}]')
    for b in range(1, 7):
        tags.add(f'[FQ{b}]')
    tags.add('[SAME]')  # marks context form with the same feats as target
    return Vocab(chars, tags)


def target_tokens(lemma, feats, dialect, yat, freq):
    toks = ['[D:' + dialect + ']'] + ['[F:' + p + ']' for p in feats.split(';')]
    toks += [f'[YAT{int(yat)}]', freq_bucket(freq)]
    toks += list(lemma) + [SEP]
    return toks


def ctx_tokens(feats, dialect, form, same):
    toks = [CTX]
    if same:
        toks.append('[SAME]')
    toks += ['[CD:' + dialect + ']'] + ['[CF:' + p + ']' for p in feats.split(';')]
    toks += list(form) + [SEP]
    return toks


def order_context(cands, feats, dialect, k=MAX_CTX):
    """cands: list of (feats, dialect, form). Priority: same feats (other dialect),
    then same dialect, then rest. Deterministic order within tiers."""
    tgt_num = 'SG' if ';SG' in feats else ('PL' if ';PL' in feats else '')
    def key(c):
        f, d, _ = c
        same_f = f == feats
        same_d = d == dialect
        same_num = (('SG' if ';SG' in f else ('PL' if ';PL' in f else '')) == tgt_num)
        return (0 if same_f else 1, 0 if same_d else 1, 0 if same_num else 1, f, d)
    cands = sorted(cands, key=key)
    return cands[:k]


def make_input(lemma, feats, dialect, yat, freq, cands):
    toks = target_tokens(lemma, feats, dialect, yat, freq)
    for f, d, form in order_context(cands, feats, dialect):
        toks += ctx_tokens(f, d, form, f == feats)
    return toks


def sample_train_context(rows, feats, dialect, pos, rng):
    """rows: list of (feats, dialect, form) for the same lemma excluding target.
    Simulates test conditions: dialect-transfer, paradigm completion, wug (seed only), unseen."""
    r = rng.random()
    if pos == 'ADJ' and r < 0.12:
        return []  # unseen-lemma condition
    if pos in ('N', 'V') and r < 0.30:
        seed_f = 'N;ACC;SG' if pos == 'N' else 'V;PRS;3;SG'
        seeds = [c for c in rows if c[0] == seed_f and c[1] == 'SEV']
        if seeds:
            return seeds
    r = rng.random()
    if r < 0.45:
        # paradigm completion: remove same-feats forms
        rows = [c for c in rows if c[0] != feats]
    if not rows:
        return []
    n = rng.randint(1, len(rows))
    return rng.sample(rows, n)
