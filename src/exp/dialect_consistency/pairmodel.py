"""Cross-dialect pair consistency model learned from train pairs of the same (lemma, feats).
score(x, y, a, b) = log P_ab(diff ops between stress-stripped x and y), ops assumed independent.
"""
import collections, difflib, itertools, math
import numpy as np

ACC = '́'


def strip(s):
    return s.replace(ACC, '')


def ops(x, y, ctx=0):
    xs, ys = strip(x), strip(y)
    if xs == ys:
        return ()
    sm = difflib.SequenceMatcher(None, xs, ys, autojunk=False)
    out = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == 'equal':
            continue
        if ctx:
            out.append((xs[max(0, i1 - ctx):i1], xs[i1:i2], ys[j1:j2], xs[i2:i2 + ctx]))
        else:
            out.append((xs[i1:i2], ys[j1:j2]))
    return tuple(out)


class PairModel:
    def __init__(self, ctx=0, floor=0.5, min_count=1):
        self.ctx, self.floor, self.min_count = ctx, floor, min_count
        self.counts = {}   # (a,b) -> Counter of ops
        self.N = {}        # (a,b) -> number of pairs
        self.ident = {}    # (a,b) -> P(identical)

    def fit(self, df):
        g = df.groupby(['lemma_ru', 'feats'])
        cnt = collections.defaultdict(collections.Counter); N = collections.Counter(); ident = collections.Counter()
        for (l, f), d in g:
            if d.dialect.nunique() < 2:
                continue
            forms = dict(zip(d.dialect, d.form_vvz))
            for a, b in itertools.combinations(sorted(forms), 2):
                N[(a, b)] += 1
                o = ops(forms[a], forms[b], self.ctx)
                if not o:
                    ident[(a, b)] += 1
                for op in o:
                    cnt[(a, b)][op] += 1
        self.counts, self.N = dict(cnt), dict(N)
        self.ident = {k: ident[k] / N[k] for k in N}
        return self

    def key(self, a, b):
        return (a, b) if (a, b) in self.N else (b, a)

    def logp(self, x, y, a, b):
        """log-probability-ish consistency of forms x (dialect a) and y (dialect b)."""
        if a == b:
            return 0.0 if strip(x) == strip(y) else math.log(self.floor / 1000.0)
        k = self.key(a, b)
        if k[0] != a:
            x, y = y, x
        o = ops(x, y, self.ctx)
        if not o:
            return math.log(self.ident[k])
        c = self.counts[k]; N = self.N[k]
        s = 0.0
        for op in o:
            n = c.get(op, 0)
            s += math.log(max(n, self.floor) / N) if n >= self.min_count else math.log(self.floor / N)
        return s
