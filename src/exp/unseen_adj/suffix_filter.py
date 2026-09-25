"""Suffix-validity constraint on beam candidates, learned from train (per pos,feats,dialect), evaluated on all segments.
Baseline = candmerge 'base' decision (Kaggle cands only, lam=2/lam_new=2, group by (lemma,feats))."""
import sys, os, pickle, collections, argparse
import numpy as np, pandas as pd
sys.path.insert(0, 'src')
from paradigm import ParadigmModel, stress_class, n_vow, NC, ACC
from joint import combine_tally, build_masses
from validate import load_splits, weighted_score
p = argparse.ArgumentParser(); p.add_argument('--k', type=int, default=3); p.add_argument('--min_count', type=int, default=3)
p.add_argument('--pen', type=float, default=0.05); p.add_argument('--repair', action='store_true'); p.add_argument('--nvprior', action='store_true')
p.add_argument('--segs', default='transfer,complete,wug,unseen')
args = p.parse_args()
train_rows, known, val = load_splits()
ext = pickle.load(open('runs/kdev2/val_cands_aligned.pkl', 'rb'))
KEY = {'transfer': 'dev_transfer', 'complete': 'dev_complete', 'wug': 'wug', 'unseen': 'unseen_adj'}
strip = lambda s: s.replace(ACC, '')
# suffix table from train_rows (safe)
suf = collections.defaultdict(collections.Counter); tot = collections.Counter()
for pos_, f, d, fm in zip(train_rows.pos, train_rows.feats, train_rows.dialect, train_rows.form_vvz):
    suf[(pos_, f, d)][strip(fm)[-args.k:]] += 1; tot[(pos_, f, d)] += 1
def valid(pos_, f, d, form):
    return suf[(pos_, f, d)][strip(form)[-args.k:]] >= args.min_count
known_map = {}
for l, f, d, fm in zip(known.lemma_ru, known.feats, known.dialect, known.form_vvz): known_map.setdefault(l, []).append((f, d, fm))
train_lemmas = set(train_rows.lemma_ru)
pms = [ParadigmModel(K=300, seed=s).fit(train_rows) for s in (0, 1, 2)]
adj = train_rows[train_rows.pos == 'ADJ'].copy(); adj['cls'] = adj.form_vvz.map(stress_class); adj = adj[adj.cls >= 0]
g = adj.groupby(['lemma_ru', 'feats']).cls.agg(lambda s: collections.Counter(s).most_common(1)[0][0]).reset_index()
g['nv'] = g.lemma_ru.map(n_vow).clip(2, 6)
t = g.groupby(['feats', 'nv', 'cls']).size().unstack(fill_value=0).reindex(columns=range(NC), fill_value=0) + 1.0
Pnv = (t.T / t.sum(axis=1)).T

def decide(df, cpm, use_filter, repair=False, nvprior=False):
    rows = list(df.itertuples())
    tallies = [combine_tally(cpm, i) for i in range(len(rows))]
    if use_filter:
        for i, r in enumerate(rows):
            t2 = collections.defaultdict(float)
            for s, q in tallies[i].items():
                ok = valid(r.pos, r.feats, r.dialect, s)
                if not ok and repair and r.pos == 'ADJ' and r.feats.endswith('LONG') and strip(s).endswith('ии') and not strip(s).endswith('иии') and valid(r.pos, r.feats, r.dialect, s + 'и'):
                    t2[s + 'и'] += q; continue
                t2[s] += q if ok else q * args.pen
            tallies[i] = t2
    masses = build_masses(tallies)
    groups = collections.defaultdict(list)
    for i, r in enumerate(rows): groups[(r.lemma_ru, r.feats)].append(i)
    cls = [0] * len(rows)
    for (lemma, feats), idx in groups.items():
        r0 = rows[idx[0]]
        obs = [(f, fm) for f, d, fm in known_map.get(lemma, []) if not (f == feats and any(d == rows[i].dialect for i in idx))]
        if nvprior and r0.pos == 'ADJ' and lemma not in train_lemmas:
            prior = Pnv.loc[(feats, min(max(n_vow(lemma), 2), 6))].values
        else:
            prior = sum(pm.class_dist(r0.pos, feats, obs) for pm in pms) / len(pms)
        sc = sum(np.log(masses[i]) for i in idx) + 2.0 * np.log(prior + 1e-9)
        c = int(sc.argmax())
        for i in idx: cls[i] = c
    out = []
    for i, r in enumerate(rows):
        best = max(((s, q) for s, q in tallies[i].items() if stress_class(s) == cls[i]), key=lambda x: x[1], default=None)
        out.append(best[0] if best else max(tallies[i].items(), key=lambda x: x[1])[0])
    return out
segs = args.segs.split(',')
golds = {s: list(val[s].form_vvz) for s in segs}
# false-rejection rate of gold under the rule
for s in segs:
    d = val[s]; fr = np.mean([not valid(a, b, c, e) for a, b, c, e in zip(d.pos, d.feats, d.dialect, d.form_vvz)])
    print(f'{s}: gold invalid under rule k={args.k} min_count={args.min_count}: {fr:.4f} (n={len(d)})')
P = {}
for name, kw in [('base', dict(use_filter=False)), ('filter', dict(use_filter=True)), ('filter+repair', dict(use_filter=True, repair=True)),
                 ('filter+repair+nvprior', dict(use_filter=True, repair=True, nvprior=True)), ('nvprior', dict(use_filter=False, nvprior=True))]:
    P[name] = {s: decide(val[s], ext[KEY[s]], **kw) for s in segs}
    per = {s: np.mean([a == b for a, b in zip(P[name][s], golds[s])]) for s in segs}
    print(f'{name:24s} ' + ' '.join(f'{s}:{per[s]:.4f}' for s in segs) + (f'  score {weighted_score(P[name], golds)[0]}' if len(segs) == 4 else ''))
# per-feats delta for unseen
if 'unseen' in segs:
    d = val['unseen'].reset_index(drop=True)
    d['b'] = [a == b for a, b in zip(P['base']['unseen'], golds['unseen'])]; d['f'] = [a == b for a, b in zip(P['filter+repair']['unseen'], golds['unseen'])]
    x = d.groupby('feats')[['b', 'f']].mean(); x['delta'] = x.f - x.b; print(x.sort_values('delta').round(3).head(4).to_string()); print(x.sort_values('delta').round(3).tail(6).to_string())
pickle.dump(P, open(f'runs/exp/unseen_adj/suffix_preds_k{args.k}_m{args.min_count}.pkl', 'wb'))
