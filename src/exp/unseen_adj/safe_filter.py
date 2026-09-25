"""Adaptive-ending-length suffix filter (only cells whose top-5 k-suffixes cover >=99% of train rows) on holdout + test counts."""
import sys, os, pickle, collections
import numpy as np, pandas as pd
sys.path.insert(0, 'src')
from paradigm import ParadigmModel, stress_class, n_vow, NC, ACC
from joint import combine_tally, build_masses
from validate import load_splits, weighted_score
strip = lambda s: s.replace(ACC, '')
def build_rules(df, cov=0.99, topn=5):
    rules = {}
    for key, d in df.groupby(['pos', 'feats', 'dialect']):
        sf = d.form_vvz.map(strip); best = None
        for k in (1, 2, 3, 4):
            vc = sf.str[-k:].value_counts()
            if vc.head(topn).sum() / len(sf) >= cov: best = (k, set(vc.head(topn).index))
            else: break
        if best: rules[key] = best
    return rules
def is_valid(rules, pos_, f, d, s):
    r = rules.get((pos_, f, d))
    return True if r is None else strip(s)[-r[0]:] in r[1]
train_rows, known, val = load_splits()
rules = build_rules(train_rows)
print('cells with rule:', len(rules), 'of', train_rows.groupby(['pos', 'feats', 'dialect']).ngroups, ' k dist:', collections.Counter(r[0] for r in rules.values()))
print('example rules:', {k: v for k, v in list(rules.items()) if k[0] == 'N' and k[2] == 'SEV'})
ext = pickle.load(open('runs/kdev2/val_cands_aligned.pkl', 'rb'))
KEY = {'transfer': 'dev_transfer', 'complete': 'dev_complete', 'wug': 'wug', 'unseen': 'unseen_adj'}
segs = list(KEY)
for s in segs:
    d = val[s]; fr = np.mean([not is_valid(rules, a, b, c, e) for a, b, c, e in zip(d.pos, d.feats, d.dialect, d.form_vvz)])
    print(f'{s}: gold false-rejection {fr:.4f}')
known_map = {}
for l, f, d, fm in zip(known.lemma_ru, known.feats, known.dialect, known.form_vvz): known_map.setdefault(l, []).append((f, d, fm))
pms = [ParadigmModel(K=300, seed=s).fit(train_rows) for s in (0, 1, 2)]
def decide(df, cpm, use_filter, pen=0.05):
    rows = list(df.itertuples()); tallies = [combine_tally(cpm, i) for i in range(len(rows))]
    if use_filter:
        for i, r in enumerate(rows):
            tallies[i] = {s: (q if is_valid(rules, r.pos, r.feats, r.dialect, s) else q * pen) for s, q in tallies[i].items()}
    masses = build_masses(tallies); groups = collections.defaultdict(list)
    for i, r in enumerate(rows): groups[(r.lemma_ru, r.feats)].append(i)
    cls = [0] * len(rows)
    for (lemma, feats), idx in groups.items():
        r0 = rows[idx[0]]
        obs = [(f, fm) for f, d, fm in known_map.get(lemma, []) if not (f == feats and any(d == rows[i].dialect for i in idx))]
        prior = sum(pm.class_dist(r0.pos, feats, obs) for pm in pms) / len(pms)
        c = int((sum(np.log(masses[i]) for i in idx) + 2.0 * np.log(prior + 1e-9)).argmax())
        for i in idx: cls[i] = c
    out = []
    for i, r in enumerate(rows):
        best = max(((s, q) for s, q in tallies[i].items() if stress_class(s) == cls[i]), key=lambda x: x[1], default=None)
        out.append(best[0] if best else max(tallies[i].items(), key=lambda x: x[1])[0])
    return out
golds = {s: list(val[s].form_vvz) for s in segs}
P = {}
for name, uf in [('base', False), ('safe_filter', True)]:
    P[name] = {s: decide(val[s], ext[KEY[s]], uf) for s in segs}
    per = {s: np.mean([a == b for a, b in zip(P[name][s], golds[s])]) for s in segs}
    print(f'{name:12s} ' + ' '.join(f'{s}:{per[s]:.4f}' for s in segs) + f'  score {weighted_score(P[name], golds)[0]}')
ch = {s: int(np.sum([a != b for a, b in zip(P['base'][s], P['safe_filter'][s])])) for s in segs}
gain = {s: int(np.sum([(a != b) and (b == g) for a, b, g in zip(P['base'][s], P['safe_filter'][s], golds[s])])) for s in segs}
loss = {s: int(np.sum([(a != b) and (a == g) for a, b, g in zip(P['base'][s], P['safe_filter'][s], golds[s])])) for s in segs}
print('holdout changed rows', ch, 'fixed', gain, 'broken', loss)
# test: rules from full train+dev, count top-candidate invalid per segment
tr = pd.read_csv('data/train.csv'); dv = pd.read_csv('data/dev.csv'); te = pd.read_csv('data/test_features.csv'); wug = pd.read_csv('data/wug_seeds.csv')
rules_full = build_rules(pd.concat([tr, dv]))
ext_te = pickle.load(open('runs/test_cands_v1v3.pkl', 'rb'))
trl = set(tr.lemma_ru); wl = set(wug.lemma_ru); lf = set(zip(tr.lemma_ru, tr.feats))
seg = np.where(te.lemma_ru.isin(trl), np.where([(l, f) in lf for l, f in zip(te.lemma_ru, te.feats)], 'transfer', 'complete'), np.where(te.lemma_ru.isin(wl), 'wug', 'unseen'))
inv = []; anyv = []
for i, r in enumerate(te.itertuples()):
    t = combine_tally(ext_te, i); s = max(t.items(), key=lambda x: x[1])[0]
    inv.append(not is_valid(rules_full, r.pos, r.feats, r.dialect, s)); anyv.append(any(is_valid(rules_full, r.pos, r.feats, r.dialect, c) for c in t))
te['seg'] = seg; te['inv'] = inv; te['anyv'] = anyv
print('TEST top-candidate invalid count by seg:', te.groupby('seg').inv.sum().to_dict(), ' rate:', te.groupby('seg').inv.mean().round(4).to_dict())
print('TEST invalid rows with a valid cand in beam:', te[te.inv].groupby('seg').anyv.sum().to_dict())
print(te[te.inv].groupby(['seg', 'feats']).size().sort_values(ascending=False).head(12).to_string())
