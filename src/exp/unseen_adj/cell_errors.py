import sys, os, pickle, collections, re
import numpy as np, pandas as pd
sys.path.insert(0, 'src')
from paradigm import ParadigmModel, stress_class, n_vow, NC, ACC
from joint import combine_tally, build_masses
from validate import load_splits
train_rows, known, val = load_splits()
vu = val['unseen'].reset_index(drop=True)
ext = pickle.load(open('runs/kdev2/val_cands_aligned.pkl', 'rb'))['unseen_adj']
tallies = [combine_tally(ext, i) for i in range(len(vu))]
gold = list(vu.form_vvz); gcls = np.array([stress_class(f) for f in gold])
def pick(i, c):
    best = max(((s, q) for s, q in tallies[i].items() if stress_class(s) == c), key=lambda x: x[1], default=None)
    return best[0] if best else max(tallies[i].items(), key=lambda x: x[1])[0]
orc = [pick(i, gcls[i]) if gcls[i] >= 0 else '' for i in range(len(vu))]
vu['orc'] = [a == b for a, b in zip(orc, gold)]; vu['pred'] = orc
vu['inbeam'] = [gold[i] in tallies[i] for i in range(len(vu))]
strip = lambda s: s.replace(ACC, '')
print(vu.groupby('feats')[['orc', 'inbeam']].mean().round(3).sort_values('orc').to_string())
# ending analysis per cell: gold ending (last 3 chars stripped) vs pred ending
def ending_stats(cell, k=3):
    d = vu[vu.feats == cell]
    ge = d.form_vvz.map(lambda s: strip(s)[-k:]); pe = d.pred.map(lambda s: strip(s)[-k:])
    print(cell, 'gold end', ge.value_counts().head(5).to_dict(), '| pred end', pe.value_counts().head(5).to_dict())
for c in ['ADJ;NOM;PL;LONG', 'ADJ;GEN;PL;LONG', 'ADJ;ACC;PL;LONG', 'ADJ;DAT;SG;LONG', 'ADJ;INS;SG;LONG', 'ADJ;LOC;SG;LONG']:
    ending_stats(c)
# train truth: for NOM;PL;LONG what fraction ends in иии / ыии etc, per dialect
tra = train_rows[train_rows.pos == 'ADJ']
for c in ['ADJ;NOM;PL;LONG', 'ADJ;NOM;PL;SHORT']:
    d = tra[tra.feats == c]
    print('TRAIN', c, d.groupby('dialect').form_vvz.apply(lambda s: s.map(lambda x: strip(x)[-3:]).value_counts(normalize=True).head(4).round(3).to_dict()).to_dict())
# error categories on orc-miss rows: geminate/triple drop? compare stripped forms
miss = vu[~vu.orc]
cat = collections.Counter()
for r in miss.itertuples():
    g, p = strip(r.form_vvz), strip(r.pred)
    if g == p: cat['stress-only (same segments)'] += 1
    elif len(g) - len(p) == 1 and any(g[:i] + g[i+1:] == p for i in range(len(g))): cat['pred drops 1 char'] += 1
    elif len(p) - len(g) == 1 and any(p[:i] + p[i+1:] == g for i in range(len(p))): cat['pred adds 1 char'] += 1
    elif len(g) == len(p) and sum(a != b for a, b in zip(g, p)) == 1: cat['1 substitution'] += 1
    elif r.lemma_ru[:2] not in g[:3]: cat['gold is random word (noise)'] += 1
    else: cat['other'] += 1
print('oracle-miss categories', dict(cat), 'of', len(miss))
# dropped char details
drops = collections.Counter()
for r in miss.itertuples():
    g, p = strip(r.form_vvz), strip(r.pred)
    if len(g) - len(p) == 1:
        for i in range(len(g)):
            if g[:i] + g[i+1:] == p:
                drops[(r.feats, g[max(0,i-1):i+2], r.dialect)] += 1; break
print('top dropped-char contexts', drops.most_common(25))
