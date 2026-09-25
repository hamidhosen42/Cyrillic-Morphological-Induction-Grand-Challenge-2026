import sys, collections
import numpy as np, pandas as pd
sys.path.insert(0, 'src')
from paradigm import ACC
tr = pd.read_csv('data/train.csv'); adj = tr[tr.pos == 'ADJ'].copy()
strip = lambda s: s.replace(ACC, '')
adj['sf'] = adj.form_vvz.map(strip)
for c in ['ADJ;ACC;PL;LONG', 'ADJ;INS;PL;LONG', 'ADJ;NOM;PL;LONG', 'ADJ;LOC;PL;LONG', 'ADJ;DAT;PL;LONG', 'ADJ;GEN;PL;LONG', 'ADJ;NOM;SG;LONG', 'ADJ;LOC;SG;LONG']:
    d = adj[adj.feats == c]
    e4 = d.sf.str[-4:].value_counts(normalize=True)
    print(c, 'last4 top:', e4.head(6).round(3).to_dict())
    # ending after consonant: what is the char before the 3-suffix?
    d2 = d.assign(pre=d.sf.str[-4], e3=d.sf.str[-3:])
    ct = pd.crosstab(d2.pre, d2.e3); ct = ct[ct.sum(1) > 30]
    print('   pre-char -> 3-suffix:', {p: ct.loc[p][ct.loc[p] > 0].to_dict() for p in ct.index[:14]})
