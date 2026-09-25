"""feature builders shared by the wug-squeeze experiments"""
import sys, os; sys.path.insert(0, os.path.dirname(__file__))
from common import *

def endtype(seed, pos):
    p = plain(seed)
    if pos == 'N':
        last = p[-1]
        return 'V:' + last if last in VOW else ('J:' + last if last in 'ъь' else 'C')
    # verb: theme vowel before the last consonant cluster
    vs = [c for c in p if c in VOW]
    return p[-3:] + '|' + (vs[-1] if vs else '')


def featurize(df, seedmap):
    F = pd.DataFrame(index=df.index)
    sd = df.lemma_ru.map(seedmap)
    F['feats'] = df.feats.values; F['dialect'] = df.dialect.values; F['pos'] = df.pos.values
    F['seed_cls'] = sd.map(stress_class).values
    F['seed_si'] = sd.map(stress_idx).values
    F['seed_nv'] = sd.map(n_vow).values
    F['lem_nv'] = df.lemma_ru.map(n_vow).values
    F['dnv'] = F.seed_nv - F.lem_nv
    F['seed_end'] = [endtype(s, p) for s, p in zip(sd, df.pos)]
    F['seed_end2'] = sd.map(lambda s: plain(s)[-2:]).values
    F['lem_end2'] = df.lemma_ru.str[-2:].values
    F['yat'] = df.yat_flag.values
    F['seed_from_end'] = F.seed_nv - 1 - F.seed_si
    return F


