"""Noise detector for labelled rows (train/dev/wug seeds).
Flags per row:
  sub     : random-word substitution (form stem unrelated to lemma AND to sibling forms)
  stressx : stress class disagrees with the same (lemma,feats) in other dialects (minority / unresolved)
  typo    : segmental disagreement with same (lemma,feats) other-dialect sibling after dialect normalisation
Outputs runs/exp/noise/<name>_noise.csv aligned to the input rows.
"""
import sys, os, collections, argparse
import numpy as np, pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein as Lev
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from paradigm import stress_class, ACC

CANON = str.maketrans({'о': 'а', 'ѣ': 'е', 'э': 'е', 'и': 'е', 'ы': 'е', 'ё': 'е', 'ѫ': 'у', 'ю': 'у', 'я': 'а', 'ъ': '', 'ь': '', 'й': ''})


def canon(s):
    s = s.replace(ACC, '').translate(CANON)
    # SEV pleophony reversal: оро->ра, ере->рѣ, оло->ла (after canon: ара->ра, ере->ре, ала->ла)
    return s


def canon_lemma(s):
    s = canon(s)
    for a, b in (('ара', 'ра'), ('ере', 'ре'), ('ала', 'ла')):
        s = s.replace(a, b)
    return s


def canon_form(s):
    s = canon(s)
    for a, b in (('ара', 'ра'), ('ере', 'ре'), ('ала', 'ла')):
        s = s.replace(a, b)
    # palatalisation / dialect consonant variants collapse
    for a, b in (('сц', 'ск'), ('ск', 'ск'), ('з', 'г'), ('ж', 'г'), ('ч', 'к'), ('ц', 'к'), ('ш', 'х'), ('с', 'х')):
        s = s.replace(a, b)
    return s


def lcp(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y: break
        n += 1
    return n


def stem_sim(form, lemma):
    """similarity of form stem to lemma: max of LCP ratio and fuzz ratio on canonicalised strings"""
    cf, cl = canon_form(form), canon_form(lemma)
    L = lcp(cf, cl)
    r1 = L / max(1, min(len(cl), len(cf)))
    r2 = fuzz.partial_ratio(cl[:max(3, len(cl) - 2)], cf) / 100.0
    return max(r1, r2)


def run(df, name):
    df = df.reset_index(drop=True).copy()
    df['cls'] = df.form_vvz.map(stress_class)
    df['nacc'] = df.form_vvz.map(lambda s: s.count(ACC))
    # --- substitution: similarity to lemma and to sibling forms of the same lemma
    sim_lem = np.array([stem_sim(f, l) for f, l in zip(df.form_vvz, df.lemma_ru)])
    by_lemma = collections.defaultdict(list)
    for i, l in enumerate(df.lemma_ru): by_lemma[l].append(i)
    sim_sib = np.zeros(len(df))
    cforms = [canon_form(f) for f in df.form_vvz]
    for l, idx in by_lemma.items():
        for i in idx:
            best = 0.0
            for j in idx:
                if j == i: continue
                a, b = cforms[i], cforms[j]
                L = lcp(a, b); r = L / max(1, min(len(a), len(b)))
                best = max(best, r)
            sim_sib[i] = best
    df['sim_lem'] = sim_lem; df['sim_sib'] = sim_sib
    df['sub'] = (sim_lem < 0.5) & (sim_sib < 0.5)
    # --- stress: compare with same (lemma,feats) other dialects
    g = df.groupby(['lemma_ru', 'feats'])
    df['n_dial'] = g.cls.transform('size')
    def stress_flag(s):
        if len(s) < 2: return pd.Series([0] * len(s), index=s.index)  # unknown
        cnt = collections.Counter(s)
        top, n = cnt.most_common(1)[0]
        if n * 2 > len(s):  # majority exists
            return pd.Series([0 if c == top else 2 for c in s], index=s.index)  # 2 = minority (noise)
        return pd.Series([1] * len(s), index=s.index)  # 1 = unresolved 2-way tie
    df['stressx'] = g.cls.transform(stress_flag) if False else pd.concat([stress_flag(s) for _, s in g.cls], axis=0).reindex(df.index)
    # --- typo: segmental disagreement with a sibling in another dialect after normalisation (excluding subs)
    cseg = [canon_form(f) for f in df.form_vvz]
    df['cseg'] = cseg
    def typo_flag(s):
        if len(s) < 2: return pd.Series([0] * len(s), index=s.index)
        cnt = collections.Counter(s)
        top, n = cnt.most_common(1)[0]
        if n * 2 > len(s):
            return pd.Series([0 if c == top else 2 for c in s], index=s.index)
        return pd.Series([1] * len(s), index=s.index)
    df['segx'] = pd.concat([typo_flag(s) for _, s in g.cseg], axis=0).reindex(df.index)
    df['typo'] = (df.segx == 2) & ~df['sub']
    out = df[['lemma_ru', 'pos', 'feats', 'dialect', 'form_vvz', 'cls', 'nacc', 'sim_lem', 'sim_sib', 'n_dial', 'sub', 'stressx', 'segx', 'typo']]
    out.to_csv(f'runs/exp/noise/{name}_noise.csv', index=False)
    print(f'{name}: n={len(df)} sub={df["sub"].mean():.4f} stress-minority={np.mean(df.stressx == 2):.4f} stress-tie={np.mean(df.stressx == 1):.4f} '
          f'seg-minority={np.mean(df.segx == 2):.4f} seg-tie={np.mean(df.segx == 1):.4f} typo(seg-minority,not sub)={df.typo.mean():.4f} nacc!=1={np.mean(df.nacc != 1):.4f}')
    return out


if __name__ == '__main__':
    tr = pd.read_csv('data/train.csv'); dv = pd.read_csv('data/dev.csv'); wug = pd.read_csv('data/wug_seeds.csv')
    a = run(tr, 'train'); b = run(dv, 'dev'); c = run(wug, 'wug')
    # histogram of sim_lem for train
    print('sim_lem hist', np.histogram(a.sim_lem, bins=[0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.01])[0])
    print('sim_sib hist', np.histogram(a.sim_sib, bins=[0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.01])[0])
    print('sub examples:'); print(a[a['sub']].head(15).to_string())
    print('borderline (0.4-0.6):'); print(a[(a.sim_lem >= 0.4) & (a.sim_lem < 0.65)].head(20).to_string())
    print('by pos sub rate', a.groupby('pos')['sub'].mean().round(4).to_dict())
    # joint train+dev pass (dev rows add dialect siblings for stress check)
    run(pd.concat([tr, dv]), 'traindev')
