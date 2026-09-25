"""(c): sparse logistic regression: seed-feature keys (as in the best lookup) +/- lemma & seed char n-grams (crossed with target cell).
Train on train seed rows, eval on wug holdout; GroupKFold CV on train for tighter CI."""
import sys, os; sys.path.insert(0, os.path.dirname(__file__))
from common import *
from feats_ab_lib import featurize
from sklearn.feature_extraction import FeatureHasher
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
import scipy.sparse as sp

train_rows, known, wug_df, seeds = get_splits()
B = pickle.load(open(os.path.join(CACHE, 'baseline_wug.pkl'), 'rb'))
pr_w, mass_w, gold_w = B['pr'], B['mass'], B['gold']
trows, tseeds = train_seed_table(train_rows); gold_t = trows.cls.values
Ft = featurize(trows, tseeds); Fw = featurize(wug_df, seeds)


def ngrams(s, n_max=3, tag=''):
    s = '^' + s + '$'
    return [f'{tag}{n}:{s[i:i+n]}' for n in range(1, n_max + 1) for i in range(len(s) - n + 1)]


def tokens(F, df, seedmap, use_ngram, use_seed_ngram, use_keys=True):
    out = []
    for i, r in enumerate(df.itertuples()):
        f = F.iloc[i]; cell = f'{r.feats}|{r.dialect}'
        t = []
        if use_keys:
            t += [f'k1:{cell}|{f.seed_cls}', f'k2:{cell}|{f.seed_si}|{f.seed_nv}|{f.seed_end}',
                  f'k3:{cell}|{f.seed_si}|{f.seed_nv}|{f.lem_nv}|{f.seed_end}', f'k4:{cell}|{f.seed_end}',
                  f'k5:{r.feats}|{f.seed_si}|{f.seed_nv}|{f.seed_end}', f'k6:{cell}|{f.seed_si}|{f.seed_nv}']
        if use_ngram:
            g = ngrams(r.lemma_ru, 3, 'L')
            t += g + [f'{cell}|{x}' for x in g]
        if use_seed_ngram:
            g = ngrams(seedmap[r.lemma_ru], 3, 'S')
            t += g + [f'{cell}|{x}' for x in g]
        out.append(t)
    return out


H = FeatureHasher(n_features=2 ** 21, input_type='string', alternate_sign=False)
gkf = GroupKFold(5)
for name, (ng, sg) in {'LR keys only': (False, False), 'LR keys + lemma ngrams': (True, False),
                       'LR keys + seed ngrams': (False, True), 'LR keys + lemma + seed ngrams': (True, True)}.items():
    Xt = H.transform(tokens(Ft, trows, tseeds, ng, sg)); Xw = H.transform(tokens(Fw, wug_df, seeds, ng, sg))
    for C in [0.3, 1.0]:
        clf = LogisticRegression(C=C, max_iter=300, solver='saga', tol=1e-3).fit(Xt, gold_t)
        p = clf.predict_proba(Xw); pw = np.full((len(Xw.shape and [0] * Xw.shape[0]), NC), 1e-4)
        for j, c in enumerate(clf.classes_): pw[:, int(c)] = p[:, j]
        pw /= pw.sum(1, keepdims=True)
        # CV on train (2 folds only to save time)
        cv = []
        for k, (a, b) in enumerate(gkf.split(Xt, groups=trows.lemma_ru)):
            if k >= 2: break
            cl = LogisticRegression(C=C, max_iter=300, solver='saga', tol=1e-3).fit(Xt[a], gold_t[a])
            cv.append(float(np.mean(cl.predict(Xt[b]) == gold_t[b])))
        ok = gold_w >= 0
        print(f'{name:32s} C={C}: wug prior-acc {acc(pw[ok], gold_w[ok]):.4f} comb-acc {comb_acc(mass_w[ok], pw[ok], gold_w[ok]):.4f} | train-CV(2 folds) {np.mean(cv):.4f} n_cv={sum(len(b) for k,(a,b) in enumerate(gkf.split(Xt, groups=trows.lemma_ru)) if k<2)}', flush=True)
