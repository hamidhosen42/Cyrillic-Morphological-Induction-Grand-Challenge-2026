"""Stress-class prior for NEW lemmas (wug: one SEV seed form; unseen ADJ: nothing) via GBM on
seed/lemma features. Returns P(class | features) to be used instead of the paradigm prior."""
import numpy as np, pandas as pd, collections
from sklearn.ensemble import HistGradientBoostingClassifier
from paradigm import stress_class, stress_idx, n_vow, NC, ACC

SEED_F = {'N': 'N;ACC;SG', 'V': 'V;PRS;3;SG'}


def _feats(lemma, feats, dialect, yat, freq, seed=None):
    parts = feats.split(';')
    f = {
        'lem_nv': n_vow(lemma), 'lem_len': len(lemma),
        'last1': lemma[-1], 'last2': lemma[-2:], 'last3': lemma[-3:], 'first1': lemma[0],
        'yat': int(yat), 'logf': np.log10(freq), 'feats': feats, 'dialect': dialect,
        'num': 'SG' if 'SG' in parts else ('PL' if 'PL' in parts else '-'),
        'ls': 'LONG' if 'LONG' in parts else ('SHORT' if 'SHORT' in parts else '-'),
        'case': parts[1] if len(parts) > 1 else '-',
    }
    if seed is not None:
        f.update({'seed_cls': stress_class(seed), 'seed_si': stress_idx(seed), 'seed_nv': n_vow(seed),
                  'seed_len': len(seed), 'seed_end': seed.replace(ACC, '')[-1], 'seed_end2': seed.replace(ACC, '')[-2:]})
    return f


class NewLemmaPrior:
    def __init__(self):
        self.models = {}
        self.cats = {}

    def _frame(self, rows, pos, fit=False):
        df = pd.DataFrame(rows)
        strcols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
        if fit:
            self.cats[pos] = {}
            for c in strcols:
                top = df[c].value_counts().index[:250].tolist()
                self.cats[pos][c] = {v: i + 1 for i, v in enumerate(top)}  # 0 = other
        for c in strcols:
            m = self.cats[pos][c]
            df[c] = df[c].map(lambda v: m.get(v, 0)).astype(int)
        return df, strcols

    def fit(self, known):
        """known: rows with lemma_ru,pos,feats,dialect,yat_flag,lemma_frequency,form_vvz."""
        known = known[known.form_vvz.map(stress_class) >= 0]
        for pos in ['N', 'V', 'ADJ']:
            d = known[known.pos == pos]
            rows, y = [], []
            if pos in SEED_F:
                seeds = d[(d.feats == SEED_F[pos]) & (d.dialect == 'SEV')].drop_duplicates('lemma_ru').set_index('lemma_ru').form_vvz
                d = d[d.lemma_ru.isin(seeds.index) & (d.feats != SEED_F[pos])]
                for r in d.itertuples():
                    rows.append(_feats(r.lemma_ru, r.feats, r.dialect, r.yat_flag, r.lemma_frequency, seeds[r.lemma_ru]))
                    y.append(stress_class(r.form_vvz))
            else:
                for r in d.itertuples():
                    rows.append(_feats(r.lemma_ru, r.feats, r.dialect, r.yat_flag, r.lemma_frequency))
                    y.append(stress_class(r.form_vvz))
            X, strcols = self._frame(rows, pos, fit=True); y = np.array(y)
            catidx = [list(X.columns).index(c) for c in strcols]
            clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06, max_leaf_nodes=31,
                                                 categorical_features=catidx, early_stopping=False, random_state=0)
            clf.fit(X.values, y)
            self.models[pos] = (clf, list(X.columns))
        return self

    def predict(self, pos, lemma, feats, dialect, yat, freq, seed=None):
        return self.predict_many(pos, [(lemma, feats, dialect, yat, freq, seed)])[0]

    def predict_many(self, pos, items):
        """items: list of (lemma, feats, dialect, yat, freq, seed). Returns (n, NC)."""
        clf, cols = self.models[pos]
        X, _ = self._frame([_feats(*it) for it in items], pos)
        pr = clf.predict_proba(X[cols].values)
        out = np.full((len(items), NC), 1e-4)
        for j, c in enumerate(clf.classes_):
            out[:, int(c)] = pr[:, j]
        return out / out.sum(1, keepdims=True)
