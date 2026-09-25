import sys, os; sys.path.insert(0, os.path.dirname(__file__))
from common import *
train_rows, known, wug_df, seeds = get_splits()
print('wug rows', len(wug_df), 'lemmas', wug_df.lemma_ru.nunique(), wug_df.pos.value_counts().to_dict())
gold = wug_df.form_vvz.map(stress_class).values
print('gold class dist', np.bincount(gold[gold >= 0], minlength=NC) / (gold >= 0).sum(), 'noaccent', (gold < 0).sum())
pms = paradigm_models(train_rows)
pr = prior_dists(pms, wug_df, seeds)
mass, tallies = wug_masses(wug_df)
ok = gold >= 0
print(f'prior acc {acc(pr[ok], gold[ok]):.4f}  nn acc {acc(mass[ok], gold[ok]):.4f}  comb acc {comb_acc(mass[ok], pr[ok], gold[ok]):.4f} n={ok.sum()}')
for pos in ['N', 'V']:
    m = ok & (wug_df.pos == pos).values
    print(f'  {pos}: prior {acc(pr[m], gold[m]):.4f} nn {acc(mass[m], gold[m]):.4f} comb {comb_acc(mass[m], pr[m], gold[m]):.4f} n={m.sum()}')
# EM of current pipeline's pick (best cand of chosen class)
sc = np.log(mass) + 2.0 * np.log(pr + 1e-9); cls = sc.argmax(1)
pred = []
for i, t in enumerate(tallies):
    best = max(((s, q) for s, q in t.items() if stress_class(s) == cls[i]), key=lambda x: x[1], default=None)
    pred.append(best[0] if best else max(t.items(), key=lambda x: x[1])[0])
em = np.mean([p == g for p, g in zip(pred, wug_df.form_vvz)])
em_c = np.mean([p == g for p, g, c, gg in zip(pred, wug_df.form_vvz, cls, gold) if c == gg])
print(f'pipeline EM {em:.4f}; EM given class correct {em_c:.4f}')
pickle.dump(dict(pr=pr, mass=mass, gold=gold), open(os.path.join(CACHE, 'baseline_wug.pkl'), 'wb'))
