"""Helper for the full pipeline: drop detected substitution rows from the observation context.
    from exp.noise.clean_known import clean_known
    known = clean_known(pd.concat([tr, dv, wug]))     # wug seeds have no flags and are always kept
Flags: runs/exp/noise/sub_flags_traindev.csv (source in {train, dev}, id, sub2). Row-level removal only;
removing whole noisy lemmas hurts (measured: -0.0026 weighted holdout)."""
import os, pandas as pd
FLAGS = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'runs', 'exp', 'noise', 'sub_flags_traindev.csv')


def clean_known(known, tr_ids=None, dv_ids=None):
    fl = pd.read_csv(FLAGS)
    bad = set(zip(fl.source[fl.sub2], fl.lemma_ru[fl.sub2], fl.feats[fl.sub2], fl.dialect[fl.sub2], fl.form_vvz[fl.sub2]))
    keep = [not any((s, l, f, d, fm) in bad for s in ('train', 'dev')) for l, f, d, fm in zip(known.lemma_ru, known.feats, known.dialect, known.form_vvz)]
    return known[keep]
