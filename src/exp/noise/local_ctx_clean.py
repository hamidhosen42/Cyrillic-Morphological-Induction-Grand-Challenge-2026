"""On holdout transfer/complete rows of NOISY lemmas: does removing detected substitution rows from the
context (paradigm prior obs + local class-conditioned model context) improve EM?  Replicates candmerge.decide
for a subset of rows.  Usage: python3 src/exp/noise/local_ctx_clean.py --models runs/loc_dev,runs/loc_dev_s1
"""
import argparse, os, sys, pickle, collections, time, math
import numpy as np, pandas as pd, torch
SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')); ROOT = os.path.dirname(SRC); sys.path.insert(0, SRC)
from paradigm import ParadigmModel, stress_class, NC
from joint import combine_tally, build_masses
from model import Seq2Seq
from validate import load_splits, DATA

p = argparse.ArgumentParser()
p.add_argument('--models', default='runs/loc_dev,runs/loc_dev_s1')
p.add_argument('--ext', default='runs/kdev2/val_cands_aligned.pkl')
p.add_argument('--noise', default='runs/exp/noise/train_noise.csv')
p.add_argument('--w_loc', type=float, default=0.5)
p.add_argument('--beam', type=int, default=4)
p.add_argument('--lam', type=float, default=2.0)
p.add_argument('--subset', default='noisy')   # noisy | all
p.add_argument('--limit', type=int, default=0)
args = p.parse_args()
device = torch.device('cpu')
src_code = open(os.path.join(SRC, 'local_train.py')).read()
exec(src_code[src_code.index("ACC = "):src_code.index("# ---------------- splits")], globals())

tr = pd.read_csv(f'{DATA}/train.csv'); nz = pd.read_csv(args.noise)
train_rows, known, val = load_splits()
sub_ids = set(tr.id[nz.sub2.values]); noisy_lemmas = set(tr.lemma_ru[nz.noisylemma.values])
wug_lemmas = set(val['wug'].lemma_ru)
km_noisy, km_clean = {}, {}
for i, l, f, d, fm in zip(known.id, known.lemma_ru, known.feats, known.dialect, known.form_vvz):
    km_noisy.setdefault(l, []).append((f, d, fm))
    if i not in sub_ids or l in wug_lemmas:
        km_clean.setdefault(l, []).append((f, d, fm))
pms = [ParadigmModel(K=300, seed=s).fit(train_rows) for s in (0, 1, 2)]

models, vocab = [], None
for md in args.models.split(','):
    meta = pickle.load(open(f'{md}/meta.pkl', 'rb')); ta = meta['args']
    if vocab is None: vocab = meta['vocab']; args.max_ctx = ta['max_ctx']
    m = Seq2Seq(len(vocab), ta['d_model'], 4, ta['layers'], ta['layers'], ta['ff'], 0.0, max_len=512).to(device)
    m.load_state_dict(torch.load(f'{md}/model_last.pt', map_location=device)); m.eval(); models.append((m, ta))


def pad(seqs):
    S = max(map(len, seqs)); s = torch.zeros(len(seqs), S, dtype=torch.long)
    for i, a in enumerate(seqs): s[i, :len(a)] = torch.tensor(a)
    return s


@torch.no_grad()
def local_beams(srcs, bs=128):
    out = [collections.defaultdict(float) for _ in srcs]
    order = np.argsort([len(s) for s in srcs])
    for m, _ in models:
        for i in range(0, len(srcs), bs):
            idx = order[i:i + bs]
            s = pad([srcs[j] for j in idx]).to(device)
            ys, sc = m.beam(s, vocab.stoi[BOS], vocab.stoi[EOS], beam=args.beam, return_all=True)
            ys = ys.cpu().tolist(); sc = sc.cpu().numpy()
            for j, yy, cc in zip(idx, ys, sc):
                pr = np.exp(cc - cc.max()); pr /= pr.sum()
                for y, q in zip(yy, pr): out[j][vocab.dec(y)] += float(q) / len(models)
    return out


def run(rows, cpm_rows, km_prior, km_local):
    tallies = [combine_tally(cpm_rows, i) for i in range(len(rows))]; masses = build_masses(tallies)
    groups = collections.defaultdict(list)
    for i, r in enumerate(rows): groups[(r.lemma_ru, r.feats)].append(i)
    cls = [0] * len(rows)
    for (lemma, feats), idx in groups.items():
        r0 = rows[idx[0]]
        obs = [(f, fm) for f, d, fm in km_prior.get(lemma, []) if not (f == feats and any(d == rows[i].dialect for i in idx))]
        prior = sum(pm.class_dist(r0.pos, feats, obs) for pm in pms) / len(pms)
        c = int((sum(np.log(masses[i]) for i in idx) + args.lam * np.log(prior + 1e-9)).argmax())
        for i in idx: cls[i] = c
    base = []
    for i, r in enumerate(rows):
        best = max(((s, q) for s, q in tallies[i].items() if stress_class(s) == cls[i]), key=lambda x: x[1], default=None)
        base.append(best[0] if best else max(tallies[i].items(), key=lambda x: x[1])[0])
    srcs = []
    for i, r in enumerate(rows):
        cands = [c for c in km_local.get(r.lemma_ru, []) if not (c[0] == r.feats and c[1] == r.dialect)]
        srcs.append(vocab.enc(make_input(r.lemma_ru, r.feats, r.dialect, r.yat_flag, r.lemma_frequency, cls[i], cands)))
    lb = local_beams(srcs)
    merged = []
    for i, r in enumerate(rows):
        pool = collections.defaultdict(float)
        for s, q in tallies[i].items():
            if stress_class(s) == cls[i]: pool[s] += q
        for s, q in lb[i].items():
            if stress_class(s) == cls[i]: pool[s] += args.w_loc * q
        merged.append(max(pool.items(), key=lambda x: x[1])[0] if pool else base[i])
    return base, merged


ext = pickle.load(open(args.ext, 'rb'))
KEY = {'transfer': 'dev_transfer', 'complete': 'dev_complete'}
for seg in ['transfer', 'complete']:
    df = val[seg].reset_index(drop=True); cpm = ext[KEY[seg]]
    sel = np.where(df.lemma_ru.isin(noisy_lemmas).values)[0] if args.subset == 'noisy' else np.arange(len(df))
    if args.limit: sel = sel[:args.limit]
    rows = [r for r in df.iloc[sel].itertuples()]; cpm_rows = [[m[i] for i in sel] for m in cpm]
    gold = [r.form_vvz for r in rows]
    t0 = time.time()
    b_nn, m_nn = run(rows, cpm_rows, km_noisy, km_noisy)   # current pipeline
    b_cn, m_cn = run(rows, cpm_rows, km_clean, km_noisy)   # clean prior ctx only
    b_cc, m_cc = run(rows, cpm_rows, km_clean, km_clean)   # clean prior + clean local ctx
    em = lambda pr: np.mean([a == g for a, g in zip(pr, gold)])
    print(f'{seg} subset={args.subset} n={len(rows)} ({time.time() - t0:.0f}s)  base: noisy {em(b_nn):.4f} clean-prior {em(b_cn):.4f} | merged(w_loc={args.w_loc}): noisy/noisy {em(m_nn):.4f} clean/noisy {em(m_cn):.4f} clean/clean {em(m_cc):.4f}', flush=True)
    # gold-noise breakdown
    dvn = pd.read_csv('runs/exp/noise/dev_noise2.csv'); dsub = set(dvn.id[dvn.sub2.values])
    cg = np.array([r.id not in dsub for r in rows])
    emc = lambda pr: np.mean([a == g for a, g, c in zip(pr, gold, cg) if c])
    print(f'   clean-gold rows n={cg.sum()}: base noisy {emc(b_nn):.4f} clean-prior {emc(b_cn):.4f} | merged noisy/noisy {emc(m_nn):.4f} clean/noisy {emc(m_cn):.4f} clean/clean {emc(m_cc):.4f}')
