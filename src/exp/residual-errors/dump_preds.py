"""Dump per-row predictions + intermediates of the current best pipeline (candmerge, w_loc=0.5) on the holdout.
Mirrors src/candmerge.py decide() exactly; nothing under src/ is modified.

  python3 src/exp/residual-errors/dump_preds.py --ext runs/kdev2/val_cands_aligned.pkl \
      --models runs/loc_dev,runs/loc_dev_s1 --device cpu --out runs/exp/residual-errors/preds.pkl
"""
import argparse, os, sys, pickle, collections, math, time
import numpy as np, pandas as pd, torch
SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
sys.path.insert(0, SRC)
from paradigm import ParadigmModel, stress_class, NC
from joint import combine_tally, build_masses
from model import Seq2Seq
from validate import load_splits, weighted_score, DATA

p = argparse.ArgumentParser()
p.add_argument('--ext', required=True)
p.add_argument('--models', default='')
p.add_argument('--out', required=True)
p.add_argument('--lam', type=float, default=2.0)
p.add_argument('--lam_new', type=float, default=2.0)
p.add_argument('--w_loc', type=float, default=0.5)
p.add_argument('--beam', type=int, default=4)
p.add_argument('--K', type=int, default=300)
p.add_argument('--device', default='cpu')
p.add_argument('--limit', type=int, default=0)
p.add_argument('--threads', type=int, default=8)
p.add_argument('--seg', default='')
p.add_argument('--offset', type=int, default=0)
args = p.parse_args()
args.mode = 'dev'
torch.set_num_threads(args.threads)
device = torch.device(args.device if (args.device != 'mps' or torch.backends.mps.is_available()) else 'cpu')

src_code = open(os.path.join(SRC, 'local_train.py')).read()
exec(src_code[src_code.index("ACC = "):src_code.index("# ---------------- splits")], globals())

train_rows, known, val = load_splits(); pm_fit = train_rows
known_map = {}
for l, f, d, fm in zip(known.lemma_ru, known.feats, known.dialect, known.form_vvz):
    known_map.setdefault(l, []).append((f, d, fm))
train_lemmas = set(pm_fit.lemma_ru)
pms = [ParadigmModel(K=args.K, seed=s).fit(pm_fit) for s in (0, 1, 2)]

model_dirs = [m for m in args.models.split(',') if m]
models, vocab = [], None
for md in model_dirs:
    meta = pickle.load(open(f'{md}/meta.pkl', 'rb')); ta = meta['args']
    if vocab is None:
        vocab = meta['vocab']; args.max_ctx = ta['max_ctx']
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
                for y, q in zip(yy, pr):
                    out[j][vocab.dec(y)] += float(q) / len(models)
    return out


def decide(df, cands_per_model, seg):
    rows = list(df.itertuples())
    tallies = [combine_tally(cands_per_model, i) for i in range(len(rows))]
    masses = build_masses(tallies)
    groups = collections.defaultdict(list)
    for i, r in enumerate(rows): groups[(r.lemma_ru, r.feats)].append(i)
    cls = [0] * len(rows); priors = [None] * len(rows); nobs = [0] * len(rows)
    for (lemma, feats), idx in groups.items():
        r0 = rows[idx[0]]
        lam = args.lam if lemma in train_lemmas else args.lam_new
        obs = [(f, fm) for f, d, fm in known_map.get(lemma, []) if not (f == feats and any(d == rows[i].dialect for i in idx))]
        prior = sum(pm.class_dist(r0.pos, feats, obs) for pm in pms) / len(pms)
        sc = sum(np.log(masses[i]) for i in idx) + lam * np.log(prior + 1e-9)
        c = int(sc.argmax())
        for i in idx: cls[i] = c; priors[i] = prior; nobs[i] = len(obs)
    base = []
    for i, r in enumerate(rows):
        c = cls[i]
        best = max(((s, q) for s, q in tallies[i].items() if stress_class(s) == c), key=lambda x: x[1], default=None)
        base.append(best[0] if best else max(tallies[i].items(), key=lambda x: x[1])[0])
    srcs = []; lb = [collections.defaultdict(float) for _ in rows]
    if models:
        for i, r in enumerate(rows):
            cands = [c for c in known_map.get(r.lemma_ru, []) if not (c[0] == r.feats and c[1] == r.dialect)]
            srcs.append(vocab.enc(make_input(r.lemma_ru, r.feats, r.dialect, r.yat_flag, r.lemma_frequency, cls[i], cands)))
        t0 = time.time(); lb = local_beams(srcs); print(f'  {seg}: local beams {len(srcs)} rows x {len(models)} models in {time.time() - t0:.0f}s', flush=True)
    w = args.w_loc
    recs = []
    for i, r in enumerate(rows):
        c = cls[i]
        pool = collections.defaultdict(float)
        for s, q in tallies[i].items():
            if stress_class(s) == c: pool[s] += q
        for s, q in lb[i].items():
            if stress_class(s) == c: pool[s] += w * q
        merged = max(pool.items(), key=lambda x: x[1])[0] if pool else base[i]
        gold = r.form_vvz
        recs.append(dict(seg=seg, i=i + args.offset, lemma=r.lemma_ru, pos=r.pos, feats=r.feats, dialect=r.dialect,
                         yat=int(r.yat_flag), freq=float(r.lemma_frequency), gold=gold, base=base[i], pred=merged,
                         cls=c, gold_cls=stress_class(gold), prior=priors[i], mass=masses[i], nobs=nobs[i],
                         tally=dict(tallies[i]), loc=dict(lb[i]), pool=dict(pool),
                         gold_in_tally=gold in tallies[i], gold_in_loc=gold in lb[i], gold_in_pool=gold in pool))
    return recs


ext = pickle.load(open(args.ext, 'rb'))
KEY = {'transfer': 'dev_transfer', 'complete': 'dev_complete', 'wug': 'wug', 'unseen': 'unseen_adj'}
all_recs = []
for seg, df in val.items():
    if args.seg and seg not in args.seg.split(','): continue
    cpm = ext[KEY[seg]]
    if args.offset: df = df.iloc[args.offset:]; cpm = [m[args.offset:] for m in cpm]
    if args.limit: df = df.head(args.limit); cpm = [m[:args.limit] for m in cpm]
    recs = decide(df, cpm, seg)
    all_recs += recs
    print(f'{seg:9s} n={len(recs)} base {np.mean([r["base"] == r["gold"] for r in recs]):.4f} merged {np.mean([r["pred"] == r["gold"] for r in recs]):.4f}', flush=True)
pb = collections.defaultdict(list); pm_ = collections.defaultdict(list); golds = collections.defaultdict(list)
for r in all_recs:
    pb[r['seg']].append(r['base']); pm_[r['seg']].append(r['pred']); golds[r['seg']].append(r['gold'])
print('BASE  ', weighted_score(pb, golds))
print('MERGED', weighted_score(pm_, golds))
pickle.dump(all_recs, open(args.out, 'wb'))
print('wrote', args.out, len(all_recs))
