"""Reproduce candmerge.py --mode dev decision (base + local beams) and cache everything.
  python3 src/exp/dialect_consistency/cache_base.py --ext runs/kdev2/val_cands_aligned.pkl --models runs/loc_dev,runs/loc_dev_s1 --out runs/exp/dialect_consistency/cache.pkl
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
p.add_argument('--beam', type=int, default=4)
p.add_argument('--K', type=int, default=300)
p.add_argument('--device', default='cpu')
p.add_argument('--limit', type=int, default=0)
p.add_argument('--segs', default='wug,unseen,transfer,complete')
args = p.parse_args()
args.mode = 'dev'
device = torch.device('cpu')
torch.set_num_threads(max(1, os.cpu_count() // 2))

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


def decide(df, cands_per_model):
    rows = list(df.itertuples())
    tallies = [dict(combine_tally(cands_per_model, i)) for i in range(len(rows))]
    masses = build_masses(tallies)
    groups = collections.defaultdict(list)
    for i, r in enumerate(rows): groups[(r.lemma_ru, r.feats)].append(i)
    cls = [0] * len(rows); priors = [None] * len(rows)
    for (lemma, feats), idx in groups.items():
        r0 = rows[idx[0]]
        lam = args.lam if lemma in train_lemmas else args.lam_new
        obs = [(f, fm) for f, d, fm in known_map.get(lemma, []) if not (f == feats and any(d == rows[i].dialect for i in idx))]
        prior = sum(pm.class_dist(r0.pos, feats, obs) for pm in pms) / len(pms)
        sc = sum(np.log(masses[i]) for i in idx) + lam * np.log(prior + 1e-9)
        c = int(sc.argmax())
        for i in idx: cls[i] = c; priors[i] = prior
    base = []
    for i, r in enumerate(rows):
        c = cls[i]
        best = max(((s, q) for s, q in tallies[i].items() if stress_class(s) == c), key=lambda x: x[1], default=None)
        base.append(best[0] if best else max(tallies[i].items(), key=lambda x: x[1])[0])
    lb = None
    if models:
        srcs = []
        for i, r in enumerate(rows):
            cands = [c for c in known_map.get(r.lemma_ru, []) if not (c[0] == r.feats and c[1] == r.dialect)]
            srcs.append(vocab.enc(make_input(r.lemma_ru, r.feats, r.dialect, r.yat_flag, r.lemma_frequency, cls[i], cands)))
        t0 = time.time(); lb = [dict(x) for x in local_beams(srcs)]; print(f'  local beams {len(srcs)} rows x {len(models)} models in {time.time() - t0:.0f}s', flush=True)
    return dict(tallies=tallies, masses=masses, cls=cls, priors=priors, base=base, loc=lb)


ext = pickle.load(open(args.ext, 'rb'))
KEY = {'transfer': 'dev_transfer', 'complete': 'dev_complete', 'wug': 'wug', 'unseen': 'unseen_adj'}
cache = {}
for seg in [s for s in args.segs.split(",") if s]:
    df = val[seg]
    if args.limit: df = df.head(args.limit)
    cpm = ext[KEY[seg]]
    if args.limit: cpm = [m[:args.limit] for m in cpm]
    t0 = time.time()
    res = decide(df, cpm)
    res['df'] = df.reset_index(drop=True)
    cache[seg] = res
    print(f'{seg:9s} base EM {np.mean([a == c for a, c in zip(res["base"], df.form_vvz)]):.4f}  ({time.time()-t0:.0f}s)', flush=True)
    pickle.dump(cache, open(args.out, 'wb'))
print('saved', args.out)
