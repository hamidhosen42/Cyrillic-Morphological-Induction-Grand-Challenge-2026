"""Candidate-level merge: Kaggle beam candidates + local class-conditioned beams.

For every row:
  1. class decision (joint per (lemma, feats)) from Kaggle class mass + paradigm prior
  2. candidate pool = Kaggle candidates of that class  ∪  local beams decoded with that class token
  3. pick argmax of  w_k·(Kaggle mass) + w_l·(mean local prob)

  python candmerge.py --mode dev  --ext runs/kdev2/val_cands_aligned.pkl --models runs/loc_dev,runs/loc_dev_s1
  python candmerge.py --mode full --ext runs/test_cands_v1v3.pkl --models runs/loc_full,runs/loc_full_s1 --out runs/sub.csv
"""
import argparse, os, sys, pickle, collections, math, time
import numpy as np, pandas as pd, torch
sys.path.insert(0, os.path.dirname(__file__))
from paradigm import ParadigmModel, stress_class, NC
from joint import combine_tally, build_masses
from model import Seq2Seq
from validate import load_splits, weighted_score, DATA

p = argparse.ArgumentParser()
p.add_argument('--mode', default='dev')
p.add_argument('--ext', required=True)
p.add_argument('--models', default='')
p.add_argument('--out', default=None)
p.add_argument('--lam', type=float, default=2.0)
p.add_argument('--lam_new', type=float, default=2.0)
p.add_argument('--w_loc', default='0.5')
p.add_argument('--beam', type=int, default=4)
p.add_argument('--K', type=int, default=300)
p.add_argument('--device', default='cpu')
p.add_argument('--limit', type=int, default=0)
args = p.parse_args()
device = torch.device(args.device if (args.device != 'mps' or torch.backends.mps.is_available()) else 'cpu')

src_code = open(os.path.join(os.path.dirname(__file__), 'local_train.py')).read()
exec(src_code[src_code.index("ACC = "):src_code.index("# ---------------- splits")], globals())

tr = pd.read_csv(f'{DATA}/train.csv'); dv = pd.read_csv(f'{DATA}/dev.csv')
wug = pd.read_csv(f'{DATA}/wug_seeds.csv'); te = pd.read_csv(f'{DATA}/test_features.csv'); wug['lemma_frequency'] = 60.0
if args.mode == 'dev':
    train_rows, known, val = load_splits(); pm_fit = train_rows
else:
    known = pd.concat([tr, dv, wug]); pm_fit = pd.concat([tr, dv]); val = {'test': te}
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
    """returns list over rows of {form: prob} averaged over local models"""
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
    tallies = [combine_tally(cands_per_model, i) for i in range(len(rows))]
    masses = build_masses(tallies)
    groups = collections.defaultdict(list)
    for i, r in enumerate(rows): groups[(r.lemma_ru, r.feats)].append(i)
    cls = [0] * len(rows)
    for (lemma, feats), idx in groups.items():
        r0 = rows[idx[0]]
        lam = args.lam if lemma in train_lemmas else args.lam_new
        obs = [(f, fm) for f, d, fm in known_map.get(lemma, []) if not (f == feats and any(d == rows[i].dialect for i in idx))]
        prior = sum(pm.class_dist(r0.pos, feats, obs) for pm in pms) / len(pms)
        sc = sum(np.log(masses[i]) for i in idx) + lam * np.log(prior + 1e-9)
        c = int(sc.argmax())
        for i in idx: cls[i] = c
    base = []
    for i, r in enumerate(rows):
        c = cls[i]
        best = max(((s, q) for s, q in tallies[i].items() if stress_class(s) == c), key=lambda x: x[1], default=None)
        base.append(best[0] if best else max(tallies[i].items(), key=lambda x: x[1])[0])
    if not models:
        return base, base
    srcs = []
    for i, r in enumerate(rows):
        cands = [c for c in known_map.get(r.lemma_ru, []) if not (c[0] == r.feats and c[1] == r.dialect and args.mode == 'dev')]
        srcs.append(vocab.enc(make_input(r.lemma_ru, r.feats, r.dialect, r.yat_flag, r.lemma_frequency, cls[i], cands)))
    t0 = time.time(); lb = local_beams(srcs); print(f'  local beams {len(srcs)} rows x {len(models)} models in {time.time() - t0:.0f}s', flush=True)
    merged = {}
    for w in [float(x) for x in str(args.w_loc).split(',')]:
        mm = []
        for i, r in enumerate(rows):
            c = cls[i]
            pool = collections.defaultdict(float)
            for s, q in tallies[i].items():
                if stress_class(s) == c: pool[s] += q
            for s, q in lb[i].items():
                if stress_class(s) == c: pool[s] += w * q
            mm.append(max(pool.items(), key=lambda x: x[1])[0] if pool else base[i])
        merged[w] = mm
    return base, merged


ext = pickle.load(open(args.ext, 'rb'))
KEY = {'transfer': 'dev_transfer', 'complete': 'dev_complete', 'wug': 'wug', 'unseen': 'unseen_adj'}
if args.mode == 'dev':
    pb, pm_, golds = {}, {}, {}
    for seg, df in val.items():
        if args.limit: df = df.head(args.limit)
        cpm = ext[KEY[seg]]
        if args.limit: cpm = [m[:args.limit] for m in cpm]
        b, mg = decide(df, cpm)
        pb[seg] = b; golds[seg] = list(df.form_vvz)
        for w, v in mg.items(): pm_.setdefault(w, {})[seg] = v
        print(f'{seg:9s} base {np.mean([a == c for a, c in zip(b, golds[seg])]):.4f} ' +
              ' '.join(f'w{w}:{np.mean([a == c for a, c in zip(v, golds[seg])]):.4f}' for w, v in mg.items()), flush=True)
    print('BASE  ', weighted_score(pb, golds)[0])
    for w in sorted(pm_): print(f'MERGED w_loc={w}', weighted_score(pm_[w], golds))
else:
    b, mg = decide(te, ext)
    w = list(mg)[0]; out = mg[w]
    pd.DataFrame({'id': te.id, 'form_vvz': out}).to_csv(args.out, index=False)
    print('changed vs base:', int(np.sum([x != y for x, y in zip(b, out)])), 'wrote', args.out)
