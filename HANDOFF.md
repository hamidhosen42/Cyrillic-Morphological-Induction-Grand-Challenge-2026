# HANDOFF — Cyrillic Morphological Induction Grand Challenge 2026

Last updated: 2026-09-26 ~02:00 UTC. Deadline: **2026-09-26 04:00 UTC**. 5 submissions per UTC day.

## Leaderboard (verified 2026-09-24 ~18:40 UTC)
| # | Team | Public |
|---|---|---|
| 1 | localAI | 0.69891 |
| 2 | Sepp Mair | 0.69834 |
| 3 | FOYSAL | 0.69435 |
| 8 | **Md. Hamid Hosen** | **0.68003** (`runs/final/var_complete_poe.csv`) |
| 10 | keeaitec | 0.67077 |

## 2026-09-26 public noise-copy leak (disclosed by the user on the forum, topic 743193)
Options in `src/final.py`: `--leak 1 --leak_ctx 1` (pool-matched copies: row overrides + prior observations),
`--leak_lem 85,5` (spelling-matched copies as extra observations), `--cache_tag` (separate local-beam cache per run).
| holdout variant | score |
|---|---|
| no leak | 0.65981 |
| overrides only | 0.66314 |
| overrides + observations | 0.67310 |
| + spelling match 85,5 | 0.67442 |
Submitted: `runs/final/sub_leak_ctx.csv` (3 local models) and `runs/final/sub_leak5_lem_b.csv` (5 models + spelling match).
| + prior weight 5 for unseen lemmas with leaked obs (`--obs_cfg`) | 0.67482 |
| (tried, no gain) leaked forms as local-model context `--leak_stem 1` | 0.67445 |
| (tried, worse) 3way λ5 joint for all unseen | 0.67067 |
Also submitted `runs/final/sub_leak5_obs.csv` (last, best holdout).
Suggested finals: `sub_leak5_obs.csv` + `var_complete_poe.csv` (no-leak hedge).

## 2026-09-25 holdout checks
- When the stress class is right, EM is .988 transfer / .990 complete / .977 wug / .964 unseen.
  Remaining error is almost all the stress-class decision for new lemmas (class right: wug .505, unseen .416).
- KAM is not weaker on holdout (wug KAM .519 vs SEV .474), so the KAM-heavy test mix does not explain the gap.
- Test wug cells are sampled uniformly (seed cell excluded in SEV only); no selection signal.
- v4 (5-model pooling) auto-builds and submits via `runs/final/auto_v4.sh` after `runs/loc_full_s5` finishes (~08:30 UTC).

## Metric (measured via probes)
`Score = 0.9*WeightedEM + 0.1*(1-WeightedCER)`; row weights by segment: transfer 1, complete 2, wug 3, unseen 4
(test rows 18000 / 15000 / 18260 / 8740 → score shares .131 / .218 / .398 / .254).

## Current best pipeline — `src/final.py`
Candidates: `runs/test_cands_v1v3.pkl` (5 Kaggle-trained char transformers, beam candidates per test row).
Per segment:
| segment | stress representation | λ (prior weight) | decision |
|---|---|---|---|
| transfer | cur5 (initial / distance-from-end), ParadigmModel K=600 × 6 seeds | 2 | joint per (lemma, feats) |
| complete | **3way** (initial / stem-final / ending vs lemma stem), MaskParadigmModel K=300 × 3 | 5 | joint |
| wug | **3way** | 5 | joint |
| unseen | cur5 | 2 | **per row** |
Selection inside the chosen class: `log(mass) + 0.6 * log P(len delta | pos, feats, dialect)`.
Options: `--ctx_clean 1` drops flagged substitution-noise rows (`runs/exp/noise/sub_flags_traindev.csv`) from the paradigm context;
`--models ... --w_loc 0.5` pools beams of local class-conditioned models (`src/local_train.py`) inside the chosen class.

### Commands
```bash
python3 src/final.py --mode dev --ctx_clean 1 --models runs/loc_dev,runs/loc_dev_s1 --w_loc 0.25,0.5,1.0   # holdout validation
python3 src/final.py --mode full --ctx_clean 1 --models runs/loc_full,runs/loc_full_s1,runs/loc_full_s3 --w_loc 0.5 --out runs/final/sub_final_v2.csv
python3 src/preflight.py runs/final/sub_final_v2.csv runs/final/sub_final_v1.csv                        # schema + change breakdown
```

## Measured results (holdout = seed-123 lemma holdouts, `src/validate.py`; test-like weighting)
| system | holdout | public LB |
|---|---|---|
| old pipeline (cur5 K=300, λ=2, joint) | 0.63898 | 0.66783 |
| + candidate-level local pooling (old) | — | 0.66871 |
| final v1 (3way for complete+wug, K600, per-row unseen, μ=0.6) | 0.65362 | **0.67826** |
| v1 + ctx_clean | 0.65458 | — |
| v2 = v1 + ctx_clean + 3-model local pooling w=0.5 | 0.6571 | 0.67904 |
| v3 = v2 + post-fixes BAC (skeleton, ADJ-long, geminate) | +0.003 (0 broken) | 0.67963 |
| v3 + product-of-experts prior for complete (`poe`, λ5=1) | complete EM .948→.9515 | **0.68003** |
| v4 = above + 5-model pooling (auto-built/submitted by `runs/final/auto_v4.sh`) | — | pending |

Headroom (holdout): gold form in beam 97–98% in every segment; stress-oracle EM .967/.973/.961/.931.
Class accuracy: transfer .97, complete .97 (near noise ceiling), wug ~.51, unseen ~.42.

## Failed / ruled-out (do not repeat without a new reason)
- Stress paradigm predictable from lemma string / n-grams / frequency rank / id / md5-sha-crc32 mod K / yat: no.
- VVZ stress = Russian stress (StressRNN dictionary + pymorphy3): chance-level match.
- Seeded-RNG reconstruction of paradigm assignment (numpy/python/PCG, many orderings, K 30–64): no significant hit.
- EM latent-class mixture; small-K k-medoids (K=20–100); nonparametric Bayes over empirical full patterns:
  wug seed-only class accuracy is capped at ~0.47 (train lemmas) — information limit, not a modelling gap.
- Joint per-lemma paradigm inference from NN class masses (cur5 and 3way): neutral/negative.
- Dialect-consistency joint decoding: negative. Stem-prefix consensus across a lemma's rows: −0.0003.
- Two-syllable unseen ADJ anomaly: LB probe 0.03762 → EM ≈ 0.58, consistent with train statistics.
- String/hash-seeded RNG template assignment (random.Random(lemma), md5/sha/crc variants, T 8–100): no hit.
- Declension-type / stem-length conditioning of the wug seed posterior: N worse, V +0.01 (≈ +0.001 LB) — skipped.
- Local-model-heavy decoding for unseen: local models drop chunks of long words; Kaggle candidates are better.
- Bigger local model, beam vs greedy for local model, geminate post-fix, per-class bias, per-POS λ (overfits).

## Artifacts
- Candidates: `runs/test_cands_v1v3.pkl` (test), `runs/kdev2/val_cands_aligned.pkl` (holdout).
- Local models: `runs/loc_full{,_s1,_s3}` (test), `runs/loc_dev{,_s1}` (holdout-safe).
- Priors cache (full mode): `runs/final/pm_*_full.pkl`.
- Analyst scripts/results: `src/exp/*`, `runs/exp/*`.

## LB-probe note
λ variants (wug λ3, unseen λ1) change ~800–900 rows; the true effect (~1% of changed rows) is ~50× smaller than public-LB noise on those rows, so they were NOT submitted — keep holdout-optimal λ.

## Next highest-value experiments
1. Product-of-experts class decision (cur5 prior × 3way prior) for wug/complete — cheap, validate on holdout.
2. More local models for pooling (each ≈ +0.0005 on LB); GPU ~15 min/epoch when free.
3. Final selection: pick the 2 final submissions by **holdout-validated** gains, not public-LB noise (sd ≈ 0.003).

## Security
`token.txt`, `data/`, model binaries are git-ignored and were purged from GitHub history. Rotate the Kaggle token.
