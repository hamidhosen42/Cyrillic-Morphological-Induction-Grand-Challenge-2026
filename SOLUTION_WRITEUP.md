# 14th Place Solution: Context-Aware Character Transformers + Explicit Stress-Paradigm Inference

**Team Hack2Publish** (hosen42, esfersami50)

## Results

| | Private LB | Public LB |
|---|---:|---:|
| **Final A:** `var_complete_poe` (selected) | **0.67529** | 0.68003 |
| **Final B:** `sub_final_v4` (selected) | 0.67500 | 0.67968 |
| **Position** | **14th of 66** | 13th of 66\* |

\* The public leaderboard shows our best public score across all submissions, 0.68648. That score came from a leak-based submission made before the host's ruling (section 13). It was not selected. Both final submissions are clean, and the private score of 0.67529 is Final A's.

## TL;DR

1. **Generation.** Five character-level transformer seq2seq models generate beam candidates. Each model sees the lemma, the target tags and the other known forms of the same lemma.
2. **The key finding.** Stress is not random per cell: each lemma follows one hidden stress paradigm. An explicit paradigm model turns a lemma's observed forms into a prior over the stress class of the requested cell.
3. **Selection.** We decide the stress class first, from neural class mass × paradigm prior, jointly across the dialect rows of a cell. Only then do we pick the spelling inside that class.
4. **Smaller gains.** Segment-specific decision rules, a length prior, class-conditioned local models pooled inside the chosen class, noise-cleaned context and three post-fixes.
5. **Validation.** Every change that improved our lemma-grouped holdout also improved the private score. The private score rose monotonically from 0.65085 to 0.67529 across our pipeline versions.
6. **Where the score is lost.** Almost all remaining error is the stress of new lemmas, where a single seed form carries little information.

## 1. Task and data

The task is to generate an inflected form of a synthetic Cyrillic language (`form_vvz`) from a Russian lemma, part of speech, grammatical features, dialect (SEV, POM, KAM), lemma frequency and a yat flag.

| File | Rows | Role |
|---|---:|---|
| `train.csv` | 300,000 | labelled |
| `dev.csv` | 20,000 | labelled |
| `wug_seeds.csv` | 4,800 | one labelled form per new lemma |
| `test_features.csv` | 60,000 | to predict |

What we learned about the language:

- **Stress** is a combining acute accent (U+0301) on exactly one vowel. An exact match needs both the spelling and the stress.
- **Regular sound rules:**
  - akanye (unstressed о → а);
  - SEV reverses pleophony (оро → ра, ере → рѣ, оло → ла);
  - KAM keeps ськ and writes и for unstressed е and in LOC -и;
  - SEV and POM turn ск into сьц;
  - `yat_flag` marks ѣ stems.
- **Noise:** about 1.7% of labelled rows are random-word substitutions, where the form belongs to a different lemma, plus some character typos.

The test set mixes four regimes, which we derived from the files:

| Regime | Evidence available | Test rows |
|---|---|---:|
| Transfer | same lemma and cell seen in another dialect | 18,000 |
| Completion | other cells of the lemma are known | 15,000 |
| Wug | one SEV seed form (N;ACC;SG or V;PRS;3;SG) for each of 4,800 new noun/verb lemmas | 18,260 |
| Unseen | nothing (1,200 new adjective lemmas) | 8,740 |

KAM is over-represented in the test set, about a third of the wug and unseen rows.

## 2. Decoding the metric with probe submissions

The Evaluation page gives `Score = 0.9 × WeightedEM + 0.1 × (1 − WeightedCER)`, but not the row weights. On day one we spent four submissions on probes:
- an all-"а" file (0.00755), which isolates the CER part;
- files keeping only one subset of the predictions: SEV transfer (0.05868), POM transfer (0.06001) and high-frequency rows (0.00098).

Together they fit segment weights of about **1 / 2 / 3 / 4** for transfer / completion / wug / unseen. Wug and unseen therefore carry about 65% of the score.

Later, a wug-only probe (0.22776) and an unseen-only probe (0.10940) of the same pipeline put test exact match at about 0.54 on wug and 0.38 on unseen. That told us precisely where the missing points were.

## 3. Validation

The holdout mimics the test:
- 500 wug-like noun/verb lemmas, keeping only the SEV seed cell;
- 250 unseen adjective lemmas, keeping nothing;
- dev rows split into transfer and completion by whether (lemma, feats) is seen in train;
- scoring with the test's segment weights and row counts.

Every change was validated there before we submitted it. Final A scores 0.65981 on this holdout, by segment:

| Transfer | Completion | Wug | Unseen |
|---:|---:|---:|---:|
| 0.960 | 0.955 | 0.496 | 0.403 |

Two caveats:
- The holdout is harsher than the test on wug, because train rows contain substitution noise while the test wug seeds are clean.
- The same holdout informed many decisions, so it overstates small gains.

## 4. The key insight: hidden stress paradigms

Stress looked random per cell at first. But across full training paradigms, each lemma follows one of a limited set of recurring stress patterns over its cells. Given the other cells of a lemma, a held-out cell's stress class is predictable about 91–97% of the time. From a single wug seed it is about 48%, and for an unseen lemma only the population prior (about 40%) remains.

We represent stress in two ways:

- **cur5:** initial stress, or distance from the last vowel (5 classes).
- **3way:** initial, stem-final or ending stress relative to the lemma stem. The stem boundary is found with a stem-vowel rule corrected for SEV pleophony. Masks handle short stems, where two classes coincide. This fit the data better than cur5 and gave the largest single gain: holdout +0.013, public +0.0096, private +0.0080.

The paradigm model is k-medoids over the stress-class vectors of training lemmas (cur5: K=600 × 6 seeds; 3way: K=300 × 3 seeds). Given a lemma's observed cells, the posterior over medoids yields `P(class | observed forms)` for the requested cell.

## 5. Candidate generation

- **Model:** a character transformer encoder-decoder.
- **Encoder input:** target tags (dialect, feature parts, yat, log-frequency bucket), the lemma characters, and up to 6 (v1) or 12 (v3) other known forms of the same lemma. Each context form carries its own tags, and a marker flags a same-feature form from another dialect.
- **Context sampling during training** mimics the four regimes: other dialects of the same cell, missing cells, seed only, or nothing.
- **v1:** d=320, 5 layers, 14.6M parameters, 3 seeds, beam 5.
- **v3:** d=384, 6 layers, about 25M parameters, 2 seeds, beam 8.
- Both were trained on Kaggle GPUs, a few hours per run.
- **Ensembling:** beam probabilities are normalized per model, averaged over the five models, and identical strings are merged into one candidate mass.

On its own the ensemble reaches only about 0.50 dev exact match: it never learned to infer the stress paradigm, and a bigger model did not change that. But the gold form is in the beam 97–98% of the time in every regime. The remaining work was selection.

## 6. Stress decision and spelling selection

For each (lemma, feats) cell, the rows of all dialects are decided together, since the stress class is shared across dialects about 93% of the time:

```
class = argmax_c [ Σ_rows log M_row(c) + λ · log P(c | observed forms) ]
```

`M_row(c)` is the candidate mass of stress class `c`. Prior weights were chosen on the holdout; for new lemmas λ=2 beat 1, 4 and 0.5 with cur5.

| Regime | Representation | λ | Decision |
|---|---|---:|---|
| Transfer | cur5, K=600 × 6 | 2 | joint across dialects |
| Completion | 3way, candidate mass reweighted by the cur5 prior (product of experts) | 5 | joint |
| Wug | 3way | 5 | joint |
| Unseen | cur5 | 2 | per row |

Inside the chosen class, the spelling is:

```
y = argmax_y [ log M(y) + 0.6 · log P(len(y) − len(lemma) | POS, feats, dialect) ]
```

## 7. Local class-conditioned models

We also trained small character transformers (5–10M parameters) that take the stress class as an input token. They ran locally on an Apple-silicon GPU (MPS), at about 13 minutes per epoch.

- **Pooling:** for each row we decode them with the chosen class token and pool their beams with the main candidates of that class, at weight 0.5.
- **Final A** uses three local models (seeds 0, 1 and 3). **Final B** adds two more (seeds 4 and 5, with a longer context and a different size).
- **Pooling at the candidate level helped;** merging at the level of final answers hurt:

| | Holdout | Public | Private |
|---|---|---:|---:|
| Candidate-level pooling | +0.0077 | +0.0009 | +0.0020 |
| Answer-level merges | — | 0.66562 and 0.66513 (vs 0.66653) | 0.66099 and 0.66180 (vs 0.66205) |

## 8. Noise-cleaned context

A detector flags substitution noise in train and dev: rows whose form is unrelated to both its lemma and its sibling forms. Flagged rows are dropped from the paradigm context, and the detector uses released labelled data only. This added +0.001 on the holdout.

## 9. Post-fixes

We added three narrow rules for residual error types found on the holdout:
- the candidate must keep the lemma's consonant skeleton;
- an adjective's long form must equal its short form plus the ending;
- geminates must be kept.

On the holdout they fixed 17 transfer, 17 completion, 7 wug and 40 unseen rows, and broke none. On the test they changed only 76 rows.

## 10. Score progression: holdout, public and private

| Version | Holdout | Public | Private |
|---|---:|---:|---:|
| Copy the same cell from another dialect (baseline) | — | 0.21939 | 0.22053 |
| v1 ensemble + paradigm re-ranking | — | 0.65492 | 0.65085 |
| Joint stress decision per (lemma, feats), K=300 × 3 | — | 0.66360 | 0.66135 |
| + v3 models (5-model ensemble) | — | 0.66653 | 0.66205 |
| Prior weight 2 for new lemmas | 0.63898 | 0.66783 | 0.66315 |
| Candidate-level pooling with 2 local models | — | 0.66871 | 0.66511 |
| 3way representation for completion and wug, per-row unseen, length prior | 0.65362 | 0.67826 | 0.67308 |
| Noise-cleaned context + 3 local models | — | 0.67904 | 0.67415 |
| Post-fixes | — | 0.67963 | 0.67455 |
| **Product of experts for completion (Final A)** | **0.65981** | **0.68003** | **0.67529** |
| 5 local models in the pool (Final B) | — | 0.67968 | 0.67500 |

What the private leaderboard showed:
- **Every step improved private.** The private score never went down across these versions, from 0.65085 to 0.67529.
- **Private tracked public closely.** It was 0.002–0.005 below public for every clean version, so we lost almost nothing in the shake-up. Four teams above us dropped 0.009–0.016.
- **Final A beat Final B on both boards.** The two finals differ on only 93 of the 60,000 rows, so choosing B added very little hedge.

## 11. What did not work

- **Predicting the paradigm from surface data:** the lemma string, character n-grams, frequency rank, row ids, hashes (md5, sha, crc32 mod K) and yat gave no signal.
- **Real Russian stress** (StressRNN dictionary + pymorphy3) agreed with this language's stress only at chance level.
- **Reconstructing a seeded random assignment** of paradigms (numpy/Python/PCG generators, several orderings, string-seeded RNGs) found nothing significant.
- **Better class models for wug:** EM latent-class mixtures, other K values, nonparametric Bayes over full patterns and a dedicated stress classifier. None beat the paradigm prior plus neural evidence.
- **Joint inference and consistency constraints:** joint inference across a lemma's cells from neural class masses, dialect-consistency decoding and stem-prefix consensus were neutral or slightly negative.
- **Product of experts for wug and unseen** was worse: wug 0.492 → 0.485, unseen 0.394 → 0.383. It only helps completion.
- **Rescoring the whole candidate pool** with the neural models (teacher forcing) gained nothing.
- **Bigger models:** a bigger local model spelled no better, and a bigger Kaggle model alone was no better than v1.
- **Tuning λ against the public board:** variants changed 800–900 rows, but the true effect was far below public-LB noise. We kept the holdout-optimal values.

## 12. Where the score is lost

With the right stress class, holdout exact match is 0.988 / 0.990 / 0.977 / 0.964 for transfer / completion / wug / unseen.

Stress-class accuracy is:
- about 0.97 on training lemmas, close to the label-noise level;
- about 0.50 for wug;
- about 0.42 for unseen.

Seed-only class accuracy for wug stops around 0.47–0.49 however it is modelled. Unseen adjective stress looks close to uniform over the three 3way classes. Most of the remaining gap is information the data does not contain, not a modelling gap.

## 13. The copied-forms issue

While cleaning the noise we found that some substitution-noise rows in train and dev are exact copies of hidden test forms, and we reported it publicly on the forum. Before the host ruled, we tested it in three submissions; the best scored 0.68648 public and 0.68233 private. The host then ruled that deliberately exploiting these copies is not permitted, and we excluded all of that work. Neither final selection uses it: both were built before any leak-related code existed. The 14th place above is the result of the clean pipeline.

## 14. Lessons

- **Separate generating candidates from deciding the stress.** The network alone never learned paradigm inference; an explicit model did.
- **Spend a few early submissions on probes** to decode the metric. Knowing that wug and unseen carry about 65% of the score set our priorities for the whole week.
- **Build validation that mirrors each test regime** and trust it over the public board. Our holdout-validated steps all carried over to private, while public differences below about 0.003 were noise.

Thanks to the host for an interesting synthetic-language task, and congratulations to the winners.
