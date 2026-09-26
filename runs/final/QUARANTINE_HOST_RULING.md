# INVALID_FOR_FINAL_SELECTION_HOST_RULING

Host ruling (2026-09-26): "Deliberately exploiting these copied forms will not be permitted."
Everything listed here is kept only as a historical record. Do not submit or select it, ensemble it,
use it as pseudo-labels, or tune anything (thresholds, priors, routing, lambdas) from it or from its public LB.

## Leak-derived submissions (submitted)
| file | public LB | submitted (UTC) |
|---|---|---|
| sub_leak_ctx.csv | 0.68648 | 2026-09-25 19:36 |
| sub_leak5_lem_b.csv | 0.68621 | 2026-09-26 01:28 |
| sub_leak5_obs.csv | 0.68581 | 2026-09-26 01:56 |

## Leak-derived files (never submitted)
sub_leak5.csv, sub_leak5_lem_a.csv

## Leak artifacts
- leak_lemma_match.pkl, dd_base.pkl.leak, dd_u3.pkl.leak
- scripts and logs: leak_*.sh, leak_*.done, leak_dev.log, full_leak*.log, dev_lem_*.log, dev_obs*.log, dev_more_*.log, preflight_leak_ctx.log
- local-beam caches written by leak runs: lb_cache_*_b4_{a,b,c,d,g1,g2,g3,g4,n}.pkl
  (the _d cache holds beams decoded with leaked forms as local-model context)

## Caches last written by a leak run (do not reuse)
- lb_cache_full_loc_full_loc_full_s1_loc_full_s3_b4.pkl (rewritten 2026-09-25 18:38 UTC by the sub_leak_ctx build)
- lb_cache_dev_loc_dev_loc_dev_s1_b4.pkl (rewritten 2026-09-25 18:37 UTC by leak_dev.sh)
These runs did not use --leak_stem, so the stored beams were decoded from clean inputs, but clean rebuilds must
still use a new --cache_tag (e.g. _clean) and never --leak, --leak_lem, --leak_stem or --obs_cfg.
