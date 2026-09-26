#!/bin/bash
# Build v4 (best config + 5-model local pooling) when loc_full_s4/s5 are trained; preflight; submit after the UTC day resets.
cd "/Users/md.hamidhosen/Documents/Kaggle/Cyrillic Morphological Induction Grand Challenge 2026"
until [ -f runs/loc_full_s5/model_last.pt ] && grep -q "^DONE" runs/loc_full_s5.log 2>/dev/null; do sleep 120; done
CFG='{"transfer":{"rep":"cur5","lam":2.0,"joint":true},"complete":{"rep":"poe","lam":5.0,"joint":true,"lam5":1.0},"wug":{"rep":"3way","lam":5.0,"joint":true},"unseen":{"rep":"cur5","lam":2.0,"joint":false}}'
python3 -u src/final.py --mode full --ctx_clean 1 --fixes BAC --cfg "$CFG" \
  --models runs/loc_full,runs/loc_full_s1,runs/loc_full_s3,runs/loc_full_s4,runs/loc_full_s5 --w_loc 0.5 --device mps \
  --out runs/final/sub_final_v4.csv > runs/final/full_v4.log 2>&1 || exit 1
python3 src/preflight.py runs/final/sub_final_v4.csv runs/final/var_complete_poe.csv > runs/final/preflight_v4.log 2>&1
grep -q "PREFLIGHT PASSED" runs/final/preflight_v4.log || exit 1
# wait for a UTC day with free quota (only submits once)
while [ "$(date -u +%Y%m%d)" = "20260924" ]; do sleep 120; done
# Authenticate with the Kaggle CLI's existing configuration or environment.
kaggle competitions submit -c cyrillic-morphological-induction-grand-challenge -f runs/final/sub_final_v4.csv \
  -m "final v4: complete-PoE config + 5-model local pooling (w 0.5) + ctx_clean + post-fixes" > runs/final/submit_v4.log 2>&1
sleep 90; kaggle competitions submissions -c cyrillic-morphological-induction-grand-challenge 2>&1 | sed -n '3p' >> runs/final/submit_v4.log
