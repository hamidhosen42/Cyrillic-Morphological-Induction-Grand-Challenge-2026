#!/bin/bash
# Holdout validation of the publicly disclosed noise-copy leak (forum topic 743193); user explicitly approved running it.
cd "/Users/md.hamidhosen/Documents/Kaggle/Cyrillic Morphological Induction Grand Challenge 2026"
CFG='{"transfer":{"rep":"cur5","lam":2.0,"joint":true},"complete":{"rep":"poe","lam":5.0,"joint":true,"lam5":1.0},"wug":{"rep":"3way","lam":5.0,"joint":true},"unseen":{"rep":"cur5","lam":2.0,"joint":false}}'
for spec in "0 1" "1 0" "1 1"; do
  set -- $spec
  echo "=== leak=$1 leak_ctx=$2"
  python3 -u src/final.py --mode dev --ctx_clean 1 --fixes BAC --cfg "$CFG" --models runs/loc_dev,runs/loc_dev_s1 --w_loc 0.5 --device mps --leak $1 --leak_ctx $2 2>&1 | grep -E "leak|CONFIG|POOLED|BEFORE|AFTER|fixed|no-pool|decoded|Error|error"
done
echo ALLDONE
