#!/bin/bash
# Publicly disclosed noise-copy leak (forum topic 743193), user explicitly asked to continue: holdout variants + test builds.
cd "/Users/md.hamidhosen/Documents/Kaggle/Cyrillic Morphological Induction Grand Challenge 2026"
CFG='{"transfer":{"rep":"cur5","lam":2.0,"joint":true},"complete":{"rep":"poe","lam":5.0,"joint":true,"lam5":1.0},"wug":{"rep":"3way","lam":5.0,"joint":true},"unseen":{"rep":"cur5","lam":2.0,"joint":false}}'
DEV="--mode dev --ctx_clean 1 --fixes BAC --models runs/loc_dev,runs/loc_dev_s1 --w_loc 0.5 --device mps --leak 1 --leak_ctx 1"
FULL="--mode full --ctx_clean 1 --fixes BAC --models runs/loc_full,runs/loc_full_s1,runs/loc_full_s3,runs/loc_full_s4,runs/loc_full_s5 --w_loc 0.5 --device mps --leak 1 --leak_ctx 1"
python3 -u src/final.py $DEV --cfg "$CFG" --leak_lem 70,10 --cache_tag _a > runs/final/dev_lem_a.log 2>&1 &
python3 -u src/final.py $DEV --cfg "$CFG" --leak_lem 85,5 --cache_tag _b > runs/final/dev_lem_b.log 2>&1 &
python3 -u src/final.py $FULL --cfg "$CFG" --cache_tag _n --out runs/final/sub_leak5.csv > runs/final/full_leak5.log 2>&1 &
python3 -u src/final.py $FULL --cfg "$CFG" --leak_lem 70,10 --cache_tag _a --out runs/final/sub_leak5_lem_a.csv > runs/final/full_leak5_lem_a.log 2>&1 &
python3 -u src/final.py $FULL --cfg "$CFG" --leak_lem 85,5 --cache_tag _b --out runs/final/sub_leak5_lem_b.csv > runs/final/full_leak5_lem_b.log 2>&1 &
wait
echo ALLDONE > runs/final/leak_lem.done
