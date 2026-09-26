#!/bin/bash
# Public noise-copy leak (forum topic 743193), user asked to keep improving: holdout checks of two refinements.
cd "/Users/md.hamidhosen/Documents/Kaggle/Cyrillic Morphological Induction Grand Challenge 2026"
CFG='{"transfer":{"rep":"cur5","lam":2.0,"joint":true},"complete":{"rep":"poe","lam":5.0,"joint":true,"lam5":1.0},"wug":{"rep":"3way","lam":5.0,"joint":true},"unseen":{"rep":"cur5","lam":2.0,"joint":false}}'
CFGU='{"transfer":{"rep":"cur5","lam":2.0,"joint":true},"complete":{"rep":"poe","lam":5.0,"joint":true,"lam5":1.0},"wug":{"rep":"3way","lam":5.0,"joint":true},"unseen":{"rep":"3way","lam":5.0,"joint":true}}'
DEV="--mode dev --ctx_clean 1 --fixes BAC --models runs/loc_dev,runs/loc_dev_s1 --w_loc 0.5 --device mps --leak 1 --leak_ctx 1 --leak_lem 85,5"
python3 -u src/final.py $DEV --cfg "$CFG" --cache_tag _b --dump runs/final/dd_base.pkl > runs/final/dev_more_base.log 2>&1 &
python3 -u src/final.py $DEV --cfg "$CFGU" --cache_tag _c --dump runs/final/dd_u3.pkl > runs/final/dev_more_u3.log 2>&1 &
python3 -u src/final.py $DEV --cfg "$CFG" --cache_tag _d --leak_stem 1 > runs/final/dev_more_stem.log 2>&1 &
wait
echo ALLDONE > runs/final/leak_more.done
