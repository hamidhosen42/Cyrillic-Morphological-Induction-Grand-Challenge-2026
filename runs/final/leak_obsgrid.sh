#!/bin/bash
# Public noise-copy leak (forum topic 743193): prior weight for unseen lemmas that received a leaked observation.
cd "/Users/md.hamidhosen/Documents/Kaggle/Cyrillic Morphological Induction Grand Challenge 2026"
CFG='{"transfer":{"rep":"cur5","lam":2.0,"joint":true},"complete":{"rep":"poe","lam":5.0,"joint":true,"lam5":1.0},"wug":{"rep":"3way","lam":5.0,"joint":true},"unseen":{"rep":"cur5","lam":2.0,"joint":false}}'
DEV="--mode dev --ctx_clean 1 --fixes BAC --models runs/loc_dev,runs/loc_dev_s1 --w_loc 0.5 --device mps --leak 1 --leak_ctx 1 --leak_lem 85,5 --cfg $CFG"
python3 -u src/final.py $DEV --cache_tag _g1 --obs_cfg '{"rep":"cur5","lam":1.0,"joint":false}' > runs/final/dev_obs1.log 2>&1 &
python3 -u src/final.py $DEV --cache_tag _g2 --obs_cfg '{"rep":"cur5","lam":3.0,"joint":false}' > runs/final/dev_obs2.log 2>&1 &
python3 -u src/final.py $DEV --cache_tag _g3 --obs_cfg '{"rep":"cur5","lam":5.0,"joint":false}' > runs/final/dev_obs3.log 2>&1 &
python3 -u src/final.py $DEV --cache_tag _g4 --obs_cfg '{"rep":"poe","lam":5.0,"joint":false,"lam5":1.0}' > runs/final/dev_obs4.log 2>&1 &
wait
echo ALLDONE > runs/final/leak_obsgrid.done
