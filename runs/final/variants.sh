#!/bin/zsh
cd "/Users/md.hamidhosen/Documents/Kaggle/Cyrillic Morphological Induction Grand Challenge 2026"
until grep -q "wrote\|Traceback" runs/final/full_v3_rebuild.log; do sleep 10; done
COMMON="--mode full --ctx_clean 1 --fixes BAC --models runs/loc_full,runs/loc_full_s1,runs/loc_full_s3 --w_loc 0.5 --device mps"
T='"transfer":{"rep":"cur5","lam":2.0,"joint":true}'; C='"complete":{"rep":"3way","lam":5.0,"joint":true}'
W='"wug":{"rep":"3way","lam":5.0,"joint":true}'; U='"unseen":{"rep":"cur5","lam":2.0,"joint":false}'
python3 -u src/final.py $COMMON --cfg "{$T,$C,\"wug\":{\"rep\":\"3way\",\"lam\":3.0,\"joint\":true},$U}" --out runs/final/var_wug_lam3.csv > runs/final/var_wug_lam3.log 2>&1
python3 -u src/final.py $COMMON --cfg "{$T,$C,$W,\"unseen\":{\"rep\":\"cur5\",\"lam\":1.0,\"joint\":false}}" --out runs/final/var_uns_lam1.csv > runs/final/var_uns_lam1.log 2>&1
python3 -u src/final.py $COMMON --cfg "{$T,\"complete\":{\"rep\":\"poe\",\"lam\":5.0,\"joint\":true,\"lam5\":1.0},$W,$U}" --out runs/final/var_complete_poe.csv > runs/final/var_complete_poe.log 2>&1
for f in var_wug_lam3 var_uns_lam1 var_complete_poe; do echo "== $f"; python3 src/preflight.py runs/final/$f.csv runs/final/sub_final_v3.csv | tail -6; done
