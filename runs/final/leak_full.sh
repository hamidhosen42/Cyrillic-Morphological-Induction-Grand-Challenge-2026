#!/bin/bash
# Test build with the publicly disclosed noise-copy leak (forum topic 743193); user explicitly approved. Not submitted here.
cd "/Users/md.hamidhosen/Documents/Kaggle/Cyrillic Morphological Induction Grand Challenge 2026"
CFG='{"transfer":{"rep":"cur5","lam":2.0,"joint":true},"complete":{"rep":"poe","lam":5.0,"joint":true,"lam5":1.0},"wug":{"rep":"3way","lam":5.0,"joint":true},"unseen":{"rep":"cur5","lam":2.0,"joint":false}}'
python3 -u src/final.py --mode full --ctx_clean 1 --fixes BAC --cfg "$CFG" --models runs/loc_full,runs/loc_full_s1,runs/loc_full_s3 --w_loc 0.5 --device mps --leak 1 --leak_ctx 1 --out runs/final/sub_leak_ctx.csv > runs/final/full_leak_ctx.log 2>&1
python3 src/preflight.py runs/final/sub_leak_ctx.csv runs/final/var_complete_poe.csv > runs/final/preflight_leak_ctx.log 2>&1
echo FULLDONE >> runs/final/full_leak_ctx.log
