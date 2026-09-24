#!/bin/zsh
cd "/Users/md.hamidhosen/Documents/Kaggle/Cyrillic Morphological Induction Grand Challenge 2026"
python3 -u src/local_train.py --mode full --epochs 9 --seed 2 --max_ctx 10 --out runs/loc_full_s2 --device mps > runs/loc_full_s2.log 2>&1
python3 -u src/local_train.py --mode dev  --epochs 8 --seed 2 --max_ctx 10 --out runs/loc_dev_s2  --device mps > runs/loc_dev_s2.log 2>&1
python3 -u src/local_train.py --mode full --epochs 9 --seed 3 --d_model 320 --layers 4 --ff 1024 --out runs/loc_full_s3 --device mps > runs/loc_full_s3.log 2>&1
