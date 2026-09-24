#!/bin/zsh
cd "/Users/md.hamidhosen/Documents/Kaggle/Cyrillic Morphological Induction Grand Challenge 2026"
until grep -q "^DONE" runs/loc_full_s1.log; do sleep 60; done
python3 -u src/local_train.py --mode dev --epochs 8 --seed 1 --out runs/loc_dev_s1 --device mps > runs/loc_dev_s1.log 2>&1
python3 -u src/local_train.py --mode dev --epochs 10 --d_model 384 --layers 4 --ff 1024 --bs 128 --lr 5e-4 --out runs/loc_dev_big --device mps > runs/loc_dev_big.log 2>&1
python3 -u src/local_train.py --mode full --epochs 10 --d_model 384 --layers 4 --ff 1024 --bs 128 --lr 5e-4 --out runs/loc_full_big --device mps > runs/loc_full_big.log 2>&1
