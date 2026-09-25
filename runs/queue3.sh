#!/bin/zsh
cd "/Users/md.hamidhosen/Documents/Kaggle/Cyrillic Morphological Induction Grand Challenge 2026"
while pgrep -f "src/final.py" >/dev/null; do sleep 30; done
python3 -u src/local_train.py --mode full --epochs 8 --seed 4 --max_ctx 10 --out runs/loc_full_s4 --device mps > runs/loc_full_s4.log 2>&1
python3 -u src/local_train.py --mode full --epochs 8 --seed 5 --d_model 320 --layers 4 --ff 1024 --out runs/loc_full_s5 --device mps > runs/loc_full_s5.log 2>&1
