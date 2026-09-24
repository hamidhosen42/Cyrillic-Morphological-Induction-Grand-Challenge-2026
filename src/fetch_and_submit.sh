#!/bin/zsh
# Usage: src/fetch_and_submit.sh <kernel-slug> <run-name> [lam] [--submit "message"]
set -e
cd "$(dirname "$0")/.."
export KAGGLE_API_TOKEN=$(head -1 token.txt | tr -d '[:space:]')
K=$1; RUN=$2; LAM=${3:-1.0}
mkdir -p runs/$RUN
kaggle kernels output hosen42/$K -p runs/$RUN
ls -la runs/$RUN
python3 src/postprocess.py --cands runs/$RUN/test_cands.pkl --mode full --lam $LAM --out runs/sub_${RUN}_lam${LAM}.csv
if [[ "$4" == "--submit" ]]; then
  kaggle competitions submit -c cyrillic-morphological-induction-grand-challenge -f runs/sub_${RUN}_lam${LAM}.csv -m "$5"
fi
