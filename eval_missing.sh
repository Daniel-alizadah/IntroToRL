#!/bin/bash
# Evaluates every weights/<cond>_seed<k>.pth for the given conditions that has no
# results/<cond>_seed<k>_scores.csv yet, three at a time (cap 20000).
#   bash eval_missing.sh fixed uniform
mkdir -p results logs/stdout
ev() {
  python evaluate.py "weights/$1_seed$2.pth" --name "$1_seed$2" --cap 20000 > "logs/stdout/$1_seed$2_eval.txt" 2>&1
  echo "evaluated $1 seed $2"
}
export -f ev
for c in "$@"; do
  for s in 0 1 2 3 4 5 6 7; do
    if [ -f "weights/${c}_seed${s}.pth" ] && [ ! -f "results/${c}_seed${s}_scores.csv" ]; then echo "$c $s"; fi
  done
done | xargs -P 3 -L 1 bash -c 'ev "$0" "$1"'
echo "EVAL DONE"
