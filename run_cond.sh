#!/bin/bash
# Train + evaluate seeds 0-7 for the given conditions, three at a time.
# Run from the folder with train.py / evaluate.py, venv active:
#   nohup caffeinate -i bash run_cond.sh strat_curriculum > run_strat.log 2>&1 &
# Example with two conditions:  bash run_cond.sh strat_curriculum strat_uniform
mkdir -p logs/stdout results

job() {
  cond="$1"; seed="$2"
  python train.py --condition "$cond" --seed "$seed" > "logs/stdout/${cond}_seed${seed}.txt" 2>&1
  python evaluate.py "weights/${cond}_seed${seed}.pth" --name "${cond}_seed${seed}" --cap 20000 \
      > "logs/stdout/${cond}_seed${seed}_eval.txt" 2>&1
  echo "finished $cond seed $seed"
}
export -f job

for c in "$@"; do
  for s in 0 1 2 3 4 5 6 7; do
    echo "$c $s"
  done
done | xargs -P 3 -L 1 bash -c 'job "$0" "$1"'

echo "ALL DONE"
