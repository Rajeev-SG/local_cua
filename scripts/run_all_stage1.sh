#!/bin/zsh
# Reproduce the full Stage-1 screen for all five models. ~15 min.
set -e
cd /Users/rajeev/Code/local_cua
: > results/stage1.jsonl
for m in fara tongui uitars uground; do
  echo "########## $m"
  .venv/bin/python -m harness.run_stage1 $m --warm-reps 3 2>&1 | grep -vE "Loading weights|Fetching|it/s|Warning"
done
for mode in direct actor; do
  echo "########## showui/$mode"
  .venv/bin/python -m harness.run_stage1_multi showui --mode $mode --warm-reps 3 2>&1 | grep -vE "Fetching|it/s|Warning"
done
echo "=== STAGE1 DONE ==="
