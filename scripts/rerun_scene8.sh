#!/bin/zsh
cd /Users/rajeev/Code/local_cua
# Re-run only scene 8 (fixture disambiguated) for every model/mode, appending rows.
for m in fara tongui uitars uground; do
  .venv/bin/python -m harness.run_stage1 $m --warm-reps 3 --scene scene8 2>&1 | grep -vE "Loading weights|Fetching|it/s|Warning"
done
for mode in direct actor; do
  .venv/bin/python -m harness.run_stage1_multi showui --mode $mode --warm-reps 3 --scene scene8 2>&1 | grep -vE "Fetching|it/s|Warning"
done
echo "=== SCENE8 RERUN DONE ==="
