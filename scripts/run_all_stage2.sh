#!/bin/zsh
cd /Users/rajeev/Code/local_cua
: > results/stage2.jsonl
echo "########## fara (agent)"
.venv/bin/python -m harness.run_stage2 fara --reps 2 2>&1 | grep -vE "Loading weights|Fetching|it/s|Warning"
echo "########## tongui (agent)"
.venv/bin/python -m harness.run_stage2 tongui --reps 2 2>&1 | grep -vE "Loading weights|Fetching|it/s|Warning"
echo "########## uitars (agent)"
.venv/bin/python -m harness.run_stage2 uitars --reps 2 2>&1 | grep -vE "Loading weights|Fetching|it/s|Warning"
echo "########## showui-direct (agent)"
.venv/bin/python -m harness.run_stage2 showui --mode direct --reps 2 2>&1 | grep -vE "Fetching|it/s|Warning"
echo "########## showui-actor (actor)"
.venv/bin/python -m harness.run_stage2 showui --mode actor --reps 2 2>&1 | grep -vE "Fetching|it/s|Warning"
echo "########## uground (grounder+executor)"
.venv/bin/python -m harness.run_stage2 uground --reps 2 2>&1 | grep -vE "Fetching|it/s|Warning"
echo "=== STAGE2 DONE ==="
