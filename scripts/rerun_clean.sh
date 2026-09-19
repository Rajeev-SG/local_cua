#!/bin/zsh
cd /Users/rajeev/Code/local_cua
echo "### showui direct (clean)"
.venv/bin/python -m harness.run_stage1_multi showui --mode direct --warm-reps 3 2>&1 | grep -vE "Fetching|it/s|Warning"
echo "### showui actor (clean)"
.venv/bin/python -m harness.run_stage1_multi showui --mode actor --warm-reps 3 2>&1 | grep -vE "Fetching|it/s|Warning"
echo "### fara stage2 (scroll fix)"
.venv/bin/python -m harness.run_stage2 fara --reps 2 2>&1 | grep -vE "Fetching|it/s|Warning"
