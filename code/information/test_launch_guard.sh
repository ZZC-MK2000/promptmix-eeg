#!/usr/bin/env bash
# Guard self-test: the first instance runs, the second must be refused.
set -u
PY=/c/Users/18439/.conda/envs/clabram312/python.exe
cd /d/Projects/CLaBraM/promptmix-eeg
TAG=TEST_guard_selftest
"$PY" code/information/launch_guard.py --tag "$TAG" --log results/.locks/selftest1.log -- sleep 8 &
FIRST=$!
sleep 2
"$PY" code/information/launch_guard.py --tag "$TAG" --log results/.locks/selftest2.log -- echo SHOULD_NOT_RUN
echo "second_exit=$?"
wait $FIRST
echo "first_exit=$?"
echo "locks_left:"; ls results/.locks/ 2>/dev/null || true
