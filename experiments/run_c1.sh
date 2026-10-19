#!/usr/bin/env bash
# C1 压测批次：双口径 × 两分支 × 交替配对（抵消跨会话漂移）。
#
#   phase1  V1    : stress(rep2) + ctrl(rep2)
#   phase2  main  : stress(rep2) + ctrl(rep2)
#   phase3  V1    : stress(rep2)      <- 第 2 个交替配对
#   phase4  main  : stress(rep2)
#
# 结束后回到 V1 并恢复被 stash 的记忆文件。
cd "$(dirname "$0")/.." || exit 1
export PYTHONPATH=$PWD TQDM_DISABLE=1
PY=.venv/bin/python
FLT='Generating|it/s|s/it|W1014|recompile|dynamo|torch._dynamo'
MEM=".workbuddy/memory/2026-10-10.md .workbuddy/memory/MEMORY.md"

to_main() {
    git stash push -q -- $MEM 2>/dev/null
    git checkout -q main
    echo "--- branch: $(git branch --show-current)"
}

to_v1() {
    git checkout -q V1
    git stash pop -q 2>/dev/null
    echo "--- branch: $(git branch --show-current)"
}

run() {   # $1=side  $2=regime  $3=round
    echo "=== $1 / $2 / r$3 ==="
    $PY experiments/c1_stress.py --tag "$1" --regime "$2" --n 256 --repeat 2 \
        --out "experiments/results/c1_${1}_${2}_r${3}.json" 2>&1 | grep -vE "$FLT" | tail -8
}

echo "########## phase1 V1 ##########"
to_v1
run V1 stress r1
run V1 ctrl   r1

echo "########## phase2 main ##########"
to_main
run main stress r1
run main ctrl   r1

echo "########## phase3 V1 (pair 2) ##########"
to_v1
run V1 stress r2

echo "########## phase4 main (pair 2) ##########"
to_main
run main stress r2

echo "########## restore ##########"
to_v1
echo "DONE_C1"
