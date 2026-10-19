#!/usr/bin/env bash
# C5：budget 扫描（重跑版）。
# 坑：定长 4864 → 19 块，max_model_len 必须 ≥ 4864，否则 CUDA graph 里
#     block_table 宽度按 max_model_len//256 固定为 16，实际要 19 → 捕获后 expand 报错。
cd "$(dirname "$0")/.." || exit 1
export PYTHONPATH=$PWD TQDM_DISABLE=1
PY=.venv/bin/python
FLT='Generating|it/s|s/it|W1014|recompile|dynamo|torch._dynamo'
MEM=".workbuddy/memory/2026-10-10.md .workbuddy/memory/MEMORY.md"

to_main() { git stash push -q -- $MEM 2>/dev/null; git checkout -q main; echo "--- $(git branch --show-current)"; }
to_v1()   { git checkout -q V1; git stash pop -q 2>/dev/null; echo "--- $(git branch --show-current)"; }

sweep() {   # $1=side
    for B in 1024 2048 4096 16384; do
        echo "=== $1 / c5 / budget=$B ==="
        $PY experiments/c1_stress.py --tag "$1" --regime ctrl --fixed-total 4864 \
            --max-model-len 8192 --pool-blocks 163 --n 32 --budget "$B" --repeat 1 \
            --out "experiments/results/c5_${1}_b${B}.json" 2>&1 | grep -vE "$FLT" | tail -4
    done
}

echo "########## C5 V1 ##########"
to_v1
sweep V1

echo "########## C5 main ##########"
to_main
sweep main

echo "########## restore ##########"
to_v1
echo "DONE_C5"
