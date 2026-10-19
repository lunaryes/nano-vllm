#!/usr/bin/env python
"""A1/A2 的数值层证据：直接比较 logits，绕过 argmax 的脆弱性。

逐 token 对齐会被两件事破坏：
  1) chunked prefill 与不同批组成改变浮点归约顺序 → logits 出现微小差异
  2) 自回归把单点差异放大成整条序列不同
所以改用「同一输入下 logits 的差异量级」判断正确性。

构造：单条 seq + 固定 prompt，唯一变量是 budget（是否分块）。
记录每次采样时 logits 的 top-5（索引 + 数值）与 top1-top2 间距 gap。

用法：
    python experiments/a_logits_probe.py --length 3200 --budgets 512 2048 16384 \
        --out experiments/results/a_logits.json
"""
import argparse
import json
import os
import random
import subprocess
import sys
import time

import torch

from nanovllm.layers import sampler as _sampler

PROBE = []


def _probe_forward(self, logits, temperatures):
    lg = logits.float()
    if lg.size(0) > 0:
        t5 = lg.topk(5, dim=-1)
        vals = t5.values[0]
        PROBE.append({
            "idx": t5.indices[0].tolist(),
            "val": [round(float(v), 6) for v in vals.tolist()],
            "gap": round(float(vals[0] - vals[1]), 6),
            "absmax": round(float(lg.abs().max()), 6),
            "mean": round(float(lg.mean()), 6),
        })
    return lg.argmax(dim=-1)


_sampler.Sampler.forward = _probe_forward

from nanovllm import LLM, SamplingParams  # noqa: E402

MODEL_PATH = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
POOL_CACHE = os.path.join(REPO_ROOT, "experiments", "a_text_pool.json")


def make_prompt(length, seed=0, mode="text"):
    if mode == "text" and os.path.exists(POOL_CACHE):
        with open(POOL_CACHE) as f:
            pool = json.load(f)
        rng = random.Random(seed)
        s = rng.randint(0, max(0, len(pool) - length - 1))
        return pool[s:s + length]
    rng = random.Random(seed)
    return [rng.randint(0, 10000) for _ in range(length)]


def run_single(args):
    prompt = make_prompt(args.length, args.seed, args.prompt_mode)
    llm = LLM(
        MODEL_PATH,
        enforce_eager=args.enforce_eager,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.budget,
    )
    PROBE.clear()  # 丢弃 warmup 阶段的记录
    sps = [SamplingParams(temperature=1.0, ignore_eos=True, max_tokens=args.max_tokens)]
    t0 = time.time()
    outs = llm.generate([prompt], sps, use_tqdm=False)
    doc = {
        "budget": args.budget,
        "length": args.length,
        "prompt_mode": args.prompt_mode,
        "max_tokens": args.max_tokens,
        "enforce_eager": args.enforce_eager,
        "token_ids": outs[0]["token_ids"],
        "probe": PROBE,
        "seconds": round(time.time() - t0, 2),
    }
    with open(args.out, "w") as f:
        json.dump(doc, f)
    print(f"  [single] budget={args.budget:6d} plen={len(prompt)} calls={len(PROBE)} "
          f"t={doc['seconds']}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--length", type=int, default=3200)
    ap.add_argument("--budgets", type=int, nargs="+", default=[512, 2048, 16384])
    ap.add_argument("--max-tokens", type=int, default=8)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--prompt-mode", choices=["random", "text"], default="text")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--enforce-eager", action="store_true")
    ap.add_argument("--out", required=True)
    ap.add_argument("--single", action="store_true")
    ap.add_argument("--budget", type=int, default=0)
    args = ap.parse_args()

    if args.single:
        run_single(args)
        return

    here = os.path.abspath(__file__)
    docs = {}
    for b in args.budgets:
        tmp = f"{args.out}.b{b}.tmp.json"
        cmd = [sys.executable, here, "--single", "--budget", str(b),
               "--length", str(args.length), "--max-tokens", str(args.max_tokens),
               "--max-model-len", str(args.max_model_len),
               "--prompt-mode", args.prompt_mode, "--seed", str(args.seed),
               "--out", tmp]
        if args.enforce_eager:
            cmd.append("--enforce-eager")
        print(f"budget={b} ...", flush=True)
        p = subprocess.run(cmd, capture_output=True, text=True)
        if p.returncode != 0:
            print(p.stdout, p.stderr, flush=True)
            sys.exit("子进程失败")
        with open(tmp) as f:
            docs[b] = json.load(f)
        os.remove(tmp)

    with open(args.out, "w") as f:
        json.dump({"length": args.length, "prompt_mode": args.prompt_mode, "runs": docs}, f)

    bs = sorted(docs.keys(), reverse=True)
    base_b = bs[0]
    base = docs[base_b]
    bp = base["probe"][0]  # 第一次采样 = prefill 收尾，位置同为 prompt 末尾

    print(f"\n=== logits 探针 | length={args.length} | prompt={args.prompt_mode} "
          f"| eager={args.enforce_eager}")
    print(f"基准 budget={base_b}（不分块）")
    print(f"  第一次采样 logits: absmax={bp['absmax']}  mean={bp['mean']}  gap={bp['gap']}")
    print(f"  top5: {list(zip(bp['idx'], bp['val']))}")
    print(f"  首个 token: {base['token_ids'][0] if base['token_ids'] else None}")
    for b in bs[1:]:
        d = docs[b]
        dp = d["probe"][0]
        same = sum(1 for x, y in zip(bp["idx"], dp["idx"]) if x == y)
        dval = max(abs(x - y) for x, y in zip(bp["val"], dp["val"]))
        rel = dval / max(abs(bp["absmax"]), 1e-9)
        print(f"\nbudget={b}（分块）")
        print(f"  第一次采样 logits: absmax={dp['absmax']}  mean={dp['mean']}  gap={dp['gap']}")
        print(f"  top5: {list(zip(dp['idx'], dp['val']))}")
        print(f"  首个 token: {d['token_ids'][0] if d['token_ids'] else None}")
        print(f"  → top5 索引一致 {same}/5   max|Δlogit|={dval:.6f}  "
              f"相对 absmax={rel:.2e}")
    print()


if __name__ == "__main__":
    main()
