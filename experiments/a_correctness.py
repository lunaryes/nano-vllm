#!/usr/bin/env python
"""A1 分块不变性 / A2 跨实现等价 —— 正确性采集脚本。

用法（在目标分支上执行，两次运行分别得到 main / V1 的结果）：
    python experiments/a_correctness.py --tag V1 --out experiments/results/a_V1.json
    python experiments/a_correctness.py --tag V1 --prompt-mode text --out experiments/results/a_V1_text.json

设计要点
--------
1. 实验期把 Sampler 换成 argmax（确定性采样）。
   原实现是 Gumbel-max + 全局 RNG，且 softmax 无 max-subtract（小 temperature 会 nan），
   逐 token 对齐必须先消除随机性。此 patch 只在本脚本内生效，不写入仓库源码。
2. 负载两种模式：
   - random：固定 seed 的随机 token id。模型处于困惑态、logits 分布平坦，
     会出现大量近乎平局的位置，放大浮点敏感性 —— 用于压力测试。
   - text：仓库自身源码的真实文本片段。logits 分布尖锐，更贴近实际使用。
3. ignore_eos=True → 每条恰好生成 max_tokens 个 token，长度对齐才能逐位置比较。
4. 每个 budget 起一个独立子进程。
   `ModelRunner.__init__` 会调用 `dist.init_process_group`，同一进程只能初始化一次。
"""
import argparse
import glob
import json
import os
import random
import subprocess
import sys
import time

import torch

# ---- 确定性采样：必须在构造 LLM 之前完成 patch ----
from nanovllm.layers import sampler as _sampler


def _greedy_forward(self, logits, temperatures):
    return logits.float().argmax(dim=-1)


_sampler.Sampler.forward = _greedy_forward

from nanovllm import LLM, SamplingParams  # noqa: E402


# 覆盖多种长度：整块 / 半块 / 跨多块；最长的用于强制多 chunk
DEFAULT_LENGTHS = [256, 512, 640, 768, 1024, 1500, 2048, 2560, 3072, 3200, 4096, 512]

MODEL_PATH = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# token 池缓存：必须固定，否则不同分支的源码文本不同 → 池不同 → prompt 不同，跨实现对比失效
POOL_CACHE = os.path.join(REPO_ROOT, "experiments", "a_text_pool.json")


def _text_token_pool():
    """真实文本 token 池。首次生成后缓存到文件，保证跨分支 / 跨次运行拿到完全相同的 prompt。"""
    if os.path.exists(POOL_CACHE):
        with open(POOL_CACHE) as f:
            return json.load(f)
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    chunks = []
    for pat in ("nanovllm/**/*.py", "*.py", "*.md"):
        for f in sorted(glob.glob(os.path.join(REPO_ROOT, pat), recursive=True)):
            with open(f, encoding="utf-8") as fh:
                chunks.append(fh.read())
    ids = tok.encode("\n".join(chunks))
    with open(POOL_CACHE, "w") as f:
        json.dump(ids, f)
    return ids


def make_prompts(mode, seed=0, lengths=None):
    lengths = list(lengths or DEFAULT_LENGTHS)
    if mode == "text":
        pool = _text_token_pool()
        rng = random.Random(seed)
        out = []
        for n in lengths:
            hi = max(0, len(pool) - n - 1)
            s = rng.randint(0, hi) if hi > 0 else 0
            out.append(pool[s:s + n])
        print(f"[prompt] text 池大小={len(pool)} tokens", flush=True)
        return out
    rng = random.Random(seed)
    return [[rng.randint(0, 10000) for _ in range(n)] for n in lengths]


def run_single(args):
    prompts = make_prompts(args.prompt_mode, lengths=args.lengths)
    t0 = time.time()
    llm = LLM(
        MODEL_PATH,
        enforce_eager=args.enforce_eager,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.budget,
    )
    sps = [
        SamplingParams(temperature=1.0, ignore_eos=True, max_tokens=args.max_tokens)
        for _ in prompts
    ]
    outs = llm.generate(prompts, sps, use_tqdm=False)
    ids = [o["token_ids"] for o in outs]
    doc = {
        "tag": args.tag,
        "budget": args.budget,
        "prompt_mode": args.prompt_mode,
        "max_tokens": args.max_tokens,
        "max_model_len": args.max_model_len,
        "enforce_eager": args.enforce_eager,
        "lengths": [len(p) for p in prompts],
        "gpu_blocks": llm.model_runner.config.num_kvcache_blocks,
        "token_ids": ids,
        "seconds": round(time.time() - t0, 2),
    }
    with open(args.out, "w") as f:
        json.dump(doc, f)
    print(f"  [single] budget={args.budget:6d} blocks={doc['gpu_blocks']} "
          f"gen={len(ids)}x{len(ids[0])} t={doc['seconds']}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, help="结果标识，一般用分支名")
    ap.add_argument("--budgets", type=int, nargs="+", default=[256, 512, 2048, 16384])
    ap.add_argument("--repeat", type=int, default=1, help="每个 budget 独立跑几次，用于验证确定性")
    ap.add_argument("--prompt-mode", choices=["random", "text"], default="random")
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--lengths", type=int, nargs="+", default=None)
    ap.add_argument("--enforce-eager", action="store_true")
    ap.add_argument("--out", required=True)
    ap.add_argument("--single", action="store_true", help="内部使用：只跑一个 budget 然后退出")
    ap.add_argument("--budget", type=int, default=0, help="内部使用：配合 --single")
    args = ap.parse_args()

    if args.single:
        run_single(args)
        return

    result = {
        "tag": args.tag,
        "prompt_mode": args.prompt_mode,
        "max_tokens": args.max_tokens,
        "max_model_len": args.max_model_len,
        "enforce_eager": args.enforce_eager,
        "lengths": None,
        "budgets": {},
    }

    here = os.path.abspath(__file__)
    for budget in args.budgets:
        runs, secs, blocks = [], [], None
        for r in range(args.repeat):
            tmp = f"{args.out}.b{budget}.r{r}.tmp.json"
            cmd = [
                sys.executable, here, "--single",
                "--tag", args.tag, "--budget", str(budget),
                "--prompt-mode", args.prompt_mode,
                "--max-tokens", str(args.max_tokens),
                "--max-model-len", str(args.max_model_len),
                "--out", tmp,
            ]
            if args.lengths:
                cmd += ["--lengths"] + [str(x) for x in args.lengths]
            if args.enforce_eager:
                cmd.append("--enforce-eager")
            print(f"[{args.tag}] budget={budget} run {r + 1}/{args.repeat} ...", flush=True)
            p = subprocess.run(cmd, capture_output=True, text=True)
            if p.returncode != 0:
                print(p.stdout, flush=True)
                print(p.stderr, flush=True)
                sys.exit(f"子进程失败：budget={budget} run={r}")
            with open(tmp) as f:
                d = json.load(f)
            os.remove(tmp)
            runs.append(d["token_ids"])
            secs.append(d["seconds"])
            blocks = d["gpu_blocks"]
            result["lengths"] = d["lengths"]

        same = all(x == runs[0] for x in runs)
        result["budgets"][str(budget)] = {
            "gpu_blocks": blocks,
            "runs": runs,
            "seconds": secs,
            "repeat_identical": same,
        }
        print(f"[{args.tag}] budget={budget:6d} blocks={blocks} "
              f"identical={same} t={secs}", flush=True)

    out_dir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f)
    print("WROTE", args.out, flush=True)


if __name__ == "__main__":
    main()
