#!/usr/bin/env python
"""A4 抢占重入验证 —— 采集脚本。

被测风险（P4）
--------------
V1 的调度器在 running 的 decode 分支里，`can_append` 失败时调用 `preempt()`：
把 seq 打回 waiting、`deallocate()` 释放全部 KV 块、`num_cached_tokens = 0`。
之后它被重新 admit 时 `can_allocate()` 走前缀哈希命中，从
`num_cached_blocks * block_size` 处**续算**。若"已缓存 token 数"与块里真实 KV 不符，
就会读脏 KV —— 这是 P4。

设计：把"KV 池大小"作为唯一自变量
----------------------------------
同一批 prompt、同一注入方式（N 条一次性全下）、同一 budget，只改 KV 池块数：

- `--force-blocks 0`   → 用自然池（≈163 块）：总需求放得下 → **不抢占**（对照）
- `--force-blocks 100` → 强制缩小池：总需求放不下 → **必然抢占**（实验）

这样 **prefill 阶段的批组成在两次运行中完全一致**（由 budget 切分同一组 seq 决定，
与池大小无关），差异只出现在 decode 阶段 —— 而那正是被测对象。
（若对照用"逐条串行"，批组成会全盘不同，噪声会淹没信号，实测已验证。）

确定性采样：实验期把 `Sampler.forward` 换成 argmax（同 A1/A2，不改仓库源码）。

记录
----
每个采样事件记录 (prompt_digest, ci, token, top5 logits, 结构量)。
`ci` = 生成序列中的下标（0 基）。仅记录"该次 forward 覆盖了 seq 全部已知 token"的采样
（等价于 V1 的 need_logits；main 无掩码时用它过滤 mid-chunk 的白算采样）。
"""
import argparse
import glob
import hashlib
import json
import os
import random
import subprocess
import sys
import time

import numpy as np
import torch

from nanovllm.layers import sampler as _sampler


PROBE = []
SAMPLES = []
PREEMPT_EVENTS = []
STATS = {"runs": 0, "uniform_runs": 0, "mixed_runs": 0,
         "can_append_fail": 0, "can_allocate_fail": 0,
         "skipped_unused_samples": 0}
RECORD_ON = False
_PREEMPTED = set()
_SEQ_LABEL = {}
FORCE_DELTA = 0
FORCE_MIN = 0
POOL_INFO = {}
TOPK = 20   # 记录 top-20：足够在两次运行间求交，用于按 token id 比较 logits


def _patch_sampler():
    def forward(self, logits, temperatures):
        lg = logits.float()
        PROBE.append(lg)
        return lg.argmax(dim=-1)
    _sampler.Sampler.forward = forward


_patch_sampler()

from nanovllm import LLM, SamplingParams                       # noqa: E402
from nanovllm.engine.model_runner import ModelRunner           # noqa: E402
from nanovllm.engine.scheduler import Scheduler                # noqa: E402
from nanovllm.engine.block_manager import BlockManager         # noqa: E402


def _label_of(seq) -> str:
    lab = _SEQ_LABEL.get(seq.seq_id)
    if lab is None:
        arr = np.asarray(seq.prompt_token_ids, dtype=np.int64)
        lab = hashlib.sha1(arr.tobytes()).hexdigest()[:16]
        _SEQ_LABEL[seq.seq_id] = lab
    return lab


def prompt_digest(tokens) -> str:
    return hashlib.sha1(np.asarray(tokens, dtype=np.int64).tobytes()).hexdigest()[:16]


# ---------------- 埋点 ----------------
_orig_run = ModelRunner.run


def _patched_run(self, seqs, is_uniform):
    PROBE.clear()
    res = _orig_run(self, seqs, is_uniform)
    if isinstance(res, tuple):                 # V1: (token_ids, need_logits)
        token_ids, need_logits = res
    else:                                      # main: token_ids
        token_ids, need_logits = res, [True] * len(seqs)
    if RECORD_ON:
        STATS["runs"] += 1
        STATS["uniform_runs" if is_uniform else "mixed_runs"] += 1
        rows = [s for s, n in zip(seqs, need_logits) if n]
        if PROBE:
            lg = PROBE[0]
            assert lg.size(0) == len(rows), (lg.size(0), len(rows))
            vals, idx = lg.topk(TOPK, dim=-1)
            absmax = lg.abs().max(dim=-1).values
            for j, s in enumerate(rows):
                nt = s.num_tokens
                nc = s.num_cached_tokens
                ns = s.num_scheduled_tokens
                if nc + ns != nt:
                    # mid-chunk 的白算采样（main 无 need_logits 掩码时会出现）
                    STATS["skipped_unused_samples"] += 1
                    continue
                SAMPLES.append({
                    "label": _label_of(s),
                    "ci": (nt - 1) - s.num_prompt_tokens + 1,   # 生成序列下标(0基)
                    "token": int(token_ids[j]),
                    "top_idx": [int(x) for x in idx[j].tolist()],
                    "top_val": [round(float(x), 6) for x in vals[j].tolist()],
                    "absmax": round(float(absmax[j]), 6),
                    "nt": nt, "nc": nc, "nb": len(s.block_table),
                    "pre": bool(s.is_prefill),
                    "preempted": s.seq_id in _PREEMPTED,
                })
    return res


_orig_preempt = Scheduler.preempt


def _patched_preempt(self, seq):
    if RECORD_ON:
        PREEMPT_EVENTS.append({
            "label": _label_of(seq),
            "seq_id": seq.seq_id,
            "len": len(seq),
            "ci_next": len(seq) - seq.num_prompt_tokens,   # 被抢占后要产出的下一个 token 下标
            "num_cached_tokens": seq.num_cached_tokens,
            "blocks": len(seq.block_table),
            "seq_blocks": seq.num_blocks,
        })
        _PREEMPTED.add(seq.seq_id)
    return _orig_preempt(self, seq)


_orig_can_append = BlockManager.can_append


def _patched_can_append(self, seq):
    r = _orig_can_append(self, seq)
    if RECORD_ON and not r:
        STATS["can_append_fail"] += 1
    return r


_orig_can_allocate = BlockManager.can_allocate


def _patched_can_allocate(self, seq):
    r = _orig_can_allocate(self, seq)
    if RECORD_ON and r == -1:
        STATS["can_allocate_fail"] += 1
    return r


_orig_sched_init = Scheduler.__init__


def _patched_sched_init(self, config):
    nat = config.num_kvcache_blocks
    if FORCE_DELTA:
        # 目标 = 自然池 - delta，但不小于 FORCE_MIN（保证 admit 组成与对照一致）
        force = max(FORCE_MIN, nat - FORCE_DELTA)
        if force < nat:
            config.num_kvcache_blocks = force
    _orig_sched_init(self, config)
    config.num_kvcache_blocks = nat
    POOL_INFO["natural_blocks"] = nat
    POOL_INFO["effective_blocks"] = len(self.block_manager.blocks)


ModelRunner.run = _patched_run
Scheduler.preempt = _patched_preempt
Scheduler.__init__ = _patched_sched_init
BlockManager.can_append = _patched_can_append
BlockManager.can_allocate = _patched_can_allocate


MODEL_PATH = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
A4_POOL = os.path.join(REPO_ROOT, "experiments", "a4_text_pool.json")
BLOCK_SIZE = 256


def build_pool(min_tokens: int) -> list[int]:
    if os.path.exists(A4_POOL):
        with open(A4_POOL) as f:
            pool = json.load(f)
        if len(pool) >= min_tokens:
            return pool
    import sysconfig
    from transformers import AutoTokenizer

    stdlib = sysconfig.get_paths()["stdlib"]
    files = sorted(glob.glob(os.path.join(stdlib, "**", "*.py"), recursive=True))
    chunks, nchar = [], 0
    for f in files:
        try:
            with open(f, encoding="utf-8") as fh:
                txt = fh.read()
        except (UnicodeDecodeError, OSError):
            continue
        chunks.append(txt)
        nchar += len(txt)
        if nchar > 1_200_000:
            break
    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    ids = tok.encode("\n".join(chunks))
    with open(A4_POOL, "w") as f:
        json.dump(ids, f)
    print(f"[pool] 新建 A4 文本池：{len(ids)} tokens（源 {stdlib}，{len(chunks)} 文件）", flush=True)
    return ids


def make_prompts(n: int, plen: int, seed: int = 20261012):
    """取自池中的 n 个**互不重叠**窗口 → prompt 之间不会前缀缓存命中。

    保留区用固定常数（与 n 无关），保证不同 n 取到的是**同一组 prompt 的前缀**，
    这样 "N 条 vs N-1 条" 的噪声地板对照才成立。
    """
    RESERVE = 64 * plen
    pool = build_pool(RESERVE + n * plen + 4096)
    rng = random.Random(seed)
    hi = max(0, len(pool) - RESERVE - 1)
    base = rng.randint(0, hi) if hi > 0 else 0
    assert base + n * plen <= len(pool), "池不足"
    return [pool[base + i * plen: base + (i + 1) * plen] for i in range(n)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--plen", type=int, default=2048)
    ap.add_argument("--o-base", type=int, default=64)
    ap.add_argument("--o-step", type=int, default=16)
    ap.add_argument("--o-fixed", type=int, default=0)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--budget", type=int, default=16384)
    ap.add_argument("--util", type=float, default=0.90)
    ap.add_argument("--force-blocks", type=int, default=0,
                    help="0 = 自然池；>0 = 强制缩小到该块数（必须 < 自然池）")
    ap.add_argument("--force-delta", type=int, default=0,
                    help=">0 = 强制池 = 自然池 - 该值（下限由 admissibility 决定）")
    ap.add_argument("--enforce-eager", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    global RECORD_ON, FORCE_DELTA, FORCE_MIN
    prompts = make_prompts(args.n, args.plen)
    o_list = ([args.o_fixed] * args.n if args.o_fixed > 0
              else [args.o_base + args.o_step * i for i in range(args.n)])
    demand_blocks = sum((args.plen + o + BLOCK_SIZE - 1) // BLOCK_SIZE for o in o_list)
    # 全部请求被 admit 所需块数（保证实验组与对照组的 admit / prefill 批组成一致）
    admit_need = sum((args.plen + BLOCK_SIZE - 1) // BLOCK_SIZE for _ in o_list)
    FORCE_DELTA = args.force_delta
    FORCE_MIN = admit_need

    llm = LLM(MODEL_PATH, enforce_eager=args.enforce_eager,
              max_model_len=args.max_model_len,
              max_num_batched_tokens=args.budget,
              gpu_memory_utilization=args.util)
    blocks = POOL_INFO["effective_blocks"]
    print(f"[{args.tag}] 自然池={POOL_INFO['natural_blocks']} 生效池={blocks}  "
          f"admit需={admit_need}  总需求≈{demand_blocks}  → "
          f"{'必然抢占' if demand_blocks > blocks else '不应抢占'}"
          f"{'（注意：池 < admit 需，admit 组成与对照不同）' if blocks < admit_need else ''}",
          flush=True)

    sps = [SamplingParams(temperature=1.0, ignore_eos=True, max_tokens=o) for o in o_list]

    RECORD_ON = True
    t0 = time.time()
    outs = llm.generate(prompts, sps, use_tqdm=False)
    outs = [o["token_ids"] for o in outs]
    RECORD_ON = False
    secs = round(time.time() - t0, 2)

    doc = {
        "tag": args.tag,
        "branch": subprocess.run(["git", "branch", "--show-current"], cwd=REPO_ROOT,
                                 capture_output=True, text=True).stdout.strip(),
        "n": args.n, "plen": args.plen, "o_list": o_list,
        "budget": args.budget, "util": args.util,
        "max_model_len": args.max_model_len, "enforce_eager": args.enforce_eager,
        "natural_blocks": POOL_INFO["natural_blocks"],
        "gpu_blocks": blocks,
        "demand_blocks": demand_blocks,
        "stats": STATS,
        "preempt_events": PREEMPT_EVENTS,
        "preempt_count": len(PREEMPT_EVENTS),
        "seconds": secs,
        "outputs": [
            {"label": prompt_digest(prompts[i]), "o": o_list[i],
             "token_ids": outs[i], "len": len(outs[i])}
            for i in range(args.n)
        ],
        "samples": SAMPLES,
    }
    with open(args.out, "w") as f:
        json.dump(doc, f)
    print(f"[{args.tag}] preempt={doc['preempt_count']} "
          f"can_append_fail={STATS['can_append_fail']} "
          f"can_alloc_fail={STATS['can_allocate_fail']} "
          f"uniform_runs={STATS['uniform_runs']} mixed_runs={STATS['mixed_runs']} "
          f"samples={len(SAMPLES)} skipped={STATS['skipped_unused_samples']} t={secs}s", flush=True)
    print("WROTE", args.out, flush=True)


if __name__ == "__main__":
    main()
