#!/usr/bin/env python
"""C2 混合批的物理基础：P × D 网格上的四量分解。

对每个 (P, D)：
    t_p            只跑 prefill（P token，一个 chunk）
    t_d            只跑 decode（D 条，历史 dlen，走 decode 入口 = paged kernel + graph）
    t_d_generic    D 条 decode 行走 prefill 入口、强制 num_decode=0 → 整批 varlen
    t_mix          真实混合批（prefill chunk + D 条 decode），当前实现
    t_mix_varlen   同批、强制 num_decode=0 → 整批 varlen（对照：不分路会怎样）

判据：
    H1  t_mix < t_p + t_d                     （混合批是否省一步）
    H2  t_mix ≈ max(t_p, t_d)                 （是否"decode 白送"）
    失效根因：t_mix 与 t_mix_varlen 的差，即「decode 行走了哪个算子」。

用法：
    .venv/bin/python experiments/c2_grid.py --ps 512 1024 2048 4096 \
        --ds 1 4 8 16 32 64 --out experiments/results/c2_grid_V1.json
"""
import argparse
import inspect
import json
import os
import statistics

import torch

torch._dynamo.config.recompile_limit = 256
torch._dynamo.config.cache_size_limit = 256

from nanovllm import LLM
from nanovllm.engine.sequence import Sequence
from nanovllm.sampling_params import SamplingParams
import nanovllm.engine.model_runner as MR
import nanovllm.layers.attention as ATTN

MODEL_PATH = os.path.expanduser("~/huggingface/Qwen3-0.6B/")

_STATE = {"force_nd": None}
_orig_set_context = MR.set_context
_orig_varlen = ATTN.flash_attn_varlen_func


def _set_context(is_prefill, num_decode=0, *args, **kwargs):
    if is_prefill and _STATE["force_nd"] is not None:
        num_decode = _STATE["force_nd"]
    return _orig_set_context(is_prefill, num_decode, *args, **kwargs)


def _varlen(q, k, v, *a, **kw):
    if q.shape[0] == 0:
        return q.new_zeros((0, q.shape[1], v.shape[-1]))
    return _orig_varlen(q, k, v, *a, **kw)


MR.set_context = _set_context
ATTN.flash_attn_varlen_func = _varlen


def pool_ids(length):
    return list(range(1000, 1000 + length))


def build_prefill(P, bsz, block_off):
    s = Sequence(pool_ids(P), SamplingParams(temperature=1e-6, max_tokens=1, ignore_eos=True))
    s.block_table = list(range(block_off, block_off + (P + bsz - 1) // bsz))
    s.num_cached_tokens = 0
    s.num_scheduled_tokens = P
    return s


def build_decode(dlen, bsz, block_off):
    s = Sequence(pool_ids(dlen), SamplingParams(temperature=1e-6, max_tokens=1, ignore_eos=True))
    s.block_table = list(range(block_off, block_off + (dlen + bsz - 1) // bsz))
    s.num_cached_tokens = dlen - 1
    s.num_scheduled_tokens = 1
    return s


def fill_kv(runner, seqs, prefill_flag):
    target = [(s.num_scheduled_tokens, s.num_cached_tokens) for s in seqs]
    for s in seqs:
        s.num_scheduled_tokens, s.num_cached_tokens = len(s), 0
    runner.run(seqs, prefill_flag)
    for s, (sq, ca) in zip(seqs, target):
        s.num_scheduled_tokens, s.num_cached_tokens = sq, ca


def measure_rr(specs, repeat, warmup):
    for _ in range(warmup):
        for _, setter, fn in specs:
            setter()
            fn()
    torch.cuda.synchronize()
    samples = {name: [] for name, _, _ in specs}
    for _ in range(repeat):
        for name, setter, fn in specs:
            setter()
            e0 = torch.cuda.Event(enable_timing=True)
            e1 = torch.cuda.Event(enable_timing=True)
            e0.record()
            fn()
            e1.record()
            torch.cuda.synchronize()
            samples[name].append(e0.elapsed_time(e1))
    return {k: {"min": min(v), "med": statistics.median(v)} for k, v in samples.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ps", type=int, nargs="+", default=[512, 1024, 2048, 4096])
    ap.add_argument("--ds", type=int, nargs="+", default=[1, 4, 8, 16, 32, 64])
    ap.add_argument("--dlen", type=int, default=2048, help="decode seq 的历史长度")
    ap.add_argument("--repeat", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--budget", type=int, default=16384)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--tag", type=str, default="V1")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    torch.manual_seed(0)
    llm = LLM(MODEL_PATH, enforce_eager=False, max_model_len=args.max_model_len,
              max_num_batched_tokens=args.budget)
    runner = llm.model_runner
    cfg = runner.config
    bsz = cfg.kvcache_block_size

    pname = list(inspect.signature(runner.run).parameters)[1]
    PREFILL, DECODE = (False, True) if pname == "is_uniform" else (True, False)
    print(f"[{args.tag}] run(seqs, {pname}) PREFILL={PREFILL} DECODE={DECODE} "
          f"| blocks={cfg.num_kvcache_blocks} bsz={bsz} dlen={args.dlen}", flush=True)

    DEC_OFF = 0                       # decode 行共用同一片块（只测时间）
    P_OFF = 32 + (args.dlen + bsz - 1) // bsz

    rows = []
    for P in args.ps:
        for D in args.ds:
            pre = build_prefill(P, bsz, P_OFF)
            dec = [build_decode(args.dlen, bsz, DEC_OFF) for _ in range(D)]
            fill_kv(runner, dec, PREFILL)

            mixed = dec + [pre]       # 行序不变量：decode 行在前

            def set_nat():
                _STATE["force_nd"] = None

            def set_gen():
                _STATE["force_nd"] = 0

            run_pre_mixed = lambda: runner.run(mixed, PREFILL)
            run_pre_dec = lambda: runner.run(dec, PREFILL)
            run_pre_p = lambda: runner.run([pre], PREFILL)
            run_dec = lambda: runner.run(dec, DECODE)

            st = measure_rr([
                ("t_p", set_nat, run_pre_p),
                ("t_d", set_nat, run_dec),
                ("t_d_generic", set_gen, run_pre_dec),
                ("t_mix", set_nat, run_pre_mixed),
                ("t_mix_varlen", set_gen, run_pre_mixed),
            ], args.repeat, args.warmup)

            tp, td = st["t_p"]["min"], st["t_d"]["min"]
            tm, tmv = st["t_mix"]["min"], st["t_mix_varlen"]["min"]
            row = {
                "plen": P, "dlen": args.dlen, "D": D,
                "prefill_tokens": P, "decode_tokens": D, "total_tokens": P + D,
                "fill_ratio": round((P + D) / args.budget, 4),
                "t_p": st["t_p"], "t_d": st["t_d"],
                "t_d_generic": st["t_d_generic"],
                "t_mix": st["t_mix"], "t_mix_varlen": st["t_mix_varlen"],
                "t_sum": round(tp + td, 4),
                "mix_over_sum": round(tm / (tp + td), 3),
                "mix_over_max": round(tm / max(tp, td), 3),
                "mix_over_p": round(tm / tp, 3),
                "mixv_over_sum": round(tmv / (tp + td), 3),
                "split_gain_ms": round(tmv - tm, 4),
            }
            rows.append(row)
            print("P=%4d D=%2d | t_p=%7.2f t_d=%6.2f sum=%7.2f | mix=%7.2f (sum×%.3f) "
                  "mix_varlen=%7.2f (sum×%.3f) | 分路省 %+6.2f ms"
                  % (P, D, tp, td, tp + td, tm, tm / (tp + td), tmv, tmv / (tp + td), tmv - tm),
                  flush=True)

    doc = {"tag": args.tag, "args": vars(args),
           "pool": {"num_blocks": cfg.num_kvcache_blocks, "block_size": bsz},
           "results": rows}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(doc, open(args.out, "w"), indent=2)
    print("->", args.out)


if __name__ == "__main__":
    main()
