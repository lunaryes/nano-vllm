#!/usr/bin/env python
"""C3b：混合批里 decode 行的「路径单价」——split 改动后的重做版。

背景
----
改动前，混合步（prefill chunk + decode）里 decode 行也走 `flash_attn_varlen_func`
+ 整块 cache；改动后，decode 行改走 `flash_attn_with_kvcache`（paged decode kernel），
prefill 行仍走 varlen。于是同一个 decode 行有三条可能的路：

    t_d_generic   PREFILL 入口，强制 num_decode=0  → 整批 varlen（改动前的行为）
    t_d_eager     PREFILL 入口，num_decode=N       → 新 split 支路，eager
    t_d_graph     DECODE  入口                      → graph replay（只在纯 decode 批可用）

这三者不是同一个入口，直接相减会把「入口/拼装差异」混进来。因此加一个隔离列：

    t_d_eager_pure  DECODE 入口，eager            → 与 t_d_graph 同入口、只差 graph

于是三笔账可以分开算（同一批 N 行、同一份 KV、同一个 T=N）：
    分路收益（本次改动） = t_d_generic - t_d_eager
    graph 收益           = t_d_eager_pure - t_d_graph
    入口/拼装开销        = t_d_eager - t_d_eager_pure

面板 B 用真实混合批（nd 条 decode + 1 条 prefill chunk，T≈1056）验证分路收益
在「批里真的有 prefill」时是否还在——这是 C6 对旧 C3 的核心批评（旧 C3 的 T 只有 32）。

用法
----
    .venv/bin/python experiments/c3b_split.py --stage a --out experiments/results/c3b_a.json
    .venv/bin/python experiments/c3b_split.py --stage b --out experiments/results/c3b_b.json
    .venv/bin/python experiments/c3b_split.py --stage check        # 数值等价性
"""
import argparse
import inspect
import json
import os
import statistics

import torch

# 固定编译状态：dynamo 默认 recompile_limit=8，本实验会出现 10+ 种形状，
# 一旦触顶就永久回退 eager，不同 D 会落在不同编译状态，横向不可比。
torch._dynamo.config.recompile_limit = 256
torch._dynamo.config.cache_size_limit = 256

from nanovllm import LLM
from nanovllm.engine.sequence import Sequence
from nanovllm.sampling_params import SamplingParams
import nanovllm.engine.model_runner as MR
import nanovllm.layers.attention as ATTN

MODEL_PATH = os.path.expanduser("~/huggingface/Qwen3-0.6B/")

# ---------------------------------------------------------------- 注入的开关
_STATE = {"force_nd": None, "calls": {"varlen": 0, "kvcache": 0}}

_orig_set_context = MR.set_context
_orig_varlen = ATTN.flash_attn_varlen_func
_orig_kvcache = ATTN.flash_attn_with_kvcache


def _set_context(is_prefill, num_decode=0, *args, **kwargs):
    """唯一被改写的东西：允许把 num_decode 强制成指定值（用于 generic 档）。"""
    if is_prefill and _STATE["force_nd"] is not None:
        num_decode = _STATE["force_nd"]
    return _orig_set_context(is_prefill, num_decode, *args, **kwargs)


def _varlen(q, k, v, *a, **kw):
    _STATE["calls"]["varlen"] += 1
    if q.shape[0] == 0:
        # 真实调度不会出现（纯 decode 步走 DECODE 入口），但合成「全是 decode 行」
        # 的批时 nd == n_seq，prefill 段为空；这里只把这一种退化调用短路掉。
        return q.new_zeros((0, q.shape[1], v.shape[-1]))
    return _orig_varlen(q, k, v, *a, **kw)


def _kvcache(q, kc, vc, *a, **kw):
    _STATE["calls"]["kvcache"] += 1
    return _orig_kvcache(q, kc, vc, *a, **kw)


ATTN.flash_attn_varlen_func = _varlen
ATTN.flash_attn_with_kvcache = _kvcache
MR.set_context = _set_context


# ---------------------------------------------------------------- 工具
def pool_ids(length):
    return list(range(1000, 1000 + length))


def build_rows(n_tok, bsz, n_seq, block_off, dlen):
    """n_seq 条 seq：本条步内算 n_tok 个 token，之前有 dlen-n_tok 个历史。"""
    nb = (dlen + bsz - 1) // bsz
    ids = pool_ids(dlen)
    seqs = []
    for _ in range(n_seq):
        s = Sequence(list(ids), SamplingParams(temperature=1e-6, max_tokens=1, ignore_eos=True))
        s.block_table = list(range(block_off, block_off + nb))
        s.num_cached_tokens = dlen - n_tok
        s.num_scheduled_tokens = n_tok
        seqs.append(s)
    return seqs


def fill_kv(runner, seqs, prefill_flag):
    """把历史 KV 真正写进池（否则读到未初始化显存，数值比较无意义）。

    跑完整 prefill 一遍，然后按调用方原本设定的 (scheduled, cached) 复原，
    这样「全 decode 行」和「decode 行 + prefill chunk」两种批都能复用。
    """
    target = [(s.num_scheduled_tokens, s.num_cached_tokens) for s in seqs]
    for s in seqs:
        s.num_scheduled_tokens, s.num_cached_tokens = len(s), 0
    runner.run(seqs, prefill_flag)
    for s, (sq, ca) in zip(seqs, target):
        s.num_scheduled_tokens, s.num_cached_tokens = sq, ca


def measure_rr(specs, repeat, warmup):
    """轮转交错计时：每一轮里把各变体各跑一次，抵消时钟/热漂移与顺序效应。

    specs: [(name, set_state_fn, run_fn)]
    """
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
    out = {}
    for name, ts in samples.items():
        ts.sort()
        out[name] = {"min": ts[0], "p25": ts[max(0, int(0.25 * len(ts)) - 1)],
                     "med": statistics.median(ts)}
    return out


def call_ratio(labels):
    """跑一次，返回 (varlen 次数, kvcache 次数)，用来证明分支真的激活了。"""
    _STATE["calls"] = {"varlen": 0, "kvcache": 0}
    labels()
    return dict(_STATE["calls"])


# ---------------------------------------------------------------- 面板 A
def panel_a(args, runner, prefill_flag, decode_flag, bsz):
    rows = []
    for N in args.ds:
        _STATE["force_nd"] = None
        seqs = build_rows(1, bsz, N, 0, args.dlen)
        fill_kv(runner, seqs, prefill_flag)

        def set_generic():
            _STATE["force_nd"] = 0
            runner.enforce_eager = True

        def set_split():
            _STATE["force_nd"] = None
            runner.enforce_eager = True

        def set_graph():
            _STATE["force_nd"] = None
            runner.enforce_eager = False

        run_pre = lambda: runner.run(seqs, prefill_flag)
        run_dec = lambda: runner.run(seqs, decode_flag)

        # 激活门：先确认每条路径真的被走到了（28 层各一次）
        gate = {}
        for name, setter, fn in (("t_d_generic", set_generic, run_pre),
                                 ("t_d_eager", set_split, run_pre),
                                 ("t_d_eager_pure", set_split, run_dec),
                                 ("t_d_graph", set_graph, run_dec)):
            setter()
            gate[name] = call_ratio(fn)

        stats = measure_rr([("t_d_generic", set_generic, run_pre),
                            ("t_d_eager", set_split, run_pre),
                            ("t_d_eager_pure", set_split, run_dec),
                            ("t_d_graph", set_graph, run_dec)],
                           args.repeat, args.warmup)

        g = stats["t_d_generic"]["min"]
        e = stats["t_d_eager"]["min"]
        ep = stats["t_d_eager_pure"]["min"]
        gr = stats["t_d_graph"]["min"]
        row = {"N": N,
               "t_d_generic": round(g, 4), "t_d_eager": round(e, 4),
               "t_d_eager_pure": round(ep, 4), "t_d_graph": round(gr, 4),
               "p25": {k: round(v["p25"], 4) for k, v in stats.items()},
               "med": {k: round(v["med"], 4) for k, v in stats.items()},
               "split_gain_ms": round(g - e, 4),
               "split_over_generic": round(e / g, 3),
               "graph_gain_ms": round(ep - gr, 4),
               "graph_over_eager": round(gr / ep, 3),
               "entry_overhead_ms": round(e - ep, 4),
               "calls": gate,
               "us_per_tok_generic": round(g * 1000.0 / N, 2),
               "us_per_tok_split": round(e * 1000.0 / N, 2)}
        rows.append(row)
        print("  N=%3d | gen=%8.3f  split=%8.3f  dec_eager=%8.3f  graph=%8.3f ms"
              " | split/gen=%.2f  graph/eager=%.2f  entry=%+.2f ms"
              % (N, g, e, ep, gr, e / g, gr / ep, e - ep), flush=True)
    return rows


# ---------------------------------------------------------------- 面板 B
def panel_b(args, runner, prefill_flag, bsz):
    """真实混合批：nd 条 decode 行 + 1 条 prefill chunk，总 T = c + nd。"""
    rows = []
    for c, nd in args.mix:
        _STATE["force_nd"] = None
        dec = build_rows(1, bsz, nd, 0, args.dlen)
        pre = build_rows(c, bsz, 1, args.prefill_block_off, c)
        seqs = dec + pre
        fill_kv(runner, seqs, prefill_flag)

        T = c + nd

        def set_generic():
            _STATE["force_nd"] = 0
            runner.enforce_eager = True

        def set_split():
            _STATE["force_nd"] = None
            runner.enforce_eager = True

        run_pre = lambda: runner.run(seqs, prefill_flag)
        gate = {}
        for name, setter in (("t_d_generic", set_generic), ("t_d_eager", set_split)):
            setter()
            gate[name] = call_ratio(run_pre)

        stats = measure_rr([("t_d_generic", set_generic, run_pre),
                            ("t_d_eager", set_split, run_pre)],
                           args.repeat, args.warmup)
        tg, ts = stats["t_d_generic"]["min"], stats["t_d_eager"]["min"]
        row = {"T": T, "chunk": c, "nd": nd,
               "t_generic_ms": round(tg, 4), "t_split_ms": round(ts, 4),
               "med": {k: round(v["med"], 4) for k, v in stats.items()},
               "gain_ms": round(tg - ts, 4),
               "split_over_generic": round(ts / tg, 4),
               "us_per_tok_generic": round(tg * 1000.0 / T, 3),
               "us_per_tok_split": round(ts * 1000.0 / T, 3),
               "calls": gate}
        rows.append(row)
        print("  T=%4d (chunk=%4d + nd=%2d) | generic=%8.3f  split=%8.3f ms"
              " | split/gen=%.4f  gain=%+.3f ms" % (T, c, nd, tg, ts, ts / tg, tg - ts),
              flush=True)
    return rows


# ---------------------------------------------------------------- 等价性
def stage_check(args, runner, prefill_flag, bsz):
    """同批数据、四条路径，比 logits 与 argmax token。"""
    cap = {}

    class _Cap:
        def __call__(self, logits, temperatures):
            cap["logits"] = logits.float().clone()
            return logits.argmax(dim=-1)

    real_sampler = runner.sampler
    runner.sampler = _Cap()

    def grab(nd_force, decode_entry, eager):
        _STATE["force_nd"] = nd_force
        runner.enforce_eager = eager
        runner.run(seqs, decode_flag if decode_entry else prefill_flag)
        return cap["logits"].clone()

    decode_flag = not prefill_flag
    out = {}
    for label, (nd, dec, eager) in {
        "generic": (0, False, True),
        "split": (None, False, True),
        "eager_pure": (None, True, True),
        "graph": (None, True, False),
    }.items():
        _STATE["force_nd"] = None
        N = 8
        seqs = build_rows(1, bsz, N, 0, args.dlen)
        fill_kv(runner, seqs, prefill_flag)
        out[label] = grab(nd, dec, eager)
    ref = out["generic"]
    print("参照 = generic（改动前的行为）")
    for label in ("split", "eager_pure", "graph"):
        d = (ref - out[label]).abs()
        tokm = (ref.argmax(-1) == out[label].argmax(-1)).float().mean().item()
        print("  %-11s max|dlogit|=%.3e  mean=%.3e  相同元素=%.4f%%  argmax 一致=%.1f%%"
              % (label, d.max(), d.mean(), 100 * (d == 0).float().mean(), 100 * tokm))

    # 混合批：generic 与 split 必须算同一件事（否则面板 B 的时间差没有意义）
    print()
    print("混合批等价性（chunk + nd 条 decode）")
    for c, nd in ((256, 8), (1024, 32)):
        _STATE["force_nd"] = None
        dec = build_rows(1, bsz, nd, 0, args.dlen)
        pre = build_rows(c, bsz, 1, args.prefill_block_off, c)
        seqs = dec + pre
        fill_kv(runner, seqs, prefill_flag)
        _STATE["force_nd"] = 0
        runner.enforce_eager = True
        runner.run(seqs, prefill_flag)
        a = cap["logits"].clone()
        _STATE["force_nd"] = None
        runner.run(seqs, prefill_flag)
        b = cap["logits"].clone()
        d = (a - b).abs()
        print("  chunk=%4d nd=%2d (T=%4d): 行数=%d  max|dlogit|=%.3e  相同元素=%.4f%%"
              % (c, nd, c + nd, a.shape[0], d.max(), 100 * (d == 0).float().mean()))

    runner.sampler = real_sampler
    runner.enforce_eager = False
    return {k: v.tolist() for k, v in out.items()}


# ---------------------------------------------------------------- 机制探针
def stage_unit(args):
    """算子级：把 N 条 1-token 的 "decode 行" 和一条长度 Q 的 prefill 行放进同一个
    varlen 调用，看 decode 行的单价是否随 Q 上涨（= 被按 max_seqlen_q 做 q 维 padding）。

    三种批：
        A = N 条 decode + 1 条 prefill(Q)
        B = 1 条 prefill(Q)
        C = N 条 decode（单独，max_seqlen_q == 1）
    A-B 就是 decode 行在「异质批」里的真实代价，和 C 对比。
    """
    from flash_attn import flash_attn_varlen_func
    H, KVH, D = 16, 8, 128
    NB, BS, DL = 160, 256, 2048
    SCALE = D ** -0.5
    torch.manual_seed(0)
    kc = torch.randn(NB, BS, KVH, D, dtype=torch.float16, device="cuda") * 0.1
    vc = torch.randn(NB, BS, KVH, D, dtype=torch.float16, device="cuda") * 0.1
    bt_dec = torch.zeros(args.unit_n, 8, dtype=torch.int32, device="cuda")
    for i in range(args.unit_n):
        bt_dec[i] = torch.arange(8, dtype=torch.int32, device="cuda")

    def timeit(fn, repeat=200, warmup=20):
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        ts = []
        for _ in range(repeat):
            e0 = torch.cuda.Event(enable_timing=True)
            e1 = torch.cuda.Event(enable_timing=True)
            e0.record()
            fn()
            e1.record()
            torch.cuda.synchronize()
            ts.append(e0.elapsed_time(e1))
        return min(ts)

    N = args.unit_n
    print("decode 行 N=%d，dlen=%d；A-B = decode 行在异质批里的实付" % (N, DL))
    print("%6s %10s %10s %10s %10s" % ("Q", "A(dec+pre)", "B(pre)", "A-B", "C(dec alone)"))
    rows = []
    for Q in args.unit_qs:
        # C：N 条 decode 单独成批（max_seqlen_q == 1），仍走 paged
        q_c = torch.randn(N, H, D, dtype=torch.float16, device="cuda") * 0.5
        cu_dec = torch.arange(0, N * DL + 1, DL, dtype=torch.int32, device="cuda")
        cu_qc = torch.arange(0, N + 1, 1, dtype=torch.int32, device="cuda")
        tC = timeit(lambda: flash_attn_varlen_func(
            q_c, kc, vc, max_seqlen_q=1, cu_seqlens_q=cu_qc,
            max_seqlen_k=DL, cu_seqlens_k=cu_dec, softmax_scale=SCALE,
            causal=True, block_table=bt_dec))

        # A：N decode + 1 prefill(Q)
        q_a = torch.randn(N + Q, H, D, dtype=torch.float16, device="cuda") * 0.5
        cu_q_a = torch.cat([torch.arange(0, N + 1, 1, dtype=torch.int32, device="cuda"),
                            torch.tensor([N + Q], dtype=torch.int32, device="cuda")])
        cu_k_a = torch.cat([cu_dec, torch.tensor([N * DL + Q], dtype=torch.int32, device="cuda")])
        bt_a = torch.zeros(N + 1, max(8, (Q + BS - 1) // BS), dtype=torch.int32, device="cuda")
        bt_a[:N, :8] = bt_dec
        bt_a[N, :(Q + BS - 1) // BS] = torch.arange(
            64, 64 + (Q + BS - 1) // BS, dtype=torch.int32, device="cuda")
        tA = timeit(lambda: flash_attn_varlen_func(
            q_a, kc, vc, max_seqlen_q=Q, cu_seqlens_q=cu_q_a,
            max_seqlen_k=DL, cu_seqlens_k=cu_k_a, softmax_scale=SCALE,
            causal=True, block_table=bt_a))

        # B：只有 prefill(Q)
        q_b = q_a[N:].contiguous()
        cu_q_b = torch.tensor([0, Q], dtype=torch.int32, device="cuda")
        cu_k_b = torch.tensor([0, Q], dtype=torch.int32, device="cuda")
        bt_b = torch.zeros(1, (Q + BS - 1) // BS, dtype=torch.int32, device="cuda")
        bt_b[0] = torch.arange(64, 64 + (Q + BS - 1) // BS, dtype=torch.int32, device="cuda")
        tB = timeit(lambda: flash_attn_varlen_func(
            q_b, kc, vc, max_seqlen_q=Q, cu_seqlens_q=cu_q_b,
            max_seqlen_k=Q, cu_seqlens_k=cu_k_b, softmax_scale=SCALE,
            causal=True, block_table=bt_b))

        rows.append({"Q": Q, "A": round(tA, 4), "B": round(tB, 4),
                     "A_minus_B": round(tA - tB, 4), "C": round(tC, 4),
                     "padding_factor": round((tA - tB) / tC, 3) if tC else None})
        print("%6d %10.3f %10.3f %10.3f %10.3f    (A-B)/C = %.2f×"
              % (Q, tA, tB, tA - tB, tC, (tA - tB) / tC), flush=True)
    return rows


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="a", choices=["a", "b", "check", "unit"])
    ap.add_argument("--dlen", type=int, default=2048)
    ap.add_argument("--ds", type=int, nargs="+", default=[1, 8, 16, 32, 64])
    ap.add_argument("--mix", type=int, nargs="+", action="append",
                    default=None, help="每项两个数：chunk 长度 nd，可重复")
    ap.add_argument("--repeat", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--rounds", type=int, default=1)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--max-num-seqs", type=int, default=64)
    ap.add_argument("--prefill-block-off", type=int, default=64)
    ap.add_argument("--unit-n", type=int, default=32)
    ap.add_argument("--unit-qs", type=int, nargs="+", default=[1, 128, 256, 512, 1024, 2048])
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    if args.mix is None:
        args.mix = [[512, 1], [512, 8], [512, 32], [1024, 32]]

    if args.stage == "unit":
        rows = stage_unit(args)
        if args.out:
            os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
            json.dump({"stage": "unit", "unit_n": args.unit_n, "rows": rows},
                      open(args.out, "w"), indent=2)
            print("-> %s" % args.out)
        return

    torch.manual_seed(0)
    llm = LLM(MODEL_PATH, enforce_eager=False,
              max_model_len=args.max_model_len,
              max_num_seqs=args.max_num_seqs,
              max_num_batched_tokens=16384,
              gpu_memory_utilization=0.85)
    runner = llm.model_runner
    bsz = runner.config.kvcache_block_size

    pname = list(inspect.signature(runner.run).parameters)[1]
    prefill_flag = pname == "is_prefill"
    decode_flag = not prefill_flag
    print("入口语义: 第二参数 %s → prefill=%s decode=%s | blocks=%d bsz=%d"
          % (pname, prefill_flag, decode_flag,
             runner.config.num_kvcache_blocks, bsz), flush=True)

    if args.stage == "check":
        res = stage_check(args, runner, prefill_flag, bsz)
        if args.out:
            os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
            json.dump(res, open(args.out, "w"))
        return

    if args.stage == "a":
        rows = panel_a(args, runner, prefill_flag, decode_flag, bsz)
        doc = {"stage": "a", "dlen": args.dlen, "repeat": args.repeat,
               "entry": pname, "rows": rows}
    else:
        rows = panel_b(args, runner, prefill_flag, bsz)
        doc = {"stage": "b", "dlen": args.dlen, "repeat": args.repeat,
               "entry": pname, "rows": rows}

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        json.dump(doc, open(args.out, "w"), indent=2)
        print("-> %s" % args.out)


if __name__ == "__main__":
    main()
