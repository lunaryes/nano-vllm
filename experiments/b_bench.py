#!/usr/bin/env python
"""B 组 benchmark：标准端到端吞吐（对齐仓库根 bench.py 口径，两分支共用）。

用法：
    python experiments/b_bench.py --tag V1 --repeat 2 --out experiments/results/b_V1_bench.json

口径（= 仓库根 bench.py；该文件两分支完全一致，git diff 为空）：
    num_seqs=256   prompt_len~U[100,1024]   output_len~U[100,1024]
    max_model_len=4096   enforce_eager=False   temperature=0.6   ignore_eos=True
    max_num_batched_tokens=16384（Config 默认）   max_num_seqs=512（不设上限）

与 B1/B2/B3 的区别：
    B1/B2/B3 用受控负载（固定 19 块 prompt + max_num_seqs=8 -> preempt=0），测的是
    「ITL 分布形状」；本实验用官方不受控负载（KV 池严重超订 -> 大量抢占），测的是
    「标准吞吐」，回答「V1 的混合调度在真实压力下是否牺牲吞吐」。

记录：wall / 总 token / 吞吐 / 抢占计数 / 步级 (dt, np, nd) / TTFT / ITL 分位。
"""
import argparse
import json
import os
import subprocess
import time
from random import Random

from nanovllm.layers import sampler as _sampler

COUNTERS = {"preempt": 0, "can_allocate_fail": 0, "can_append_fail": 0}
RECORD_ON = False


def _patch_sampler_argmax():
    def forward(self, logits, temperatures):
        return logits.float().argmax(dim=-1)
    _sampler.Sampler.forward = forward


from nanovllm import LLM, SamplingParams                       # noqa: E402
from nanovllm.engine.scheduler import Scheduler                # noqa: E402
from nanovllm.engine.block_manager import BlockManager         # noqa: E402


def _patch_counters():
    _pre = Scheduler.preempt

    def pre(self, seq):
        if RECORD_ON:
            COUNTERS["preempt"] += 1
        return _pre(self, seq)

    Scheduler.preempt = pre

    _ca = BlockManager.can_allocate

    def ca(self, seq):
        r = _ca(self, seq)
        if RECORD_ON and r == -1:
            COUNTERS["can_allocate_fail"] += 1
        return r

    BlockManager.can_allocate = ca

    _cp = BlockManager.can_append

    def cp(self, seq):
        r = _cp(self, seq)
        if RECORD_ON and not r:
            COUNTERS["can_append_fail"] += 1
        return r

    BlockManager.can_append = cp


_patch_counters()

# 官方 bench.py 的负载（seed(0) 下的 randint 序列）
BENCH = dict(n=256, in_max=1024, out_max=1024, ml=4096, budget=16384, temp=0.6)
MODEL = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def pct(xs, p):
    if not xs:
        return None
    xs = sorted(xs)
    k = int(round((p / 100.0) * (len(xs) - 1)))
    return xs[min(max(k, 0), len(xs) - 1)]


def mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def stdev(xs):
    if len(xs) < 2:
        return 0.0
    m = mean(xs)
    return (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5


def summarize(vs):
    if not vs:
        return None
    m = mean(vs)
    return {"n": len(vs), "mean": round(m, 4), "p50": round(pct(vs, 50), 4),
            "p90": round(pct(vs, 90), 4), "p99": round(pct(vs, 99), 4),
            "max": round(max(vs), 4), "std": round(stdev(vs), 4),
            "cv": round(stdev(vs) / m, 4) if m else None}


def run_once(llm, prompts, o_list, temp):
    """返回单轮：wall、步级数据、emit 时刻、抢占增量、prefill/decode token 数。"""
    global RECORD_ON
    c0 = dict(COUNTERS)
    emit, last_k = {}, {}

    RECORD_ON = True
    t0 = time.perf_counter()
    for p, o in zip(prompts, o_list):
        llm.add_request(p, SamplingParams(temperature=temp, ignore_eos=True, max_tokens=o))
    arrival = {s.seq_id: t0 for s in llm.scheduler.waiting}

    step_dt, step_p, step_d = [], [], []
    prev = time.perf_counter()
    while not llm.is_finished():
        res = llm.step()
        t = time.perf_counter()
        out = res[0]
        if len(res) == 3:
            np_, nd_ = res[1], res[2]
        else:
            v = res[1]
            np_, nd_ = (v, 0) if v > 0 else (0, -v)

        for seq in llm.scheduler.running:
            k = seq.num_completion_tokens
            pk = last_k.get(seq.seq_id, 0)
            if k > pk:
                for _ in range(pk, k):
                    emit.setdefault(seq.seq_id, []).append(t)
                last_k[seq.seq_id] = k
        for sid, toks in out:
            k = len(toks)
            pk = last_k.get(sid, 0)
            if k > pk:
                for _ in range(pk, k):
                    emit.setdefault(sid, []).append(t)
                last_k[sid] = k

        step_dt.append((t - prev) * 1000.0)
        step_p.append(np_)
        step_d.append(nd_)
        prev = t

    wall = prev - t0
    RECORD_ON = False

    dcnt = {k: COUNTERS[k] - c0[k] for k in COUNTERS}
    return dict(wall=wall, step_dt=step_dt, step_p=step_p, step_d=step_d,
                emit=emit, arrival=arrival, preempt=dcnt,
                tok_p=sum(step_p), tok_d=sum(step_d))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--n", type=int, default=BENCH["n"])
    ap.add_argument("--max-input-len", type=int, default=BENCH["in_max"])
    ap.add_argument("--max-output-len", type=int, default=BENCH["out_max"])
    ap.add_argument("--max-model-len", type=int, default=BENCH["ml"])
    ap.add_argument("--budget", type=int, default=BENCH["budget"])
    ap.add_argument("--max-num-seqs", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=BENCH["temp"])
    ap.add_argument("--argmax", action="store_true", help="patch 采样为 argmax（确定性，偏离官方口径）")
    ap.add_argument("--enforce-eager", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    if args.argmax:
        _patch_sampler_argmax()

    rng = Random(args.seed)
    prompts = [[rng.randint(0, 10000) for _ in range(rng.randint(100, args.max_input_len))]
               for _ in range(args.n)]
    o_list = [rng.randint(100, args.max_output_len) for _ in range(args.n)]

    llm = LLM(MODEL, enforce_eager=args.enforce_eager, max_model_len=args.max_model_len,
              max_num_batched_tokens=args.budget, max_num_seqs=args.max_num_seqs)
    llm.generate(["Benchmark: "], SamplingParams())
    pool = len(llm.scheduler.block_manager.blocks)

    runs = []
    for r in range(args.repeat):
        res = run_once(llm, prompts, o_list, args.temperature)
        emit, arrival = res["emit"], res["arrival"]
        ttft = sorted((emit[s][0] - arrival[s]) * 1000.0 for s in arrival if emit.get(s))
        itl = []
        for s, e in emit.items():
            if len(e) < 2:
                continue
            itl.extend((e[i] - e[i - 1]) * 1000.0 for i in range(1, len(e)))
        n_steps = len(res["step_dt"])
        mixed = sum(1 for a, b in zip(res["step_p"], res["step_d"]) if a and b)
        pure_p = sum(1 for a, b in zip(res["step_p"], res["step_d"]) if a and not b)
        pure_d = sum(1 for a, b in zip(res["step_p"], res["step_d"]) if not a and b)
        runs.append(dict(
            idx=r, wall_s=round(res["wall"], 3),
            tokens=dict(prefill=res["tok_p"], decode=res["tok_d"], total=res["tok_p"] + res["tok_d"]),
            throughput=dict(
                out_tok_s=round(res["tok_d"] / res["wall"], 2),
                all_tok_s=round((res["tok_p"] + res["tok_d"]) / res["wall"], 2),
            ),
            preempt=res["preempt"],
            steps=dict(total=n_steps, pure_prefill=pure_p, pure_decode=pure_d, mixed=mixed),
            ttft_ms=summarize(ttft), itl_ms=summarize(itl),
            step_dt_ms=summarize(res["step_dt"]),
            step_dt_raw=[round(x, 3) for x in res["step_dt"]],
            step_p_raw=res["step_p"], step_d_raw=res["step_d"],
        ))
        print(f"  [{args.tag} rep{r}] wall={res['wall']:.2f}s "
              f"out={runs[-1]['throughput']['out_tok_s']}tok/s "
              f"preempt={res['preempt']['preempt']} "
              f"steps={n_steps}(P{pure_p}/D{pure_d}/M{mixed})")

    walls = [r["wall_s"] for r in runs]
    outs = [r["throughput"]["out_tok_s"] for r in runs]
    alls = [r["throughput"]["all_tok_s"] for r in runs]
    report = {
        "tag": args.tag,
        "branch": subprocess.run(["git", "branch", "--show-current"], cwd=REPO_ROOT,
                                 capture_output=True, text=True).stdout.strip(),
        "config": dict(n=args.n, seed=args.seed, repeat=args.repeat,
                       p_range=[100, args.max_input_len], o_range=[100, args.max_output_len],
                       budget=args.budget, max_model_len=args.max_model_len,
                       max_num_seqs=args.max_num_seqs, temperature=args.temperature,
                       argmax=args.argmax, enforce_eager=args.enforce_eager,
                       pool_blocks=pool),
        "total_output_tokens": sum(o_list),
        "runs": runs,
        "agg": dict(
            wall_s=dict(mean=round(mean(walls), 3), min=round(min(walls), 3),
                        max=round(max(walls), 3), std=round(stdev(walls), 3)),
            out_tok_s=dict(mean=round(mean(outs), 2), min=round(min(outs), 2),
                           max=round(max(outs), 2), std=round(stdev(outs), 2)),
            all_tok_s=dict(mean=round(mean(alls), 2), min=round(min(alls), 2),
                           max=round(max(alls), 2), std=round(stdev(alls), 2)),
        ),
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False)
    print(f"[{args.tag}] branch={report['branch']} pool={pool} "
          f"out_tok_s mean={report['agg']['out_tok_s']['mean']} "
          f"({report['agg']['out_tok_s']['min']}~{report['agg']['out_tok_s']['max']}) "
          f"wall mean={report['agg']['wall_s']['mean']}s")
    print(f"  -> {args.out}")


if __name__ == "__main__":
    main()
