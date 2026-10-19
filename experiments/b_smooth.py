#!/usr/bin/env python
"""B 组：目的达成（ITL 平滑 / TTFT 不牺牲 / 利用率）采集。

用法：
    python experiments/b_smooth.py --tag V1 --scenario base --out experiments/results/b_V1_base.json

场景（v3 重设计）：三场景**唯一自变量 = prompt 长度分布**，其余全同
    short  N=96  每条总长 768  token（3 块）   budget=1024  （模拟短 prompt）
    long   N=96  每条总长 4864 token（19 块）  budget=1024  （长 prompt）
    mix    N=96  奇偶交错 short / long          budget=1024  （短长混合）

每条 seq 的 O~U[64,255]，P = 总长 − O → **每条 seq 的块数恒定**（short 3 / long 19）。

budget 固定 1024（= 参考数据的 Chunk Size）：
    long 场景一条 prompt 要 5 步才预填完（main 这 5 步连续不产 token，V1 每步插 decode）。

并发必须锁死（否则两版不可比）：main 的 waiting 优先会一直 admit 直到池满、
**不受 max_num_seqs 约束**（实测 short 负载下 main 吃到 ~32 条 → 池打满 → 抢占 12 次，
而 V1 被 max_num_seqs 限在 8 条）。故本组改为：
    - max_num_seqs 放开（512），两版都由**池**约束；
    - 池强制 = 前 8 条负载的块数之和（short 24 / long 152 / mix 88）
      → 两版稳态并发都恰好 8 条，且因每条块数恒定，decode 无增长 → 抢占 = 0。
    负载校核的实测过程见实验报告 B 组注释。

约定（对齐 实验方案v2 §1）：
    - 采样补丁同 A 组：Sampler.forward -> argmax（只 patch 类方法，不改仓库源码，两版共用）
    - 全程监控抢占；B/C 要求 preempt == 0，否则调低并发重跑
    - 单进程独占 GPU；每轮先 warmup（图捕获 / 编译在 warmup 内完成，不计入）

记录（step 分辨率）：
    - 每条 seq 的 completion token 发射时刻 -> 逐请求 ITL / TTFT
    - 每步 (dt, prefill tokens, decode tokens)
"""
import argparse
import json
import os
import subprocess
import time
from random import Random

import torch

from nanovllm.layers import sampler as _sampler

COUNTERS = {"can_append_fail": 0, "can_allocate_fail": 0, "preempt": 0}
RECORD_ON = False


SAMPLER_MODE = "argmax"


def _patch_sampler():
    """argmax：patch 掉随机采样内核（与 A 组口径一致）。
    normal：**完全不 patch** —— 保留仓库的 Gumbel-max 随机采样与 torch.compile。
    """
    if SAMPLER_MODE != "argmax":
        return

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

from nanovllm.engine.model_runner import ModelRunner       # noqa: E402


def _patch_pool(blocks):
    """固定 KV 池块数（None = 按显存反推）。

    池大小由 `mem_get_info` 反推，会随环境 free 波动（实测 163 / 177，约 ±8%），
    跨进程比较显存时不干净。固定后两版同池，峰值差异只剩"权重 + 激活"部分。
    """
    if not blocks:
        return

    def alloc(self):
        hf = self.config.hf_config
        num_kv_heads = hf.num_key_value_heads // self.world_size
        head_dim = getattr(hf, "head_dim", hf.hidden_size // hf.num_attention_heads)
        self.config.num_kvcache_blocks = blocks
        self.kv_cache = torch.empty(2, hf.num_hidden_layers, blocks, self.block_size,
                                    num_kv_heads, head_dim)
        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                layer_id += 1

    ModelRunner.allocate_kv_cache = alloc


LM_ROWS: list = []


def _patch_lm_rows():
    """记录每步 lm_head 的实际行数（= 采样行数）。

    V1 用 `need_logits` 把"预填充中间 chunk"的行筛掉 → 0 行；
    main 无此机制 → 每步都按 prefill/decode 全量出行。差值即"无用 lm_head"。
    """
    _ps = ModelRunner.prepare_sample

    def ps(self, *a, **kw):
        r = _ps(self, *a, **kw)
        if RECORD_ON:
            LM_ROWS.append(int(r.shape[0]))
        return r

    ModelRunner.prepare_sample = ps


_patch_lm_rows()

SHORT_TOTAL = 768      # 3 块
LONG_TOTAL = 4864      # 19 块

SCEN = {
    # 三场景唯一自变量 = prompt 长度分布；N / budget / ml / 池推导规则 全同
    "short": dict(n=96, total=SHORT_TOTAL, budget=1024, ml=8192),
    "long":  dict(n=96, total=LONG_TOTAL,  budget=1024, ml=8192),
    "mix":   dict(n=96, total=SHORT_TOTAL, budget=1024, ml=8192, mix_total=LONG_TOTAL),
}

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


def main():
    global RECORD_ON, SAMPLER_MODE
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--scenario", default="base", choices=list(SCEN))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n", type=int, default=0, help="覆盖场景的 N")
    ap.add_argument("--max-num-seqs", type=int, default=512, help="并发上限（保抢占=0 用）")
    ap.add_argument("--pool-blocks", type=int, default=0,
                    help="固定 KV 池块数（0 = 按显存反推）")
    ap.add_argument("--budget", type=int, default=0, help="覆盖场景的 budget")
    ap.add_argument("--max-model-len", type=int, default=0)
    ap.add_argument("--sampler", default="argmax", choices=["argmax", "normal"],
                    help="argmax=patch 成贪心（A 组口径）；normal=不 patch（真实随机采样）")
    ap.add_argument("--independent-o", action="store_true",
                    help="P 与 O 独立采样（复现初版 v2 负载：decode 期跨块 → 允许抢占）")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    SAMPLER_MODE = args.sampler
    _patch_sampler()          # 必须在 LLM 实例化前完成

    cfg = dict(SCEN[args.scenario])
    if args.n:
        cfg["n"] = args.n
    if args.budget:
        cfg["budget"] = args.budget
    if args.max_model_len:
        cfg["ml"] = args.max_model_len

    rng = Random(args.seed)
    mix_total = cfg.get("mix_total")
    plens, o_list, kinds = [], [], []
    for i in range(cfg["n"]):
        is_long = bool(mix_total) and (i % 2 == 1)     # mix：奇偶交错，长短各半
        total = mix_total if is_long else cfg["total"]
        o = rng.randint(64, 256) if args.independent_o else rng.randint(64, 255)
        o_list.append(o)
        if args.independent_o:
            # 初版 v2 负载：P、O 独立 → P+O 跨 256 倍数 → decode 期追加块 → 允许抢占
            plens.append(rng.randint(4609, 4864))
        else:
            plens.append(total - o)                    # P = 总长 − O → 每条块数恒定
        kinds.append("long" if is_long else ("short" if mix_total else args.scenario))
    prompts = [[rng.randint(0, 10000) for _ in range(L)] for L in plens]

    # 池 = 前 8 条负载的块数之和 → 两版稳态并发都恰好 8 条（详见文件头注释）
    per_seq_blocks = [(L + 255) // 256 for L in plens]
    want_pool = args.pool_blocks or sum(per_seq_blocks[:8])

    _patch_pool(want_pool)
    llm = LLM(MODEL, enforce_eager=False, max_model_len=cfg["ml"],
              max_num_batched_tokens=cfg["budget"], max_num_seqs=args.max_num_seqs)
    llm.generate(["Benchmark: "], SamplingParams())
    pool = len(llm.scheduler.block_manager.blocks)

    # 显存基线 = 权重 + KV 池（推理开始前）；peak 从这一刻重新统计
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    resident_mb = torch.cuda.memory_allocated() / 2 ** 20

    LM_ROWS.clear()
    RECORD_ON = True
    emit = {}
    last_k = {}

    t0 = time.perf_counter()
    for p, o in zip(prompts, o_list):
        llm.add_request(p, SamplingParams(temperature=1.0, ignore_eos=True, max_tokens=o))
    arrival = {s.seq_id: t0 for s in llm.scheduler.waiting}

    step_dt, step_p, step_d = [], [], []
    step_run, step_wait = [], []
    prev = time.perf_counter()
    while not llm.is_finished():
        res = llm.step()
        t = time.perf_counter()
        step_run.append(len(llm.scheduler.running))
        step_wait.append(len(llm.scheduler.waiting))
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
    torch.cuda.synchronize()
    peak_alloc_mb = torch.cuda.max_memory_allocated() / 2 ** 20
    peak_reserved_mb = torch.cuda.max_memory_reserved() / 2 ** 20

    ttft_ms = sorted((emit[sid][0] - t0) * 1000.0 for sid in arrival if emit.get(sid))
    itl_ms, per_cv = [], []
    for sid, e in emit.items():
        if len(e) < 2:
            continue
        d = [(e[i] - e[i - 1]) * 1000.0 for i in range(1, len(e))]
        itl_ms.extend(d)
        m = mean(d)
        if m > 0:
            per_cv.append(stdev(d) / m)

    thr = 3 * (pct(itl_ms, 50) or 0)
    holes = [x for x in itl_ms if x > thr]

    n_steps = len(step_dt)
    mixed = sum(1 for a, b in zip(step_p, step_d) if a and b)
    pure_p = sum(1 for a, b in zip(step_p, step_d) if a and not b)
    pure_d = sum(1 for a, b in zip(step_p, step_d) if not a and b)
    with_p = [i for i in range(n_steps) if step_p[i] > 0]
    fill = mean([(step_p[i] + step_d[i]) / cfg["budget"] for i in with_p]) if with_p else 0.0
    tok_p, tok_d = sum(step_p), sum(step_d)

    report = {
        "tag": args.tag,
        "scenario": args.scenario,
        "branch": subprocess.run(["git", "branch", "--show-current"], cwd=REPO_ROOT,
                                 capture_output=True, text=True).stdout.strip(),
        "n": cfg["n"], "seed": args.seed,
        "total_len": cfg["total"], "mix_total": cfg.get("mix_total"), "o_range": [64, 255],
        "budget": cfg["budget"], "max_model_len": cfg["ml"], "pool_blocks": pool,
        "max_num_seqs": args.max_num_seqs,
        "plen": {"mean": round(mean(plens), 1), "min": min(plens), "max": max(plens),
                 "n_long": sum(1 for k in kinds if k == "long"),
                 "n_short": sum(1 for k in kinds if k == "short")},
        "concurrency": {
            "running_mean": round(mean(step_run), 2),
            "running_max": max(step_run),
            "running_p50": round(pct(step_run, 50), 1),
            "waiting_mean": round(mean(step_wait), 1),
            "waiting_min": min(step_wait),
        },
        "memory": {
            "resident_mb": round(resident_mb, 1),          # 权重 + KV 池（推理前基线）
            "pool_mb": round(pool * 28.0, 1),
            "peak_alloc_mb": round(peak_alloc_mb, 1),
            "peak_reserved_mb": round(peak_reserved_mb, 1),
            "activation_peak_mb": round(peak_alloc_mb - resident_mb, 1),
        },
        "preempt": dict(COUNTERS),
        "lm_head": {
            "rows_total": sum(LM_ROWS),
            "sample_steps": len(LM_ROWS),
            "rows_per_step_mean": round(mean(LM_ROWS), 2) if LM_ROWS else 0,
            "rows_max": max(LM_ROWS) if LM_ROWS else 0,
        },
        "sampler": args.sampler,
        "independent_o": bool(args.independent_o),
        "wall_s": round(wall, 3),
        "tokens": {"prefill": tok_p, "decode": tok_d, "total": tok_p + tok_d},
        "throughput": {
            "out_tok_s": round(tok_d / wall, 2),
            "all_tok_s": round((tok_p + tok_d) / wall, 2),
            "prompt_tok_s": round(tok_p / wall, 2),
        },
        "steps": {
            "total": n_steps, "pure_prefill": pure_p, "pure_decode": pure_d, "mixed": mixed,
            "mixed_ratio": round(mixed / n_steps, 4) if n_steps else None,
            "prefill_step_ratio": round(len(with_p) / n_steps, 4) if n_steps else None,
            "mean_tokens_per_step": round((tok_p + tok_d) / n_steps, 1) if n_steps else None,
            "prefill_fill_rate": round(fill, 4),
            "step_dt_p50": round(pct(step_dt, 50), 3),
            "step_dt_p99": round(pct(step_dt, 99), 3),
            "step_dt_p999": round(pct(step_dt, 99.9), 3),
            "step_dt_max": round(max(step_dt), 3),
            "step_dt_mean": round(mean(step_dt), 3),
        },
        "ttft_ms": summarize(ttft_ms),
        "itl_ms": summarize(itl_ms),
        "holes": {"threshold_ms": round(thr, 4), "count": len(holes),
                  "frac": round(len(holes) / len(itl_ms), 4) if itl_ms else None},
        "per_request_cv": {"median": round(pct(per_cv, 50), 4) if per_cv else None,
                           "max": round(max(per_cv), 4) if per_cv else None,
                           "n": len(per_cv)},
        "raw": {"itl_ms": [round(x, 3) for x in itl_ms],
                "ttft_ms": [round(x, 3) for x in ttft_ms],
                "step_dt_ms": [round(x, 3) for x in step_dt],
                "step_p": step_p, "step_d": step_d},
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False)

    itl = report["itl_ms"]
    tt = report["ttft_ms"]
    st = report["steps"]
    print(f"[{args.tag} {args.scenario}] branch={report['branch']} wall={report['wall_s']}s "
          f"preempt={report['preempt']['preempt']} "
          f"steps={n_steps}(P{pure_p}/D{pure_d}/M{mixed}) "
          f"conc={report['concurrency']['running_mean']}/{report['concurrency']['running_max']}")
    print(f"  ITL(ms) p50={itl['p50']} p90={itl['p90']} p99={itl['p99']} max={itl['max']} "
          f"cv={itl['cv']}  holes={len(holes)}({report['holes']['frac']})")
    print(f"  step(ms) p50={st['step_dt_p50']} p99={st['step_dt_p99']} "
          f"p999={st['step_dt_p999']} max={st['step_dt_max']}")
    print(f"  TTFT(ms) p50={tt['p50']} p90={tt['p90']} p99={tt['p99']}  "
          f"thr={report['throughput']['out_tok_s']}tok/s  fill={st['prefill_fill_rate']}  "
          f"pool={pool} peak={report['memory']['peak_alloc_mb']}MB")
    lh = report["lm_head"]
    print(f"  lm_head rows: total={lh['rows_total']} mean/step={lh['rows_per_step_mean']} "
          f"max={lh['rows_max']} sample_steps={lh['sample_steps']} sampler={args.sampler}")
    print(f"  -> {args.out}")


if __name__ == "__main__":
    main()
