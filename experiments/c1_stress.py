#!/usr/bin/env python
"""C1 压测：双口径全指标面板（应用层 + 硬件层）。

两个口径
    --regime stress   官方 bench.py 口径原样：N=256、P~U[100,1024]、O~U[100,1024]、
                      budget=16384、max_model_len=4096、temp=0.6、池按显存反推(~163 块)。
                      **池严重超订** → 抢占发生、排队深 → 测「真实压力下的端到端表现」。
                      注：本机池仅 41728 token，256 条提交后实际 running 只有几十条，
                      「256 并发」应读作「提交 256 条」。

    --regime ctrl     定长负载：每条总长 1024（O~U[64,255]、P = 总长 − O），块数恒为 4。
                      池 163 块 → 稳态并发 40 条 → **抢占 = 0**。
                      测「无重算干扰下的调度形态与硬件利用率」。

    两口径的负载形状不同（变量长 vs 定长），故**跨口径比较的是"压力水平"，
    口径内比较的才是"分支"**。定长是抢占归零的充分条件，理由是代码级的：
    `Scheduler.schedule()` 只在 decode 分支 `can_append(seq)` 失败时调用 `preempt()`，
    而 `can_append` 要求 `len(free_block_ids) >= (len(seq) % block_size == 1)`。
    每条 seq 总长固定 → 块数全程不变 → 该条件恒真 → 永不抢占。

指标面板（齐见 docs 侧「指标可得性声明」）
    延迟：TTFT / ITL / E2E / TPOT 的 p50/p90/p99/p999/max/CV
    吞吐：输入 TPS、输出 TPS、总 TPS、RPS、RPM
    队列：running、waiting（逐步采样）
    缓存：KV 利用率 = used_blocks / pool（逐步采样）
    硬件：nvidia-smi 流式采样 GPU 利用率、显存控制器利用率、设备显存占用
    显存：peak_allocated / peak_reserved / 池 / 激活峰值
    其余：步延迟分位、步类型构成、抢占计数、lm_head 行数

不可得项（本机物理/工具链限制，报告需声明）
    SM 占用率、真实显存带宽(GB/s) —— 需 ncu/DCGM，本机无
    NVLink / PCIe 跨卡带宽      —— 单卡，无此链路

用法
    .venv/bin/python experiments/c1_stress.py --tag V1 --regime stress --repeat 2 \
        --out experiments/results/c1_V1_stress.json
"""
import argparse
import json
import os
import subprocess
import threading
import time
from random import Random

import torch

from nanovllm.layers import sampler as _sampler

COUNTERS = {"can_append_fail": 0, "can_allocate_fail": 0, "preempt": 0}
RECORD_ON = False
SAMPLER_MODE = "normal"

MODEL = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _patch_sampler():
    """argmax：patch 成贪心（A 组口径，确定性）。
    normal：完全不 patch —— 保留仓库的 Gumbel-max 随机采样与 torch.compile（官方口径）。
    """
    if SAMPLER_MODE != "argmax":
        return

    def forward(self, logits, temperatures):
        return logits.float().argmax(dim=-1)

    _sampler.Sampler.forward = forward


from nanovllm import LLM, SamplingParams                       # noqa: E402
from nanovllm.engine.scheduler import Scheduler                # noqa: E402
from nanovllm.engine.block_manager import BlockManager         # noqa: E402
from nanovllm.engine.model_runner import ModelRunner           # noqa: E402


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


def _patch_pool(blocks):
    """固定 KV 池块数（0 = 按显存反推）。固定后两版同池，显存差异只剩权重+激活。"""
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
    """记录每步 lm_head 实际行数：V1 用 need_logits 筛掉预填充中间 chunk（0 行），
    main 无此机制 → 每步全量出行。差值即"无用 lm_head"。"""
    _ps = ModelRunner.prepare_sample

    def ps(self, *a, **kw):
        r = _ps(self, *a, **kw)
        if RECORD_ON:
            LM_ROWS.append(int(r.shape[0]))
        return r

    ModelRunner.prepare_sample = ps


class SmiSampler:
    """nvidia-smi 流式采样（单进程 -lms，避免反复 spawn 扰动）。

    只拿得到粗粒度量：utilization.gpu 是"有 kernel 在执行的时间占比"，
    utilization.memory 是显存控制器忙碌度（非 GB/s）。
    """

    def __init__(self, interval_ms):
        self.interval_ms = interval_ms
        self.samples = []          # (t, gpu_util, mem_util, dev_used_mb)
        self.proc = None
        self.th = None

    def start(self):
        try:
            self.proc = subprocess.Popen(
                ["nvidia-smi",
                 "--query-gpu=utilization.gpu,utilization.memory,memory.used",
                 "--format=csv,noheader,nounits", "-lms", str(self.interval_ms)],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1)
        except Exception:
            self.proc = None
            return False
        self.th = threading.Thread(target=self._loop, daemon=True)
        self.th.start()
        return True

    def _loop(self):
        try:
            for line in self.proc.stdout:
                t = time.perf_counter()
                p = [x.strip() for x in line.split(",")]
                if len(p) < 3:
                    continue
                try:
                    self.samples.append((t, float(p[0]), float(p[1]), float(p[2])))
                except ValueError:
                    continue
        except Exception:
            pass

    def stop(self):
        if self.proc is None:
            return
        try:
            self.proc.terminate()
            self.proc.wait(timeout=2)
        except Exception:
            try:
                self.proc.kill()
            except Exception:
                pass


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
            "p999": round(pct(vs, 99.9), 4), "max": round(max(vs), 4),
            "std": round(stdev(vs), 4), "cv": round(stdev(vs) / m, 4) if m else None}


def build_load(args):
    """官方口径：逐字复刻 bench.py 的随机数消耗顺序（先 P 再 O，逐条 interleave）。"""
    rng = Random(args.seed)
    plens, prompts = [], []
    if args.load == "official":
        for _ in range(args.n):
            L = rng.randint(100, args.max_input_len)
            plens.append(L)
            prompts.append([rng.randint(0, 10000) for _ in range(L)])
        olens = [rng.randint(100, args.max_output_len) for _ in range(args.n)]
    else:                                   # fixed：定长 1024，块数恒 4
        olens = [rng.randint(64, 255) for _ in range(args.n)]
        plens = [args.fixed_total - o for o in olens]
        prompts = [[rng.randint(0, 10000) for _ in range(L)] for L in plens]
    return prompts, plens, olens


def run_once(llm, prompts, olens, temp, smi, budget):
    global RECORD_ON
    c0 = dict(COUNTERS)
    emit, last_k = {}, {}

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    LM_ROWS.clear()
    smi.samples.clear()

    RECORD_ON = True
    smi.start()
    t0 = time.perf_counter()
    for p, o in zip(prompts, olens):
        llm.add_request(p, SamplingParams(temperature=temp, ignore_eos=True, max_tokens=o))
    arrival = {s.seq_id: t0 for s in llm.scheduler.waiting}

    step_dt, step_p, step_d = [], [], []
    step_run, step_wait, step_kv = [], [], []
    pool = len(llm.scheduler.block_manager.blocks)
    bm = llm.scheduler.block_manager

    prev = time.perf_counter()
    while not llm.is_finished():
        res = llm.step()
        t = time.perf_counter()
        step_run.append(len(llm.scheduler.running))
        step_wait.append(len(llm.scheduler.waiting))
        step_kv.append(len(bm.used_block_ids) / pool)

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
    smi.stop()
    torch.cuda.synchronize()
    peak_alloc = torch.cuda.max_memory_allocated() / 2 ** 20
    peak_reserved = torch.cuda.max_memory_reserved() / 2 ** 20

    return dict(wall=wall, emit=emit, arrival=arrival, step_dt=step_dt, step_p=step_p,
                step_d=step_d, step_run=step_run, step_wait=step_wait, step_kv=step_kv,
                pool=pool, preempt={k: COUNTERS[k] - c0[k] for k in COUNTERS},
                tok_p=sum(step_p), tok_d=sum(step_d),
                peak_alloc=peak_alloc, peak_reserved=peak_reserved,
                smi=list(smi.samples), t0=t0, lm_rows=list(LM_ROWS))


def main():
    global RECORD_ON, SAMPLER_MODE
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--regime", default="stress", choices=["stress", "ctrl"])
    ap.add_argument("--load", default="", choices=["", "official", "fixed"])
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--fixed-total", type=int, default=1024)
    ap.add_argument("--max-input-len", type=int, default=1024)
    ap.add_argument("--max-output-len", type=int, default=1024)
    ap.add_argument("--budget", type=int, default=0, help="0 = 16384（官方默认）")
    ap.add_argument("--max-model-len", type=int, default=0, help="0 = 4096")
    ap.add_argument("--max-num-seqs", type=int, default=512)
    ap.add_argument("--pool-blocks", type=int, default=0, help="0 = 按显存反推")
    ap.add_argument("--sampler", default="normal", choices=["normal", "argmax"])
    ap.add_argument("--smi-ms", type=int, default=500)
    ap.add_argument("--no-smi", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    load = args.load or ("fixed" if args.regime == "ctrl" else "official")
    args.load = load                       # build_load 读的是 args.load，必须写回
    budget = args.budget or 16384
    ml = args.max_model_len or 4096

    SAMPLER_MODE = args.sampler
    _patch_sampler()
    _patch_counters()
    _patch_lm_rows()
    _patch_pool(args.pool_blocks)

    prompts, plens, olens = build_load(args)

    llm = LLM(MODEL, enforce_eager=False, max_model_len=ml,
              max_num_batched_tokens=budget, max_num_seqs=args.max_num_seqs)
    llm.generate(["Benchmark: "], SamplingParams())
    pool = len(llm.scheduler.block_manager.blocks)

    torch.cuda.synchronize()
    resident_mb = torch.cuda.memory_allocated() / 2 ** 20

    smi = SmiSampler(args.smi_ms)
    if args.no_smi:
        smi.start = lambda: False

    runs = []
    for r in range(args.repeat):
        res = run_once(llm, prompts, olens, 0.6, smi, budget)
        t0 = res["t0"]
        sid_done = [s for s in res["arrival"] if res["emit"].get(s)]
        ttft = [(res["emit"][s][0] - t0) * 1000.0 for s in sid_done]
        e2e = [(res["emit"][s][-1] - t0) * 1000.0 for s in sid_done]
        tpot = [((res["emit"][s][-1] - res["emit"][s][0]) * 1000.0 / (len(res["emit"][s]) - 1))
                for s in sid_done if len(res["emit"][s]) > 1]
        itl = []
        for s in sid_done:
            e = res["emit"][s]
            itl.extend((e[i] - e[i - 1]) * 1000.0 for i in range(1, len(e)))

        # nvidia-smi：只取落在 [t0, t0+wall] 窗口内的样本
        t1 = t0 + res["wall"]
        win = [s for s in res["smi"] if t0 <= s[0] <= t1]
        gpu = [s[1] for s in win]
        memu = [s[2] for s in win]
        dev = [s[3] for s in win]

        n_steps = len(res["step_dt"])
        sp, sd = res["step_p"], res["step_d"]
        mixed = sum(1 for a, b in zip(sp, sd) if a and b)
        pure_p = sum(1 for a, b in zip(sp, sd) if a and not b)
        pure_d = sum(1 for a, b in zip(sp, sd) if not a and b)

        runs.append(dict(
            idx=r, wall_s=round(res["wall"], 3),
            throughput=dict(
                prompt_tok_s=round(res["tok_p"] / res["wall"], 2),
                out_tok_s=round(res["tok_d"] / res["wall"], 2),
                all_tok_s=round((res["tok_p"] + res["tok_d"]) / res["wall"], 2),
                rps=round(args.n / res["wall"], 3),
                rpm=round(args.n / res["wall"] * 60.0, 1),
                req_done=len(sid_done)),
            ttft_ms=summarize(ttft), itl_ms=summarize(itl),
            e2e_ms=summarize(e2e), tpot_ms=summarize(tpot),
            queue=dict(
                running=summarize(res["step_run"]),
                running_p50=round(pct(res["step_run"], 50), 1),
                running_p90=round(pct(res["step_run"], 90), 1),
                running_max=max(res["step_run"]) if res["step_run"] else 0,
                waiting=summarize(res["step_wait"]),
                waiting_max=max(res["step_wait"]) if res["step_wait"] else 0,
                waiting_nonzero_frac=round(
                    sum(1 for x in res["step_wait"] if x > 0) / max(n_steps, 1), 4)),
            kv=dict(util=summarize(res["step_kv"]),
                    used_blocks_max=int(round(max(res["step_kv"], default=0) * res["pool"])),
                    pool_blocks=res["pool"]),
            hw=dict(gpu_util=summarize(gpu), mem_util=summarize(memu),
                    dev_mem_mb=summarize(dev), samples=len(win)),
            memory=dict(resident_mb=round(resident_mb, 1),
                        pool_mb=round(res["pool"] * 28.0, 1),
                        peak_alloc_mb=round(res["peak_alloc"], 1),
                        peak_reserved_mb=round(res["peak_reserved"], 1),
                        activation_peak_mb=round(res["peak_alloc"] - resident_mb, 1)),
            preempt=res["preempt"],
            steps=dict(total=n_steps, pure_prefill=pure_p, pure_decode=pure_d, mixed=mixed,
                       mean_tokens_per_step=round((res["tok_p"] + res["tok_d"]) / n_steps, 1),
                       dt_p50=round(pct(res["step_dt"], 50), 3),
                       dt_p90=round(pct(res["step_dt"], 90), 3),
                       dt_p99=round(pct(res["step_dt"], 99), 3),
                       dt_p999=round(pct(res["step_dt"], 99.9), 3),
                       dt_max=round(max(res["step_dt"]), 3)),
            lm_rows=dict(total=sum(res["lm_rows"]), steps=len(res["lm_rows"]),
                         per_step=round(mean(res["lm_rows"]), 2) if res["lm_rows"] else 0,
                         mx=max(res["lm_rows"]) if res["lm_rows"] else 0),
            raw=dict(itl_ms=[round(x, 3) for x in itl],
                     ttft_ms=[round(x, 3) for x in ttft],
                     e2e_ms=[round(x, 3) for x in e2e],
                     tpot_ms=[round(x, 3) for x in tpot],
                     step_dt_ms=[round(x, 3) for x in res["step_dt"]],
                     step_p=res["step_p"], step_d=res["step_d"],
                     step_run=res["step_run"], step_wait=res["step_wait"],
                     step_kv=[round(x, 4) for x in res["step_kv"]],
                     smi=[(round(s[0] - t0, 3), s[1], s[2], s[3]) for s in win])))

        rr = runs[-1]
        print(f"  [{args.tag} {args.regime} r{r}] wall={rr['wall_s']}s "
              f"out={rr['throughput']['out_tok_s']}tok/s rps={rr['throughput']['rps']} "
              f"preempt={rr['preempt']['preempt']} conc={rr['queue']['running']['mean']}"
              f"/{rr['queue']['running_max']} kv={rr['kv']['util']['mean']} "
              f"gpu={rr['hw']['gpu_util']['mean']}% mem={rr['hw']['mem_util']['mean']}% "
              f"steps={rr['steps']['total']}(P{pure_p}/D{pure_d}/M{mixed})")

    out = dict(
        tag=args.tag, regime=args.regime, load=load,
        branch=subprocess.run(["git", "branch", "--show-current"], cwd=REPO_ROOT,
                              capture_output=True, text=True).stdout.strip(),
        config=dict(n=args.n, seed=args.seed, repeat=args.repeat,
                    p_range=([100, args.max_input_len] if load == "official"
                             else [args.fixed_total - 255, args.fixed_total - 64]),
                    o_range=([100, args.max_output_len] if load == "official"
                             else [64, 255]),
                    fixed_total=(args.fixed_total if load == "fixed" else None),
                    budget=budget, max_model_len=ml, max_num_seqs=args.max_num_seqs,
                    sampler=args.sampler, pool_blocks=pool, smi_ms=args.smi_ms),
        plen=dict(mean=round(mean(plens), 1), min=min(plens), max=max(plens)),
        olen=dict(mean=round(mean(olens), 1), min=min(olens), max=max(olens)),
        runs=runs)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)
    print(f"[{args.tag}] branch={out['branch']} regime={args.regime} load={load} "
          f"pool={pool} peak={runs[-1]['memory']['peak_alloc_mb']}MB")
    print(f"  -> {args.out}")


if __name__ == "__main__":
    main()
