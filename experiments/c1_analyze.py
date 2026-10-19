#!/usr/bin/env python
"""C1 汇总：把压测 JSON 变成指标面板表（results/c1_summary.md）。

读 experiments/results/c1_{V1,main}_{stress,ctrl}_r{1,2}.json。
用法：.venv/bin/python experiments/c1_analyze.py
"""
import glob
import json
import os
import statistics as st

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
OUT = os.path.join(RES, "c1_summary.md")
TAGS = ["V1", "main"]
REGIMES = [("stress", "真实压力口径（官方 256 提交 · 池超订 · 抢占发生）"),
           ("ctrl", "受控口径（定长 1024 · 抢占 = 0）")]


def load(tag, regime):
    rs = []
    for f in sorted(glob.glob(os.path.join(RES, f"c1_{tag}_{regime}_r*.json"))):
        d = json.load(open(f, encoding="utf-8"))
        for r in d["runs"]:
            r["_file"] = os.path.basename(f)
            r["_cfg"] = d["config"]
            r["_plen"] = d["plen"]
            rs.append(r)
    return rs


def med(rs, path):
    """按点分路径取各次重复的中位数。"""
    vs = []
    for r in rs:
        v = r
        for k in path.split("."):
            v = v[int(k)] if isinstance(v, list) else v[k]
        if v is not None:
            vs.append(v)
    return st.median(vs) if vs else None


def fmt(v, nd=2):
    if v is None:
        return "—"
    if isinstance(v, int):
        return str(v)
    return f"{v:.{nd}f}"


def ratio(a, b):
    if a is None or b is None or b == 0:
        return "—"
    return f"{a / b:.3f}"


def main():
    L = []
    for regime, desc in REGIMES:
        data = {t: load(t, regime) for t in TAGS}
        n_rep = {t: len(data[t]) for t in TAGS}
        cfg = data["V1"][0]["_cfg"]
        L.append(f"## 口径 {regime}")
        L.append("")
        L.append(f"{desc}")
        L.append("")
        L.append(f"配置：N={cfg['n']} · budget={cfg['budget']} · max_model_len="
                 f"{cfg['max_model_len']} · 池={cfg['pool_blocks']} 块(41728 token) · "
                 f"max_num_seqs={cfg['max_num_seqs']} · sampler={cfg['sampler']} · "
                 f"p∈{cfg['p_range']} · o∈{cfg['o_range']} · 每分支 {n_rep['V1']} 次重复")
        L.append("")
        L.append("### 延迟（per-request）")
        L.append("")
        L.append("| 指标 | V1 p50 / p99 / p999 | main p50 / p99 / p999 | main÷V1 (p99) | main÷V1 (p999) |")
        L.append("| :--- | ---: | ---: | ---: | ---: |")
        for key, name, nd in [("ttft_ms", "TTFT (ms)", 2), ("itl_ms", "ITL (ms)", 2),
                              ("e2e_ms", "E2E 请求延迟 (ms)", 1), ("tpot_ms", "TPOT (ms)", 2)]:
            row = [name]
            rs = {}
            for t in TAGS:
                p50, p99, p999 = (med(data[t], f"{key}.p50"), med(data[t], f"{key}.p99"),
                                  med(data[t], f"{key}.p999"))
                rs[t] = (p50, p99, p999)
                row.append(f"{fmt(p50, nd)} / {fmt(p99, nd)} / {fmt(p999, nd)}")
            row.append(ratio(rs["main"][1], rs["V1"][1]))
            row.append(ratio(rs["main"][2], rs["V1"][2]))
            L.append("| " + " | ".join(row) + " |")
        L.append("")

        L.append("### 吞吐")
        L.append("")
        L.append("| 指标 | V1 | main | main÷V1 |")
        L.append("| :--- | ---: | ---: | ---: |")
        for key, name in [("throughput.prompt_tok_s", "输入 TPS (tok/s)"),
                          ("throughput.out_tok_s", "输出 TPS (tok/s)"),
                          ("throughput.all_tok_s", "总 TPS (tok/s)"),
                          ("throughput.rps", "RPS (req/s)"),
                          ("throughput.rpm", "RPM (req/min)"),
                          ("wall_s", "墙钟 (s)")]:
            a, b = med(data["V1"], key), med(data["main"], key)
            nd = 3 if key.endswith("rps") else (1 if key.endswith("rpm") or key == "wall_s" else 2)
            L.append(f"| {name} | {fmt(a, nd)} | {fmt(b, nd)} | {ratio(b, a)} |")
        L.append("")

        L.append("### 队列 / KV / 硬件")
        L.append("")
        L.append("| 指标 | V1 | main | 说明 |")
        L.append("| :--- | ---: | ---: | :--- |")
        rows = [
            ("queue.running.mean", "running 均值", "并发驻留数"),
            ("queue.running.max", "running 峰值", ""),
            ("queue.waiting.mean", "waiting 均值", "排队深度"),
            ("queue.waiting_max", "waiting 峰值", ""),
            ("queue.waiting_nonzero_frac", "waiting 非空步占比", "有排队的时间比例"),
            ("kv.util.mean", "KV 利用率 均值", "used_blocks / pool"),
            ("kv.util.max", "KV 利用率 峰值", ""),
            ("hw.gpu_util.mean", "GPU 利用率 均值 (%)", "nvidia-smi，粗粒度"),
            ("hw.mem_util.mean", "显存控制器利用率 均值 (%)", "非 GB/s"),
            ("hw.dev_mem_mb.max", "设备显存占用 峰值 (MB)", ""),
            ("memory.peak_alloc_mb", "torch 峰值 allocated (MB)", "权重+池+激活"),
            ("memory.resident_mb", "常驻（权重+池）(MB)", ""),
            ("memory.activation_peak_mb", "激活峰值 (MB)", "peak − 常驻"),
            ("preempt.preempt", "抢占次数", ""),
            ("preempt.can_allocate_fail", "can_allocate 失败", ""),
        ]
        for key, name, note in rows:
            a, b = med(data["V1"], key), med(data["main"], key)
            L.append(f"| {name} | {fmt(a)} | {fmt(b)} | {note} |")
        L.append("")

        L.append("### 步级")
        L.append("")
        L.append("| 指标 | V1 | main |")
        L.append("| :--- | ---: | ---: |")
        for key, name in [("steps.total", "步总数"), ("steps.pure_prefill", "纯 prefill 步"),
                          ("steps.pure_decode", "纯 decode 步"), ("steps.mixed", "混合步"),
                          ("steps.mean_tokens_per_step", "每步平均 token"),
                          ("steps.dt_p50", "步延迟 p50 (ms)"),
                          ("steps.dt_p90", "步延迟 p90 (ms)"),
                          ("steps.dt_p99", "步延迟 p99 (ms)"),
                          ("steps.dt_p999", "**步延迟 p999 (ms)**"),
                          ("steps.dt_max", "步延迟 max (ms)"),
                          ("lm_rows.total", "lm_head 总行数"),
                          ("lm_rows.steps", "采样步数"),
                          ("lm_rows.per_step", "lm_head 行数/步")]:
            a, b = med(data["V1"], key), med(data["main"], key)
            L.append(f"| {name} | {fmt(a)} | {fmt(b)} |")
        L.append("")

        # 逐次明细
        L.append("### 逐次明细")
        L.append("")
        L.append("| 分支 | 文件 | wall (s) | 输出 TPS | 抢占 | 步数 | 步 p999 (ms) | 步 max (ms) |")
        L.append("| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |")
        for t in TAGS:
            for r in data[t]:
                L.append(f"| {t} | {r['_file']} | {r['wall_s']:.3f} | "
                         f"{r['throughput']['out_tok_s']:.2f} | {r['preempt']['preempt']} | "
                         f"{r['steps']['total']} | {r['steps']['dt_p999']:.1f} | "
                         f"{r['steps']['dt_max']:.1f} |")
        L.append("")

        # 交替配对比值
        L.append("### 交替配对比值（V1 ÷ main，>1 = V1 更慢）")
        L.append("")
        v1 = sorted(data["V1"], key=lambda r: r["_file"])
        mn = sorted(data["main"], key=lambda r: r["_file"])
        L.append("| 配对 | V1 wall | main wall | V1÷main | V1 out TPS | main out TPS |")
        L.append("| :--- | ---: | ---: | ---: | ---: | ---: |")
        for i, (a, b) in enumerate(zip(v1, mn)):
            L.append(f"| {i + 1} | {a['wall_s']:.3f} | {b['wall_s']:.3f} | "
                     f"{a['wall_s'] / b['wall_s']:.3f} | {a['throughput']['out_tok_s']:.2f} | "
                     f"{b['throughput']['out_tok_s']:.2f} |")
        aw = [r["wall_s"] for r in v1]
        bw = [r["wall_s"] for r in mn]
        L.append(f"| **均值** | {st.mean(aw):.3f} | {st.mean(bw):.3f} | "
                 f"**{st.mean(aw) / st.mean(bw):.3f}** | "
                 f"{st.mean([r['throughput']['out_tok_s'] for r in v1]):.2f} | "
                 f"{st.mean([r['throughput']['out_tok_s'] for r in mn]):.2f} |")
        L.append("")
        L.append(f"分支内散布：V1 wall std = {st.stdev(aw):.3f} s，main = {st.stdev(bw):.3f} s")
        L.append("")

    with open(OUT, "w", encoding="utf-8") as f:
        f.write("# C1 压测汇总：双口径全指标面板\n\n")
        f.write("\n".join(L))
    print(f"-> {OUT}")
    print("\n".join(L[:4]))


if __name__ == "__main__":
    main()
