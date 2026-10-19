#!/usr/bin/env python
"""C5 budget 扫描汇总 -> results/c5_summary.md"""
import json
import os

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
OUT = os.path.join(RES, "c5_summary.md")
BUDGETS = [1024, 2048, 4096, 16384]
TAGS = ["V1", "main"]


def load(tag, B):
    p = os.path.join(RES, f"c5_{tag}_b{B}.json")
    if not os.path.exists(p):
        return None
    return json.load(open(p, encoding="utf-8"))["runs"][0]


def main():
    L = ["# C5 budget 扫描汇总（定长 4864 · 池 163 · n=32 · 并发 ~7 · 抢占 0）", ""]
    L.append("| budget | 分支 | 墙钟 s | 输出 TPS | 步 p50 | 步 p90 | 步 p99 | **步 p999** | 步 max | 激活峰值 MB | 峰值 MB | 混合步 | 纯P步 | lm_head 行 |")
    L.append("| ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for B in BUDGETS:
        for t in TAGS:
            r = load(t, B)
            if not r:
                continue
            s, m, lh = r["steps"], r["memory"], r["lm_rows"]
            L.append(f"| {B} | {t} | {r['wall_s']:.2f} | {r['throughput']['out_tok_s']:.1f} | "
                     f"{s['dt_p50']:.2f} | {s['dt_p90']:.2f} | {s['dt_p99']:.2f} | "
                     f"**{s['dt_p999']:.2f}** | {s['dt_max']:.2f} | {m['activation_peak_mb']:.1f} | "
                     f"{m['peak_alloc_mb']:.1f} | {s['mixed']} | {s['pure_prefill']} | {lh['total']} |")
    L.append("")

    L.append("## 峰值显存随 budget 的变化（相对 b=1024）")
    L.append("")
    L.append("| 分支 | b=1024 | b=2048 | b=4096 | b=16384 |")
    L.append("| :--- | ---: | ---: | ---: | ---: |")
    for t in TAGS:
        base = load(t, 1024)["memory"]["peak_alloc_mb"]
        cells = []
        for B in BUDGETS:
            v = load(t, B)["memory"]["peak_alloc_mb"]
            cells.append(f"{v:.0f} ({100 * (v / base - 1):+.1f}%)")
        L.append(f"| {t} | " + " | ".join(cells) + " |")
    L.append("")
    L.append("## 步延迟 P999 随 budget 的变化")
    L.append("")
    L.append("| 分支 | b=1024 | b=2048 | b=4096 | b=16384 |")
    L.append("| :--- | ---: | ---: | ---: | ---: |")
    for t in TAGS:
        cells = [f"{load(t, B)['steps']['dt_p999']:.1f}" for B in BUDGETS]
        L.append(f"| {t} | " + " | ".join(cells) + " |")
    L.append("")
    L.append("## 同 budget 下两版的峰值显存（证明 −3.0% 是配置效应、非调度效应）")
    L.append("")
    L.append("| budget | V1 峰值 MB | main 峰值 MB | 差 |")
    L.append("| ---: | ---: | ---: | ---: |")
    for B in BUDGETS:
        a = load("V1", B)["memory"]["peak_alloc_mb"]
        b = load("main", B)["memory"]["peak_alloc_mb"]
        L.append(f"| {B} | {a:.1f} | {b:.1f} | {100 * (a / b - 1):+.2f}% |")
    L.append("")

    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
