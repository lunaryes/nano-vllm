#!/usr/bin/env python
"""B 组 v3（三场景）汇总：读 results/b3_{V1,main}_{short,long,mix}_r*.json，
输出 markdown 表到 results/b3_summary.md（多次重复取中位数，附极差）。

用法：~/venvs/fig/bin/python experiments/b_summary3.py
"""
import json
import os
import statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
SCEN = [("short", "short：短 prompt（768 token / 3 块）"),
        ("long", "long：长 prompt（4864 token / 19 块）"),
        ("mix", "mix：短长混合（奇偶交错）")]
TAGS = ["V1", "main"]


def load(tag, scen):
    out = []
    for r in (1, 2):
        p = os.path.join(RES, f"b3_{tag}_{scen}_r{r}.json")
        if os.path.exists(p):
            with open(p, encoding="utf-8") as f:
                out.append(json.load(f))
    return out


def med(runs, fn):
    vs = [fn(r) for r in runs]
    return st.median(vs), min(vs), max(vs)


def fnum(v, nd=2):
    return f"{v:.{nd}f}"


def main():
    runs = {(t, s): load(t, s) for t in TAGS for s, _ in SCEN}

    lines = ["# B 组 v3 三场景汇总（多次重复取中位数）", ""]
    for key in ("preempt", "pool", "conc", "budget", "plen_mean"):
        vals = []
        for t in TAGS:
            for s, _ in SCEN:
                rs = runs[(t, s)]
                if not rs:
                    continue
                if key == "preempt":
                    v = rs[0]["preempt"]["preempt"]
                elif key == "pool":
                    v = rs[0]["pool_blocks"]
                elif key == "conc":
                    v = f"{rs[0]['concurrency']['running_mean']}/{rs[0]['concurrency']['running_max']}"
                elif key == "budget":
                    v = rs[0]["budget"]
                else:
                    v = rs[0]["plen"]["mean"]
                vals.append(f"{t}/{s}={v}")
        lines.append(f"- {key}: " + "  ".join(vals))
    lines.append("")

    SPECS = [
        ("ITL p50 (ms)", lambda r: r["itl_ms"]["p50"], 2),
        ("ITL p90 (ms)", lambda r: r["itl_ms"]["p90"], 2),
        ("ITL p99 (ms)", lambda r: r["itl_ms"]["p99"], 2),
        ("ITL max (ms)", lambda r: r["itl_ms"]["max"], 1),
        ("ITL CV", lambda r: r["itl_ms"]["cv"], 3),
        ("ITL p99÷p50", lambda r: r["itl_ms"]["p99"] / r["itl_ms"]["p50"], 2),
        ("空洞占比 (>3×p50)", lambda r: r["holes"]["frac"], 4),
        ("步延迟 P999 (ms)", lambda r: r["steps"]["step_dt_p999"], 2),
        ("步延迟 max (ms)", lambda r: r["steps"]["step_dt_max"], 1),
        ("TTFT p50 (ms)", lambda r: r["ttft_ms"]["p50"], 0),
        ("TTFT p99 (ms)", lambda r: r["ttft_ms"]["p99"], 0),
        ("墙钟 (s)", lambda r: r["wall_s"], 2),
        ("输出吞吐 (tok/s)", lambda r: r["throughput"]["out_tok_s"], 1),
        ("步数 (总/P/D/M)", None, 0),
        ("预填预算填充率", lambda r: r["steps"]["prefill_fill_rate"], 4),
        ("峰值显存 (MB)", lambda r: r["memory"]["peak_alloc_mb"], 0),
    ]

    for scen, title in SCEN:
        lines += [f"## {title}", "",
                  "| 指标 | V1 | main | V1/main |", "| :--- | ---: | ---: | ---: |"]
        for name, fn, nd in SPECS:
            row = []
            for t in TAGS:
                rs = runs[(t, scen)]
                if not rs:
                    row.append("—")
                    continue
                if fn is None:
                    s = rs[0]["steps"]
                    row.append(f"{s['total']}/{s['pure_prefill']}/{s['pure_decode']}/{s['mixed']}")
                else:
                    m, lo, hi = med(rs, fn)
                    v = fnum(m, nd)
                    if abs(hi - lo) > (10 ** -nd):
                        v += f" ±{(hi - lo) / 2:.{nd}f}"
                    row.append(v)
            ratio = ""
            if fn is not None and "—" not in row:
                a, b = float(row[0].split(" ")[0]), float(row[1].split(" ")[0])
                ratio = fnum(a / b, 3) if b else "—"
            lines.append(f"| {name} | {row[0]} | {row[1]} | {ratio} |")
        lines.append("")

    out = os.path.join(RES, "b3_summary.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
