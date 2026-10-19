#!/usr/bin/env python
"""把 results/c2_grid_V1.json 渲染成实验报告要用的几张 markdown 表。

用法：.venv/bin/python experiments/c2_table.py [json 路径]
"""
import json
import os
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PATH = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "results", "c2_grid_V1.json")


def main():
    doc = json.load(open(PATH, encoding="utf-8"))
    rows = doc["results"]
    print(f"<!-- {os.path.basename(PATH)} | args={doc['args']} -->")
    print(f"<!-- pool={doc['pool']} -->")

    Ps = sorted({r["plen"] for r in rows})
    Ds = sorted({r["D"] for r in rows})

    def cell(P, D, key):
        return next(r[key] for r in rows if r["plen"] == P and r["D"] == D)

    # 表 0：两个基线量
    print("\n### 基线：t_p（按 P）/ t_d（按 D），min ms\n")
    print("| P | " + " | ".join(str(p) for p in Ps) + " |")
    print("| ---: |" + " ---: |" * len(Ps))
    tp = [st.median([cell(p, d, "t_p")["min"] for d in Ds]) for p in Ps]
    print("| t_p | " + " | ".join(f"{v:.2f}" for v in tp) + " |")
    print("\n| D | " + " | ".join(str(d) for d in Ds) + " |")
    print("| ---: |" + " ---: |" * len(Ds))
    td = [st.median([cell(p, d, "t_d")["min"] for p in Ps]) for d in Ds]
    print("| t_d | " + " | ".join(f"{v:.2f}" for v in td) + " |")

    def matrix(title, key, fmt, sign=False):
        print(f"\n### {title}\n")
        print("| D \\ P | " + " | ".join(str(p) for p in Ps) + " |")
        print("| ---: |" + " ---: |" * len(Ps))
        for d in Ds:
            vals = [cell(p, d, key) for p in Ps]
            print(f"| **{d}** | " + " | ".join(fmt.format(v) for v in vals) + " |")

    matrix("分路：t_mix ÷ (t_p + t_d)", "mix_over_sum", "{:.3f}")
    matrix("整批 varlen：t_mix ÷ (t_p + t_d)", "mixv_over_sum", "{:.3f}")
    matrix("分路省下 (ms) = varlen − split", "split_gain_ms", "{:+.2f}")

    # 补充：斜率
    print("\n### 斜率（D 从 8 → 64，按 P 线性拟合 ms/行）\n")
    print("| P | 整批 varlen | 分路 | 差 |")
    print("| ---: | ---: | ---: | ---: |")
    for p in Ps:
        x = [8, 16, 32, 64]
        yv = [cell(p, d, "t_mix_varlen")["min"] for d in x]
        ys = [cell(p, d, "t_mix")["min"] for d in x]
        # 最小二乘
        mx = sum(x) / len(x)
        kv = sum((a - mx) * (b - sum(yv) / len(yv)) for a, b in zip(x, yv)) / sum((a - mx) ** 2 for a in x)
        ks = sum((a - mx) * (b - sum(ys) / len(ys)) for a, b in zip(x, ys)) / sum((a - mx) ** 2 for a in x)
        print(f"| {p} | {kv:.3f} | {ks:.3f} | {kv - ks:.3f} |")

    # 汇总统计
    def stats(key, sub=None):
        vs = [r[key] for r in rows if sub is None or sub(r)]
        return (f"n={len(vs)}  min={min(vs):.3f}  med={st.median(vs):.3f}  "
                f"max={max(vs):.3f}  ≤1 的格数={sum(1 for v in vs if v <= 1)}")

    print("\n### 汇总\n")
    print(f"- 分路 mix/sum 全部 24 格：{stats('mix_over_sum')}")
    print(f"- 分路 mix/sum D≥8：{stats('mix_over_sum', lambda r: r['D'] >= 8)}")
    print(f"- 分路 mix/sum D≥32：{stats('mix_over_sum', lambda r: r['D'] >= 32)}")
    print(f"- 整批 varlen mix/sum 全部 24 格：{stats('mixv_over_sum')}")
    print(f"- 整批 varlen mix/sum D≤16：{stats('mixv_over_sum', lambda r: r['D'] <= 16)}")
    print(f"- 整批 varlen mix/sum D≥32：{stats('mixv_over_sum', lambda r: r['D'] >= 32)}")
    for d_ in (1, 4, 8, 16, 32, 64):
        g = [r["split_gain_ms"] for r in rows if r["D"] == d_]
        print(f"- 分路省下 D={d_:>2}：med={st.median(g):+.2f} ms "
              f"（{min(g):+.2f} ~ {max(g):+.2f}）")


if __name__ == "__main__":
    main()
