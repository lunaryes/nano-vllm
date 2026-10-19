#!/usr/bin/env python
"""C3b 出图：decode 行三条路径的单价（split 改动后的重做版）。

    ~/venvs/fig/bin/python experiments/c3b_fig.py
    -> experiments/figs/c3b_split.png
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

FONT = os.path.expanduser("~/.fonts/simhei.ttf")
if os.path.exists(FONT):
    font_manager.fontManager.addfont(FONT)
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=FONT).get_name()
plt.rcParams["axes.unicode_minus"] = False

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "experiments", "results")
FIGS = os.path.join(ROOT, "experiments", "figs")

C_GEN, C_SPLIT, C_PURE, C_GRAPH = "#b03030", "#204080", "#8090a0", "#1f8a4c"


def jload(name):
    with open(os.path.join(RES, name)) as f:
        return json.load(f)


def fig_a(ax, d):
    rows = d["rows"]
    ns = [r["N"] for r in rows]
    for key, color, label, mk in (
            ("t_d_generic", C_GEN, "t_d_generic（整批 varlen）", "o"),
            ("t_d_eager", C_SPLIT, "t_d_eager（分路 eager）", "s"),
            ("t_d_eager_pure", C_PURE, "t_d_eager_pure（decode 入口 eager）", "^"),
            ("t_d_graph", C_GRAPH, "t_d_graph（decode 入口 + graph）", "D")):
        ax.plot(ns, [r[key] for r in rows], marker=mk, color=color, label=label, lw=1.8)
    for r in rows:
        ax.annotate("%.2f×" % (r["t_d_graph"] / r["t_d_eager_pure"]),
                    (r["N"], r["t_d_graph"]), textcoords="offset points",
                    xytext=(0, -14), ha="center", fontsize=8, color=C_GRAPH)
    ax.set_xscale("log", base=2)
    ax.set_xticks(ns)
    ax.set_xticklabels([str(n) for n in ns])
    ax.set_xlabel("批内 decode 行数 N（每行 1 token，历史 2048）")
    ax.set_ylabel("整步延迟 ms（min of 60）")
    ax.set_title("(a) 纯 decode 批：四条路径的单价", fontsize=11)
    ax.grid(alpha=0.3, ls=":")
    ax.legend(fontsize=7.5, loc="upper left")


def fig_b_nd(ax, d):
    rows = d["rows"]
    nds = [r["nd"] for r in rows]
    ax.plot(nds, [r["t_generic_ms"] for r in rows], marker="o", color=C_GEN,
            label="generic（decode 行也走 varlen）", lw=1.8)
    ax.plot(nds, [r["t_split_ms"] for r in rows], marker="s", color=C_SPLIT,
            label="split（decode 行走 paged decode kernel）", lw=1.8)
    ax.set_xlabel("混合批里的 decode 行数 nd（chunk 固定 1024）")
    ax.set_ylabel("整步延迟 ms（min of 60）")
    ax.set_title("(b) 真实混合批：nd 扫描", fontsize=11)
    ax.grid(alpha=0.3, ls=":")
    ax.legend(fontsize=7.5, loc="upper left")

    ax2 = ax.twinx()
    ax2.bar(nds, [r["gain_ms"] for r in rows], width=4.0, alpha=0.25, color="#c08030")
    ax2.set_ylabel("split 省下的 ms", color="#a06010")
    ax2.tick_params(axis="y", colors="#a06010")
    for r in rows:
        ax2.annotate("%.1f" % r["gain_ms"], (r["nd"], max(0, r["gain_ms"])),
                     textcoords="offset points", xytext=(0, 3), ha="center",
                     fontsize=7.5, color="#a06010")


def fig_b_chunk(ax, d):
    rows = d["rows"]
    xs = [r["chunk"] for r in rows]
    gains = [r["gain_ms"] for r in rows]
    bars = ax.bar(range(len(xs)), gains, color=C_SPLIT, alpha=0.8, width=0.6)
    for i, (b, r) in enumerate(zip(bars, rows)):
        ax.annotate("%.1f ms\n%.2f×" % (r["gain_ms"], r["split_over_generic"]),
                    (b.get_x() + b.get_width() / 2, b.get_height()),
                    textcoords="offset points", xytext=(0, 3), ha="center", fontsize=7.5)
    ax.set_xticks(range(len(xs)))
    ax.set_xticklabels([str(x) for x in xs])
    ax.set_xlabel("prefill chunk 长度（nd 固定 32）")
    ax.set_ylabel("split 省下的 ms")
    ax.set_title("(c) 收益与 chunk 长度无关（≈常数）", fontsize=11)
    ax.grid(alpha=0.3, ls=":", axis="y")
    ax.set_ylim(0, max(gains) * 1.25)


def main():
    da = jload("c3b_a.json")
    dnd = jload("c3b_b_nd.json")
    dch = jload("c3b_b_chunk.json")

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.4))
    fig_a(axes[0], da)
    fig_b_nd(axes[1], dnd)
    fig_b_chunk(axes[2], dch)
    fig.suptitle("C3b  decode 行的三条路径：varlen（generic）/ paged decode kernel（split）/ graph"
                 "   ·  Qwen3-0.6B · RTX 4060 8G · 历史 2048", fontsize=11.5)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = os.path.join(FIGS, "c3b_split.png")
    fig.savefig(out, dpi=160)
    print("->", out)


if __name__ == "__main__":
    main()
