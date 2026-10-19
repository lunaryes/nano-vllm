#!/usr/bin/env python
"""B 组（v3 三场景）绘图：把 results/b3_{V1,main}_{short,long,mix}_r*.json 画成 PNG。

用法：
    ~/venvs/fig/bin/python experiments/b_fig.py
产出（experiments/figs/）：
    b_itl_dist.png    三场景 ITL 经验分布（ECDF，log-x），V1 vs main
    b_metrics.png     尾部量对比：ITL CV / p99÷p50 / ITL max / 步延迟 P999
    b_steps_mem.png   步类型构成 + 峰值显存
"""
import json
import os
import statistics as st
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager, pyplot as plt  # noqa: E402
from matplotlib.ticker import FixedLocator, NullLocator, ScalarFormatter  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
FIGS = os.path.join(HERE, "figs")
SCEN = [("short", "短 prompt"), ("long", "长 prompt"), ("mix", "混合")]
TAGS = ["V1", "main"]
COLOR = {"V1": "#d62728", "main": "#1f77b4"}   # 红 = V1（涨/优化方）

FONT = os.path.expanduser("~/.fonts/simhei.ttf")
if os.path.exists(FONT):
    font_manager.fontManager.addfont(FONT)
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=FONT).get_name()
plt.rcParams.update({"axes.unicode_minus": False, "figure.dpi": 140,
                     "axes.grid": True, "grid.alpha": 0.3, "font.size": 9})


def load(tag, scen, reps=(1, 2)):
    out = []
    for r in reps:
        p = os.path.join(RES, f"b3_{tag}_{scen}_r{r}.json")
        if os.path.exists(p):
            with open(p, encoding="utf-8") as f:
                out.append(json.load(f))
    return out


def pick_rep(runs, key, fn):
    """多次重复取中位数那一次的代表值 + 极差。"""
    vs = [fn(r) for r in runs]
    return st.median(vs), (min(vs), max(vs))


def ecdf(xs):
    xs = sorted(xs)
    return xs, [(i + 1) / len(xs) for i in range(len(xs))]


# ---------- 图 1：ITL ECDF ----------
def fig_itl(runs):
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.4), sharey=True)
    for ax, (scen, label) in zip(axes, SCEN):
        for tag in TAGS:
            rs = runs[(tag, scen)]
            if not rs:
                continue
            itl = rs[0]["raw"]["itl_ms"]
            x, y = ecdf(itl)
            ax.plot(x, y, color=COLOR[tag], lw=1.4, label=tag)
            # 尾部关键点
            p50 = rs[0]["itl_ms"]["p50"]
            ax.axvline(p50, color=COLOR[tag], ls=":", lw=0.8, alpha=0.7)
        ax.set_xscale("log")
        ax.xaxis.set_major_locator(FixedLocator([5, 10, 20, 50, 100, 200, 500, 1000]))
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_xlim(4, 1100)
        ax.set_xlabel("ITL (ms, log)")
        ax.set_title(f"{label}（budget=1024）")
        ax.set_ylim(0, 1.02)
    axes[0].set_ylabel("累计占比")
    axes[0].legend(loc="lower right", framealpha=0.9)
    fig.suptitle("三场景 ITL 经验分布：V1 vs main（虚线 = 各自 p50）", y=0.99)
    fig.tight_layout()
    out = os.path.join(FIGS, "b_itl_dist.png")
    fig.savefig(out)
    plt.close(fig)
    return out


# ---------- 图 2：尾部指标 ----------
def fig_metrics(runs):
    specs = [
        ("ITL CV（越小越平滑）", lambda r: r["itl_ms"]["cv"], "CV"),
        ("ITL p99 ÷ p50", lambda r: r["itl_ms"]["p99"] / r["itl_ms"]["p50"], "倍"),
        ("ITL max（ms）", lambda r: r["itl_ms"]["max"], "ms"),
        ("步延迟 P999（ms）", lambda r: r["steps"]["step_dt_p999"], "ms"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(12.5, 3.2))
    w = 0.36
    xs = range(len(SCEN))
    for ax, (title, fn, unit) in zip(axes, specs):
        for i, tag in enumerate(TAGS):
            vals, errs = [], []
            for scen, _ in SCEN:
                rs = runs[(tag, scen)]
                if rs:
                    m, (lo, hi) = pick_rep(rs, tag, fn)
                    vals.append(m)
                    errs.append((m - lo, hi - m))
                else:
                    vals.append(0)
                    errs.append((0, 0))
            ax.bar([x + (i - 0.5) * w for x in xs], vals, w, color=COLOR[tag],
                   label=tag, yerr=list(zip(*errs)), capsize=2.5,
                   error_kw=dict(lw=0.8, alpha=0.7))
            for x, v in zip(xs, vals):
                ax.text(x + (i - 0.5) * w, v, f"{v:.2f}" if v < 100 else f"{v:.0f}",
                        ha="center", va="bottom", fontsize=7.5)
        ax.set_xticks(list(xs))
        ax.set_xticklabels([l for _, l in SCEN])
        ax.set_title(title, fontsize=9)
        ax.set_ylabel(unit)
        ax.margins(y=0.22)
    axes[0].legend(loc="upper right", framealpha=0.9)
    fig.suptitle("尾部指标：三场景 × 两版（误差棒 = 2 次重复极差）", y=1.0)
    fig.tight_layout()
    out = os.path.join(FIGS, "b_metrics.png")
    fig.savefig(out)
    plt.close(fig)
    return out


# ---------- 图 3：步构成 + 峰值显存 ----------
def fig_steps(runs):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 3.3),
                                   gridspec_kw={"width_ratios": [1.35, 1]})
    xs, labels = [], []
    k = 0
    for scen, label in SCEN:
        for tag in TAGS:
            xs.append(k)
            labels.append(f"{label}\n{tag}")
            k += 1
        k += 0.6
    bot = [0.0] * len(xs)
    comp = [("pure_prefill", "纯 prefill", "#8c564b"),
            ("pure_decode", "纯 decode", "#7f7f7f"),
            ("mixed", "混合", "#d62728")]
    for key, lab, c in comp:
        vals = []
        for scen, _ in SCEN:
            for tag in TAGS:
                rs = runs[(tag, scen)]
                vals.append(rs[0]["steps"][key] if rs else 0)
        ax1.bar(xs, vals, 0.72, bottom=bot, color=c, label=lab)
        bot = [a + b for a, b in zip(bot, vals)]
    ax1.set_xticks(xs)
    ax1.set_xticklabels(labels, fontsize=8)
    ax1.set_ylabel("步数")
    ax1.set_title("步类型构成")
    ax1.legend(fontsize=8, framealpha=0.9)

    for i, tag in enumerate(TAGS):
        vals = []
        for scen, _ in SCEN:
            rs = runs[(tag, scen)]
            vals.append(rs[0]["memory"]["peak_alloc_mb"] if rs else 0)
        ax2.bar([x + (i - 0.5) * 0.36 for x in range(len(SCEN))], vals, 0.36,
                color=COLOR[tag], label=tag)
        for x, v in zip(range(len(SCEN)), vals):
            ax2.text(x + (i - 0.5) * 0.36, v, f"{v:.0f}", ha="center",
                     va="bottom", fontsize=7.5)
    ax2.set_xticks(range(len(SCEN)))
    ax2.set_xticklabels([l for _, l in SCEN])
    ax2.set_ylabel("MB")
    ax2.set_title("峰值显存（torch peak_allocated）")
    ax2.legend(fontsize=8, framealpha=0.9)
    fig.tight_layout()
    out = os.path.join(FIGS, "b_steps_mem.png")
    fig.savefig(out)
    plt.close(fig)
    return out


def fig_tradeoff(runs):
    """频率-幅度权衡：V1 用「抖动更频繁」换「单次停顿更小」。"""
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    MK = {"short": "o", "long": "s", "mix": "^"}
    for scen, label in SCEN:
        for tag in TAGS:
            rs = runs[(tag, scen)]
            if not rs:
                continue
            x = rs[0]["holes"]["frac"] * 100
            y = rs[0]["itl_ms"]["max"]
            ax.scatter(x, y, color=COLOR[tag], marker=MK[scen], s=90,
                       edgecolor="k", linewidth=0.6, zorder=3)
            ax.annotate(f"{label}\n{tag}", (x, y), textcoords="offset points",
                        xytext=(7, 5 if tag == "V1" else -14), fontsize=8)
    ax.set_yscale("log")
    ax.set_xlabel("空洞占比（频率量，%）")
    ax.set_ylabel("ITL max（幅度量，ms，log）")
    ax.set_title("频率–幅度权衡：V1 把「少而巨」的停顿换成「多而小」", fontsize=10)
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([], [], color=COLOR[t], marker="o", ls="",
                              label=t) for t in TAGS], framealpha=0.9)
    fig.tight_layout()
    out = os.path.join(FIGS, "b_tradeoff.png")
    fig.savefig(out)
    plt.close(fig)
    return out


def main():
    os.makedirs(FIGS, exist_ok=True)
    runs = defaultdict(list)
    for tag in TAGS:
        for scen, _ in SCEN:
            runs[(tag, scen)] = load(tag, scen)
    for (tag, scen), rs in runs.items():
        print(f"  {tag:>4} {scen:<6} reps={len(rs)}")
    for fn in (fig_itl, fig_metrics, fig_steps, fig_tradeoff):
        print("  ->", fn(runs))


if __name__ == "__main__":
    main()
