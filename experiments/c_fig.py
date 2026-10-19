#!/usr/bin/env python
"""C 组绘图：把 c1/c3/c5 + c2 的结果画成 PNG（experiments/figs/）。

用法：~/venvs/fig/bin/python experiments/c_fig.py
产出：
    c_panel.png       双口径指标面板（吞吐/RPS/墙钟/ITL p999/步 p999/峰值显存）
    c_step_ecdf.png   步延迟 ECDF：V1 的 P999 反转（与参考数据 −36.1% 方向相反）
    c_queue_kv.png    队列、KV 利用率、GPU 利用率时间序列 + 步构成
    c_budget.png      budget 扫描：步延迟 / 激活显存 / 吞吐 / 每步 token
    c_window.png      C2 的收益窗口（分路 vs 整批 varlen 两条 mix/sum 曲线 + 本实验工作点）
"""
import glob
import json
import os
import statistics as st

import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager, pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
FIGS = os.path.join(HERE, "figs")
TAGS = ["V1", "main"]
COLOR = {"V1": "#d62728", "main": "#1f77b4"}
REGIMES = [("stress", "真实压力口径"), ("ctrl", "受控口径（抢占 0）")]
BUDGETS = [1024, 2048, 4096, 16384]

FONT = os.path.expanduser("~/.fonts/simhei.ttf")
if os.path.exists(FONT):
    font_manager.fontManager.addfont(FONT)
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=FONT).get_name()
plt.rcParams.update({"axes.unicode_minus": False, "figure.dpi": 140,
                     "axes.grid": True, "grid.alpha": 0.3, "font.size": 9})


def jload(path):
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_c1(tag, regime):
    out = []
    for p in sorted(glob.glob(os.path.join(RES, f"c1_{tag}_{regime}_r*.json"))):
        d = jload(p)
        for r in d["runs"]:
            r["_file"] = os.path.basename(p)
            out.append(r)
    return out


def med(rs, path, nd=None):
    vs = []
    for r in rs:
        v = r
        for k in path.split("."):
            v = v[k]
        if v is not None:
            vs.append(v)
    if not vs:
        return None
    m = st.median(vs)
    return round(m, nd) if nd is not None else m


def rng(rs, path):
    vs = []
    for r in rs:
        v = r
        for k in path.split("."):
            v = v[k]
        vs.append(v)
    return min(vs), max(vs)


def ecdf(xs):
    xs = sorted(xs)
    return xs, [(i + 1) / len(xs) for i in range(len(xs))]


def bar2(ax, labels, vals, ylabel, title, logy=False, fmt="{:.1f}", annot=True):
    """两分支分组柱：x = labels，每组两根（V1/main）。"""
    import math
    xs = range(len(labels))
    w = 0.36
    for i, tag in enumerate(TAGS):
        v = [vals[l][tag] for l in labels]
        vv = [x if x and x > 0 else 0 for x in v]
        bars = ax.bar([x + (i - 0.5) * w for x in xs], vv, w * 0.94,
                      color=COLOR[tag], label=tag)
        if annot:
            for b, x in zip(bars, v):
                if x is None:
                    continue
                ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                        fmt.format(x), ha="center", va="bottom", fontsize=7.2)
    ax.set_xticks(list(xs))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8.5)
    ax.set_title(title, fontsize=9)
    if logy:
        ax.set_yscale("log")
    ax.legend(fontsize=7.5)


# ---------- 图 1：双口径面板 ----------
def fig_panel(d):
    fig, axes = plt.subplots(2, 3, figsize=(12.0, 6.2))
    labels = [r for r, _ in REGIMES]
    specs = [
        ("throughput.out_tok_s", "输出 TPS (tok/s)", "输出吞吐", False, "{:.0f}"),
        ("throughput.rps", "RPS (req/s)", "请求吞吐", False, "{:.2f}"),
        ("wall_s", "墙钟 (s)", "端到端墙钟", False, "{:.1f}"),
        ("itl_ms.p999", "ITL p999 (ms)", "ITL 尾部（越平滑越好）", False, "{:.0f}"),
        ("steps.dt_p999", "步延迟 p999 (ms)", "步延迟 P999", False, "{:.0f}"),
        ("memory.peak_alloc_mb", "峰值显存 (MB)", "torch 峰值 allocated", False, "{:.0f}"),
    ]
    for ax, (key, ylab, title, lg, fmt) in zip(axes.ravel(), specs):
        vals = {l: {t: med(d[(t, l)], key, 1) for t in TAGS} for l in labels}
        bar2(ax, labels, vals, ylab, title, logy=lg, fmt=fmt)
    fig.suptitle("C1 双口径指标面板（V1 vs main；柱高 = 各次重复中位数；红 = V1）", fontsize=10)
    fig.tight_layout()
    out = os.path.join(FIGS, "c_panel.png")
    fig.savefig(out)
    plt.close(fig)
    return out


# ---------- 图 2：步延迟 ECDF ----------
def fig_step_ecdf(d):
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 3.8))
    for ax, (regime, label) in zip(axes, REGIMES):
        for tag in TAGS:
            rs = d[(tag, regime)]
            if not rs:
                continue
            dt = rs[0]["raw"]["step_dt_ms"]
            x, y = ecdf(dt)
            ax.plot(x, y, color=COLOR[tag], lw=1.3, label=f"{tag}  (n={len(dt)} 步)")
            p999 = rs[0]["steps"]["dt_p999"]
            ax.axvline(p999, color=COLOR[tag], ls=":", lw=0.9, alpha=0.8)
        ax.set_xscale("log")
        ax.set_xlabel("步延迟 (ms, log)", fontsize=8.5)
        ax.set_ylabel("累积占比", fontsize=8.5)
        ax.set_title(f"{label}", fontsize=9)
        ax.legend(fontsize=7.5, loc="upper left")
    axes[0].annotate("main 的 P999 更低\n（相位分离让每步更短）", xy=(0.98, 0.55),
                     xycoords="axes fraction", ha="right", fontsize=7.5,
                     color="#1f77b4")
    axes[0].annotate("V1 的 P999 更高\n（混合步 = prefill + decode）", xy=(0.98, 0.40),
                     xycoords="axes fraction", ha="right", fontsize=7.5,
                     color="#d62728")
    fig.suptitle("步延迟分布：V1 的 P999 更高（参考数据报 -36.1%，本仓 stress 为 +31%）", fontsize=10)
    fig.tight_layout()
    out = os.path.join(FIGS, "c_step_ecdf.png")
    fig.savefig(out)
    plt.close(fig)
    return out


# ---------- 图 3：队列 / KV / 硬件 ----------
def fig_queue_kv(d):
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 6.0))
    regime = "stress"
    ax = axes[0][0]
    for tag in TAGS:
        r = d[(tag, regime)][0]["raw"]
        ax.plot(r["step_run"], color=COLOR[tag], lw=0.9, label=f"{tag} running")
        ax.plot(r["step_wait"], color=COLOR[tag], lw=0.9, ls="--", alpha=0.6,
                label=f"{tag} waiting")
    ax.set_xlabel("调度步序号", fontsize=8.5)
    ax.set_ylabel("请求数", fontsize=8.5)
    ax.set_title("队列深度（真实压力口径）", fontsize=9)
    ax.legend(fontsize=7)

    ax = axes[0][1]
    for tag in TAGS:
        r = d[(tag, regime)][0]["raw"]
        ax.plot(r["step_kv"], color=COLOR[tag], lw=0.9, label=tag)
    ax.set_xlabel("调度步序号", fontsize=8.5)
    ax.set_ylabel("used / pool", fontsize=8.5)
    ax.set_title("KV 缓存利用率（池 163 块被压满）", fontsize=9)
    ax.legend(fontsize=7.5)

    ax = axes[1][0]
    for tag in TAGS:
        smi = d[(tag, regime)][0]["raw"]["smi"]
        ts = [s[0] for s in smi]
        ax.plot(ts, [s[1] for s in smi], color=COLOR[tag], lw=0.9, label=f"{tag} GPU%")
        ax.plot(ts, [s[2] for s in smi], color=COLOR[tag], lw=0.9, ls=":", alpha=0.7,
                label=f"{tag} 显存控制器%")
    ax.set_xlabel("相对时间 (s)", fontsize=8.5)
    ax.set_ylabel("%", fontsize=8.5)
    ax.set_ylim(0, 105)
    ax.set_title("硬件利用率（nvidia-smi，粗粒度）", fontsize=9)
    ax.legend(fontsize=7, ncol=2)

    ax = axes[1][1]
    comp = ["pure_prefill", "pure_decode", "mixed"]
    xs = range(len(TAGS))
    bottom = [0] * len(TAGS)
    cols = {"pure_prefill": "#8c8c8c", "pure_decode": "#1f77b4", "mixed": "#d62728"}
    names = {"pure_prefill": "纯 prefill", "pure_decode": "纯 decode", "mixed": "混合"}
    for c in comp:
        vals = [med(d[(t, regime)], f"steps.{c}", 0) or 0 for t in TAGS]
        ax.bar(list(xs), vals, 0.5, bottom=bottom, color=cols[c], label=names[c])
        bottom = [b + v for b, v in zip(bottom, vals)]
    for x, t in zip(xs, TAGS):
        ax.text(x, bottom[x], f"{int(bottom[x])}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(list(xs))
    ax.set_xticklabels(TAGS)
    ax.set_ylabel("步数", fontsize=8.5)
    ax.set_title("步类型构成（真实压力口径）", fontsize=9)
    ax.legend(fontsize=7.5)

    fig.suptitle("C1 队列 / KV / 硬件利用率（真实压力口径，首次重复）", fontsize=10)
    fig.tight_layout()
    out = os.path.join(FIGS, "c_queue_kv.png")
    fig.savefig(out)
    plt.close(fig)
    return out


# ---------- 图 4：budget 扫描 ----------
def fig_budget(d5):
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 6.0))
    specs = [
        ("steps.dt_p999", "步延迟 p999 (ms)", "步延迟 P999 vs budget", False),
        ("memory.activation_peak_mb", "激活峰值 (MB)", "激活显存 vs budget（池固定 163）", False),
        ("throughput.out_tok_s", "输出 TPS (tok/s)", "输出吞吐 vs budget", False),
        ("steps.mean_tokens_per_step", "每步平均 token", "每步承载的 token 数", False),
    ]
    for ax, (key, ylab, title, lg) in zip(axes.ravel(), specs):
        for tag in TAGS:
            xs, ys = [], []
            for B in BUDGETS:
                rs = d5.get((tag, B))
                if not rs:
                    continue
                xs.append(B)
                ys.append(med(rs, key))
            if xs:
                ax.plot(xs, ys, "o-", color=COLOR[tag], lw=1.3, ms=4, label=tag)
        ax.axvline(4609, color="#888", ls="--", lw=0.9)
        ax.text(4609, 0.97, "  prompt≈4609\n  （chunked 生效边界）", fontsize=6.8,
                va="top", color="#555", transform=ax.get_xaxis_transform())
        ax.set_xscale("log", base=2)
        ax.set_xticks(BUDGETS)
        ax.set_xticklabels([str(b) for b in BUDGETS])
        ax.set_xlabel("max_num_batched_tokens (budget)", fontsize=8.5)
        ax.set_ylabel(ylab, fontsize=8.5)
        ax.set_title(title, fontsize=9)
        ax.legend(fontsize=7.5)
    fig.suptitle("C5 budget 扫描（定长 4864 · 池 163 · 并发 ~7 · 抢占 0）", fontsize=10)
    fig.tight_layout()
    out = os.path.join(FIGS, "c_budget.png")
    fig.savefig(out)
    plt.close(fig)
    return out


# ---------- 图 5：收益窗口 ----------
def fig_window(d, c2):
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.0))
    ax = axes[0]
    if c2:
        rows = c2["results"]
        Ps = sorted({r["plen"] for r in rows})

        def get(P, D, key):
            return next(r[key] for r in rows if r["plen"] == P and r["D"] == D)

        for i, P in enumerate(Ps):
            col = plt.cm.viridis(i / max(len(Ps) - 1, 1))
            xs = sorted(r["D"] for r in rows if r["plen"] == P)
            ax.plot(xs, [get(P, D, "mixv_over_sum") for D in xs], "s--", lw=1.0,
                    ms=3.2, color=col, alpha=0.7, label=f"P={P}　整批 varlen")
            ax.plot(xs, [get(P, D, "mix_over_sum") for D in xs], "o-", lw=1.5,
                    ms=3.8, color=col, label=f"P={P}　分路")
    ax.axhline(1.0, color="#444", lw=1.0)
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 4, 8, 16, 32, 64])
    ax.set_xticklabels(["1", "4", "8", "16", "32", "64"])
    ax.set_xlabel("混合步里的 decode 条数 D", fontsize=8.5)
    ax.set_ylabel("t_mix ÷ (t_p + t_d)", fontsize=8.5)
    ax.set_title("C2：混合步的收益窗口（<1 = 合并更省）", fontsize=9)
    ax.legend(fontsize=6.2, ncol=2, loc="upper left")

    ax = axes[1]
    labels, ratios, notes = [], [], []
    for regime, short in [("ctrl", "受控\n并发≈36"), ("stress", "真实压力\n并发≈33")]:
        a = med(d[("V1", regime)], "wall_s")
        b = med(d[("main", regime)], "wall_s")
        labels.append(short)
        ratios.append(a / b)
        notes.append(f"抢占 {int(med(d[('V1', regime)], 'preempt.preempt', 0))}")
    BASE = 0.97
    bars = ax.bar(labels, [r - BASE for r in ratios], 0.45, bottom=BASE,
                  color=["#d62728", "#d62728"])
    for b, r, n in zip(bars, ratios, notes):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + BASE,
                f"{r:.3f}\n({n})", ha="center", va="bottom", fontsize=7.5)
    ax.axhline(1.0, color="#444", lw=1.0, ls="--")
    ax.set_ylabel("V1 墙钟 ÷ main 墙钟", fontsize=8.5)
    ax.set_ylim(BASE, max(ratios) * 1.02)
    ax.set_title("本实验工作点：两个口径 V1 都不慢于 main", fontsize=9)
    fig.suptitle("C2/C1：合并的收益由混合步的 decode 条数决定；分路把整条曲线压到 1 以下", fontsize=10)
    fig.tight_layout()
    out = os.path.join(FIGS, "c_window.png")
    fig.savefig(out)
    plt.close(fig)
    return out


# ---------- 图 6：CUDA graph ----------
def fig_c3(d3):
    g, e = d3.get("graph"), d3.get("eager")
    if not g or not e:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.7))
    ax = axes[0]
    for doc, tag, lab in [(g, "V1", "graph（enforce_eager=False）"),
                          (e, "V1", "eager（enforce_eager=True）")]:
        xs = [r["D"] for r in doc["results"]]
        ys = [r["t_d"]["gpu_ms_min"] for r in doc["results"]]
        ax.plot(xs, ys, "o-", lw=1.3, ms=4, label=lab)
    xs = [r["D"] for r in g["results"]]
    ax.plot(xs, [r["t_d_generic"]["gpu_ms_min"] for r in g["results"]], "s--",
            lw=1.1, ms=3.5, color="#8c8c8c", label="通用分支（混合步里 decode 实走）")
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 8, 16, 32, 64])
    ax.set_xticklabels(["1", "8", "16", "32", "64"])
    ax.set_xlabel("纯 decode 批的条数 D", fontsize=8.5)
    ax.set_ylabel("步时间 (ms)", fontsize=8.5)
    ax.set_title("纯 decode 步：graph vs eager", fontsize=9)
    ax.legend(fontsize=7)

    ax = axes[1]
    xs = [r["D"] for r in g["results"]]
    eg = {r["D"]: r["t_d"]["gpu_ms_min"] for r in e["results"]}
    ratio = [eg[D] / next(r["t_d"]["gpu_ms_min"] for r in g["results"] if r["D"] == D)
             for D in xs]
    ax.bar([str(x) for x in xs], ratio, 0.5, color="#1f77b4")
    for i, r in enumerate(ratio):
        ax.text(i, r, f"{r:.2f}×", ha="center", va="bottom", fontsize=7.5)
    ax.axhline(1.0, color="#444", lw=0.8, ls="--")
    ax.set_xlabel("D", fontsize=8.5)
    ax.set_ylabel("eager ÷ graph", fontsize=8.5)
    ax.set_title("eager 相对 graph 的代价（V1 混合步必付）", fontsize=9)
    fig.suptitle("C3：CUDA graph 的收益量化", fontsize=10)
    fig.tight_layout()
    out = os.path.join(FIGS, "c_c3.png")
    fig.savefig(out)
    plt.close(fig)
    return out


def main():
    os.makedirs(FIGS, exist_ok=True)
    d = {(t, r): load_c1(t, r) for t in TAGS for r, _ in REGIMES}
    for k, v in d.items():
        print(f"  c1 {k}: {len(v)} 次")
    d5 = {}
    for t in TAGS:
        for B in BUDGETS:
            doc = jload(os.path.join(RES, f"c5_{t}_b{B}.json"))
            if doc:
                d5[(t, B)] = doc["runs"]
                print(f"  c5 {t} b{B}: {doc['runs'][0]['steps']['total']} 步 "
                      f"preempt={doc['runs'][0]['preempt']['preempt']}")
    c2 = jload(os.path.join(RES, "c2_grid_V1.json"))
    for fn, arg in [(fig_panel, d), (fig_step_ecdf, d), (fig_queue_kv, d),
                    (fig_budget, d5), (fig_window, None)]:
        try:
            out = fn(arg) if fn is not fig_window else fig_window(d, c2)
            if out:
                print("  ->", out)
        except Exception as ex:
            print(f"  !! {fn.__name__}: {type(ex).__name__}: {ex}")


if __name__ == "__main__":
    main()
