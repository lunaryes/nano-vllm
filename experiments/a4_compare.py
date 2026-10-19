#!/usr/bin/env python
"""A4 比对：抢占运行 vs 不抢占对照。

用法：
    python experiments/a4_compare.py <preempt.json> <control.json> \
        [--repeat <control_repeat.json>] [--noise <n_minus_1.json>]

判据
----
L1  功能健全性   全部完成、长度 == max_tokens、输出非空、实验组抢占 > 0 且对照组 == 0
L1' 结构不变式   len(block_table)*256 >= num_tokens 且 num_cached_tokens <= len(block_table)*256
L2  数值层等价   匹配前缀内 logits 差异量与「批组成噪声地板」同量级
L3  无 KV 污染   被抢占 seq 的最长一致前缀必须**越过**抢占点
                （若读了脏 KV，重算续算产出的那个 token 必定错 → 首异位置 == 抢占点 ci_next）

噪声地板怎么来
--------------
对照组(N=16) 与 N=15 的对照跑（prompt 相同、只有解码批 16 行 vs 15 行）之间的差异，
就是"批组成变化"造成的纯浮点噪声。抢占组与对照组的差异必须落在这个量级内。
"""
import argparse
import json
from collections import defaultdict


def load(p):
    with open(p) as f:
        return json.load(f)


def first_div(a, b):
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return -1 if len(a) == len(b) else n


def build_samples(doc):
    S = defaultdict(dict)
    for s in doc["samples"]:
        S[s["label"]].setdefault(s["ci"], s)
    return S


def compare_sample(a, b):
    """按 token id 求交比较 logits。返回 (top1一致, 交集大小, 交集内max|Δ|, 相对absmax)。"""
    da = dict(zip(a["top_idx"], a["top_val"]))
    db = dict(zip(b["top_idx"], b["top_val"]))
    common = set(da) & set(db)
    dmax = max((abs(da[i] - db[i]) for i in common), default=0.0)
    rel = dmax / max(abs(a["absmax"]), 1e-9)
    return a["top_idx"][0] == b["top_idx"][0], len(common), dmax, rel


def collect(S1, S2, labels):
    """逐 (label, ci) 比较，直到 token 首次分歧。返回明细与聚合。"""
    per_label = {}
    agg = {"n": 0, "top1_same": 0, "overlap": [], "rel": [], "rel_corrupt_proxy": []}
    for lab in labels:
        a_tok = S1["tokens"][lab]
        b_tok = S2["tokens"][lab]
        div = first_div(a_tok, b_tok)
        lcp = len(a_tok) if div == -1 else div
        det = {"div": div, "lcp": lcp, "o": len(a_tok), "rel_max": 0.0,
               "top1_bad": 0, "overlap_min": 20, "n": 0}
        for ci in range(lcp):
            sa, sb = S1["samples"][lab].get(ci), S2["samples"][lab].get(ci)
            if not sa or not sb:
                continue
            t1, ncom, dmax, rel = compare_sample(sa, sb)
            det["n"] += 1
            agg["n"] += 1
            det["rel_max"] = max(det["rel_max"], rel)
            agg["rel"].append(rel)
            agg["overlap"].append(ncom)
            det["overlap_min"] = min(det["overlap_min"], ncom)
            if t1:
                agg["top1_same"] += 1
            else:
                det["top1_bad"] += 1
        per_label[lab] = det
    return per_label, agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("preempt")
    ap.add_argument("control")
    ap.add_argument("--repeat", default=None, help="对照的重复跑（确定性检查）")
    ap.add_argument("--noise", default=None, help="N-1 条对照跑（噪声地板）")
    args = ap.parse_args()

    P, C = load(args.preempt), load(args.control)
    R = load(args.repeat) if args.repeat else None
    N1 = load(args.noise) if args.noise else None

    def prep(doc):
        return {"tokens": {o["label"]: o["token_ids"] for o in doc["outputs"]},
                "samples": build_samples(doc),
                "o": {o["label"]: o["o"] for o in doc["outputs"]}}

    pP, pC = prep(P), prep(C)

    print("### A4 抢占重入比对")
    print(f"- 实验 `{P['tag']}`: branch={P['branch']} 池={P['gpu_blocks']}/{P['natural_blocks']} 块"
          f"  需求≈{P['demand_blocks']} 块")
    print(f"- 对照 `{C['tag']}`: branch={C['branch']} 池={C['gpu_blocks']}/{C['natural_blocks']} 块"
          f"  需求≈{C['demand_blocks']} 块")
    print(f"- N={P['n']} plen={P['plen']} budget={P['budget']} util={P['util']} "
          f"O∈[{min(P['o_list'])},{max(P['o_list'])}]")
    print()

    # ---------- 确定性 ----------
    if R is not None:
        pR = prep(R)
        eq = all(pC["tokens"][k] == pR["tokens"][k] for k in pC["tokens"])
        _, agg = collect(pC, pR, list(pC["tokens"]))
        print(f"**确定性**：对照重跑 token 完全一致={eq}  "
              f"logits 点数={agg['n']} top1全同={agg['top1_same']==agg['n']} "
              f"max rel Δ={max(agg['rel']) if agg['rel'] else 0:.2e}")
        print()

    # ---------- 噪声地板 ----------
    if N1 is not None:
        pN = prep(N1)
        lab = [l for l in pC["tokens"] if l in pN["tokens"]]
        _, aggn = collect(pN, pC, lab)
        print(f"**噪声地板**（`{N1['tag']}` N={N1['n']} vs `{C['tag']}` N={C['n']}，仅批组成不同）：")
        print(f"- 比较点数 {aggn['n']}，top1 一致 {aggn['top1_same']}/{aggn['n']}"
              f"（{100.0*aggn['top1_same']/max(aggn['n'],1):.1f}%）")
        print(f"- top-20 交集均值 {sum(aggn['overlap'])/max(len(aggn['overlap']),1):.1f}/20")
        print(f"- **max 相对 Δ = {max(aggn['rel']) if aggn['rel'] else 0:.2e}**")
        NOISE = max(aggn["rel"]) if aggn["rel"] else 0.0
        print()

    # ---------- L1 ----------
    n = P["n"]
    len_ok = all(len(o["token_ids"]) == o["o"] for o in P["outputs"])
    print("## L1 功能健全性")
    print("| 项 | 实验(抢占) | 对照 |")
    print("| :--- | :--- | :--- |")
    print(f"| 完成 / 长度正确 / 非空 | {len(P['outputs'])}/{n} / {len_ok} / "
          f"{all(len(o['token_ids'])>0 for o in P['outputs'])} | "
          f"{len(C['outputs'])}/{n} / "
          f"{all(len(o['token_ids'])==o['o'] for o in C['outputs'])} / "
          f"{all(len(o['token_ids'])>0 for o in C['outputs'])} |")
    print(f"| **抢占次数** | **{P['preempt_count']}** | **{C['preempt_count']}** |")
    print(f"| can_append 失败 | {P['stats']['can_append_fail']} | {C['stats']['can_append_fail']} |")
    print(f"| can_allocate 失败 | {P['stats']['can_allocate_fail']} | {C['stats']['can_allocate_fail']} |")
    print(f"| 步数 / 混合步 | {P['stats']['runs']} / {P['stats']['mixed_runs']} | "
          f"{C['stats']['runs']} / {C['stats']['mixed_runs']} |")
    print(f"| 耗时 | {P['seconds']}s | {C['seconds']}s |")
    l1 = (len(P["outputs"]) == n and len_ok and P["preempt_count"] > 0
          and C["preempt_count"] == 0)
    print(f"\n**L1 {'通过' if l1 else '不通过'}**\n")

    # ---------- L1' ----------
    bad = []
    for tag, doc in (("实验", P), ("对照", C)):
        for s in doc["samples"]:
            cap = s["nb"] * 256
            if not (cap >= s["nt"] and s["nc"] <= cap and s["nb"] > 0):
                bad.append((tag, s["label"][:8], s["ci"], s["nt"], s["nc"], s["nb"]))
    print("## L1' 结构不变式")
    print(f"- 检查 {len(P['samples']) + len(C['samples'])} 个采样点，违反 **{len(bad)}**")
    for b in bad[:8]:
        print(f"  - {b}")
    print()

    # ---------- L2 / L3 ----------
    all_labels = list(pP["tokens"])
    per_label, agg = collect(pP, pC, all_labels)

    events = defaultdict(list)
    for e in P["preempt_events"]:
        events[e["label"]].append(e)

    pre_seqs = [l for l in all_labels if events[l]]
    clean = [l for l in pre_seqs if per_label[l]["div"] == -1]
    at_px, before_px, after_px = [], [], []
    for l in pre_seqs:
        d = per_label[l]["div"]
        pxs = [e["ci_next"] for e in events[l]]
        if d == -1:
            continue
        if d in pxs:
            at_px.append(l)
        elif d < min(pxs):
            before_px.append(l)
        else:
            after_px.append(l)

    print("## L2 数值层等价（按 token id 求交比较 logits）")
    print(f"- 比较点数 {agg['n']}，top1 一致 **{agg['top1_same']}/{agg['n']}**"
          f"（{100.0*agg['top1_same']/max(agg['n'],1):.1f}%）")
    print(f"- top-20 交集均值 {sum(agg['overlap'])/max(len(agg['overlap']),1):.1f}/20")
    print(f"- **max 相对 Δlogit = {max(agg['rel']) if agg['rel'] else 0:.2e}**")
    if N1 is not None:
        print(f"- 噪声地板 max 相对 Δ = {NOISE:.2e}  →  实验/地板 = "
              f"{(max(agg['rel'])/NOISE if NOISE else float('inf')):.2f}×")
    l2 = bool(agg["rel"]) and (N1 is None or max(agg["rel"]) <= max(NOISE * 2, 1e-2))
    print(f"\n**L2 {'通过' if l2 else '不通过'}**\n")

    print("## L3 无 KV 污染特征")
    print(f"- 被抢占 seq：**{len(pre_seqs)}**/{n}")
    print(f"  - 输出与对照**完全一致**：{len(clean)}")
    print(f"  - 首异位置 **== 抢占点**（污染特征）：**{len(at_px)}** ← 应为 0")
    print(f"  - 首异位置 **早于** 抢占点（抢占前就已噪声分歧，无法评价重算）：{len(before_px)}")
    print(f"  - 首异位置 **晚于** 抢占点（重算续算那段是对的）：{len(after_px)}")
    for l in at_px:
        print(f"    ! {l[:8]} div={per_label[l]['div']} 抢占点={[e['ci_next'] for e in events[l]]}")
    l3 = len(at_px) == 0
    print(f"\n**L3 {'通过' if l3 else '不通过'}**\n")

    print("## 逐 seq 明细")
    print("| label | O | 抢占 | 抢占点 ci_next | 被抢占时 len/已缓存块 | 首异位置 | 抢占前Δrel | top1不一致 | 交集min |")
    print("| :--- | ---: | ---: | :--- | :--- | ---: | ---: | ---: | ---: |")
    for l in all_labels:
        d = per_label[l]
        evs = events.get(l, [])
        pxs = ",".join(str(e["ci_next"]) for e in evs) or "-"
        lens = ",".join(f"{e['len']}/{e['num_cached_tokens']//256}" for e in evs) or "-"
        print(f"| {l[:8]} | {d['o']} | {len(evs)} | {pxs} | {lens} | {d['div']} | "
              f"{d['rel_max']:.2e} | {d['top1_bad']} | {d['overlap_min']} |")
    print()

    if P["preempt_events"]:
        ev = P["preempt_events"]
        cl = [e["num_cached_tokens"] // 256 for e in ev]
        print(f"## 抢占事件（{len(ev)} 次）")
        print(f"- 生成进度 ci_next ∈ [{min(e['ci_next'] for e in ev)}, {max(e['ci_next'] for e in ev)}]")
        print(f"- 已缓存块数 ∈ [{min(cl)}, {max(cl)}]  （>0 = 重入走前缀命中续算）")
        print(f"- 被抢占时 len ∈ [{min(e['len'] for e in ev)}, {max(e['len'] for e in ev)}]")
        print()
        print("## 抢占重入的续算点（重入后第一次采样）")
        print("| label | 抢占点 ci_next | 被抢占时已缓存块 | 重入后 nc/nt | 重算 token 数 | 块数 |")
        print("| :--- | ---: | ---: | :--- | ---: | ---: |")
        for e in ev:
            lab = e["label"]
            cand = [s for s in P["samples"]
                    if s["label"] == lab and s["ci"] >= e["ci_next"] and s["preempted"]]
            if not cand:
                print(f"| {lab[:8]} | {e['ci_next']} | {e['num_cached_tokens']//256} | "
                      f"（无后期采样） | - | - |")
                continue
            s = min(cand, key=lambda x: x["ci"])
            print(f"| {lab[:8]} | {e['ci_next']} | {e['num_cached_tokens']//256} | "
                  f"{s['nc']}/{s['nt']} | **{s['nt']-s['nc']}** | {s['nb']} |")
        print()

    print(f"**A4 总判定：{'PASS' if (l1 and l2 and l3 and not bad) else 'FAIL'}**")


if __name__ == "__main__":
    main()
