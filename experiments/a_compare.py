#!/usr/bin/env python
"""A1 / A2 结果比较。

用法：
    python experiments/a_compare.py experiments/results/a_main.json experiments/results/a_V1.json

A1 分块不变性：同一 tag 内，以最大 budget（几乎不分块）为基准，比较其他 budget。
A2 跨实现等价：同一 budget 下，跨 tag 逐 token 比对。

注意：token 一致率会被自回归放大——一个位置翻转会让该 seq 后续全变。
因此同时报告「完全一致的 seq 数」，它不受放大影响。
"""
import json
import sys


def compare(a, b):
    res = {"n_eq": 0, "n_tot": 0, "first": None, "bad": [], "n_seq": len(a)}
    if len(a) != len(b):
        res["first"] = ("seq_count_mismatch", len(a), len(b), 0)
        return res
    for si, (sa, sb) in enumerate(zip(a, b)):
        if len(sa) != len(sb):
            res["bad"].append((si, "len_mismatch"))
            res["first"] = res["first"] or (si, "len_mismatch", len(sa), len(sb))
        marked = False
        for pi, (ta, tb) in enumerate(zip(sa, sb)):
            res["n_tot"] += 1
            if ta == tb:
                res["n_eq"] += 1
            else:
                if not marked:
                    marked = True
                    res["bad"].append((si, pi))
                if res["first"] is None:
                    res["first"] = (si, pi, ta, tb)
    return res


def fmt(res):
    rate = (100.0 * res["n_eq"] / res["n_tot"]) if res["n_tot"] else 0.0
    status = "PASS" if res["n_eq"] == res["n_tot"] else "DIFF"
    ok_seq = res["n_seq"] - len(res["bad"])
    return f"{rate:7.3f}% ({res['n_eq']}/{res['n_tot']}) {status}  seq一致={ok_seq}/{res['n_seq']}"


def main():
    paths = sys.argv[1:]
    if len(paths) < 1:
        print(__doc__)
        sys.exit(1)
    docs = []
    for p in paths:
        with open(p) as f:
            docs.append(json.load(f))

    for d in docs:
        print(f"=== A1 分块不变性 | tag={d['tag']} | eager={d['enforce_eager']} "
              f"| max_tokens={d['max_tokens']}")
        print(f"    lengths={d['lengths']}")
        bs = sorted(d["budgets"].keys(), key=int)
        base_b = bs[-1]
        base = d["budgets"][base_b]["runs"][0]
        print(f"  基准 budget={base_b}（blocks={d['budgets'][base_b]['gpu_blocks']}）")
        for b in bs[:-1]:
            info = d["budgets"][b]
            res = compare(base, info["runs"][0])
            extra = "" if info.get("repeat_identical", True) else "  [repeat 不一致→采样不确定]"
            print(f"  budget={int(b):6d} vs {base_b}: {fmt(res)}"
                  f"  blocks={info['gpu_blocks']}  首个翻转={res['first']}{extra}")
            if res["bad"]:
                s = ", ".join(f"seq{i}@{p}" for i, p in res["bad"])
                print(f"           翻转的 seq: {s}")
        print()

    if len(docs) >= 2:
        print("=== A2 跨实现等价")
        tags = [d["tag"] for d in docs]
        common = set.intersection(*(set(d["budgets"]) for d in docs))
        for b in sorted(common, key=int):
            base = docs[0]["budgets"][b]["runs"][0]
            for d in docs[1:]:
                res = compare(base, d["budgets"][b]["runs"][0])
                print(f"  budget={int(b):6d}  {tags[0]} vs {d['tag']}: {fmt(res)}"
                      f"  首个翻转={res['first']}")
                if res["bad"]:
                    s = ", ".join(f"seq{i}@{p}" for i, p in res["bad"])
                    print(f"             翻转的 seq: {s}")
        print()


if __name__ == "__main__":
    main()
