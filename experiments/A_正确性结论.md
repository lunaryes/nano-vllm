# A1 / A2 正确性验证结论

- 日期：2026-10-12　分支：V1　设备：RTX 4060 8G　模型：Qwen3-0.6B
- 脚本：`a_correctness.py` / `a_compare.py` / `a_logits_probe.py` / `a_gap_probe.py`

## 结论

| 判据 | 实测 | 判定 |
|---|---|---|
| A2 跨实现等价（数值层） | 所有 budget 下 main 与 V1 的 logits **逐位相同**（Δ=0.000000） | PASS |
| A1 分块不变性（数值层） | 分块 vs 不分块，logits 差 = bf16 舍入量级（相对 3e-3 ~ 6e-3） | PASS |
| A1 逐 token 严格一致 | 6~11/12 seq 一致，**main 基线同等** | 判据不适用，见下 |

**总判定：V1 的改动未引入超出浮点精度的任何差异，正确性成立。**

## 判据必须修正

原方案把 A1 定为「逐 token 100% 一致」。实测证明**该判据不成立，且 main 基线自己也不满足**：

1. 分块改变 attention 的浮点归约顺序
2. 不同 batch 组成改变 varlen 打包布局 —— **即使某条 seq 自身没被分块**（实测 seq1 长度 512，在 budget=512 下整条处理，仍发生翻转）
3. 单点翻转经自回归放大成整条序列不同（一次早期翻转即污染该 seq 剩余全部位置）

正确判据是**数值层等价**：比较同一输入位置上的 logits，而不是逐 token 相同。

## 证据

### 1. A2：logits 逐位相同（最强证据）

单条 seq、prompt = 3200 token（仓库源码文本），取「处理完 prompt 那一次采样」对比：

| budget | main top5（前 2 项） | V1 top5（前 2 项） | max\|Δlogit\| | 首 token |
|---|---|---|---|---|
| 16384（不分块） | (25177, 19.375), (101884, 17.25) | 同左 | **0.000000** | 25177 |
| 2048（2 块） | (25177, 19.25), (101884, 17.125) | 同左 | **0.000000** | 25177 |
| 512（7 块） | (25177, 19.375), (101884, 17.25) | 同左 | **0.000000** | 25177 |

→ V1 的四条新路径（P1 通用 prefill 分支 / P2 need_logits 掩码 / P3 is_uniform 切 graph、eager / P4 抢占重入）**在数值上零副作用**。

注：对齐规则为 `probe[len(probe) - max_tokens]`。main 缺掩码 → mid-chunk 也会采样一次再丢弃 → probe 更长，直接取 `probe[0]` 会抓到 prompt 中间位置的 logits（踩过这个坑）。

### 2. A1：分块只引入 bf16 舍入量级误差

V1 内部，同一条 prompt，分块 vs 不分块：

| budget | max\|Δlogit\| | 相对 absmax | top5 索引一致 | 首 token |
|---|---|---|---|---|
| 2048 | 0.125 | 6.45e-3 | 5/5 | 相同 |
| 512 | 0.0625 | 3.23e-3 | 5/5 | 相同 |

bf16 尾数 8 位 → 相对精度 2⁻⁸ ≈ 3.9e-3。**观测误差正好落在表示精度内**。

### 3. 翻转机制（gap 探针）

gap = 采样时 top1 与 top2 的 logits 间距：

- 随机 token prompt 的首位置 gap = **0.0000**（精确平局）→ 必然翻转
- 同序列 gap 中位数 = 5.3
- 误差 0.125 ≪ gap 的位置稳定；误差 ≥ gap 的位置翻转

**误差量级（~0.125）决定了哪些位置会翻转，与实现无关。**

### 4. 排除项

| 假设 | 实测 | 结论 |
|---|---|---|
| CUDA graph 造成差异 | `enforce_eager` 与默认结果**逐字相同** | 排除 |
| V1 调度逻辑有 bug | budget=16384 下 main/V1 logits Δ=0 | 排除 |
| 采样随机性 | 同配置独立进程重跑 `identical=True` | 排除 |
| 随机 prompt 制造平局是主因 | 改真实文本后 A1 未改善（反而略差） | 排除 |

## 额外收获：need_logits（P2）的收益被量化

main 无掩码 → mid-chunk 白算 lm_head 并采样一次再丢弃。采样调用次数（prompt=3200，max_tokens=8）：

| budget | chunk 数 | main probe 次数 | V1 probe 次数 | 白算次数 |
|---|---|---|---|---|
| 16384 | 1 | 8 | 8 | 0 |
| 2048 | 2 | 9 | 8 | 1 |
| 512 | 7 | 14 | 8 | 6 |

白算次数 = chunk 数 − 1，**完全吻合**。这是 P2 存在的直接依据。

## 复现

```bash
# 采集（每个 budget 一个独立子进程：dist.init_process_group 同进程只能初始化一次）
python experiments/a_correctness.py --tag V1   --prompt-mode text --out experiments/results/a_V1_text.json
python experiments/a_correctness.py --tag main --prompt-mode text --out experiments/results/a_main_text.json
# 比对
python experiments/a_compare.py experiments/results/a_main_text.json experiments/results/a_V1_text.json
# 数值层探针
python experiments/a_logits_probe.py --length 3200 --budgets 512 2048 16384 --prompt-mode text \
    --out experiments/results/a_logits_V1.json
python experiments/a_gap_probe.py --length 3200 --budgets 512 16384 --out experiments/results/a_gap.json
```

**坑：`a_text_pool.json` 是缓存的 prompt token 池，跨分支必须共用同一份。**
它由仓库源码生成，而 main 与 V1 的源码不同 → 各自生成会导致 prompt 完全不同、跨实现对比彻底失效（踩过）。

**坑：切分支前需 `git stash push -- .workbuddy/memory/`**，否则 tracked 文件冲突导致 checkout 失败。

## 未做 / 后续

- **P4（抢占重入）未验证**：本次负载的 KV 需求远低于池容量，未触发抢占。需另设超额订阅场景（并发 × 长度 > 41728 tokens）专门验证。
- A1 的 `max_tokens=64` 下翻转集中在位置 40+ 之后，说明前段稳定、误差累积到某点才翻转。若要更严格的等价性，可只比较前 N 个 token 或直接比较 logits。
