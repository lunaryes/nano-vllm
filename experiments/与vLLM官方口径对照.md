# 我们的结果与 vLLM 官方口径对照

> 起因：有人做了和我们同样的优化（混合调度 + chunked prefill），报出 Qwen3-0.6B / 256 并发 / chunk=1024 下
> **吞吐 +4.6%、P999 步延迟 −36.1%、峰值显存 −3.0%**。因此要问：**我们的实测与 vLLM 官方在 V1 中引入
> 混合调度 + chunked prefill 后的收益吻合吗？**
>
> 本文件所用的每一个「我们」的数字，都由 `experiments/vllm_ref_compare.py` 从 `results/*.json` 现算
> （输出见 `results/vllm_ref_compare.md`），不引用记忆或旧文档里的转述。
>
> **§2「混合批 × CUDA graph」是后加的，依据 C6/C6b 新测**（`c6_prefill_graph.py` / `c6b_bisect.py`
> → `results/c6_prefill_graph.json`、`figs/c6_prefill_graph.png`），它**修正**了 §2 原有的结论。

---

## 0. 先明确「官方收益」到底指什么 —— 三条口径必须分开

这是回答该问题的全部关键。vLLM 侧的「收益」不是一个数，而是**三份性质完全不同的主张**：

| # | 出处 | 主张 | 归因 |
| :--- | :--- | :--- | :--- |
| **①** | [vLLM V1 alpha 博客](https://blog.vllm.ai/2025/01/27/v1-alpha-release.html)（2025-01-27） | 「up to **1.7x** higher throughput compared to V0」 | 原文：「These dramatic performance gains stem from comprehensive **CPU overhead reductions** across the entire stack」；并特别说明「the kernels used for V0 and V1 are **almost identical**, the performance difference is mainly due to the architectural improvements (**reduced CPU overheads**)」 |
| **②** | [vLLM 官方调优文档 · Chunked Prefill](https://docs.vllm.ai/en/latest/configuration/optimization.html) | 两条收益：**(a) 改善 ITL**（因为 decode 被优先）；**(b) 更好的 GPU 利用率**（把 compute-bound 的 prefill 与 memory-bound 的 decode 放进同一批） | 调度策略本身 |
| **③** | [SARATHI / Sarathi-Serve, OSDI'24](https://www.usenix.org/conference/osdi24/presentation/agrawal)（chunked prefill 的源头论文） | 消除 vLLM 的**数秒级 generation stall**、显著降低 **P99 TBT**；明确「**prioritizing TBT over TTFT**」；吞吐是「same or better」。在**尾延迟 SLO 约束下的服务容量** 2.6×（Mistral-7B/单 A100）/ 3.7×（Yi-34B/2×A100）/ 5.6×（Falcon-180B/PP） | 调度策略本身 |

**三条主张里，只有 ②③ 是关于「混合调度 + chunked prefill」的；① 的 1.7× 是引擎层重建（多进程 EngineCore、持久化输入、torch.compile、prefix caching、API server 重叠）的功劳。** 博客里讲统一调度器（unified scheduler）那一节，**一个性能数字都没有**，只讲 `{request_id: num_tokens}` 这个表示法如何优雅地同时支持 chunked prefill / prefix caching / 推测解码。

---

## 1. 逐维度对照

### 1.1 吞吐：不吻合，而且**本来就不该吻合**

| | 数值 |
| :--- | :--- |
| vLLM 官方（V1 vs V0，Llama 3.1 8B / ShareGPT） | **最高 1.7×** |
| 我们 · 官方 bench 口径（stress，提交 256 条） | **0.990×**（V1/main 墙钟 1.011×） |
| 我们 · 受控口径（ctrl，抢占 = 0） | **0.958×**（墙钟 1.044×） |

不吻合的根因不是实验失败，而是**改的层不同**：

- 官方 1.7× 的杠杆是 **CPU 开销**。而 nano-vllm 是**单进程离线批处理**：没有 OpenAI API server、没有流式解码、没有分词/去分词进程、没有 ZeroMQ IPC、也没有推测解码与 LoRA。① 的主要杠杆在本仓**几乎不存在**。
- 我们的 V1 分支只改了 **Scheduler + ModelRunner 的调度路径**，等价于 vLLM V1 的「调度器那一半」，而 ① 来自「执行循环那一半」。
- 更直接的旁证：vLLM 文档自己写「**For optimal throughput, we recommend setting `max_num_batched_tokens` > 8192**」——**吞吐最优的设置恰恰是把 chunk 关掉（或几乎关掉）的区间**。官方从没把 chunked prefill 当成吞吐优化来卖。

> 因此：**吞吐这一项"不吻合"是预期内的，不构成矛盾。** 真正该拿来对照的是 ② 和 ③。

### 1.2 尾延迟（ITL / TBT）：**强吻合**，而且是逐条吻合

B 组受控三场景（budget = 1024，两版抢占均为 0）：

| 场景 | 分支 | ITL CV | ITL p99 | ITL max | 停顿步 | 最长连续停顿 |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| 短 prompt（768） | V1 | 0.490 | 35.6 ms | 45 ms | **1** | 1 |
| 短 prompt（768） | main | 0.674 | 42.3 ms | 234 ms | 96 | 8 |
| 长 prompt（4864） | V1 | 0.612 | 92.5 ms | **99 ms** | **5** | 5 |
| 长 prompt（4864） | main | 1.686 | 309.2 ms | **1985 ms** | 480 | **40** |
| 混合 | V1 | 0.657 | 80.4 ms | **84 ms** | **1** | 1 |
| 混合 | main | 1.777 | 303.1 ms | **1222 ms** | 288 | 24 |

- `main ÷ V1` 的 **ITL max = 5.16× / 20.11× / 14.46×**；**ITL p99 = 1.19× / 3.34× / 3.77×**。
- main 的 480 个停顿步、最长连续 40 步 × ≈55 ms ≈ **2.2 s**，正对上实测 ITL max **1985 ms**。

SARATHI 论文的 Fig.2 描述（"effectively eliminates the multi-second generation stalls observed in vLLM"、"maintains a significantly lower P99 TBT as the query rate increases"）与本表**在现象、量级、指标选择上都是同一件事**。main 的「整步只做 prefill、decode 完全停摆」正是论文所说的 generation stall。

旁证：vLLM 文档给的两条收益我们都测到了 —— ITL ✓；**GPU 利用率 95.9%、KV 利用率 82.6~86.6%** ✓（混合批把两类负载放进同一步）。

### 1.3 TTFT：吻合，且能精确解释「为什么我们没变差」

| `main ÷ V1` | TTFT p50 | TTFT p99 | ITL p50 | ITL p99 | ITL max |
| :--- | ---: | ---: | ---: | ---: | ---: |
| 短 | 1.04× | 1.03× | 1.00× | 1.19× | 5.16× |
| 长 | **1.00×** | 1.00× | 0.99× | 3.34× | **20.11×** |
| 混合 | 1.03× | 1.02× | 1.00× | 3.77× | **14.46×** |

TTFT 几乎不动（1.00~1.04×），而 ITL 尾部差 5~20×。这与文档「budget 小 → ITL 好 / budget 大 → TTFT 好」以及 SARATHI「**prioritizing TBT over TTFT**」的框架一致，但有一处需要解释清楚：

**SARATHI 是牺牲 TTFT 换 TBT，我们这里 TTFT 连牺牲都没有。** 原因是可算的：并发 8 条 seq，每步 decode 只吃 8 个 token，而 budget 是 1024 —— **挪用比例 8/1024 = 0.78%**。实测长场景 TTFT p50 从 35 614 ms 变到 35 477 ms（−0.39%，在噪声内）。也就是说：**我们的 V1 分支没有重排 prefill 的推进顺序（每步仍只吃一个 chunk、admit 次序不变），只把"已出 token 的 seq"插进同一步** —— 所以 prefill 吞吐几乎不变 ⇒ TTFT 不变；被改变的是"已出 token 的 seq 能否在 prefill 步里继续出字" ⇒ ITL 尾部。

这也正是 §2.6（`抖动来源与实验效度.md`）那条指标归属结论的官方版注脚。

### 1.4 官方文档自己承认的边界，我们实测复现了

vLLM 文档原文：

> 「If `max_num_batched_tokens` is the same as `max_model_len`, that's **almost the equivalent to the V0 default scheduling policy** (except that it still prioritizes decodes).」

这与我们踩过的坑**逐字对应**：

- B 组旧 base（budget 16384 ≥ prompt 4700）→ chunked 机制**未激活**，两版无差；
- C5 budget 扫描（budget = `max_num_batched_tokens`）：

| budget | V1 步 p999 | main 步 p999 |
| ---: | ---: | ---: |
| 1024 | 102.3 ms | 71.6 ms |
| 16384 | 950.1 ms | 744.6 ms |

**budget 自身的效应 = 9.29×（V1）/ 10.40×（main）**，而同 budget 下分支差异只有 1.13~1.43×。文档说的「budget 是 ITL⇄TTFT 的总旋钮」在我们这里有精确的数值印证。

### 1.5 步延迟 P999 与峰值显存：官方**没有**主张，不参与对照

- vLLM 博客**未给** P999 步延迟，也**未给**显存数字（那两项属于第三方那组数据，见 C 组 §C5 的辨析）。
- 我们这边：同 budget 下 **步 P999 V1 高 1.13~1.43×**（即更差）、**峰值显存逐位相同（1.0000×）**。
- SARATHI 的「uniform batches 消除 pipeline bubbles」目标是**流水线并行（多卡）**场景；本机是单卡，没有 bubble 可消。
- ⚠ **「V1 的步更长」不能归因给"混合步牺牲了 CUDA graph"** —— C6 实测（见 §2）这笔代价只有 **0.3~1.3 ms/步（1.00~1.02×）**，相对 74~112 ms 的步长是 1% 量级。V1 的步更长是因为**一步做的事更多**（prefill chunk 与 decode 合并进同一步，而 main 把它们拆成两步），不是 graph 的锅。**代价指标应看墙钟/吞吐（0.96~1.04×），不要看"步延迟"** —— 两版的"一步"不是同一个工作量单位。

---

## 2. 官方有、我们「看似」没有的那台机器：混合批 × CUDA graph

vLLM 博客两处原文：

> 「Given the **dynamism in V1—such as combining prefill and decode within the same batch**—a flexible and high-performance attention kernel was essential. **FlashAttention 3** effectively addresses this requirement.」

> 「V1 introduces **piecewise CUDA graphs** to alleviate the limitations of CUDA graphs.」

nano-vllm 没有这套机制：decode 走 graph 要求"固定 batch + 静态 buffer"，而 `llm_engine.step()` 里

```python
is_uniform = all(seq.is_prefill == 0 for seq in seqs)      # 只要有一条 prefill，整步就 eager
```

于是任何混合步都整步回退 eager。**这一节原本的结论是「vLLM 把 chunked prefill 的代价压到了几乎免费，我们这里是一笔要算的账」。C6/C6b 把它推翻了 —— 我们不需要那台机器也能拿到几乎同样的结果。** 完整机制与优化路线见 **`CUDA图与混合批_机制与优化路线.md`**，下面只放结论。

### 2.1 先把机制说准：vLLM V1 到底靠什么让混合批不牺牲 graph

不是"把混合批也录进图"这一个技巧，而是**四件事叠起来**：

| # | 机制 | 作用 |
| :--- | :--- | :--- |
| ① | **统一的扁平批表示**（`{request_id: num_tokens}` + 张量化元数据） | 批里没有"prefill 分支 / decode 分支"，只有"总 token 数"这一个自由量。混合批因此退化成"形状由 T 决定的普通批" |
| ② | **按 num_tokens / batch_size 分桶 + padding**（`cudagraph_capture_sizes`、`max_cudagraph_capture_size`） | 实际批填到最近的桶，多出来的行用 dummy 掩盖。**形状静态化**是录图的前提 |
| ③ | **在 attention 边界切图**（piecewise）：图**外** eager 跑 attention，图**内**跑 attention 之间的一切 | attention 之间是 token-wise 的，形状只由 T 决定；attention 的形状随每个请求的 seq_len 变。把不稳定的一段摘出去，其余全进图 |
| ④ | **attention 输出张量作为输入传入**（output-as-input）+ `torch._C.Tag.cudagraph_unsafe` 标记 + inductor graph partition | 让图自己拥有 buffer（修法③的必要条件）；N 个 attention 自动切出 N+1 段，不用手写切分 |

官方文档对 ③ 的原话：

> 「…we only capture the cudagraph for the piece of graph between attention operations… This is based on a common observation: **computation between attentions are usually token-wise and easy to deal with for cudagraph; while the attention operation is non-trivial to be cudagraph compatible.** Thus, by running the attention operation in eager mode while the rest operations in cudagraph, we keep the flexibility of the attention operation. The piecewise cudagraph also has fine-grained memory management. … **This is why the attention operation in V1 has the output tensor as the input of the attention.**」

而且 V1 早就不是"只有 piecewise"了 —— `CUDAGraphMode` 现在有 5 档（官方文档）：

| 模式 | 含义 | 适用 |
| :--- | :--- | :--- |
| `NONE` | 关图 | 调试 |
| `PIECEWISE` | attention 留图外，其余进图（**需要 piecewise 编译**） | 通用、最灵活 |
| `FULL` | **对非均匀批（prefill / 混合）也录整图**，同 batch_size 的纯 decode 批复用这张图 | 官方注明「good for **small models or workloads with small prompts**」 |
| `FULL_DECODE_ONLY` | 只给纯 decode 录整图 | P/D 分离里的 decode 实例 |
| `FULL_AND_PIECEWISE` | **V1 默认**：纯 decode 走整图，prefill/混合走分段图 | 官方称通常最快，但最吃显存、捕获最慢 |

```python
# vLLM 的术语：max_query_len == 1 叫 uniform decode batch，反之（prefill 或混合）叫 non-uniform batch
# 运行时由 CUDA Graphs dispatcher 按批组成自动选模式
```

官方还明确两条边界：**只有 FA3 支持对非均匀批做 full cudagraph**（FlashInfer / FlashMLA / Mamba 等只支持纯 decode 的整图）；**cascade attention 不图安全**，用到它时一律降级到 PIECEWISE。

### 2.2 我们实测：这个"机器"值不值 5.77 ms？

C3 曾给出「混合步里的 decode 拿不到 graph 减免，D=32 时每步多花 **5.77 ms**」。C6/C6b 把这个数字拆开了，结论**方向相反**：

**C6b（二分定位）** —— 把 prefill/混合形状固定下来录图，第一次就失败：

```
cap_model（只捕获 28 层主干，含 FA varlen + triton store_kvcache）  PASS
cap_logits（bool 掩码索引）                                        FAIL  ← 「operation not permitted when stream is capturing」
cap_logits + 静态 gather 索引                                      PASS
full（整步，bool 掩码）                                            FAIL
full + 静态 gather 索引                                            PASS       ← 唯一阻塞被修掉，整步可捕获
```

阻塞点是 `nanovllm/layers/embed_head.py:60`：

```python
last_indices = context.cu_seqlens_q[1:] - 1
last_indices = last_indices[context.need_logits]      # ← bool 掩码索引
```

布尔掩码索引内部走 `nonzero`，要一次 device→host 的计数同步 → 捕获流里非法。换成**捕获外预计算、长度固定的 1D int64 gather 索引**（即 vLLM 的 `logits_indices` + pad 到桶）就过了。**注意这是用户自己那三项优化里「跳过采样」引入的算子**——它同时是混合步进图的唯一阻塞。

**C6（量化收益）** —— 10 个形状全部捕获成功，同形状、同 buffer 对比：

| 形状 | T | eager | graph | 省 | 倍数 |
| :--- | ---: | ---: | ---: | ---: | ---: |
| decode×1 | 1 | 24.63 ms | 5.85 ms | **18.78 ms** | **4.21×** |
| decode×8 | 8 | 27.43 | 9.30 | 18.12 | 2.95× |
| chunk32 | 32 | 25.81 | 5.56 | **20.25** | **4.64×** |
| decode×32 | 32 | 28.20 | 20.97 | 7.22 | 1.34× |
| chunk1024（无 prefix） | 1024 | 40.36 | 39.83 | 0.53 | 1.01× |
| chunk1024 + prefix cache | 1024 | 74.82 | 73.59 | 1.23 | 1.02× |
| **mix1024+16** | 1040 | 80.69 | 79.82 | **0.87** | 1.01× |
| **mix1024+32** | 1056 | 91.54 | 90.27 | **1.27** | 1.01× |
| **mix1024+64** | 1088 | 111.41 | 111.11 | 0.31 | 1.00× |
| chunk4096 + prefix | 4096 | 365.98 | 364.68 | 1.29 | 1.00× |

![C6](figs/c6_prefill_graph.png)

**怎么读这张表**：graph 省掉的是**固定开销 ≈ 19 ms/步**（kernel launch + CPU 侧，与 T 无关）；GPU 计算量随 T 线性增长。T ≤ 32 时计算填不满这 19 ms，所以能省 3~20 ms；**T ≈ 1024 时 GPU 单步要 40~365 ms，launch 早已被计算掩盖，只剩 1%**。真实混合步就是后者。

> **结论修正**：C3 的「多花 5.77 ms」是在 **T=32 的小 T 区间**测出来的（把 D 条 decode 单独喂进通用分支），把它外推到 T≈1050 的真实混合步上不成立。C6 直接测真实形状：**"混合步牺牲 CUDA graph" 的实际代价是 0.3~1.3 ms/步（1.00~1.02×）**。
>
> 也就是说：**vLLM 的 piecewise CUDA graph 不是为"长 prompt 的 chunk 混合步"服务的**（那种形状下它也只值 1%），它服务的是官方文档点名的另一类负载 —— **小模型 / 小 prompt / decode 主导**，即 T 小到 CPU launch 主导的区间。而在那个区间里，**nano-vllm 已经在吃这份收益了**（纯 decode 步本来就整批走 graph，见 C3 的 `t_d`）。
>
> 因此 §2 原来那句「vLLM 把代价压到几乎免费、我们这里要算账」应改为：**这件事我们这里本来就不算账 —— 混合步走 eager 只贵 1%。**

### 2.3 那么真正该抄的是什么

C6 顺带证明了一件与官方文档相反的事：**固定形状下，FA 2.8.3 的 varlen attention 是可以被录进图的**（`cap_model` PASS，含 `flash_attn_varlen_func` + `store_kvcache` triton kernel + prefix cache 的 `block_table` 路径）。官方说"attention 非图安全"，前提是它要在**同一张图里服务多种形状**（分桶复用）；我们只要**按桶各录一张**就行。

所以差距不在"能不能录图"，而在 **C3 表里剩下那 4.29 ms**：

```
t_d_generic（通用分支，D=32）  29.974 ms
t_d_eager  （decode 分支，D=32）25.687 ms      ← 差 4.287 ms = 74% 的"税"
t_d_graph  （decode 分支 + graph）24.207 ms     ← graph 只占 1.48 ms = 26%
```

**混合步里 decode 行的真正代价是"走了 varlen+整 cache 的 attention 路径"，不是"没走 graph"。** vLLM 用一个统一 attention 算子按 metadata 分组派发后端（prefill 组走 varlen/cascade、decode 组走 paged decode kernel）把这一项抹平；这才是我们该抄的那一条。具体做法见 `CUDA图与混合批_机制与优化路线.md` §4 Tier 1。

---

## 3. 总判定

| 维度 | vLLM 官方主张 | 我们实测 | 判定 |
| :--- | :--- | :--- | :--- |
| 吞吐 | V1 vs V0 最高 1.7×（**归因 CPU 开销**） | 0.958~0.990× | **不吻合，但不构成矛盾** —— 官方 1.7× 来自执行循环重建，我们只改调度器；且官方自陈吞吐最优是 budget > 8192（chunk 近乎关闭） |
| ITL / TBT 尾部 | chunked prefill 的核心卖点：消除数秒级 stall、P99 TBT 大降 | ITL max 1/20；停顿步 480→**5**，最长连续 40→**5** | **强吻合**（现象、量级、指标选择一致） |
| GPU 利用率 | 混合批提高利用率 | 95.9%（显存控制器 96.3%） | **吻合** |
| TTFT | 小 budget 牺牲 TTFT 换 ITL | 1.00~1.04×（几乎不变） | **吻合且可解释** —— 挪用比例仅 8/1024 = 0.78% |
| budget ⇄ ITL/TTFT 权衡 | 文档明写 budget 是总旋钮；budget == max_model_len ≈ V0 默认策略 | budget 效应 9.29~10.40×；同 budget 分支差异 1.13~1.43×；budget=16384 时退化 | **逐字吻合** |
| 步 P999 / 峰值显存 | **官方未主张** | 步 P999 V1 高 1.13~1.43×；显存逐位相同 | 不参与对照（SARATHI 的 uniform batch 是为 PP 消 bubble，单卡不适用；"步更长"是因为一步做的事更多，见 §1.5） |
| 混合批 × CUDA graph | V1 用 piecewise/FA3 让混合批不牺牲 graph；官方点名 FULL 模式适合**小模型/小 prompt** | 混合步整步 eager 的代价实测 **0.3~1.3 ms/步（1.00~1.02×）**；T≤32 区间 graph 值 1.34~4.64×，而那里我们已经走图 | **不吻合但无需吻合** —— piecewise 服务的是 launch 主导的小 T 区间，长 prompt 混合步里它只值 1%（§2） |
| **SLO 下服务容量** | SARATHI：2.6× / 3.7× / 5.6× | **未测** | ⚠ **唯一真实缺口**，见 §4 |

**一句话回答**：**机制层面吻合得相当好，吞吐数字不吻合且本来就不该吻合；而"混合批 × CUDA graph"这条我们曾经以为的差距（§2），实测下来只有 1%。**

我们复现的是 vLLM V1 的**调度器那一半**（统一调度 + decode 优先 + chunked prefill + 跨步维护增量 KV），它兑现的正是官方文档与 SARATHI 给 chunked prefill 派的那两件事 —— **尾延迟与 GPU 利用率**；而 vLLM V1 的 1.7× 是**执行循环那一半**（多进程 EngineCore、持久输入、torch.compile、piecewise CUDA graph、prefix caching）的功劳，那部分我们没做、在本仓也没有对应杠杆。其中 **piecewise CUDA graph 这一项经 C6 实测确认：在我们的负载形状下只值 1%，不构成差距**。

> **待办优先级**（由本轮结论推出）：① SLO 下服务容量（§4，唯一真实缺口）；② **统一 attention 按 query 长度分组派发**（C3 里 74% 的混合步 decode 代价在这里，见 `CUDA图与混合批_机制与优化路线.md` §4 Tier 1）；③ 静态 `logits_indices`（顺手的零成本改动）；④ ~~piecewise CUDA graph~~（已否决）。

---

## 4. 唯一真实的缺口：SLO 约束下的服务容量

SARATHI 的 2.6× / 3.7× / 5.6× **不是**"固定工作量下比墙钟"，而是**"给定 P99 TBT ≤ SLO，系统能扛多大请求速率"**（capacity under tail-latency constraints）。

而我们现有全部实验（C1 stress / C1 ctrl / C0 bench）都是**固定工作量**口径 —— C1 ctrl 下两版 prefill / decode token 逐位相同（差 −1 / −0），**结构上不可能测出容量差异**。所以这条不是"结论冲突"，而是**度量轴不同 + 该轴未测**。

要补它，需要一个新的实验形状（现无脚本）：

1. **泊松到达 + 固定速率**，扫 RPS（例如 2/4/6/8 req/s），每档跑够稳态时长；
2. 对每档记录 **P99 ITL（或 P99 TBT）**；
3. 画「RPS → P99 ITL」曲线，取 **P99 ITL ≤ SLO（如 200 ms / 500 ms）** 的最大 RPS；
4. 两版在同一 SLO 下的最大 RPS 之比 = 可以**直接对标 SARATHI 的 2.6×** 的那个数。

预期方向（基于现有数据的推断，尚未实证）：由于两版**吞吐几乎相同**（0.96~0.99×）而 V1 的 **ITL 尾部好 5~20×**，在**严格的 ITL SLO** 下 V1 的容量应显著更高；SLO 越松，两者越接近。**这也正是"这个优化是低并发 / 长 prompt / 流式交互的优化，不是高并发吞吐优化"这句话的定量版本。**

---

## 5. 引用来源

1. vLLM Team, *vLLM V1: A Major Upgrade to vLLM's Core Architecture*, 2025-01-27.
   https://blog.vllm.ai/2025/01/27/v1-alpha-release.html
2. vLLM Docs, *Optimization and Tuning*（Chunked Prefill / Preemption 两节）。
   https://docs.vllm.ai/en/latest/configuration/optimization.html
3. vLLM Docs, *torch.compile integration · Cudagraph Capture*（piecewise cudagraph / output-as-input / `cudagraph_capture_sizes`）。
   https://docs.vllm.ai/en/latest/design/torch_compile.html
4. vLLM Docs, *CUDA Graphs*（5 种 `CUDAGraphMode`、uniform/non-uniform batch、dispatcher、FA3 才支持 full cudagraph）。
   https://docs.vllm.ai/en/latest/design/cuda_graphs.html
5. Agrawal et al., *Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve*, OSDI 2024.
   https://www.usenix.org/conference/osdi24/presentation/agrawal
6. 本仓库数据：`experiments/vllm_ref_compare.py` → `results/vllm_ref_compare.md`
   （源：`results/b3_*.json`、`results/c1_*.json`、`results/c3_*.json`、`results/c5_*.json`）
7. 本仓库数据：`experiments/c6_prefill_graph.py` / `c6b_bisect.py` → `results/c6_prefill_graph.json`
   （固定形状的 prefill/混合步 eager vs graph；`c6_fig.py` → `figs/c6_prefill_graph.png`）
8. 本仓库文档：`experiments/CUDA图与混合批_机制与优化路线.md`（机制全文 + 抄作业清单）
