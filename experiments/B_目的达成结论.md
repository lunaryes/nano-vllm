# B 组：目的达成结论

- 日期：2026-10-13　分支：V1 / main　设备：RTX 4060 8G　模型：Qwen3-0.6B
- 脚本：`b_smooth.py`（采集）/ `b_compare.py`（比对）/ `b_bench.py`（标准吞吐采集）/ `b_bench_compare.py`（吞吐比对）
- 数据：`results/b_{V1,main}_{base,s2,s3}.json`、`results/b_compare_{base,s2,s3}.md`
- 补跑（10-13）：s3 各重复 2 次（`b_{V1,main}_s3_r{2,3}.json`）；标准 benchmark —— `b_{V1,main}_bench.json`（3 次重复）、`b_{V1,main}_bench_raw.json`（步级明细）、`b_{V1,main}_bench_p{1,2,3}.json`（交替配对）、`b_compare_bench.md`

## 结论速览

| 目的 | 判据 | base（budget 16384） | s2（同 budget，O 缩小） | s3（budget 2048） |
| :--- | :--- | :--- | :--- | :--- |
| 1 ITL 平滑 | CV 与 p99/p50 低于 main | **未达成**（CV 1.607 vs 1.589） | 持平（1.691 vs 1.699） | **达成**（CV 0.903 vs 2.241；p99/p50 6.17 vs 12.50） |
| 1' 空洞 | ITL > 3×p50 占比低于 main | 持平（4.11% vs 4.26%） | 持平（10.52% vs 10.61%） | **未达成**（**10.94% vs 4.82%，更差**） |
| 2 TTFT | 增幅 ≤ 10% | 达成（**+0.4%**） | 达成（**+2.0%**） | 达成（**+0.8%**） |
| 3 利用率 | 混合步 > 0、每步 token 更多 | 达成（87 vs 0；239.9 vs 230.7） | 达成（85 vs 0；624.8 vs 567.1） | 达成（276 vs 0；235.7 vs 210.4） |
| 旁证 吞吐（受控） | 输出吞吐 ≥ main | 未达成（0.99×） | 未达成（0.98×） | 持平（1.00×） |
| 旁证 吞吐（官方） | 标准 benchmark | 见 §8：**无稳定差异**（配对 V1/main = 0.991，跨会话漂移 3~6% > 效应量） | 同 | 同 |

**总判定：部分达成。** TTFT 与利用率达成；**ITL 平滑只在 chunk 小（budget=2048）时达成，且达成方式是"削掉极端毛刺"，不是"消除空洞"——空洞反而更多。** 三个场景抢占均为 0。

## 1. 前提：抢占必须为 0（踩坑记录）

方案 §B2 的池估算用的是**均值**（13 块/条 = (3072+160)/256），漏了**峰值**（P=4096 + O=256 → 4352 → 17 块/条）。按原负载（P~U[2048,4096]）实测：

| 负载 | V1 抢占 | main 抢占 |
| :--- | ---: | ---: |
| P~U[2048,4096]，默认 max_num_seqs | 2 | 4 |

根因在 main 的调度：`schedule()` 的 prefill 段**按「prompt 块数」准入、不预留后续增长**——每条 seq 的 decode 每跨一次 256 边界都要追加一块，于是池子必然被推到接近满，任何一条要新块时就 `can_append` 失败 → 抢占。

**修正：让 prompt 块数固定在 19**（P ~ U[4609,4864]），此时 `floor(163/19) = 8`，而每条 seq 全程只占 **20** 块（19 + decode 增长 1），`8 × 20 = 160 ≤ 163`。配 `max_num_seqs=8`，两版并发都是 **8**、完全对称，抢占 = 0。一般条件：

> `floor(池 / prompt块数) × (prompt块数 + 1) ≤ 池`

## 2. base（budget=16384）：默认配置下没有收益

| | V1 | main |
| :--- | ---: | ---: |
| ITL CV | 1.6069 | **1.5885** |
| ITL p99/p50 | 12.18 | **11.95** |
| ITL max (ms) | 990 | **1344** |
| 空洞占比 | 4.11% | 4.26% |
| 墙钟 (s) | 72.84 | 72.22 |

V1 在 CV 与 p99/p50 上**略差**。机制在步级数据里一目了然（`raw.step_dt_ms`）：

| | V1 | main |
| :--- | ---: | ---: |
| 纯 decode 步 | n=1869，mean **23.79** ms | n=1944，mean **23.90** ms |
| 承载 prefill 的步 | 混合 n=87，mean **315.89** ms | 纯 prefill n=91，mean **283.02** ms |
| 解码 seq 感受的空洞 | **315.89** ms | 283.02 + 23.90 = **306.92** ms |

**V1 的混合步（315.9ms）≈ main 的「纯 prefill 步 + 后续 decode 步」（306.9ms），还高 3%。** 因为 budget=16384 时 V1 把**整个剩余预算（≈4700 token，等于整条 prompt）**塞进同一步，这一步的长度和 main 的纯 prefill 步基本一样，只是额外加了 decode 的时间。

→ **V1 只是把停顿"搬进"了同一步，没有消除它。** 这是本次 B 组最关键的一条机制。

## 3. s2（同 budget、提高 prefill 占比）：负面对照

把 O 缩到 [32,96]，prefill 步占比升高。结果两版**同步恶化**、且仍然持平：

| | V1 | main |
| :--- | ---: | ---: |
| ITL p90 (ms) | 293.25 | 285.87 |
| ITL p99 (ms) | 311.12 | 315.71 |
| ITL max (ms) | 999.02 | 1352.04 |
| ITL CV | 1.6905 | 1.6992 |
| 空洞占比 | 10.52% | 10.61% |

p90 从 base 的 ~26ms 跳到 ~290ms（两版都跳），说明**恶化由 budget 决定、与调度无关**。

→ **驱动力不是 prefill/decode 比，而是每一个 prefill 步承载的 chunk 大小。** s2 的意义就是排除这个备择解释。

## 4. s3（budget=2048）：收益出现，但形态是"换尾"

| | V1 | main | |
| :--- | ---: | ---: | :--- |
| ITL CV | **0.9028** | 2.2406 | V1 好 60% |
| ITL p99/p50 | **6.17** | 12.50 | V1 好 51% |
| ITL p99 (ms) | **154.6** | 311.6 | |
| ITL max (ms) | **162.1** | 2285.2 | V1 好 93% |
| 空洞占比（>3×p50） | 10.94% | **4.82%** | **V1 更差** |
| 墙钟 (s) | 73.21 | 73.20 | 完全一致 |

步级构成：

| | V1 | main |
| :--- | ---: | ---: |
| 纯 decode 步 | n=1713，mean 23.68 ms | n=1944，mean 23.84 ms |
| 承载 prefill 的步 | 混合 n=**276**，mean **117.04** ms | 纯 prefill n=**288**，mean **93.21** ms |

**per 3-chunk 窗口（= 预填满一条 prompt 所需的 3 个 2048 块）：**

- **V1**：3 个混合步，各 117ms，每步都产出 1 个 token → ITL = `(117, 117, 117)`
- **main**：3 个纯 prefill 步（各 93ms，**不产出 token**）+ 1 个 decode 步（24ms）→ ITL = `(303, 24, 24)`

两者总时长相同（都约 351ms / 3 token，与墙钟一致），**差别只在分布形状**。而 main 的突发会更大：多条 seq 同时完成时集中释放块，main 会连续做 ~24 个 prefill 步 → 单次空洞 **2285ms**（≈ 24 × 92ms）。

→ **V1 的持仓特征：最坏 ITL ≈ `T_prefill(chunk) + T_decode`；main 的最坏 ITL ≈ `T_prefill(积压的整条 prompt) + T_decode`。** 所以收益比 ≈ `chunk / prompt`。budget=2048 / prompt≈4700 → 约 2.3×，实测把 max 从 2285 压到 162（14×，因为 main 的突发可超过单条 prompt）。budget=16384 ≥ prompt → 比值 ≈ 1 → **无收益**（base 的结论）。

**代价**：V1 每个混合步都延后一次 token，于是抖动**频率**从 main 的 4.8% 升到 10.9%，且 p90 从 26ms 升到 78ms。**用户感受到的"平均流畅度"变差，最坏情况变好。**

**三次重复（run-to-run 对照）**：

| 运行 | V1 CV | V1 p99(ms) | V1 max(ms) | V1 墙钟(s) | main CV | main p99(ms) | main max(ms) | main 墙钟(s) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| #1 | 0.903 | 154.6 | 162.1 | 73.21 | 2.241 | 311.6 | 2285.2 | 73.19 |
| #2 | 0.912 | 152.0 | 160.0 | 71.65 | 2.246 | 307.6 | 2217.2 | 71.56 |
| #3 | 0.916 | 152.2 | 166.3 | 71.52 | 2.247 | 306.5 | 2222.7 | 71.48 |

三次运行的步数**完全一致**（V1 1992 = P3/D1713/M276；main 2232 = P288/D1944/M0），抢占均为 0。**V1 的 max 稳定在 160~166ms，main 稳定在 2217~2285ms** —— 两版约 14× 的差距远超各自重复散布（V1 ±4ms、main ±34ms）→ §4 结论可复现。

## 5. TTFT / 利用率 / 吞吐

- **TTFT**：三场景 V1 相对 main 的 p50 增幅 +0.4% / +2.0% / +0.8%，p99 增幅 +0.8% / +1.7% / +1.5% —— 全部 ≪ 10% 阈值 → **目的 2 达成**。（绝对值很大（p50 ≈ 21~35s）是"waiting 长期非空"的设计效果：resident 仅 8，96 条排队，TTFT 由排队主导。两版排队机制相同，故对比有效。）
- **利用率**：混合步 V1 = 87 / 85 / 276，main **恒 0**；每步平均 token 数 V1 更高（239.9 vs 230.7；624.8 vs 567.1；235.7 vs 210.4）；prefill 预算填充率 V1 略高（0.316 vs 0.305；0.799 vs 0.771）→ **目的 3 达成**。
- **吞吐旁证**：输出吞吐 0.99× / 0.98× / 1.00×，墙钟 1.01× / 1.02× / 1.00× → **没有吞吐收益**，与 main 持平。混合调度不省算力，只重排时间（官方口径的标准 benchmark 另见 §8。）

## 6. 判据修正：空洞计数在本实验里失效

§B3 定的「空洞 = ITL > 3×p50」在 s3 下给出了**与尾部结论相反**的答案。原因：

- 阈值 3×p50 ≈ 75ms，恰好落在 V1 混合步（78ms）的**刚过**位置；
- main 的停顿全部**远超**阈值（≥300ms，占比 4.82%），数量少但每个都很大；
- 所以该指标衡量的是**抖动频率**，不是**最坏情况**。V1 恰恰是"用频率换幅度"。

→ **正确做法：尾部量（CV、p99/p50、max）与频率量（空洞占比）必须并列报告，不可互相替代。** 只报其中一个都会得出片面结论。这也解释了为什么 §B3 把"空洞"写进判据时预设"V1 → 0"是错的：V1 的设计目标是**不让 decode 停摆**，不是**不让单步变长**。

## 7. 与 C 组的接口

base 无收益的机制（混合步 ≈ 纯 prefill 步 + decode 步）正是 C2 要量化的对象：V1 的收益 = `t_p + t_d − t_mix`。本次给出的经验关系是

> V1 最坏 ITL ≈ `T_prefill(chunk) + T_decode`，main 最坏 ITL ≈ `T_prefill(积压) + T_decode`

即 **B 组的收益完全由"chunk 能塞多满"决定**，而 chunk 大小由 budget 控制——这直接指向 C6（budget 敏感性）与 §C2 的 H1。

## 8. 补充：标准 benchmark（官方吞吐口径）

口径 = 仓库根 `bench.py`（两分支**逐字节相同**）：num_seqs=256、prompt_len~U[100,1024]、output_len~U[100,1024]、max_model_len=4096、budget=16384、max_num_seqs 不设上限、temperature=0.6。

与 B1/B2/B3 的受控负载（preempt=0）相反，此负载下 KV 池严重超订：**重算量占 prefill 的 1/3**（V1 71,743 / main 67,579 token），抢占 166 / 172 次。端到端时间由「排队 + 抢占重算」主导。

**吞吐：无稳定差异。**

| 批 | V1 wall(s) | main wall(s) | V1/main |
| :--- | ---: | ---: | ---: |
| 同进程 3 次重复 | 98.61（std 0.47） | 95.43（std 0.70） | 0.968 |
| 单次 raw | 95.93 | 100.42 | 1.047 |
| 交替配对 ×3 | 98.85 / 98.85 / 98.93 | 100.31 / 100.66 / 98.59 | **0.986 / 0.982 / 1.003（均值 0.991）** |

- **会话内极稳、会话间漂移大**：同一进程内 3 次重复的 std 仅 0.5%，但跨会话漂移 V1 2.7% / main 5.2%，前两批方向**完全相反**。
- **V1 的墙钟比 main 稳得多**：V1 配对三次仅 98.85~98.94s（0.1%），main 95.4~100.7s。而步数、抢占数、token 数逐次完全相同 → 差异纯属环境（GPU 时钟/占用），不是调度。
- 设计最严谨的**交替配对批**（相邻运行抵消漂移）给出 V1/main = **0.991**，即 V1 略快不到 1%，与噪声同量级。

**结构层面确定的三个力，方向相反、量级相当（各约 ±1.5%）：**

| 力 | 方向 | 量 |
| :--- | :--- | :--- |
| 混合步被迫整批回退 eager（V1 独有） | 拖慢 V1 | **7,854 个 decode token（占全部 decode 5.88%）在 eager 下执行**；main 为 0 |
| 步总数更少 | 加快 V1 | 3,996 vs 4,195，**少 199 步**（省 CPU/发射开销） |
| 抢占重算更多 | 拖慢 V1 | 71,743 vs 67,579，**多 4,164 prefill token**（+1.2% 总工作量） |

步级构成（结构量，逐次运行完全一致）：

| 分支 | 步总数 | 纯 P | 纯 D | 混合 | 混合步承载 |
| :--- | ---: | ---: | ---: | ---: | :--- |
| V1 | 3,996 | 1 | 3,793 | **202** | 198,186 prefill（92.4%）+ 7,854 decode（5.88%） |
| main | 4,195 | 204 | 3,991 | **0** | — |

→ **混合调度在官方压力口径下既不构成可测量的吞吐代价，也不带来吞吐收益。** 它的收益在 ITL 尾部（§4），不在吞吐。

**口径警告**：本实验的 KV 池（163 块 = 41,728 token）对 256 条 seq 严重超订，测的是**真实压力下的标准吞吐**，不是「无抢占稳态吞吐」；且会话间漂移（3~6%）大于效应量——要给出吞吐代价的确切数值需锁频或改用受控负载。

## 附：判据达成度的另一面

| 指标 | 方向 | 说明 |
| :--- | :--- | :--- |
| 最坏 ITL / CV / p99 | V1 **更好** | 只在 chunk 小的时候 |
| 抖动频率 / p90 | V1 **更差** | 混合步把 decode 拖慢 |
| TTFT | 基本无代价 | 三场景 ≤ +2% |
| 利用率 | V1 **更好** | 混合步 > 0 |
| 吞吐（受控负载） | 持平 | 不省算力 |
| 吞吐（官方 benchmark） | 持平 | 无稳定差异；eager 惩罚 / 少 199 步 / 多重算互相抵消 |

## 复现

```bash
export PYTHONPATH=$PWD
# V1 侧
python experiments/b_smooth.py --tag V1 --scenario base --max-num-seqs 8 --out experiments/results/b_V1_base.json
python experiments/b_smooth.py --tag V1 --scenario s2   --max-num-seqs 8 --out experiments/results/b_V1_s2.json
python experiments/b_smooth.py --tag V1 --scenario s3   --max-num-seqs 8 --out experiments/results/b_V1_s3.json
# main 侧（git checkout main 后；切分支前需 git stash push -- .workbuddy/memory/2026-10-10.md .workbuddy/memory/MEMORY.md）
python experiments/b_smooth.py --tag main --scenario base --max-num-seqs 8 --out experiments/results/b_main_base.json
python experiments/b_smooth.py --tag main --scenario s2   --max-num-seqs 8 --out experiments/results/b_main_s2.json
python experiments/b_smooth.py --tag main --scenario s3   --max-num-seqs 8 --out experiments/results/b_main_s3.json
# 比对
python experiments/b_compare.py experiments/results/b_V1_base.json experiments/results/b_main_base.json --scenario base --out experiments/results/b_compare_base.md
python experiments/b_compare.py experiments/results/b_V1_s2.json   experiments/results/b_main_s2.json   --scenario s2   --out experiments/results/b_compare_s2.md
python experiments/b_compare.py experiments/results/b_V1_s3.json   experiments/results/b_main_s3.json   --scenario s3   --out experiments/results/b_compare_s3.md

# s3 重复跑（各再加 2 次）
python experiments/b_smooth.py --tag V1   --scenario s3 --max-num-seqs 8 --out experiments/results/b_V1_s3_r2.json
python experiments/b_smooth.py --tag V1   --scenario s3 --max-num-seqs 8 --out experiments/results/b_V1_s3_r3.json
python experiments/b_smooth.py --tag main --scenario s3 --max-num-seqs 8 --out experiments/results/b_main_s3_r2.json
python experiments/b_smooth.py --tag main --scenario s3 --max-num-seqs 8 --out experiments/results/b_main_s3_r3.json

# 标准 benchmark（官方口径）
python experiments/b_bench.py --tag V1   --repeat 3 --out experiments/results/b_V1_bench.json
python experiments/b_bench.py --tag main --repeat 3 --out experiments/results/b_main_bench.json
python experiments/b_bench_compare.py --out experiments/results/b_compare_bench.md
```

## 未做

- ITL 的绝对值受**step 分辨率**限制（同一 step 内产出的 token 共享同一时刻）。b_smooth 记录的是"发射时刻 - 上一发射时刻"，粒度 = 步长。若要亚步粒度需在 `model_runner` 内逐 token 打点，本次未做。
- （已补）**重复跑**：s3 各补 2 次，见 §4；**标准 benchmark** 见 §8。
- **剩余**：benchmark 的会话间漂移（3~6%）大于 V1/main 的效应量，故「吞吐代价」的符号不可判；要定论需锁频/更长窗口，或改用受控负载在更大并发下测稳态吞吐。
