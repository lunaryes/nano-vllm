# nano-vllm 项目长期笔记

## 环境事实

### WSL 控制(2026-10-10 实测)
- WorkBuddy **可直接控制 WSL**,发行版 `Ubuntu-24.04`(WSL2),用户 `ll`。cwd 自动映射到 `/mnt/c/...`。
- **标准调用模板**(必须这样写,否则踩坑):
  ```bash
  MSYS_NO_PATHCONV=1 WSL_UTF8=1 wsl.exe -- bash -s <<'WSEOF'
  <script>
  WSEOF
  ```
- 两个坑:`MSYS_NO_PATHCONV=1` 防止 Git Bash 改写 Linux 绝对路径;脚本走 stdin heredoc 防止 `$变量` 被吞。`WSL_UTF8=1` 防 UTF-16 乱码。
- WSL 内 GPU 可见(RTX 4060 8G,驱动 617.14);后台进程跨调用存活,端口可从 Windows `localhost` 访问。
- `sudo` 需密码,需提权的操作做不了。

### 实验工具链(2026-10-13 实测)
- 仓库 `.venv` 由 **uv** 管理(**无 pip**,`python -m pip` 也报 No module named pip)。装包用 `uv pip install --python <venv>/bin/python ...`。
- 绘图用**独立 venv** `~/venvs/fig`,不与实验 venv 混装(避免动 numpy/torch)。**必须走国内镜像**:`uv pip install --python ~/venvs/fig/bin/python -i https://pypi.tuna.tsinghua.edu.cn/simple matplotlib`(官方源 9 分钟未完成,镜像 6 秒)。
- **WSL 内无 CJK 字体**(`fc-list :lang=zh` 为空):从 `/mnt/c/Windows/Fonts/simhei.ttf` 拷到 `~/.fonts/`,matplotlib 里显式加载 `font_manager.fontManager.addfont(p)` + `plt.rcParams["font.family"] = FontProperties(fname=p).get_name()`,并设 `axes.unicode_minus=False`。
- 仓库内实现在 `~/projects/nano-vllm`(V1/main 两分支);实验脚本在 `experiments/`,结果在 `experiments/results/`,图在 `experiments/figs/`。
- **构造合成批 / 录图的坑(2026-10-15 实测)**:
  - **块表必须落在 `config.num_kvcache_blocks` 内**。Qwen3-0.6B(28 层/8 KV heads/128 head_dim/fp16)**每块 29.4 MB**,`gpu_memory_utilization=0.8` 时只有 **148 块**;越界 → `store_kvcache` 写飞 → **illegal memory access**(首版脚本就这么挂的)。
  - **直接调 `runner.model(...)` 必须包 `torch.inference_mode()`**:`activation`/`rotary_embedding`/`layernorm` 挂了 `@torch.compile`,grad 模式下 AOT autograd 会在 `silu_and_mul` 的 inplace 上报错。
  - **CUDA sticky error 污染上下文**:捕获一失败,后续所有数字不可信 → **每个阶段独立进程**(循环写进 `.sh` 文件,别写在 `wsl.exe -lc` 命令行里,`$var` 会被吞)。
  - `import torch._dynamo` 写在函数里会把 `torch` 变成函数局部名 → `UnboundLocalError`;写 `import torch._dynamo as _dynamo`。

### 调度对比的硬事实(2026-10-13 实测)
- **main 的 resident 不受 max_num_seqs 约束**:`Scheduler.schedule()` 每步从零计数去 admit waiting,waiting 长期非空时 resident 会涨到**池满**(短 prompt 负载实测 main ~32 条 vs V1 8 条)。V1 的 running 段与 waiting 段共享同一计数上限 → resident 恒被压在 max_num_seqs。**V1 vs main 受控对比必须靠池控并发**(池 = 前 N 条负载块数之和 + 每条块数恒定)。
- **收益比 ≈ prompt / chunk**：budget ≥ prompt 时 V1 也一步吃完、与 main 无差别。要让 chunked prefill 生效，budget 必须 < prompt 长度。

### 指标归属的硬事实(2026-10-14 实测,详见 `experiments/抖动来源与实验效度.md` §2.6)
- **相位分离的代价记在 ITL 尾部,不记在 TTFT**:三场景 `main÷V1` 的 TTFT 全落在 **1.00~1.04×**(逐位相同),ITL max 达 **5~20×**。根因:**TTFT 窗口在首 token 处关闭** → 同一段 prefill 步,对已出 token 的 seq 记入 ITL,对未出 token 的记入 TTFT。指标归属取决于该 seq 首 token 是否已产出。
- **指标瓶颈分工**:TTFT ← prefill **吞吐**(chunked 几乎不动);ITL/TPOT 尾部 ← prefill 与 decode 的**交织方式**(chunked 的战场)。**只盯 TTFT 会完全看不到该优化的价值。**
- **实验口径的两条通则（2026-10-14 B4 实测，详见 `experiments/实验设计工作流.md` §7）**：① **能被设定的量才能当条件，观测出来的量只能当判据** ——「抢占次数差不多」是判据，「池 / `max_num_seqs` 多大」才是条件；两版同受**池**约束时抢占**自然都是 0**（实测 `prefill_tok`/`decode_tok` 逐位相同，并发 7.4~7.5 对称）。② **判断 patch 是否掩盖效应，去数它前面的量** —— V1 的 `need_logits` 跳过在 `ParallelLMHead.forward`（lm_head **之前**），patch 打在 `Sampler.forward`（**之后**），两者不相交；实测 lm_head 行数在 argmax/normal 下**逐位相同**（V1 15491 / main 15875），采样模式只影响墙钟 0.6~0.9%。`main − V1 = 384 行` 无用 lm_head ≈ 27 ms（占 75 s 的 0.04%）。

### 实验设计门禁(2026-10-14 复盘 B 组设计失败后沉淀,详见 `experiments/实验设计工作流.md`)
- **三种结局必须可区分**:① 机制**未激活** / ② 激活但**无效** / ③ 激活**有效**。**不能排除 ① 的判据设计就是不可用的**。旧 B 组把 ① 当 ② 报了「无收益」。
- **四道门(全部在放量前完成纸面推演)**:
  1. **激活门** —— 触发条件写成不等式(chunked 生效 ⟺ `budget < prompt`;旧设置 16384 > 4700 恒假)。指定可观测量(chunk 数/混合步数)作活证据。
  2. **对称门** —— **控制变量 = 控制行为,不是控制参数**。同名参数语义必审计(`max_num_seqs` 在 main 不约束 resident;`ModelRunner.run()` 第 2 参数两版语义相反)。不一致 → 换成两版语义相同的量(**池**)。
  3. **容量门** —— 用**峰值**不用均值:`floor(池/块数_peak) × (块数_peak + 1) ≤ 池`;跑后必须实测抢占 = 0。
  4. **判据门** —— 先写预期失效模式再选判据;**阈值先在数据分布上标定**。**尾部量与频率量必须并列报告**。
- **采集三纪律**:① 冒烟先于放量(只看结构量);② 噪声先于效应(**跨会话**测漂移,同进程重复是假精度);③ 归因先于结论(与外部参考分歧先查**基线差异**)。
- **元教训**：知道陷阱 ≠ 流程里有门禁。方案 v2 §B1/§B4 已写下坑,没强制 → 照样踩。

### 压测层指标与 C 组结论(2026-10-15 实测)

- **本机指标可得性(RTX 4060 单卡 8G)**:只有 `nvidia-smi`(**无 ncu/nvtop/dcgmi**)。`utilization.gpu` = "有 kernel 在执行的时间占比"(**非 SM occupancy**);`utilization.memory` = 显存控制器忙碌度(**非 GB/s**)。**NVLink 不存在**(`nvidia-smi nvlink -s` → "Device does not have or support Nvlink");**PCIe 跨卡带宽不适用**(单卡)。`nvidia-smi ... -lms 500` **流式采样可用**(单进程每 500ms 一行),别反复 spawn(单次 69ms)。
- **「256 并发」是误读**:池仅 41728 token → 256 条提交后 running 峰值 65(stress)/40(ctrl),其余排队。准确写法「**提交 256 条**」。
- **配置效应 vs 调度效应(C5 budget 扫描,ctrl + 定长 4864 + 池 163 + n=32)**:

| budget | V1 步 p999 | main 步 p999 | V1 峰值 MB | main 峰值 MB |
| ---: | ---: | ---: | ---: | ---: |
| 1024 | 102.3 | 71.6 | 5766.2 | 5766.2 |
| 2048 | 161.8 | 143.1 | 5794.3 | 5794.3 |
| 4096 | 276.9 | 215.9 | 5850.3 | 5850.3 |
| 16384 | 950.1 | 744.6 | 6186.5 | 6128.7 |

  - **步延迟 P999 的主因是 budget**:16384→1024 两版都降约 **90%**(main 744.6→71.6ms)。外部参考的「−36.1%」就是这条杠杆,只是 nano-vllm 里拧它的是 `max_num_batched_tokens`,不是调度。
  - **峰值显存的主因也是 budget**:16384→1024,main 省 6.3%、V1 省 7.3%。**同 budget 下两版逐位相同(+0.00%)** → **显存是配置效应,不是调度效应**(KV 池按 `gpu_memory_utilization` 启动时静态分配 5738MB,与调度无关;16384 档 V1 反高 0.94%)。
  - **同 budget 下 V1 的步 P999 恒比 main 高 13~43%**(混合步叠加 decode)。
- **CUDA graph 与混合步的代价(C3,纯 decode,dlen 2048)**:`eager÷graph` = **3.15×(D=1)/2.17×(8)/1.48×(16)/1.06×(32)/0.96×(64)** → graph 收益是固定开销摊销,**D ≥ 32 后消失**。混合步里 decode 走通用分支(varlen+paged KV),比 graph 慢 **+5.8ms/步**(D=32)→ 受控口径 190 混合步 ≈ 1.1s ≈ 3.2%。与 C2 的 **D ≤ 16** 窗口一致。
- **C1 双口径压测结论**:**stress**(官方 256 提交,抢占 166/172)两版持平,V1÷main 墙钟 **1.010**(交替配对 1.013/1.020/1.001/1.007);**ctrl**(定长,抢占 0)**V1 慢 4.4%**,且 prefill/decode token **逐位相同**(221442/221437、40454/40453)→ 差异纯来自调度形态。
- **ITL p99 与 p999 交叉**:stress 下 `main÷V1` 在 p99 = 0.950(main 好)、p999 = **1.192**(V1 好 19%)。→ **单报 p99 或单报 max 都会误导**。
- **受控口径锁抢占的代码条件**:`schedule()` 只在 decode 分支 `can_append` 失败时 `preempt()`;**每条 seq 块数恒定 → `can_append` 恒真 → 永不抢占**(实现 = 定长负载 O~U[64,255]、P = 总长 − O)。
- **坑**:`--fixed-total` > `--max-model-len` → CUDA graph 的 block_table 宽度按 `max_model_len//256` 固定 → `expand` 报错(`[8,16]` vs `[8,19]`)。定长负载必须同时放大 `max_model_len`。

## 与 vLLM 官方口径的关系(2026-10-15 查证官方原文)
- **vLLM 的「收益」是三份不同主张**:① **V1 alpha 博客的「最高 1.7× 吞吐(V1 vs V0)」原文归因是「comprehensive CPU overhead reductions」**,并明说 V0/V1 的 kernel almost identical → 1.7× 来自**执行循环重建**(EngineCore 多进程/持久输入/torch.compile/prefix caching/API 重叠),**不是 chunked prefill**;博客讲统一调度器那节**一个性能数字都没有**,也没给 ITL/P999/显存数字。② **官方文档给 chunked prefill 只派了两条收益:改善 ITL、提高 GPU 利用率**。③ **SARATHI(OSDI'24)** 主张是**尾延迟**(消除数秒级 stall、降 P99 TBT,明确 prioritizing TBT over TTFT,吞吐 same or better)。
- **官方文档自己写明退化边界**:「If `max_num_batched_tokens` is the same as `max_model_len`, that's almost the equivalent to the V0 default scheduling policy」+「For optimal throughput we recommend `max_num_batched_tokens` > 8192」→ **吞吐最优即 chunk 近乎关闭的区间**。我们 B-base 无收益、C5 b=16384 退化是这条的实测复现。
- **吻合度判定**:吞吐**不吻合但不矛盾**(我们 stress 0.990×/ctrl 0.958×,杠杆在不同层);**ITL 尾部强吻合**(停顿步 main 480→V1 5、ITL max 差 20.11×);GPU 利用率吻合(95.9%);**TTFT 吻合且可算**(1.00~1.04×:8 条 decode 只吃 8/1024=0.78% budget → 仅 −0.39%);**budget 是总旋钮**(效应 9.29~10.40× ≫ 同 budget 分支差 1.13~1.43×)。
- **官方有我们没有的机器**:vLLM V1 靠 **piecewise CUDA graph + FlashAttention 3** 使**混合批不必牺牲 graph 加速**;机制 = ①统一扁平批 ②按 num_tokens 分桶 padding(`cudagraph_capture_sizes`)③**在 attention 边界切图**(图外 eager 跑 attention、图内跑 attention 之间 token-wise 的部分)④attention 出参化 + `cudagraph_unsafe` 标记;配套 5 档 `CUDAGraphMode`(默认 `FULL_AND_PIECEWISE`;`FULL` 对非均匀批也录整图,官方注明适合**小模型/小 prompt**)+ 运行时 dispatcher;**只有 FA3 支持对非均匀批做整图**,cascade attention 不图安全。
- **⚠ C6 修正(2026-10-15,详见 `experiments/CUDA图与混合批_机制与优化路线.md`)**:旧说法「混批 decode 每步多花 +24.17ms(D=1)/+5.77(D=32)」是**在 T=32 小 T 区间**测的(把 D 条 decode 单独喂进通用分支),**外推到真实混合步不成立**。C6 同形状同 buffer 实测:graph 省的是**固定开销 ≈19 ms/步**(与 T 无关),T=1/8/32 时值 4.21×/2.95×/1.34×(**这个区间我们已经吃到**),而 **T≥1024 时只剩 1.00~1.02%(mix1024+32 仅 1.27 ms/步)** → **piecewise 不值得做,已否决**;"步 P999 V1 高 13~43%"也**不能**归因给 graph,那是"一步做的事更多"。
- **C6/C6b 两条硬结论**:① **固定形状下整步是可以被 CUDA graph 捕获的**(FA2.8.3 varlen + block_table + triton store_kvcache 全过);② **唯一阻塞**是 `nanovllm/layers/embed_head.py:60` 的 **bool 掩码索引**(`nonzero` 要一次 D→H 计数同步,捕获流里报 `operation not permitted when stream is capturing`),换成**捕获外预计算的固定长度 1D int64 gather 索引**(= vLLM `logits_indices` + pad)即通过,与 attention/`torch.compile` 均无关。
- **该抄的不是 piecewise,是"统一 attention 按 query 长度分组派发后端"**:C3 里 `t_d_generic − t_d_eager = 4.287 ms(D=32) = 混合步这笔代价的 74%`,graph 只占 1.48 ms(26%)→ 把 decode 行从 varlen+整 cache 换成 paged decode kernel,预期步长 −4.7%。
- **唯一真实缺口 = SLO 约束下的服务容量**:SARATHI 的 2.6×/3.7×/5.6× 是「给定 P99 TBT ≤ SLO 能扛多大速率」,而 C1/C0 全是**固定工作量**口径(ctrl 两版 token 逐位相同)→ **度量轴不同,非结论冲突**;补法是泊松到达 + 扫 RPS 取 SLO 下最大 RPS。
