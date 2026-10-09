## 为什么改
在vllm-V0中采取的方案：

- **prefill 优先**。新请求来了先跑 prefill 批次，跑完 prefill 得到 KV cache，请求进入 decode 队列。

- 底层kernel对于一个batch只能跑一种任务，要么是prefill，要么是decode

优点：TTFT 好

缺点：

1. decode 被打断，TPOT 抖动;
2. GPU 利用率一般;（如果某步只有 3 个 decode 请求（3 token），剩下 2045 预算全浪费。）
3. prefill和decode逻辑分离，无法统一拓展。
4. 公平性：一个 8192 token 的长 prompt，如果不分块，它一步占满所有预算，其他请求干等。分块后，每一步让多个请求轮流推进。

## 改的是什么
**为什么prefill和decode要分离？**
考虑prefill和decode，其实它们做的本质都是同样一件事：
计算已缓存块数 -> 计算本轮被调度token数
区别在于已缓存块数和本轮被调度token数的数量区别。
如果能把它们合并起来，混合地调度它们，在保持TTFT的前提下，使TPOT变得稳定与平滑，也能够提高gpu的利用率。
此时问题就变成了在混合调度的策略下，每一步给每个序列的调度token数应该如何计算？
如果希望TPOT变得平滑，那么我们就需要在每一步连续地进行decode；保持TTFT希望每一步公平地进行prefill；提高gpu利用率希望我们每一步尽可能填满预算。
考量之下，按顺序去处理请求即FCFS是比较好的选择，这样可以在平滑地处理先到来地序列的情况下不浪费gpu的利用率。

## 先对最关键的scheduler.schedule()进行考虑

**对于它的输入：running和waiting队列**
在当前考量下，它们的语义应该发生变化，running应该表示正在进行prefill和decode的序列，waiting表示还没进行任何操作的序列。

实现running优先的混合调度，running里一定是decode再到chunked prefill。

其他内存分配、计算被调度token等操作与原来相似。

## 底层算子需要支持prefill和decode的混合操作