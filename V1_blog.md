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

> 此处保留一个问题，在长序列存在的情况，可能出现阻塞后续请求时间过长的情况，在vllm中的实现里添加了`long_prefill_token_threshold`字段作为分块限制。它的机制是这样的：请求数为1时不启用；请求数小于阈值时随着请求数自适应；请求数大于阈值时，截断为定值。

## 先对最关键的scheduler.schedule()进行考虑

**对于它的输入：running和waiting队列**
在当前考量下，它们的语义应该发生变化，running应该表示正在进行prefill和decode的序列，waiting表示还没进行任何操作的序列。

实现running优先的混合调度，running里一定是decode再到chunked prefill。

其他内存分配、计算被调度token等操作与原来相似。

**函数返回值**
原来会返回一个布尔值表示这个batch是需要decode还是prefill，现在这个混合调度不再需要这个标志。
在step中应用一个标志is_uniform标志这是混合批还是decode only批次。

## modelrunner
is_uniform标志是isprefill的语义反转。
**prepare部分**
混合批复用prepare_prefill,decode_only批复用prepare_deocde。
decode_only使用graph，混合批使用eager。
其他部分注意语义的反转即可。

## lm_head相关优化
进行混批后，context的is_prefill发生语义变化，变为not is_uniform.
在现在chunked prefil的语境下，不是每个prefill序列都需要输出logits，为了减少无用计算，在modelRunner模块的prepare_*阶段引入need_logits字段来判断batch中的每个字段是否需要进行logits计算、采样、后处理阶段写入采样token。