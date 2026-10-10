# nano-vllm 项目长期笔记

## 教学模式（用户明确要求）
- 以 `C:\Users\20479\Desktop\ai policy.txt` 为准则：助教身份，讲解 + 引导提问 + review 学生代码；不写完整实现、不补 TODO、单次示例代码 ≤5 行。
- 用户是新学生：CS336 部分基础、PyTorch 不熟、无 CUDA 经验、单卡 RTX 4060。目标是理解推理框架并做魔改。
- 讲解优先级：先调度/内存管理（scheduler、block_manager），后计算（model_runner、layers）；涉及 PyTorch/CUDA 时先补基础概念。

## 项目关键结构
- `nanovllm/llm.py` 仅是 `LLMEngine` 空壳；真正的入口是 `engine/llm_engine.py::generate/step`。
- 跨层传参靠全局变量：`utils/context.py` 的 `set_context/get_context`，model_runner 写、layers/attention 读。这是魔改时最容易踩的地方。
- KV cache 块大小默认 256，块数与显存由 `model_runner.allocate_kv_cache` 依据 gpu_memory_utilization 反推。
- 前缀缓存用 xxhash（block_manager.compute_hash，链式 hash）+ 引用计数。
- 依赖方向：engine → models → layers → utils；layers 不反向依赖 engine。

## 改造目标（vLLM V1 混合调度 + chunked prefill）
- **定案方向：两条路共存 + 按批分发**，不是"统一 varlen 一条路"。
  - 纯 decode 批（所有 seq `num_scheduled_tokens == 1`，即 max_query_len==1）→ `flash_attn_with_kvcache` + full CUDA graph（保持原样）。
  - 含 prefill / 混合的批 → 统一 `flash_attn_varlen_func`（attention eager）。
- 理由：`with_kvcache` 的元数据（slot_mapping / context_lens / block_tables）全在 `graph_vars` 里，可重放；varlen 额外需要 `cu_seqlens`，它不是静态 buffer → 上不了 graph。统一 varlen = 连带丢掉 decode 的 cudagraph。
- 参考实现 `anjoj0/nano-vllm-v1` 选择了"彻底统一"，代价就是丢 cudagraph（且其 capture 未设 cu_seqlens，graph 分支存疑）。
