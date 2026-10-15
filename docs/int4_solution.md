# nano-vllm INT4 量化改造方案 (方案 B · 真·INT4 计算)

> **目标**：在保留 nano-vllm 极简哲学（<2000 行）的前提下，端到端实现一套 **W4A16 + KV4** 的 INT4 推理流水线，并补齐 YaRN 长上下文支持。本文档是工程**设计方案**，不包含代码；具体实现请按本文第六节的步骤逐项推进。
>
> **核心承诺**：
> - 权重走 INT4：量化为 INT4 per-group（group_size=128，非对称）后存盘；推理时通过 Triton kernel 融合反量化 + GEMM；
> - KV 缓存走 INT4（可选启用）：量化粒度为 per-token × per-head，沿 head_dim 分 32 维 group；
> - 激活维持 FP16/BF16：与现有 flash-attn 路径兼容；
> - 补齐 YaRN：从 HF `rope_scaling` 字段读 NTK-by-parts 参数，构建独立 RoPE。

---

## 目录

1. [现状速览](#一现状速览)
2. [设计原则与关键决策](#二设计原则与关键决策)
3. [改造总体目录结构](#三改造总体目录结构)
4. [模块 A：离线权重量化](#四模块-a离线权重量化)
5. [模块 B：W4A16 GEMM](#五模块-bw4a16-gemm)
6. [模块 C：INT4 KV Cache](#六模块-cint4-kv-cache)
7. [模块 D：YaRN RoPE](#七模块-dyarn-rope)
8. [模块 E：Checkpoint Loader 改造](#八模块-echeckpoint-loader-改造)
9. [模块 F：配置集成与 CLI](#九模块-f配置集成与-cli)
10. [测试与验证计划](#十测试与验证计划)
11. [风险登记与回退策略](#十一风险登记与回退策略)
12. [实施路线图](#十二实施路线图)
13. [参考资源清单](#十三参考资源清单)

---

## 一、现状速览

### 1.1 nano-vllm 关键代码地图

| 文件                                                   | 行数  | 关键作用                                |
| ------------------------------------------------------ | ----- | --------------------------------------- |
| `nanovllm/config.py`                                   | ~30   | `Config` dataclass，含 TP/EOS/kvcache 等 |
| `nanovllm/engine/model_runner.py`                      | ~260  | KV cache 分配、CUDA Graph capture       |
| `nanovllm/engine/llm_engine.py`                        | ~90   | step 调度                                |
| `nanovllm/layers/linear.py`                            | ~160  | 4 种 TP-aware Linear（Col/Merged/QKV/Row） |
| `nanovllm/layers/attention.py`                         | ~80   | `store_kvcache_kernel` + flash-attn 调用 |
| `nanovllm/layers/rotary_embedding.py`                  | ~60   | 现版 RoPE                                |
| `nanovllm/layers/layernorm.py`、`activation.py`、`sampler.py`、`embed_head.py` | ~120 | 各小模块                         |
| `nanovllm/models/qwen3.py`                             | ~220  | Qwen3ForCausalLM 模型定义                |
| `nanovllm/utils/loader.py`                             | ~30   | safetensors 装载（支持 packed_mapping）  |

### 1.2 关键依赖与"硬约束"

- **Flash-Attention**：使用 `flash_attn.flash_attn_varlen_func` / `flash_attn_with_kvcache`，K/V 必须是 FP16/BF16。这意味着 KV 量化需要替换 `flash_attn_with_kvcache` 调用，改为自研 fused kernel。
- **CUDA Graph**：`enforce_eager=False` 时 decode 路径被 CUDA Graph 捕获。一旦插入 INT4 fused kernel，必须**重抓图**或强制 `enforce_eager=True`。建议引入 INT4 路径时**强制 `enforce_eager=True`**（同 vLLM 的 W4A16 量化端做法）。
- **TP 通信**：`RowParallelLinear` 内置 `dist.all_reduce`；INT4 路径必须复用，不能让 `F.linear` 退化为在 host 端反量化。
- **Paged KV Cache**：现有 KV 缓冲是单个大张量 `(2, L, blocks, block_size, H, D)`，dtype 一致。INT4 KV 必须**用更小的物理 dtype + scale/zero 张量替换**该逻辑。
- **Packed modules**：`qkv_proj` / `gate_up_proj` 在加载时被融合；量化器与加载器都必须保留这套语义。

### 1.3 量化改造的"零侵入点"

| 现存逻辑               | INT4 化后做什么                                          |
| ---------------------- | -------------------------------------------------------- |
| `F.linear(x, w)`       | 改为 `int4_gemm(x, w_packed, scales, zeros)`             |
| `store_kvcache_kernel` | 改为 `store_int4_kv_kernel`（FP16→INT4 量化写入）         |
| `flash_attn_with_kvcache` | 改为自研 `int4_flash_attn_decode`（解码侧）          |
| `flash_attn_varlen_func`（prefill 路径） | INT4 KV 时需 dequantize 到 FP16 再走 flash-attn，**或** 自研 varlen kernel |
| 现版 `RotaryEmbedding` | 扩展为 `YaRNRotaryEmbedding`，按 rope_scaling 重算 inv_freq |

---

## 二、设计原则与关键决策

### 2.1 量化方案选型（已锁定）

| 维度           | 选型                                                 | 理由                                                                                              |
| -------------- | ---------------------------------------------------- | ------------------------------------------------------------------------------------------------- |
| 权重量化精度   | **INT4 非对称 per-group（group_size=128）**          | INT4 值域只有 16，per-group 是 INT4 W4A16 公认的最佳折中                                            |
| 权重量化时机   | **离线量化（一次性脚本）**                           | 避免在线 PTQ 拖首 token 延迟；torchao 的 `Int4WeightOnlyConfig(layout=Int4CPULayout())` 已成熟   |
| 激活精度       | **FP16（BF16 兜底）**                                | 与现有 flash-attn / RMSNorm / TP reduce 一致                                                      |
| KV 精度        | **INT4 per-token × per-head 沿 head_dim group=32**   | 与 vLLM PR #40835 的 `INT4_KV_CACHE` 同款，反量化开销小、精度 OK                                  |
| 反量化位置     | **GEMM 内部融合**；KV 内部 fused attention 融合      | 读写 HBM 的带宽是真正瓶颈，反量化必须 fused                                                       |
| Kernel 实现    | **Triton（GPU only）** + Torch 参考实现 fallback     | 与现版 `store_kvcache_kernel` 风格一致                                                            |
| YaRN           | **NTK-by-parts + attention scaling**                 | Qwen3 HF config 已支持 YaRN `rope_scaling` 字段，直接消费                                          |
| 与 torchao 集成 | **producer = torchao（标准格式），consumer = 自写**   | 避免运行时强依赖 torchao，loader 用纯 torch+triton 即可                                           |

### 2.2 序列化格式（落盘契约）

每条 Linear 权重落盘形态：

```text
<layer>.<proj>.weight.packed   # INT32 (out_features, in_features // 8)
<layer>.<proj>.weight.scales   # FP16 (out_features, in_features // 128)
<layer>.<proj>.weight.zeros    # FP16 (out_features, in_features // 128)
<layer>.<proj>.weight.shape    # INT32 [out_features, in_features]   (用于 sanity check)
```

规则：

- 8 个连续 INT4 nibble 打包为 1 个 INT32，**little-endian**（最低 4 位对应输入维度的最前一个元素）。这是 torchao `Int4CPULayout()` 的默认布局，也是 `torch._weight_int4pack_mm` 期望的布局。
- 非对称量化：quants 范围 `[0, 15]`，反量化公式 `x_fp = (q - zero) * scale`，scale/zero 都是 FP16。
- `packed_modules_mapping`：Q/K/V 与 gate/up 保持原状——量化脚本把分立的 `q/k/v` 沿 output 维拼起来再量化；loader 端继续按 `q_proj → q` 这条 `packed_modules_mapping` 去取 weight，但 weight 形态从 `concat([Wq, Wk, Wv])` 整体量化。
- **不量化**：`embed_tokens`（会拉爆 vocab 而不省 weight 内存）、`lm_head`（同前）。可在 quantize 时通过 `--exclude` 排除。
- 对 skipped 层，写原始 FP16/BF16 weight 即可（保持 checkpoint 仍可被加载）。

### 2.3 显存预算与 KV 布局

原始 KV 一条 token 占用：`2 × num_layers × num_kv_heads × head_dim × dtype_bytes`。

INT4 缓存一条 token 占用：

```text
k_packed:    num_kv_heads × head_dim // 8    INT32     (= head_dim // 2 bytes)
k_scales:    num_kv_heads × head_dim // 32  FP16
k_zeros:     num_kv_heads × head_dim // 32  FP16
（v 同理）
```

Qwen3-0.6B：`head_dim=128, num_kv_heads=8`，一条 token 物理字节：`2 × (128/2 + 128/32 + 128/32) × 8 × 2 = 1088 bytes`。相比 FP16 的 `2 × 128 × 8 × 2 = 4096 bytes`，节省 **约 73%**。

> 模型 runner 中需要把现有的 `torch.empty(2, L, blocks, block_size, H, D)` 替换为 4 个张量：`k_packed, v_packed, k_scales, v_scales, k_zeros, v_zeros`（共 6 个，packed 为 INT32，其余为 FP16）。这会让 `prepare_prefill` / `prepare_decode` 中计算 `slot_mapping` 的逻辑**基本不变**（依旧是物理 slot 索引），但 `self.k_cache = ...`、`store_kvcache` 的输入需要切到新缓冲。

### 2.4 TP 兼容策略（INT4 切片规则）

| Linear 类型               | INT4 切片对象                                                                                        |
| ------------------------- | ----------------------------------------------------------------------------------------------------- |
| `ColumnParallelLinear` (`QKV/Merged/Regular`) | `weight.packed` 沿 out 维切：`packed = packed[start_out:end_out]`；scales/zeros 沿 out 维切 |
| `RowParallelLinear`       | `weight.packed` 沿 in 维切：因 packed 是按 in 维方向打包，**先 unpack 成 dense 再 narrow 是错的**。正确做法：把切片表达为 `packed[start_in_pack:end_in_pack]`，其中 `start_in_pack = start_in // 8`；scales/zeros 沿 group 维切 |
| `MergedColumnParallel`/`QKVParallel` 的 shard_id 仍然用 `'q' / 'k' / 'v' / 0 / 1`，但 weight 是 concat 后的整块再取 shard | 与现有 `weight_loader` 行为一致 |

> scales/zeros 的 TP 装载由新的 `Int4*Linear.weight_loader` 内部处理，不要让 loader 直接对它们 split。

### 2.5 与 CUDA Graph / `enforce_eager` 的边界

- 引入 `quantization != None` 时，**强制 `enforce_eager = True`**（即使用户传入 `False`）。
- 在 `config.py.__post_init__` 里加上：`if self.quantization is not None: self.enforce_eager = True`。
- 文档里写明：triton kernel autotune 与 cuda graph capture 不兼容，加上 `enforce_eager` 是与 vLLM 一致的工程选择。

### 2.6 不在本方案范围

- **Online PTQ**（运行时量化）：不做首 token 量化，模型必须先经过 `nanovllm.tools.quantize`；
- **GPTQ / AWQ**：与本 INT4 互斥；如未来要并存，可以做成不同 `quantization="awq_int4"` 分支；
- **INT8 KV**：不实现，INT4 KV 已经覆盖 73% 收益，INT8 缓存逻辑类似没必要重复；
- **MoE 整量化**：项目当前主推 dense 模型，MoE 留作未来扩展。

---

## 三、改造总体目录结构

```
nanovllm/                                            (仅新增两个文件)
├── config.py                                        MOD
├── engine/model_runner.py                           MOD
├── layers/
│   ├── linear.py                                    MOD (新增 Int4* 4 个类)
│   ├── attention.py                                 MOD (新增 INT4 路径)
│   ├── rotary_embedding.py                          MOD (新增 YaRNRotaryEmbedding)
│   └── quant.py                                     NEW (Triton kernel + torch 参考)
├── models/qwen3.py                                  MOD (qkv_proj/o_proj 等智能选择 linear 类型)
├── utils/loader.py                                  MOD (分支加载 int4 vs fp16)
└── tools/
    ├── __init__.py                                  NEW
    └── quantize.py                                  NEW (离线量化入口)

tests/test_int4.py                                   NEW (eager 路径 unit + parity)
docs/int4_solution.md                                NEW (本文件)
```

**预算行数**：核心代码增加预计 < 600 行。

---

## 四、模块 A：离线权重量化

### 4.1 入口：`nanovllm/tools/quantize.py`

CLI 用法：

```bash
python -m nanovllm.tools.quantize \
    --src  ~/huggingface/Qwen3-0.6B \
    --dst  ~/huggingface/Qwen3-0.6B-INT4 \
    --group-size 128 \
    --exclude embed_tokens lm_head
```

功能流：

```text
Load HF model on CPU (float32) ──► for each 2D Linear:
       ├─ if proj ∈ {q,k,v,gate,up}: 沿 output 维 concat 后量化
       └─ else: 直接量化
       ※ skip 任何名字含 exclude 子串的层，保留原 FP16/BF16 weight
Dump as one safetensors file (model.int4.safetensors)
Write sidecar JSON (quantize.json) 记录 group_size / targets / format
Copy tokenizer / config files so模型可被 HF AutoModelForCausalLM 跳过兼容
```

### 4.2 核心 PyTorch 参考量化函数（**伪代码**——不落到代码层）

```text
quantize_int4_per_group(weight_2D, group_size=128):
    # weight: (out, in), in % group_size == 0
    grouped = weight.view(out, in//G, G)
    wmin    = grouped.amin(-1)        # (out, in//G)
    wmax    = grouped.amax(-1)
    span    = (wmax - wmin).clamp_min(1e-8)
    scale   = span / 15               # FP32
    zero    = (-wmin / scale).round().clamp(0, 15)
    q       = (grouped / scale.unsqueeze(-1) + zero.unsqueeze(-1)).round().clamp(0, 15)
    q       = q.to(uint8).view(out, in)        # 每个 byte 是两个 nibble
    # pack成 INT32（little-endian：低 nibble = 较前元素）
    nibbles = q.view(out, in//2, 2)             # (low, high)
    packed  = ((nibbles[..., 1] << 4) | nibbles[..., 0]).view(torch.int32)
    return packed.contiguous(),
           scale.to(FP16).contiguous(),
           zero.to(FP16).contiguous()
```

### 4.3 与 torchao 的关系

- **本工具优先调用 torchao**（如果用户已安装）：`torchao.dtypes.uintx.int4.Int4WeightOnlyConfig(group_size=128)` → `quantize_(model, config)`。
- **torchao 不可用时**回退到上述 PyTorch 参考实现。两个产物的**落盘布局必须一致**（包成 INT32 little-endian + FP16 scales/zeros），否则后续 loader 无法通用。
- 序列化时如使用 torchao，让 `torch.save({"state_dict": model.state_dict(), "format": "int4_w4a16_v1"}, ...)`；PyTorch fallback 则走 `safetensors`。

### 4.4 校验闭环

量化脚本结束后追加：

1. 反读量化产物，对其中一组权重调用 `int4_gemm_torch(x_dummy, w_packed, scales, zeros, group_size)`，对比与原 FP16 `F.linear(x_dummy, w)` 的最大相对误差（应在 `< 5e-2`）。
2. 把 M_max 调整到原 hidden_size，写入 `quantize.json` 作为参考指标。

---

## 五、模块 B：W4A16 GEMM

### 5.1 接口契约

```text
int4_gemm(
    a:           Tensor (M, K) FP16,
    b_packed:    Tensor (N, K // 8) INT32,
    scales:      Tensor (N, K // G) FP16,
    zeros:       Tensor (N, K // G) FP16,
    out:         Optional[Tensor] = None,
) -> Tensor (M, N) FP16
```

- Triton kernel：`_int4_gemm_kernel(A, B, S, Z, C, M, N, K, ..., BLOCK_M, BLOCK_N, BLOCK_K=G)`。
- Autotune over `{BLOCK_M ∈ {16,32,64,128}, BLOCK_N ∈ {64,128,256}, BLOCK_K=32 or 64 (G), num_stages ∈ {2,3,4}}`。
- PyTorch 参考：`int4_gemm_torch`，仅用于 CPU 单元测试 / 数值对齐。

### 5.2 Triton kernel 设计要点

1. **2-D tile**：以 `(BLOCK_M, BLOCK_N)` 切输出矩阵；`BLOCK_M` 对应 token、`BLOCK_N` 对应 output 通道。
2. **沿 K 维 group 内整块加载**：group_size = `BLOCK_K`，每个 program 一次处理一个 group；这样 scale/zero 都是标量（per-row），可以广播。
3. **INT4 unpack**：每个 INT32 拆 4 位 × 8 次。

   ```text
   # Triton 伪代码
   b_pk = tl.load(B_ptr + offs_n[:, None] * stride_bn + offs_kpk[None, :])   # (BLOCK_N, BLOCK_K//8) int32
   low  = b_pk & 0x0F
   high = (b_pk >> 4) & 0x0F
   b_unp = tl.interleave(low, high).reshape((BLOCK_N, BLOCK_K))  # 低位先
   ```

4. **dequant**：

   ```text
   b_dq = (b_unp - zeros[:, None]) * scales[:, None]   # FP16 (BLOCK_N, BLOCK_K)
   ```

5. **accumulate**：`acc += tl.dot(A_tile, tl.trans(b_dq))`，FP32 accumulator。
6. **写回**：FP16 cast → store。

> 关键正确性保证：**scale、zero 沿 output 维广播**，每个 group 仅一个 (scale, zero) for each row。

### 5.3 与 `layers/linear.py` 的接入

新增四个类（与现有的 4 个基类**一一对应**）：

| 现有类                       | 新增对应                                                   |
| ---------------------------- | ---------------------------------------------------------- |
| `ReplicatedLinear`           | `Int4ReplicatedLinear`                                     |
| `ColumnParallelLinear`       | `Int4ColumnParallelLinear`                                 |
| `MergedColumnParallelLinear` | `Int4MergedColumnParallelLinear`（带 shard_id 路由）       |
| `QKVParallelLinear`          | `Int4QKVParallelLinear`（带 shard_id ∈ {q,k,v} 路由）      |
| `RowParallelLinear`          | `Int4RowParallelLinear`                                    |

统一抽到：

```text
class _Int4LinearBase(_LinearBase):
    def __init__(self, ..., group_size=128):
        self.weight_packed = nn.Parameter(torch.empty(N, K // 8), requires_grad=False)
        self.weight_packed.weight_loader = self._weight_loader   # 自动路由到 packed 字段
        self.scales        = nn.Parameter(torch.empty(N, K // G), requires_grad=False)
        self.zeros         = nn.Parameter(torch.empty(N, K // G), requires_grad=False)

    def forward(self, x):   # 使用 int4_gemm
        return int4_gemm(x, self.weight_packed, self.scales, self.zeros)
```

> 现有 `LinearBase.weight = nn.Parameter(torch.empty(out, in))` 的封装形态非常克制，新增子类不再破坏这个抽象：把 `weight` 字段含义扩展成 `(weight_packed, scales, zeros)` 三元组。

### 5.4 TP 边角情形

- **ColumnParallelLinear**：output 维切。`weight_packed` 是 (out_total/8/tp, K)，因此直接 narrow 第 0 维。`scales`、`zeros` 同样 narrow 第 0 维。
- **RowParallelLinear**：input 维切。`weight_packed` 的第 1 维是 `K // 8`，`narrow(1, start//8, shard//8)`；`scales/zeros` 在 group 维 narrow。
- **Merged + QKV**：保留现有 `weight_loader(param, weight, shard_id)` 三参签名；shard_id 只决定 narrow 起始位置。
- **all_reduce**：`RowParallelLinear.forward` 末尾 `dist.all_reduce(y)` 保留，对 FP16 输出做 reduced 不会因上游用 INT4 GEMM 而有误。

### 5.5 伪代码：模型端选用 Linear 类

```text
def _make_linear(qconfig):
    if qconfig == "int4":
        if kind == "qkv":      return Int4QKVParallelLinear(...)
        elif kind == "merge":  return Int4MergedColumnParallelLinear(...)
        elif kind == "col":    return Int4ColumnParallelLinear(...)
        elif kind == "row":    return Int4RowParallelLinear(...)
        else:                  return Int4ReplicatedLinear(...)
    else:
        # 走原始类
```

在 `models/qwen3.py` 的 `Qwen3ForCausalLM.__init__` 里读 `config.quantization`，把 `self.qkv_proj` / `self.o_proj` / `self.gate_up_proj` / `self.down_proj` 实例化为对应 INT4 类；embed_tokens、lm_head 保留 FP16。

> 推荐：用**构造函数标志位**直接决定类，**不要**在 init 里做 monkey-patch。

---

## 六、模块 C：INT4 KV Cache

### 6.1 物理布局

把现有 `self.kv_cache = torch.empty(2, L, blocks, block_size, H, D)` 替换为 6 个张量：

```text
k_cache_packed:  INT32 (L, blocks, block_size, H, D // 8)
v_cache_packed:  INT32 (L, blocks, block_size, H, D // 8)
k_scales:        FP16 (L, blocks, block_size, H, D // 32)
v_scales:        FP16
k_zeros:         FP16
v_zeros:         FP16
```

> 注意：原 cache 没有 `L` 这个维（layers 直接做 outer），建议新 cache 仍按 `(L, blocks, ...)` 排布，让 layer_id 能直接索引。

**一个 slot 的总字节**（Qwen3-0.6B：`D=128, H=8, group=32`）：

```text
packed = H × D/8 × 4B = 8 × 16 × 4 = 512B
scales = H × D/32 × 2B = 8 × 4 × 2  =  64B
zeros  = 同上           =  64B
tot / k = 640B
k + v   = 1280B
FP16 等价 = 2 × (H × D × 2) = 4096B
→ 节省约 68.75% (实际略高于 73% 的全局预估，因为还有 scale/zero 存储)
```

### 6.2 写入 kernel

**接口**：

```text
store_int4_kv(
    key: (N, H, D) FP16,
    value: (N, H, D) FP16,
    k_cache_packed, v_cache_packed,
    k_scales, v_scales, k_zeros, v_zeros,
    slot_mapping: (N,) INT32,
    group_size=32,
)
```

**Triton kernel 算法**：

- Grid = `(N, H)`。每个 program 处理一个 (token, head)。
- 内部 for-loop 沿 head_dim 按 `group=32` 切片。
- 每片内：
  - `min_k = tl.min(offs_kv)`，`max_k = tl.max(...)`；
  - `scale_k = (max_k - min_k) / 15`，`zero_k = -min_k / scale_k`；
  - `qk = ((kv - min_k) / scale_k).round().int32.clamp(0, 15)`；
  - 打包：`for j in 0..group_size//8: pack = sum(nibble_s << 4*s)`，`store(packed, slot, head, group_idx, j)`；
  - 同时 `store(scale_k.to(FP16), k_scales, slot, head, group_idx)` 等。
- 防御：`if scale_k < 1e-8: scale_k = 1.0`，避免除零（此时整个 group 量化结果全为 0）。

**Triton unpacking helpers**（不需要 `tl.gather`）：

```text
nibbles = tl.interleave(qk[::2], qk[1::2])   # 把 (low, high, ...) 拉直
b_int32 = (nibbles[:, 7] << 28) | (nibbles[:, 6] << 24) | ... | nibbles[:, 0]
```

> 因为 `group=32` 的情况下每个 group 是 4 个 INT32，可以一次 `tl.store` 一个 INT32（一行 pack 一个 int32）。

### 6.3 Attention 路径

#### 6.3.1 Prefill：`flash_attn_varlen_func` 仍然吃 FP16

- 走 INT4 KV 时，在调用 flash-attn 之前**把整段 K/V 反量化回 FP16 buffer**（prefill 不频繁，cost 接受）。
- 反量化使用 Triton kernel：`dequant_int4_to_fp16(cache_packed, scales, zeros) -> fp16_tensor`，按 block 内部连续 offset 切。
- 这意味着 `Attention.__init__` 里需要同时保留 `self.k_cache_fp16` 作为临时缓冲，每次 prefill allocate。

> 等价方案：重写 prefill 路径的 varlen flash-attn 到 triton，self-host 也支持 INT4。**这一改代价太高**，推荐第一种。

#### 6.3.2 Decode：自研 fused kernel

`int4_flash_attn_decode`：

```text
@triton.jit
_kernel(Q, K_pkg, V_pkg, K_sc, V_sc, K_zr, V_zr, block_tables, context_lens, O, scale, ...):
    pid_bs = program_id(0)        # seq idx
    pid_h  = program_id(1)        # head idx
    L      = context_lens[pid_bs]
    n_blk  = ceil(L / BLOCK_SIZE)
    Q      = load Q[pid_bs, pid_h, :]                                  # (D,)
    for blk in range(n_blk):
        slot_base = block_tables[pid_bs, blk] * BLOCK_SIZE
        for off in range(BLOCK_SIZE):                                  # 该 slot
            if blk*BLOCK_SIZE+off >= L: continue
            # 为该 (token, head) on-the-fly 反量化
            K = dequant(K_pkg[slot_base+off, pid_h, :], K_sc[..., g], K_zr[..., g])
            V = dequant(...)
            score = (Q * K).sum() * scale
            # online softmax
            m_new = max(m_i, score); alpha = exp(m_i - m_new); p = exp(score - m_new)
            l_i   = l_i * alpha + p
            acc   = acc * alpha + p * V
    out[pid_bs, pid_h, :] = acc / l_i
```

> 这是**论文式描述**——实际实现时把"rebuild 全 token K/V row" 改成 tile 化的版本（一次处理 BLOCK_K_HEAD 个 token，并把它们的 K 累加后再 score）。这是 flash-attn 的标准 online-softmax 写法，可参考 `tri Dao / vLLM-Ascend INT4 KV`。

### 6.4 在 `Attention` 里切换路径

- `Attention.__init__` 接收 `kv_quant_mode: Literal["none", "int4"]`。由 `Qwen3Attention.__init__` 根据 `config.quantization` 转发。
- `Attention.forward` 内：

```text
if self.kv_quant_mode == "int4" and context.is_prefill:
    # dequant K/V 到 self._k_fp16_buffers 的对应位置
    dequant_int4_to_fp16(...)
    # 然后调 flash_attn_varlen_func
elif self.kv_quant_mode == "int4" and not context.is_prefill:
    return int4_flash_attn_decode(q, ...)
else:
    # 旧路径 (FP16)
```

> `Attention` 类内部还要**对每个 `(K, V)` 写法**：旧逻辑下 `k_cache.numel()` 用作"是否分配了 cache"的判断，INT4 下需要拆成 6 个缓冲的 `numel() and numel() and ...`。

### 6.5 `model_runner` 调整

1. **`allocate_kv_cache`**：
   - 根据 `config.kv_quantization` 选择：
     - "none" → 旧 1 个张量；
     - "int4" → 6 个张量。
   - 然后 `for module in model.modules(): if hasattr(module, "k_cache"): module.k_cache = ...` 之类的循环需要扩展到 6 个字段。可重写为：定义 `Module.attach_kv_cache(k_packed, v_packed, k_s, v_s, k_z, v_z)`，或者干脆把 cache 字段名集中到一个 dict / 一个辅助对象。

2. **`prepare_prefill` / `prepare_decode`**：
   - `slot_mapping` 计算逻辑**不变**（物理 slot 索引）。
   - 这两个函数返回的 `(input_ids, positions)` 也**不变**。

3. **`run_model`**：不变。

4. **`capture_cudagraph`**：INT4 时强制跳过（前面约定）。

### 6.6 伪代码：层次划分

```text
class Attention(nn.Module):
    def __init__(..., kv_quant='none', group_size=32):
        if kv_quant == 'int4':
            self.k_cache = self.v_cache = None
            self.k_cache_packed = self.v_cache_packed = None
            self.k_cache_scales = ... = None
            self._k_fp16_buf = None        # prefill 时复用
        else:
            self.k_cache = self.v_cache = torch.tensor([])   # 旧形式

    def forward(self, q, k, v):
        ctx = get_context()
        if self.kv_quant == 'int4':
            store_int4_kv(k, v, self.k_cache_packed, ..., ctx.slot_mapping)
            if ctx.is_prefill:
                # dequant K/V to fp16 buffer,然后 flash-attn
                ...
            else:
                return int4_flash_attn_decode(...)
        else:
            # 旧路径
            store_kvcache(k, v, ...)
            return flash_attn_with_kvcache(...)
```

---

## 七、模块 D：YaRN RoPE

### 7.1 数学要点

YaRN 的核心修改：

1. **NTK-by-parts**：将 inv_freq 分两段：
   - 高频段（波长 λ ≤ original_max_pos）按 **Linear scaling**：`inv_freq' = inv_freq / s`；
   - 低频段（波长 λ > original_max_pos）按 **NTK**：`inv_freq' = inv_freq × (s^(D / (D-2)))` 的反推（用 `ramp_factor` 渐变）。
2. **attention scaling**：在 softmax(QKᵀ / √d) 之前乘 `1 / t`，其中 `t = 0.1 * ln(s) + 1`（NTK-aware 形式）。在 `apply_rotary_emb` 末尾对 `query`/`key` 乘 `t` 即可影响 attention。

### 7.2 HF `rope_scaling` 字段示例（Qwen3）

```jsonc
{
  "rope_scaling": {
    "rope_type": "yarn",
    "factor": 4.0,
    "original_max_position_embeddings": 32768,
    "beta_fast": 32,
    "beta_slow": 1,
    "mscale": 0.0,                       // (可选) attention scaling, 缺省即用 0.1*ln(s)+1
    "mscale_all_dim": false
  }
}
```

### 7.3 模块：`YaRNRotaryEmbedding(nn.Module)`

```text
def __init__(head_size, rotary_dim, max_position, base, rope_scaling):
    self.rope_scaling  = rope_scaling or {}
    self.orig_max_pos  = rope_scaling.get('original_max_position_embeddings', max_position)
    self.factor        = rope_scaling.get('factor', 1.0)
    self.beta_fast     = rope_scaling.get('beta_fast', 32)
    self.beta_slow     = rope_scaling.get('beta_slow', 1)
    self.mscale        = rope_scaling.get('mscale', 0.1 * math.log(self.factor) + 1)
    inv_freq = 1.0 / (base ** (arange(0, rotary_dim, 2) / rotary_dim))
    if self.factor != 1.0:
        inv_freq = _yarn_correction(inv_freq, self.factor, self.beta_fast, self.beta_slow, self.orig_max_pos)
    # Cache cos/sin
    t = arange(max_position)
    f = einsum('i,j -> ij', t, inv_freq)
    self.cos_sin_cache = cat((f.cos(), f.sin()), dim=-1).unsqueeze(1)    # FP32

def forward(positions, query, key):
    # 标准 RoPE 取 cos/sin 后 rotary
    cos_sin = self.cos_sin_cache[positions]
    cos, sin = chunk(cos_sin, 2, -1)
    query = apply_rotary_emb(query, cos, sin)
    key   = apply_rotary_emb(key,   cos, sin)
    # attention scaling (YaRN 关键)
    query = query * self.mscale
    key   = key   * self.mscale
    return query, key
```

`_yarn_correction(inv_freq, factor, beta_fast, beta_slow, orig_max_pos)`：

```text
def _find_correction_dim(num_rot, base, orig_max_pos):
    return (num_rot * log(orig_max_pos / (num_rot * 2π)) / log(base)).floor()
def _find_correction_range(rot_dim, beta_fast, beta_slow, orig_max_pos, base):
    # 标准 YaRN 公式
    low  = floor(rot_dim * log(orig_max_pos / beta_slow) / (2π * log(base)))
    high = floor(rot_dim * log(orig_max_pos / beta_fast) / (2π * log(base)))
    return max(low, 0), min(high, rot_dim - 1)

inv_freq_extrapolation = inv_freq / factor               # linear scaling
inv_freq_interpolation = 1.0 / (factor * base ** (...))   # NTK-by-parts ramp
low, high = _find_correction_range(...)
ramp = linspace(0, 1, high - low)
mask = where(arange(rot_dim//2) < low, 1.0, where(between(...), 1 - ramp, 0.0))
return inv_freq_interpolation * (1 - mask) + inv_freq_extrapolation * mask
```

### 7.4 在 `RotaryEmbedding` 上的复用

- 保留现有 `RotaryEmbedding` 类（default 用），把 `YaRNRotaryEmbedding` 作为子类。
- `get_rope` 工厂读 `rope_scaling`，**当 `rope_scaling` 是 `dict` 且 `rope_type == "yarn"` 时返回 `YaRNRotaryEmbedding`**，否则返回基础类。
- `Qwen3Attention.__init__` 已接收 `rope_scaling`，自然接入。

### 7.5 与 attention scaling 的微妙点

- YaRN 在 `query*key^T / √d` 之前对 `q, k` 各乘 `mscale`，等价于对 `score = q·k` 乘 `mscale²`（因为 softmax 是 `score / √d`，所以最终效果 `score → mscale² × score / √d`）。**但经验做法是只用 `mscale` 对 q/k 各乘一次（提升注意力聚焦）**，论文原意即如此；要严格遵循原公式。
- 实现里直接把 `q *= self.mscale; k *= self.mscale` 即可。
- 务必让现版 `apply_rotary_emb` 不被改动；缩放是 YaRN 在 rotary 输出后再乘。

---

## 八、模块 E：Checkpoint Loader 改造

### 8.1 现版 `loader.py` 行为回顾

```text
for file in *.safetensors:
    for weight_name in keys():
        if weight_name 命中 packed_mapping:    # q_proj / k_proj / v_proj / gate_proj / up_proj
            v, shard_id = packed_mapping[short_proj]
            param_name  = weight_name.replace(short_proj, v)
            param       = model.get_parameter(param_name)
            weight_loader(param, weight, shard_id)
        else:
            param = model.get_parameter(weight_name)
            weight_loader(param, weight)
```

### 8.2 INT4 checkpoint 的额外键

- `xxx.weight.packed`
- `xxx.weight.scales`
- `xxx.weight.zeros`
- `xxx.weight.shape`

### 8.3 改造点

1. 增加一个 `try: param = model.get_parameter(name + '.packed')` 探测；若存在，认为 `name` 是一个 INT4 线性，把 weight/scales/zeros 分别按以下优先级注入：
   - 若匹配 `packed_modules_mapping` 中的短名 → 用 `_packed_loader`（再分发 shard_id 给 INT4 类）；
   - 否则为单 Linear。
2. 每个 `Int4*Linear` 类需要：

   ```text
   def weight_loader(self, packed_param, scales_param, zeros_param, shape_param, loaded_weight_dict, shard_id=None):
       # 解析 shape
       # narrow / split weight_packed、scales、zeros
       # copy_ 到各自的 Parameter.data
       ...
   ```

3. **新加一个 `model.__class__.quant_kind` 字段或 `quantization_module_classes` 列表**，让 loader 能识别哪些模块是 INT4、哪些是普通 FP16。
4. 注意：跳过 `embed_tokens.weight` / `lm_head.weight`（保持 FP16），loader 默认行为已能处理（不命中 INT4 fallback 路径）。

### 8.4 与 TP 的协作

- 多 rank 场景：`dist.barrier()` 已在 `model_runner.py` 加了之后，loader 只在 rank=0 进程内跑；其余 rank 通过 `torch.distributed.broadcast` 拉取。
- 当前 nano-vllm 的 TP 模型是**整体在每个 rank 跑 model 实例**，每个 rank 用同一个 path 装载自己的 shard；int4 loader 要保证**每个 rank 只拿自己那份 shard**：

  - rank_i 加载时按 `narrow(self.tp_rank)` 的方式取自己的切片；
  - 由 `Int4*Linear` 内部的 `weight_loader` 实现。

### 8.5 兼容 checkpoint

新 loader 必须：

- 用 fp16 路径时跑现版同样代码 → 行为 100% 一致；
- 用 int4 路径时增加分支 → 不影响 FP16；
- 错误检测：INT4 weight 但 group_size 不匹配 → 友好报错。

---

## 九、模块 F：配置集成与 CLI

### 9.1 `config.py`

新增三个字段：

```text
quantization:    str | None = None    # "int4" | None
kv_quantization: str | None = None    # "int4" | None
rope_scaling:    dict | None = None   # 通常从 hf_config 自动读
```

`__post_init__` 增加：

- `assert self.quantization in (None, "int4")`；
- `assert self.kv_quantization in (None, "int4")`；
- 若 `quantization == "int4": self.enforce_eager = True`；
- `self.rope_scaling = self.hf_config.rope_scaling if hasattr(self.hf_config, "rope_scaling") else None`。

`LLM(model, quantization="int4", kv_quantization="int4", ...)` 作为顶层 API。

### 9.2 `LLM/__init__` 导出

- `from nanovllm.tools.quantize import quantize_checkpoint as quantize`（高层便捷函数）；
- 或建议用户直接 `python -m nanovllm.tools.quantize ...`。

### 9.3 `example.py`

保持不变，但加一段注释：

```python
# INT4 推理用法：
#   (1) 先离线量化：python -m nanovllm.tools.quantize --src Qwen3-0.6B --dst Qwen3-0.6B-INT4
#   (2) llm = LLM("~/huggingface/Qwen3-0.6B-INT4", quantization="int4",
#                 kv_quantization="int4")   # 后者可选
```

### 9.4 YAML / 命令行风格

不引入新的配置系统；保持 nano-vllm 极简 dataclass 风格。

---

## 十、测试与验证计划

### 10.1 数值一致性

| 测试                                       | 方法                                                                                  | 期望                                                              |
| ------------------------------------------ | ------------------------------------------------------------------------------------- | ----------------------------------------------------------------- |
| `int4_gemm_torch` vs `F.linear`            | 同一随机 weight，比较 max_rel_err                                                | `< 5e-2`                                                         |
| `int4_gemm_torch` vs Triton `int4_gemm`    | 同输入、同 seed，对比 FP16 输出                                                     | max abs diff `< 1e-2`（允许微小差异）                            |
| `store_int4_kv_torch` vs `dequant_kv_torch` | 写入 → 反量化，对比原始 K/V                                                        | max abs diff `< scale_k * 0.5`（由 round 误差）                  |
| `YaRNRotaryEmbedding` vs HF Transformers  | 同 positions、同 hidden，对比 rotary 后 `query`/`key`                                | max abs `< 1e-3`（FP16 浮点精度内）                              |
| INT4 end-to-end                             | 量化后跑 `example.py`，与 FP16 输出比较 greedy 前 50 token 的 diff                    | diff tokens 数 `< 5 / 50`（轻微不一致是预期的，由量化误差造成）    |
| INT4 KV end-to-end                          | 长 prompt (1k+ tokens) 比对 prefill 显存占用                                         | 显存下降 ~50%+                                                   |
| YaRN 长 prompt                              | prefill + decode 4k+ tokens，看 rotary 一致                                          | rotary shape/values 与 HF 一致                                   |

### 10.2 端到端 smoke

`tests/test_int4.py` 框架：

- 用小模型（Qwen3-0.6B）量化；
- 写 8 个简单 prompt，量化前 vs 量化后跑相同 sampling config；
- 接受断言：≤20% 长度内结果应该至少"读得通"。

### 10.3 Bench 增项

`bench.py` 增加 `--quantization int4` `--kv-quantization int4` 参数，输出与原版对照的吞吐量与显存峰值。

---

## 十一、风险登记与回退策略

| 风险                                            | 缓解                                                                              |
| ----------------------------------------------- | --------------------------------------------------------------------------------- |
| Triton kernel 数值与 PyTorch 参考实现不一致    | 用 `tests/test_int4.py` 加显式容忍阈值，定位差异到 BLOCK 配置 / autotune            |
| INT4 W4A16 在小 hidden_size (如 head_dim<64) 发散 | 退到 per-tensor INT4 或 INT8；保留 `quantization="int8"` 备选                       |
| KV INT4 vs FP16 在 decode latency 上未提升      | 关闭 `kv_quantization="int4"` 即回退；检查 store_int4_kv 与 attention fused 端是否真有 fusion |
| YaRN 在 nano-vllm 的 `cos_sin_cache` 占用过大  | cap `max_position`（如 64k × factor 内存大致 ≤ 100MB，可接受）                    |
| LoRA + INT4 冲突                                | 本方案不支持；遇到时直接报"unsupported"                                            |
| TP 同步问题（int4 切片与 scales 不一致）         | 在 `weight_loader` 中显式 assertion；提供 `model.sanity_check()` 自检              |
| CUDA Graph 不兼容                               | INT4 路径强制 `enforce_eager=True`（已写进 config 默认）                          |
| Long-context 性能（仅 decode path 关注的 KV）   | 保证 decode fused kernel 真的 fused（不要在中途 HBM round-trip）                   |

---

## 十二、实施路线图

> 推荐顺序：M1 → M2 → M3 → M4，每步 PR 都可独立验证与回退。

### M1：INT4 权重（最长 1000 行）

1. 写 `nanovllm/layers/quant.py`：`int4_gemm` Triton + `_int4_gemm_torch` 参考。
2. 扩展 `nanovllm/layers/linear.py`：新增 5 个 `Int4*Linear` 类。
3. 扩展 `nanovllm/utils/loader.py`：支持 INT4 packed 字段。
4. 扩展 `nanovllm/config.py` / `Qwen3Attention`、`Qwen3MLP`：根据 `quantization` 选择类。
5. 写 `nanovllm/tools/quantize.py`：离线量化脚本。
6. 写 `tests/test_int4.py`。
7. 验收：`quantization="int4"` 时仍能跑通 example，但**速度可能不变**（下一步提速）。

### M2：INT4 KV Cache（500 行）

1. 写 `store_int4_kv` Triton + torch 参考。
2. 写 `int4_flash_attn_decode` Triton（**fused**，避免 HBM round-trip）。
3. 改 `Attention.forward` 走两个分支。
4. 改 `ModelRunner.allocate_kv_cache`：分配 6 个缓冲；非 layer-aware 调度逻辑。
5. 验收：长序列显存下降；decode 速度有提升。

### M3：YaRN（200 行）

1. 写 `YaRNRotaryEmbedding` 与 `_yarn_correction`。
2. 改 `get_rope`：根据 `rope_scaling` 选择类。
3. 验收：`max_model_len > 32k` 时位置编码与 HF 一致。

### M4：集成与文档（200 行）

1. CLI 文档、变更记录。
2. `bench.py` 加 INT4 路径。
3. 在 `魔改方案.md` 中追加"INT4 改造"章节。

### 累计行数估计

| 阶段 | 新增行（代码） | 新增行（注释/文档） | 备注                       |
| ---- | -------------- | ------------------ | -------------------------- |
| M1   | ~600           | ~150               | 包含 Triton kernel 主体   |
| M2   | ~500           | ~120               | Attention 分支与 ModelRunner |
| M3   | ~150           | ~50                |                            |
| M4   | ~50            | ~300（文档）       |                            |
| **合计** | ~1300        | ~620               | nano-vllm 现版 ~1200 行 |

> **关键承诺**：实施后总计行数 ~2500，仍属"教学可读"级别。

---

## 十三、参考资源清单

### 13.1 直接参考代码

| 模块                | 关键仓库 / 路径                                                                                                              |
| ------------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| torchao INT4 layout | `torchao/dtypes/uintx/int4_tensor.py` (`Int4CPULayout`)                                                                      |
| INT4 W4A16 kernel   | vLLM `vllm/model_executor/layers/quantization/utils/w4a16_utils.py` 的 `triton_w4a16_gemm_kernel`                              |
| INT4 权重量化       | vLLM `vllm/model_executor/layers/quantization/quark_w4a16_int4.py`、`vllm/model_executor/layers/quantization/gptq_marlin.py` |
| INT4 KV cache       | vLLM PR #40835（参考 INT4 per-token-head-group 量化）                                                                       |
| Per-token quant    | vLLM `vllm/attention/ops/triton_flash_attention.py` 中关于 `kv_cache_quant_algo = "int4"`                                   |
| YaRN 实现           | HuggingFace Transformers `transformers/modeling_rope_utils.py` 中的 `ROPE_INIT_FUNCTIONS["yarn"]`                          |
| nano-vllm 启发       | `chengy-sysu/nano-vllm-swap`、`CefBoud/nano-vllm`（仅借鉴项目结构，不直接复用代码）                                          |

### 13.2 关键论文 / 文档

- Dettmers et al., *LLM.int4(): Pushing the Limit of Quantization for LLMs*（per-group INT4 的奠基）；
- Su et al., *YaRN: Efficient Attention for Long Context*；
- Frantar et al., *GPTQ: Accurate Post-training Quantization for Generative Pre-trained Transformers*（per-group quantization 的工业落地）；
- *SmoothQuant*（activation 量化参考，本项目不需要，但提及以备扩展）。

### 13.3 推荐工具链

```bash
# 离线量化
pip install "torchao>=0.7"        # torchao 与 torch 兼容性按需调整

# 开发
pip install "triton>=3.0"
pip install "flash-attn"          # 现版仍依赖（prefill 路径）

# CPU 测试 fallback
pip install numpy pytest
```

### 13.4 验证硬件推荐

- 单卡 ≥ 16GB（如 RTX 4090 / A4000），可跑 Qwen3-0.6B + KV4；
- ≥ 24GB（如 RTX 4090 / A5000 / L4）跑 Qwen3-4B + KV4；
- 7B+ 模型 INT4 KV 推荐 ≥ 40GB 显存（A100 / A6000）。

---

## 附录 A：关键伪代码汇总

> 这些伪代码片段在工程阶段可作为目标参考；不构成最终实现的标准。

### A.1 INT4 GEMM Triton kernel（要素版）

```text
@triton.autotune(configs=[...], key=["M","N","K"])
@triton.jit
def _int4_gemm_kernel(A, B, S, Z, C, M, N, K, ..., BLOCK_M, BLOCK_N, BLOCK_K):
    pid = tl.program_id(0)
    pid_m, pid_n = pid // tl.cdiv(N, BLOCK_N), pid % tl.cdiv(N, BLOCK_N)
    offs_m, offs_n = pid_m*BLOCK_M + tl.arange(0,BLOCK_M), pid_n*BLOCK_N + tl.arange(0,BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    offs_k = tl.arange(0, BLOCK_K)        # = group_size
    for g in range(K // BLOCK_K):
        a = tl.load(A + offs_m[:,None]*stride_am + (g*BLOCK_K + offs_k)[None,:]*stride_ak)   # (BLOCK_M, BLOCK_K)
        b_pk = tl.load(B + (pid_n*BLOCK_N + offs_n)[:,None]*stride_bn + (g*BLOCK_K//8 + tl.arange(0,BLOCK_K//8))[None,:]*stride_bk)
        b_unp = tl.interleave(b_pk & 0xF, (b_pk >> 4) & 0xF).reshape((BLOCK_N, BLOCK_K))
        s = tl.load(S + (pid_n*BLOCK_N + offs_n)*stride_sn + g*stride_sk)   # (BLOCK_N,)
        z = tl.load(Z + (pid_n*BLOCK_N + offs_n)*stride_zn + g*stride_zk)
        b_dq = (b_unp.to(tl.float16) - z[:,None]) * s[:,None]
        acc += tl.dot(a, tl.trans(b_dq), allow_tf16=False)
    c = acc.to(tl.float16)
    tl.store(C + offs_m[:,None]*stride_cm + offs_n[None,:]*stride_cn, c, mask=...)
```

### A.2 Store INT4 KV（要素版）

```text
@triton.jit
def _store_int4_kv(K, V, slot_map, Kc, Vc, Ks, Vs, Kz, Vz, H, D, GROUP):
    pid_t, pid_h = tl.program_id(0), tl.program_id(1)
    slot = tl.load(slot_map + pid_t);  if slot < 0: return
    NG = D // GROUP
    for g in tl.static_range(NG):
        offs = g*GROUP + tl.arange(0,GROUP)
        k = tl.load(K + pid_t*H*D + pid_h*D + offs)
        v = tl.load(V + pid_t*H*D + pid_h*D + offs)
        mn_k, mx_k = tl.min(k), tl.max(k); sk = (mx_k-mn_k)/15; zk = -mn_k/(sk+1e-12)
        mn_v, mx_v = tl.min(v), tl.max(v); sv = (mx_v-mn_v)/15; zv = -mn_v/(sv+1e-12)
        tl.store(Ks + slot*H*NG + pid_h*NG + g, sk.to(tl.float16))
        tl.store(Kz + slot*H*NG + pid_h*NG + g, zk.to(tl.float16))   # similar for V
        qk = (((k-mn_k)/(sk+1e-12)).round().clamp(0,15)).to(tl.int32)
        qv = (((v-mn_v)/(sv+1e-12)).round().clamp(0,15)).to(tl.int32)
        # pack GROUP nibbles into GROUP//8 int32s
        for j in tl.static_range(GROUP // 8):
            sub = j*8 + tl.arange(0, 8)
            packed_k = 0
            for s in tl.static_range(8):
                packed_k = packed_k | (tl.sum(tl.where(tl.arange(0, GROUP)==sub[s], qk, 0)) << (4*s))
            tl.store(Kc + slot*H*(D//8) + pid_h*(D//8) + g*(GROUP//8) + j, packed_k)
        # similarly for V
```

### A.3 YaRN Correction（要素版）

```text
def _yarn_correction(inv_freq, factor, beta_fast, beta_slow, orig_max_pos, base):
    rot_dim = inv_freq.shape[0] * 2
    def find_corr_dim(n_rot, base, orig):
        return (n_rot * math.log(orig/(n_rot*2*math.pi)) / math.log(base)).floor()
    def find_corr_range(rot, fast, slow, orig, b):
        low  = math.floor(rot * math.log(orig/slow)/(2*math.pi*math.log(b)))
        high = math.floor(rot * math.log(orig/fast)/(2*math.pi*math.log(b)))
        return max(low, 0), min(high, rot - 1)
    low, high = find_corr_range(rot_dim, beta_fast, beta_slow, orig_max_pos, base)
    inv_freq_extrap = inv_freq / factor
    inv_freq_interp = inv_freq * (factor ** ((rot_dim - 2*(inv_freq.shape[0]-low)) / (rot_dim - 2*low)))
    ramp = torch.linspace(0, 1, high-low)
    mask = (torch.arange(rot_dim//2) < low) | ((torch.arange(rot_dim//2) > high))
    inv_freq_mixed = inv_freq_interp * (1 - ramp*(1-mask))  # 简化版
    return inv_freq_mixed
```

---

## 附录 B：与现版 `魔改方案.md` 中"优先级 P5 · 多 LoRA / 量化"的衔接

- 本方案对应 P5 中"量化"分支的**W4A16 + KV4**落地路径；
- 不实现多 LoRA（基于 `魔改方案.md` 判断多 LoRA 优先级靠后）；
- 与 P0（Swap）/ P1（V1 调度）/ P4（KV 压缩）正交：INT4 KV 与 KV 压缩是不同维度的优化，未来如要做 KV 压缩，可基于 INT4 后的压缩 base 继续做。

---

文档版本：v1.0 起草。本文件在每个 M 阶段开始前可被补充/修订；M 阶段结束后在文档末尾追加"实施记录"章节登记偏差与决策变更。
