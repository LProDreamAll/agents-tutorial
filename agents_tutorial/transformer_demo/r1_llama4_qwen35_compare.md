# DeepSeek-R1 / Llama4 / Qwen3.5 推理流程对比

## 0. 范围说明

本文基于当前仓库中的三个示例骨架进行对比：

- `deepseek_r1_demo.py`
- `llama4.py`
- `qwen3_demo.py`（这里用作 Qwen3.5 的近似流程说明）

这是一份“工程流程对比”，不是精确还原各官方实现细节。

## 1. DecoderLayer 核心实现差异

| 维度 | DeepSeek-R1 | Llama4 | Qwen3.5（以 `qwen3_demo.py` 为准） |
| --- | --- | --- | --- |
| 主体结构 | `Norm -> Attention -> Residual -> Norm -> MoE/FFN -> Residual` | `Norm -> Attention -> Residual -> Norm -> FFN/MoE -> Residual` | `Norm -> Attention -> Residual -> Norm -> MLP -> Residual` |
| 归一化 | RMSNorm 占位 | RMSNorm 占位 | RMSNorm 占位 |
| 注意力路径 | 自注意力 + cache（占位） | 自注意力 + RoPE + kv_cache（可扩 GQA/MQA） | 自注意力 + GQA + RoPE + Sliding Window（注释定义） |
| 前馈子层 | 默认强调 MoE（`DeepSeekR1MoE`） | 可选 Dense MLP 或 MoE（`use_moe` 开关） | Dense MLP（SwiGLU） |
| 稀疏性 | 主要来自 MoE 路由稀疏 | 取决于是否启用 MoE | 主要是注意力侧（如滑窗），FFN 为 dense |
| 工程关注点 | Router/专家负载均衡、路由开销、token 分发 | 通用性强，需兼容多配置（dense/moe） | 实现相对直接，重点在 GQA+滑窗+缓存一致性 |

### 1.1 三者 DecoderLayer 的“最核心区别”

1. DeepSeek-R1：重点在 `MoE` 路由和“推理型两阶段输出”联动，层内计算通常更偏向“稀疏专家”。
2. Llama4：层结构最通用，注意力和 FFN 设计都可配置，既能做密集也能做专家化。
3. Qwen3.5：更偏“稳定 dense 解码层 + 高效注意力机制（GQA/滑窗）”，工程上更强调长上下文成本控制。

## 2. 推理流程差异（inference pipeline）

## 2.1 DeepSeek-R1：两阶段推理

在当前 `DeepSeekR1ReasoningPipeline` 中是明确分两步：

1. 先生成 reasoning/think token（内部思考阶段）。
2. 再基于思考上下文生成最终 answer token。

特点：

- 目标是“先想再答”，可显式控制思考长度。
- 线上工程里需要额外处理“思考阶段”的长度、截断、可见性和成本。

## 2.2 Llama4：标准 prefill + decode

`llama4.py` 的 `generate` 注释对应典型自回归流程：

1. `prefill`：整段 prompt 过模型，建立 KV Cache。
2. `decode`：每步只输入新 token，复用 cache 迭代生成。

特点：

- 是最常见、最通用的推理主流程。
- 重点优化点是 cache 命中、注意力实现和采样策略。

## 2.3 Qwen3.5：同样是 prefill + decode，但强调注意力效率

在 `qwen3_demo.py` 里，推理虽未单独写 `generate`，但前向结构已体现关键点：

1. GQA：减少 KV 头数，降低 cache 成本。
2. Sliding Window（可选）：降低超长上下文下的注意力计算量。
3. RoPE + cache：保持自回归解码一致性。

特点：

- 流程形态与 Llama4 类似（都是 prefill/decode）。
- 差异主要体现在注意力算子与内存/吞吐优化策略上，而不是“先想后答”的阶段拆分。

## 3. 一句话总结

1. DecoderLayer 角度：DeepSeek-R1 更偏 MoE 稀疏专家，Llama4 偏可配置通用骨架，Qwen3.5 偏 dense + 高效注意力。
2. 推理流程角度：DeepSeek-R1 强调“两阶段（think -> answer）”；Llama4 和 Qwen3.5 主体都是“prefill -> decode”的标准自回归流程。
