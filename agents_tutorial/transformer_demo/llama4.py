import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """
    Llama 系列常见的 RMSNorm。
    这里只保留接口，不写具体公式实现。
    """

    def __init__(self, hidden_size, eps=1e-6):
        super(RMSNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x):
        # TODO: 实现 RMSNorm 计算逻辑
        pass


class RotaryEmbedding(nn.Module):
    """
    RoPE 位置编码占位模块。
    """

    def __init__(self, head_dim):
        super(RotaryEmbedding, self).__init__()
        self.head_dim = head_dim

    def forward(self, query, key, position_ids):
        # TODO: 在 q/k 上应用旋转位置编码
        pass


class Llama4Attention(nn.Module):
    """
    Llama 4 注意力流程骨架。
    可按需要扩展 GQA/MQA、KV Cache、Flash Attention 等能力。
    """

    def __init__(self, hidden_size, num_heads, num_kv_heads):
        super(Llama4Attention, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads

        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.o_proj = nn.Linear(hidden_size, hidden_size)

        self.rotary_emb = RotaryEmbedding(head_dim=hidden_size // num_heads)

    def forward(self, hidden_states, attention_mask=None, position_ids=None, kv_cache=None):
        # 1. 线性投影得到 q/k/v
        # 2. reshape 到多头形状
        # 3. 对 q/k 应用 RoPE
        # 4. 如果有 kv_cache，则拼接历史 k/v（解码阶段）
        # 5. 计算注意力权重并应用 mask
        # 6. 与 v 相乘并合并多头
        # 7. 输出投影
        pass


class Llama4MLP(nn.Module):
    """
    Llama 4 前馈网络占位。
    常见实现可使用 SwiGLU/GeGLU 等。
    """

    def __init__(self, hidden_size, intermediate_size):
        super(Llama4MLP, self).__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size)
        self.up_proj = nn.Linear(hidden_size, intermediate_size)
        self.down_proj = nn.Linear(intermediate_size, hidden_size)

    def forward(self, x):
        # 1. gate/up 双分支投影
        # 2. 激活并门控融合
        # 3. down 投影回 hidden_size
        pass


class Llama4MoE(nn.Module):
    """
    MoE 路由占位模块（若模型配置启用专家网络）。
    不实现具体路由细节，仅描述流程。
    """

    def __init__(self, hidden_size, intermediate_size, num_experts):
        super(Llama4MoE, self).__init__()
        self.router = nn.Linear(hidden_size, num_experts)
        self.experts = nn.ModuleList(
            [Llama4MLP(hidden_size, intermediate_size) for _ in range(num_experts)]
        )

    def forward(self, x):
        # 1. router 计算每个 token 的专家得分
        # 2. 选择 top-k 专家（如 top-1/top-2）
        # 3. 分发 token 到对应专家
        # 4. 聚合专家输出
        pass


class Llama4DecoderLayer(nn.Module):
    """
    Decoder 层骨架：Pre-Norm + Attention + FFN/MoE + Residual。
    """

    def __init__(
        self,
        hidden_size,
        num_heads,
        num_kv_heads,
        intermediate_size,
        use_moe=False,
        num_experts=8,
    ):
        super(Llama4DecoderLayer, self).__init__()
        self.input_norm = RMSNorm(hidden_size)
        self.post_attn_norm = RMSNorm(hidden_size)

        self.self_attn = Llama4Attention(hidden_size, num_heads, num_kv_heads)
        self.ffn = (
            Llama4MoE(hidden_size, intermediate_size, num_experts)
            if use_moe
            else Llama4MLP(hidden_size, intermediate_size)
        )

    def forward(self, hidden_states, attention_mask=None, position_ids=None, kv_cache=None):
        # 1. Self Attention 子层（带残差）
        residual = hidden_states
        hidden_states = self.input_norm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            kv_cache=kv_cache,
        )
        hidden_states = residual + hidden_states

        # 2. FFN / MoE 子层（带残差）
        residual = hidden_states
        hidden_states = self.post_attn_norm(hidden_states)
        hidden_states = self.ffn(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class Llama4Model(nn.Module):
    """
    Llama 4 主体流程骨架（Decoder-Only）。
    """

    def __init__(
        self,
        vocab_size,
        hidden_size,
        num_layers,
        num_heads,
        num_kv_heads,
        intermediate_size,
        use_moe=False,
    ):
        super(Llama4Model, self).__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList(
            [
                Llama4DecoderLayer(
                    hidden_size=hidden_size,
                    num_heads=num_heads,
                    num_kv_heads=num_kv_heads,
                    intermediate_size=intermediate_size,
                    use_moe=use_moe,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_norm = RMSNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, input_ids, attention_mask=None, position_ids=None, kv_cache=None):
        # 1. token embedding
        hidden_states = self.embed_tokens(input_ids)

        # 2. 逐层 decoder
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                kv_cache=kv_cache,
            )

        # 3. final norm + lm head
        hidden_states = self.final_norm(hidden_states)
        logits = self.lm_head(hidden_states)
        return logits

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens, eos_token_id=None):
        """
        推理流程占位：
        - prefill: 先把 prompt 全量过一遍，初始化 cache
        - decode: 每步只输入最新 token，复用 cache
        """
        # 1. 初始化 kv_cache / position_ids
        # 2. prefill 阶段：处理输入 prompt
        # 3. decode 循环：采样下一个 token，更新 cache
        # 4. 命中 eos 或达到长度上限后结束
        pass
