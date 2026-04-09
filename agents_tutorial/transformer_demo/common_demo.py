import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """
        位置编码模块
    """

    def forward(self, x):
        pass


class MultiHeadAttention(nn.Module):
    """
        多头注意力机制模块
    """

    def __init__(self, hidden_size, num_heads):
        super(MultiHeadAttention, self).__init__()
        # hidden_size 越大，说明每个词能携带的信息越丰富，模型的表达能力越强，但计算量和显存占用也会大幅增加。
        assert hidden_size % num_heads == 0, " 嵌入维度 hidden_size 必须能被 num_heads 整除"
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.h_k = hidden_size // num_heads

        # 定义qkv,和线性变换层
        self.W_q = nn.Linear(hidden_size, hidden_size)
        self.W_k = nn.Linear(hidden_size, hidden_size)
        self.W_v = nn.Linear(hidden_size, hidden_size)
        self.W_o = nn.Linear(hidden_size, hidden_size)

    def forward(self, Q, K, V, mask=None):
        # 1. 对 Q, K, V 进行线性变换
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))
        # 2. 计算缩放点积注意力
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        # 3. 合并多头输出并进行线性变换
        output = self.W_o(self.combine_heads(attn_output))
        return output

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        # 1. 计算注意力得分 (QK^T)
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.h_k)
        # 2. 应用掩码
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float("-inf"))
        # 3. 计算注意力权重 (softmax)
        attn_probs = torch.softmax(attn_scores, dim=-1)
        # 4. 加权求和得到注意力输出
        attn_output = torch.matmul(attn_probs, V)
        return attn_output

    def split_heads(self, x):
        # 将输入 x 的形状从 (batch_size, seq_length, d_model)
        # 变换为 (batch_size, num_heads, seq_length, d_k)
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)

    def combine_heads(self, x):
        # 将输入 x 的形状从 (batch_size, num_heads, seq_length, d_k)
        # 变回 (batch_size, seq_length, d_model)
        batch_size, num_heads, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)


class PositionWiseFeedForward(nn.Module):
    """
        位置前馈网络模块
    """

    def forward(self, x):
        pass


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention()  # 待实现
        self.feed_forward = PositionWiseFeedForward()  # 待实现
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        # 残差连接与层归一化将在 3.1.2.4 节中详细解释
        # 1. 多头自注意力
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))

        # 2. 前馈网络
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))

        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention()  # 待实现
        self.cross_attn = MultiHeadAttention()  # 待实现
        self.feed_forward = PositionWiseFeedForward()  # 待实现
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        # 1. 掩码多头自注意力 (对自己)
        attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))

        # 2. 交叉注意力 (对编码器输出)
        cross_attn_output = self.cross_attn(x, encoder_output, encoder_output, src_mask)
        x = self.norm2(x + self.dropout(cross_attn_output))

        # 3. 前馈网络
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))

        return x
