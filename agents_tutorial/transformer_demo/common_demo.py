import torch
import torch.nn as nn
import math

"""
残差连接与层归一化

在 Transformer 的每个编码器和解码器层中，所有子模块（如多头注意力和前馈网络）都被一个 Add & Norm 操作包裹。
这个组合是为了保证 Transformer 能够稳定训练。
    残差连接 (Add)：该操作将子模块的输入 x 直接加到该子模块的输出 Sublayer(x) 上。
这一结构解决了深度神经网络中的梯度消失 (Vanishing Gradients) 问题。在反向传播时，梯度可以绕过子模块直接向前传播，从而保证了即使网络层数
很深，模型也能得到有效的训练。其公式可以表示为：Output=x+Sublayer(x)。
    层归一化 (Norm)：该操作对单个样本的所有特征进行归一化，使其均值为0，方差为1。这解决了模型训练过程中的内部协变量偏移 
(Internal Covariate Shift) 问题，使每一层的输入分布保持稳定，从而加速模型收敛并提高训练的稳定性。


位置编码

我们已经了解，Transformer 的核心是自注意力机制，它通过计算序列中任意两个词元之间的关系来捕捉依赖。然而，这种计算方式有一个固有的问题：
它本身不包含任何关于词元顺序或位置的信息。对于自注意力来说，“agent learns” 和 “learns agent” 这两个序列是完全等价的，
因为它只关心词元之间的关系，而忽略了它们的排列。为了解决这个问题，Transformer 引入了位置编码 (Positional Encoding) 。
位置编码的核心思想是，为输入序列中的每一个词元嵌入向量，都额外加上一个能代表其绝对位置和相对位置信息的“位置向量”。
这个位置向量不是通过学习得到的，而是通过一个固定的数学公式直接计算得出。这样一来，即使两个词元（例如，两个都叫 agent 的词元）
自身的嵌入是相同的，但由于它们在句子中的位置不同，它们最终输入到 Transformer 模型中的向量就会因为加上了不同的位置编码而变得独一无二。
原论文中提出的位置编码使用正弦和余弦函数来生成，其公式如下：
    
"""


class PositionalEncoding(nn.Module):
    """
        位置编码模块
        位置编码使用正弦和余弦函数生成，公式如下：
        PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
        其中 pos 是位置，i 是维度索引，d_model 是模型维度。
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        # 创建一个足够长的位置编码矩阵
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        # pe (positional encoding) 的大小为 (max_len, d_model)
        pe = torch.zeros(max_len, d_model)

        # 偶数维度使用 sin, 奇数维度使用 cos
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # 将 pe 注册为 buffer，这样它就不会被视为模型参数，但会随模型移动（例如 to(device)）
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        # x.size(1) 是当前输入的序列长度
        # 将位置编码加到输入向量上
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


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
        前馈神经网络
    在每个 Encoder 和 Decoder 层中，多头注意力子层之后都跟着一个逐位置前馈网络(Position-wise Feed-Forward Network, FFN) 。
    如果说注意力层的作用是从整个序列中“动态地聚合”相关信息，那么前馈网络的作用从这些聚合后的信息中提取更高阶的特征。
    这个名字的关键在于“逐位置”。它意味着这个前馈网络会独立地作用于序列中的每一个词元向量。换句话说，对于一个长度为 seq_len 的序列，
    这个 FFN 实际上会被调用 seq_len 次，每次处理一个词元。重要的是，所有位置共享的是同一组网络权重。这种设计既保持
    了对每个位置进行独立加工的能力，又大大减少了模型的参数量。这个网络的结构非常简单，由两个线性变换和一个 ReLU 激活函数组成：
      FFN(x) = max(0, xW_1 + b_1)W_2 + b_2
    其中，x是注意力子层的输出。W_1、b_1、W_2、b_2是可学习的参数。通常，第一个线性层的输出维度 d_ff 会远大于输入的维度 d_model（例如 d_ff = 4 * d_model），
    经过 ReLU 激活后再通过第二个线性层映射回 d_model 维度。这种“先扩大再缩小”的模式，被认为有助于模型学习更丰富的特征表示。
    """

    def __init__(self, hidden_size, d_ff, dropout=0.1):
        super(PositionWiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(hidden_size, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, hidden_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x 形状: (batch_size, seq_len, hidden_size)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        # 最终输出形状: (batch_size, seq_len, hidden_size)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, hidden_size, num_heads, d_ff, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention()  # 待实现
        self.feed_forward = PositionWiseFeedForward()  # 待实现
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
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
