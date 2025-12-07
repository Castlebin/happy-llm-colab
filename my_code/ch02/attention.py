import torch
import math

"""
注意力计算函数

注意力机制的本质是对两段序列的元素依次进行相似度计算，
寻找出一个序列的每个元素对另一个序列的每个元素的相关度，
然后基于相关度进行加权，即分配注意力
"""
def attention(query, key, value, dropout=None):
    """
    query: 查询矩阵
    key: 键矩阵
    value: 值矩阵
    """

    # 获取键向量的维度。键向量的维度和值向量的维度相同
    d_k = query.size(-1)
    # 计算 Q 与 K 的内积，并除以 sqrt(d_k) 进行缩放   Q * K^T / sqrt(d_k)
    # transpose(-2, -1) 将最后两个维度进行交换。在这里，-2 代表序列长度维度，-1 代表特征维度
    # 所以 transpose 在这里就是矩阵转置
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    # 对 scores 应用 softmax 函数，得到注意力权重
    p_attn = torch.softmax(scores, dim=-1)

    # 如果提供了 dropout，则对注意力权重应用 dropout
    if dropout is not None:
        p_attn = dropout(p_attn)

    # 计算加权和，得到最终的注意力输出
    # 返回加权和结果以及注意力权重
    return torch.matmul(p_attn, value), p_attn

