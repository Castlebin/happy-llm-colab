from transformers import PretrainedConfig
import torch
from torch import nn

# 定义模型配置类（模型的超参数配置类）
class ModelConfig(PretrainedConfig):
    model_type = "Tiny-K"

    def __init__(
        self,
        dim: int = 768, # 模型维度
        n_layers: int = 12, # Transformer 的层数
        n_heads: int = 16, # 多头注意力的头数
        n_kv_heads: int = 8, # 键值注意力的头数
        vocab_size: int = 6144, # 词汇表大小
        hidden_dim: int = None, # 隐藏层维度
        multiple_of: int = 64, # 模块维度
        norm_eps: float = 1e-5, # 归一化层的 epsilon
        max_seq_len: int = 512, # 最大序列长度
        dropout: float = 0.0, # Dropout 率
        flash_attn: bool = True, # 是否使用 Flash Attention
        **kwargs,
    ):
        self.dim = dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.multiple_of = multiple_of
        self.norm_eps = norm_eps
        self.max_seq_len = max_seq_len
        self.dropout = dropout
        self.flash_attn = flash_attn

        super().__init__(**kwargs)
    

# 实现 RMSNorm
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
    
        # eps 是为了防止除以 0 的情况
        self.eps = eps
        # weight 是一个可学习的参数，全部初始化为 1
        self.weight = nn.Parameter(torch.ones(dim))
    
    def _norm(self, x):
        # 计算 RMSNorm 的核心部分
        # x.pow(2).mean(-1, keepdim=True) 计算 x 的平方的均值
        # torch.rsqrt 是平方根的倒数，这样就得到了 RMSNorm 的分母部分，再加上 eps 防止分母为 0
        # 最后乘以 x，得到 RMSNorm 的结果
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
    
    def forward(self, x):
        # forward 函数是模型的前向传播
        # 首先将输入 x 转为 float 类型，然后进行 RMSNorm，最后再转回原来的数据类型
        # 最后乘以 weight，这是 RMSNorm 的一个可学习的缩放因子
        output = self._norm(x.float()).type_as(x)
        return output * self.weight
        

#%% 测试一下 RMSNorm

def test_rmsnorm():
    # 创建一个 RMSNorm 层
    norm = RMSNorm(dim=768, eps=1e-5)
    # 创建一个随机输入
    x = torch.randn(1, 50, 768)
    print(x.shape)
    # 计算 RMSNorm
    output = norm(x)
    # 打印结果 (可以看到 RMSNorm 的输出形状与输入形状相同，没有改变输入的形状)
    print(output.shape)
    

if __name__ == "__main__":
    test_rmsnorm()


