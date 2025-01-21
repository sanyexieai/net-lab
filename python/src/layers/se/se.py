import torch
import torch.nn as nn
from torch.nn import init

class SELayer(nn.Module):
    """Squeeze-and-Excitation Layer (SE层)
    
    通过自适应地重新校准通道特征响应来提高网络的表示能力。
    
    工作流程:
    1. Squeeze: (B,C,H,W) -> (B,C,1,1) -> (B,C)
       通过全局平均池化压缩空间维度，获取每个通道的全局信息
    2. Excitation: (B,C) -> (B,C) -> (B,C,1,1)
       通过两个全连接层学习通道间的非线性关系，生成通道权重
    3. Scale: (B,C,H,W) * (B,C,1,1) -> (B,C,H,W)
       将学习到的通道权重应用到特征图上，增强重要通道，抑制不重要通道
    
    参数:
        channel (int): 输入特征图的通道数
        reduction (int): 通道数的压缩比例，用于降低计算量，默认为16
    """
    def __init__(self, channel: int, reduction: int = 16):
        super().__init__()
        # 全局平均池化层，将每个通道的特征图压缩为一个数值
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # 两个全连接层的序列，用于学习通道间的关系
        self.fc = nn.Sequential(
            # 第一个全连接层：降维，减少参数量
            nn.Linear(channel, channel // reduction, bias=False),
            # ReLU激活函数
            nn.ReLU(inplace=True),
            # 第二个全连接层：升维，恢复通道数
            nn.Linear(channel // reduction, channel, bias=False),
            # Sigmoid激活函数，将输出压缩到0-1之间作为权重
            nn.Sigmoid()
        )
        # 初始化权重
        self.init_weights()

    def init_weights(self):
        """初始化模块的权重
        
        使用较小的标准差(0.001)初始化权重，确保SE模块在训练初期影响较小，
        让网络能够逐渐学习通道间的关系。
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # 使用正态分布初始化权重，标准差为0.001
                init.normal_(m.weight, std=0.001)
                # 如果有偏置项，初始化为0
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播
        
        参数:
            x (torch.Tensor): 输入特征图，形状为 (batch_size, channels, height, width)
            
        返回:
            torch.Tensor: 经过通道注意力调整后的特征图，形状与输入相同
        """
        # 获取输入的形状信息
        b, c, _, _ = x.size()
        # Squeeze: 空间维度压缩，得到通道描述符
        y = self.avg_pool(x).view(b, c)
        # Excitation: 通过全连接层学习通道权重
        y = self.fc(y).view(b, c, 1, 1)
        # Scale: 将权重应用到原始特征图上
        return x * y.expand_as(x) 