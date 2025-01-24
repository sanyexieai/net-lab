import torch
import torch.nn as nn
from torch.nn import init

class SKLayer(nn.Module):
    """Selective Kernel Networks (SK层)
    
    通过自适应地融合不同卷积核大小的特征图来增强网络的表示能力。
    
    工作流程:
    1. Split: 使用不同大小的卷积核处理输入特征
    2. Fuse: 将不同分支的特征进行融合，获取全局信息
    3. Select: 为每个分支生成注意力权重，动态选择最优的特征组合
    
    参数:
        channels (int): 输入特征图的通道数
        branches (int): 分支数量，即使用不同卷积核的数量，默认为2
        reduction (int): 通道数的压缩比例，用于降低计算量，默认为16
        stride (int): 卷积步长，默认为1
    """
    def __init__(self, channels: int, branches: int = 2, reduction: int = 16, stride: int = 1):
        super().__init__()
        
        self.branches = branches
        
        # 创建多个不同kernel size的卷积分支，并考虑stride
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=3+i*2, 
                         stride=stride, padding=(3+i*2)//2,
                         groups=32, bias=False),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True)
            ) for i in range(branches)
        ])
        
        # 全局信息融合
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # 特征融合和选择模块
        d = max(channels // reduction, 32)
        self.fc = nn.Linear(channels, d)
        self.fcs = nn.ModuleList([nn.Linear(d, channels) for _ in range(branches)])
        self.softmax = nn.Softmax(dim=1)
        
        # 初始化权重
        self.init_weights()
        
    def init_weights(self):
        """初始化模块的权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播
        
        参数:
            x (torch.Tensor): 输入特征图，形状为 (batch_size, channels, height, width)
            
        返回:
            torch.Tensor: 经过选择性核注意力调整后的特征图，形状与输入相同
        """
        batch_size = x.size(0)
        
        # Split: 通过不同卷积核处理输入
        feats = [conv(x) for conv in self.convs]
        
        # Fuse: 特征融合
        feats_U = torch.stack(feats, dim=1)
        feats_U = feats_U.mean(dim=1)
        feats_S = self.gap(feats_U).view(batch_size, -1)
        
        # Select: 生成注意力权重
        feats_Z = self.fc(feats_S)
        attention_weights = []
        for fc in self.fcs:
            attention_weights.append(fc(feats_Z))
        attention_weights = torch.stack(attention_weights, dim=1)
        attention_weights = self.softmax(attention_weights)
        
        # 将注意力权重应用到各个分支
        feats_V = torch.stack(feats, dim=1)  # (B, branches, C, H, W)
        attention_weights = attention_weights.unsqueeze(-1).unsqueeze(-1)
        feats_V = feats_V * attention_weights
        
        # 融合所有分支的输出
        return feats_V.sum(dim=1) 