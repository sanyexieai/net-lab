import torch
import torch.nn as nn
from ...layers.sk import SKLayer

class SKNet(nn.Module):
    """SKNet (Selective Kernel Networks)
    
    基于选择性卷积核的深度神经网络。通过自适应地选择不同大小的卷积核，
    增强网络对多尺度特征的建模能力。
    
    参数:
        num_classes (int): 分类类别数
        input_channels (int): 输入图像的通道数，默认为3
        base_channels (int): 基础通道数，默认为64
        layers (list): 每个阶段的基本块数量，默认为[3, 4, 6, 3]
        branches (int): SK模块的分支数，默认为2
    """
    def __init__(self, num_classes: int, input_channels: int = 3,
                 base_channels: int = 64, layers: list = [3, 4, 6, 3],
                 branches: int = 2):
        super().__init__()
        
        self.base_channels = base_channels
        
        # 初始卷积层
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, base_channels, kernel_size=7, stride=2,
                     padding=3, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # 构建主干网络的四个阶段
        self.stage1 = self._make_layer(base_channels, base_channels, layers[0],
                                     branches, stride=1)
        self.stage2 = self._make_layer(base_channels, base_channels * 2, layers[1],
                                     branches, stride=2)
        self.stage3 = self._make_layer(base_channels * 2, base_channels * 4,
                                     layers[2], branches, stride=2)
        self.stage4 = self._make_layer(base_channels * 4, base_channels * 8,
                                     layers[3], branches, stride=2)
        
        # 全局平均池化和分类器
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(base_channels * 8, num_classes)
        
        # 初始化权重
        self._initialize_weights()
        
    def _make_layer(self, in_channels: int, out_channels: int, blocks: int,
                    branches: int, stride: int = 1) -> nn.Sequential:
        """构建网络层
        
        参数:
            in_channels (int): 输入通道数
            out_channels (int): 输出通道数
            blocks (int): 块的数量
            branches (int): SK模块的分支数
            stride (int): 第一个块的步长
        """
        layers = []
        
        # 添加第一个块（可能需要降采样）
        layers.append(SKBottleneck(in_channels, out_channels, branches, stride))
        
        # 添加剩余的块
        for _ in range(1, blocks):
            layers.append(SKBottleneck(out_channels, out_channels, branches))
            
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        """初始化网络权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播
        
        参数:
            x (torch.Tensor): 输入图像张量，形状为 (B, C, H, W)
            
        返回:
            torch.Tensor: 分类结果，形状为 (B, num_classes)
        """
        x = self.conv1(x)
        x = self.maxpool(x)
        
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x


class SKBottleneck(nn.Module):
    """SKNet的基本构建块
    
    包含SK模块的残差块，用于自适应地选择不同尺度的特征。
    
    参数:
        in_channels (int): 输入通道数
        out_channels (int): 输出通道数
        branches (int): SK模块的分支数
        stride (int): 步长，默认为1
        reduction (int): 通道压缩比例，默认为16
    """
    expansion = 1
    
    def __init__(self, in_channels: int, out_channels: int, branches: int,
                 stride: int = 1, reduction: int = 16):
        super().__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                     stride=1, bias=False),  # 第一个卷积不做下采样
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # SK模块需要考虑stride
        self.sk = SKLayer(out_channels, branches=branches, 
                         reduction=reduction, stride=stride)
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1, 
                     stride=1, bias=False),  # 最后的1x1卷积不做下采样
            nn.BatchNorm2d(out_channels)
        )
        
        self.relu = nn.ReLU(inplace=True)
        
        # shortcut连接
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                         stride=stride, bias=False),  # 在shortcut上进行下采样
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播
        
        参数:
            x (torch.Tensor): 输入特征图
            
        返回:
            torch.Tensor: 输出特征图
        """
        identity = self.shortcut(x)
        
        out = self.conv1(x)
        out = self.sk(out)  # SK模块会处理stride
        out = self.conv2(out)
        
        out += identity
        out = self.relu(out)
        
        return out 