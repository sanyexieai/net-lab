import torch
import torch.nn as nn
from neural_networks.layers.se import SELayer
import torch.nn.functional as F

class SEBasicBlock(nn.Module):
    """SENet的基本构建块
    
    在ResNet的基本块基础上增加了SE层，结构为：
    Conv -> BN -> ReLU -> Conv -> BN -> SE -> Add -> ReLU
    
    参数:
        inplanes (int): 输入通道数
        planes (int): 输出通道数
        stride (int): 卷积步长，用于下采样
        reduction (int): SE模块的压缩比例
        downsample: 用于残差连接的下采样模块
    """
    expansion = 1  # 输出通道数相对于planes的倍数

    def __init__(self, inplanes: int, planes: int, stride: int = 1, 
                 reduction: int = 16, downsample=None):
        super().__init__()
        # 第一个卷积块：通道数变换，可能包含下采样
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        
        # 第二个卷积块：保持通道数不变
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, 
                              padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        
        # SE注意力模块
        self.se = SELayer(planes, reduction)
        
        # 下采样模块（用于残差连接）
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x  # 保存输入，用于残差连接

        # 主路径
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        
        # 应用SE注意力机制
        out = self.se(out)

        # 残差连接
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)

        return out

class SENet(nn.Module):
    """SENet-18 网络结构
    
    基于ResNet-18架构，但使用SE模块增强特征表达。网络结构：
    1. 初始层：conv7x7 -> bn -> relu -> maxpool
    2. 4个阶段的SE块，每个阶段通道数翻倍，空间尺寸减半
    3. 全局平均池化和全连接分类层
    
    参数:
        block: 基本构建块类型（默认为SEBasicBlock）
        layers: 每个阶段的块数量
        num_classes: 分类类别数
        in_channels: 输入通道数（默认为3，RGB图像）
    """
    def __init__(self, block=SEBasicBlock, layers=[2, 2, 2, 2], 
                 num_classes: int = 1000, in_channels: int = 3):
        super().__init__()
        self.inplanes = 64
        
        # 第一个卷积层，可以处理不同的输入通道数
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 四个阶段的SE层，通道数依次为64,128,256,512
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # 分类头
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes: int, blocks: int, 
                   stride: int = 1) -> nn.Sequential:
        """创建包含多个SE块的层
        
        参数:
            block: 块类型
            planes: 基础通道数
            blocks: 块的数量
            stride: 第一个块的步长（用于下采样）
        """
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                         kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        # 第一个块可能需要下采样
        layers.append(block(self.inplanes, planes, stride, downsample=downsample))
        self.inplanes = planes * block.expansion
        # 后续块保持特征图尺寸不变
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 初始特征提取
        print(f"输入: {x.shape}")
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        print(f"conv1 -> bn1 -> relu 输出: {x.shape}")
        
        x = self.maxpool(x)
        print(f"maxpool 输出: {x.shape}")
        
        # 四个SE阶段
        x = self.layer1(x)
        print(f"layer1 输出: {x.shape}")
        x = self.layer2(x)
        print(f"layer2 输出: {x.shape}")
        x = self.layer3(x)
        print(f"layer3 输出: {x.shape}")
        x = self.layer4(x)
        print(f"layer4 输出: {x.shape}")
        
        # 分类头
        x = self.avgpool(x)
        print(f"avgpool 输出: {x.shape}")
        x = torch.flatten(x, 1)
        print(f"flatten 输出: {x.shape}")
        x = self.fc(x)
        print(f"fc 输出: {x.shape}")
        
        return x 