import torch
import pytest
from neural_networks.models.senet import SENet, SEBasicBlock, SELayer

@pytest.fixture
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TestSELayer:
    def test_initialization(self):
        channel = 64
        reduction = 16
        layer = SELayer(channel, reduction)
        
        # 检查结构
        assert isinstance(layer.avg_pool, torch.nn.AdaptiveAvgPool2d)
        assert isinstance(layer.fc, torch.nn.Sequential)
        assert len(layer.fc) == 4  # Linear -> ReLU -> Linear -> Sigmoid
        
        # 检查通道压缩
        fc1 = layer.fc[0]
        fc2 = layer.fc[2]
        assert fc1.in_features == channel
        assert fc1.out_features == channel // reduction
        assert fc2.in_features == channel // reduction
        assert fc2.out_features == channel

    def test_forward(self, device):
        batch_size, channels = 2, 64
        height, width = 32, 32
        layer = SELayer(channels).to(device)
        x = torch.randn(batch_size, channels, height, width).to(device)
        
        output = layer(x)
        
        # 检查输出尺寸
        assert output.shape == x.shape
        # 检查输出范围（因为有 sigmoid，所以值应该在 0-1 之间）
        assert torch.all(output <= x * 1.0)
        assert torch.all(output >= x * 0.0)

class TestSEBasicBlock:
    def test_initialization(self):
        inplanes, planes = 64, 128
        block = SEBasicBlock(inplanes, planes, stride=2)
        
        # 检查基本组件
        assert isinstance(block.conv1, torch.nn.Conv2d)
        assert isinstance(block.bn1, torch.nn.BatchNorm2d)
        assert isinstance(block.conv2, torch.nn.Conv2d)
        assert isinstance(block.bn2, torch.nn.BatchNorm2d)
        assert isinstance(block.se, SELayer)
        
        # 检查下采样
        assert isinstance(block.downsample, torch.nn.Sequential)
        assert len(block.downsample) == 2  # Conv2d + BatchNorm2d

    def test_forward(self, device):
        inplanes, planes = 64, 128
        batch_size = 2
        height, width = 32, 32
        
        block = SEBasicBlock(inplanes, planes, stride=2).to(device)
        x = torch.randn(batch_size, inplanes, height, width).to(device)
        
        output = block(x)
        
        # 检查输出尺寸（stride=2 会减半空间维度）
        expected_shape = (batch_size, planes, height//2, width//2)
        assert output.shape == expected_shape

class TestSENet:
    def test_initialization(self):
        model = SENet(SEBasicBlock, [2, 2, 2, 2], num_classes=1000)
        
        # 检查主要组件
        assert isinstance(model.conv1, torch.nn.Conv2d)
        assert isinstance(model.bn1, torch.nn.BatchNorm2d)
        assert isinstance(model.layer1, torch.nn.Sequential)
        assert isinstance(model.layer2, torch.nn.Sequential)
        assert isinstance(model.layer3, torch.nn.Sequential)
        assert isinstance(model.layer4, torch.nn.Sequential)
        assert isinstance(model.fc, torch.nn.Linear)

    def test_forward(self, device):
        batch_size = 2
        model = SENet(SEBasicBlock, [2, 2, 2, 2], num_classes=1000).to(device)
        x = torch.randn(batch_size, 3, 224, 224).to(device)
        
        output = model(x)
        
        # 检查输出尺寸
        assert output.shape == (batch_size, 1000)

    @pytest.mark.parametrize("num_classes", [10, 100, 1000])
    def test_different_num_classes(self, num_classes, device):
        batch_size = 2
        model = SENet(SEBasicBlock, [2, 2, 2, 2], num_classes=num_classes).to(device)
        x = torch.randn(batch_size, 3, 224, 224).to(device)
        
        output = model(x)
        
        assert output.shape == (batch_size, num_classes)

    def test_feature_map_sizes(self, device):
        model = SENet(SEBasicBlock, [2, 2, 2, 2]).to(device)
        x = torch.randn(1, 3, 224, 224).to(device)
        
        # 检查每一层的特征图大小
        x = model.conv1(x)
        assert x.shape == (1, 64, 112, 112)  # 第一次下采样
        
        x = model.maxpool(model.relu(model.bn1(x)))
        assert x.shape == (1, 64, 56, 56)    # 最大池化下采样
        
        x = model.layer1(x)
        assert x.shape == (1, 64, 56, 56)    # 保持尺寸
        
        x = model.layer2(x)
        assert x.shape == (1, 128, 28, 28)   # 下采样
        
        x = model.layer3(x)
        assert x.shape == (1, 256, 14, 14)   # 下采样
        
        x = model.layer4(x)
        assert x.shape == (1, 512, 7, 7)     # 下采样 