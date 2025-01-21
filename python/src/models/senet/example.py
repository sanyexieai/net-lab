import torch
from .senet import SENet

def main():
    # 创建 SENet-18
    model = SENet(layers=[2, 2, 2, 2], num_classes=1000)
    
    # 创建示例输入
    batch_size, channels, height, width = 1, 3, 224, 224
    x = torch.randn(batch_size, channels, height, width)
    
    # 前向传播
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")

if __name__ == "__main__":
    main() 