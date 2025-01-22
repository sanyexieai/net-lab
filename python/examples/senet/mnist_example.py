import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from neural_networks.models.senet import SENet
import os

def main():
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 设置数据集保存路径 - 使用绝对路径
    data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data'))
    os.makedirs(data_path, exist_ok=True)
    
    print(f"数据集路径: {data_path}")
    
    # 加载 MNIST 数据集
    transform = transforms.Compose([
        transforms.Resize(224),  # SENet 期望 224x224 输入
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # 检查数据集文件是否已存在
    mnist_file = os.path.join(data_path, 'MNIST', 'raw', 'train-images-idx3-ubyte')
    if os.path.exists(mnist_file):
        print(f"发现已下载的数据集: {mnist_file}")
        download = False
    else:
        print(f"数据集不存在，将下载到: {data_path}")
        download = True
    
    train_dataset = datasets.MNIST(data_path, train=True, download=download, transform=transform)
    test_dataset = datasets.MNIST(data_path, train=False, download=download, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32)
    
    # 创建模型
    model = SENet(
        layers=[2, 2, 2, 2],  # SENet-18 配置
        num_classes=10  # MNIST 有 10 个类别
    ).to(device)
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 训练模型
    num_epochs = 5
    for epoch in range(num_epochs):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            if batch_idx % 100 == 0:
                print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                      f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
    
    # 测试模型
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    test_loss /= len(test_loader)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f'\nTest set: Average loss: {test_loss:.4f}, '
          f'Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)')

if __name__ == '__main__':
    main() 