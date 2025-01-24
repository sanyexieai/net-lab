import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from neural_networks.models.senet import SENet
from safetensors.torch import save_model, load_model
import os
import sys
# 添加项目根目录到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from models.model_performance import ModelPerformance

def main():
    print(f"PyTorch版本：{torch.__version__}")
    print(f"CUDA是否可用：{torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA版本：{torch.version.cuda}")
        print(f"GPU设备：{torch.cuda.get_device_name(0)}")
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 设置数据集保存路径 - 使用绝对路径
    data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data'))
    os.makedirs(data_path, exist_ok=True)
    
    print(f"数据集路径: {data_path}")
    
    # 检查压缩包和解压目录
    cifar_tar = os.path.join(data_path, 'cifar-10-python.tar.gz')
    cifar_dir = os.path.join(data_path, 'cifar-10-batches-py')
    
    # 如果压缩包存在但损坏，删除它
    if os.path.exists(cifar_tar):
        try:
            print(f"发现压缩包: {cifar_tar}")
            if not os.path.exists(cifar_dir):
                print("压缩包未解压，尝试手动解压...")
                import tarfile
                with tarfile.open(cifar_tar, 'r:gz') as tar:
                    tar.extractall(data_path)
                print("手动解压完成")
        except (EOFError, tarfile.ReadError) as e:
            print(f"压缩包损坏: {e}")
            print("删除损坏的压缩包...")
            os.remove(cifar_tar)
            if os.path.exists(cifar_dir):
                import shutil
                print("删除不完整的解压目录...")
                shutil.rmtree(cifar_dir)
    
    # 加载 CIFAR-10 数据集
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    print("准备数据集...")
    train_dataset = datasets.CIFAR10(data_path, train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(data_path, train=False, download=True, transform=transform)
    print("数据集准备完成")
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32)
    
    # 创建模型
    model = SENet(
        layers=[2, 2, 2, 2],  # SENet-18 配置
        num_classes=10,       # CIFAR-10 有 10 个类别
        in_channels=3         # CIFAR-10 是 RGB 图像
    ).to(device)
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 创建保存模型的目录
    model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../models/cifar/se'))
    os.makedirs(model_dir, exist_ok=True)
    model_path_pth = os.path.join(model_dir, 'senet_cifar.pth')
    model_path_safetensors = os.path.join(model_dir, 'senet_cifar.safetensors')
    
    # 如果存在已训练的模型，则加载
    if os.path.exists(model_path_safetensors):
        print(f"加载已训练的模型 (safetensors): {model_path_safetensors}")
        load_model(model, model_path_safetensors)
    elif os.path.exists(model_path_pth):
        print(f"加载已训练的模型 (PyTorch): {model_path_pth}")
        model.load_state_dict(torch.load(model_path_pth))
    else:
        print("开始训练新模型...")
        # 创建性能记录器
        perf = ModelPerformance()
        
        # 训练模型
        num_epochs = 5
        best_acc = 0.0
        
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
            
            # 在每个epoch后评估并保存最佳模型
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
            
            print(f'\nEpoch {epoch}: Test loss: {test_loss:.4f}, '
                  f'Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)')
            
            # 评估后记录性能
            is_best = accuracy > best_acc
            
            # 更新性能记录
            perf.update_training_record(
                model_type="senet",
                dataset="cifar10",
                epochs=epoch + 1,
                accuracy=accuracy,
                is_best=is_best
            )
            
            if is_best:
                best_acc = accuracy
                print(f"保存最佳模型，准确率: {accuracy:.2f}%")
                # 保存 PyTorch 格式
                torch.save(model.state_dict(), model_path_pth)
                # 保存 safetensors 格式
                save_model(model, model_path_safetensors)
                print(f"模型已保存为 PyTorch 格式: {model_path_pth}")
                print(f"模型已保存为 safetensors 格式: {model_path_safetensors}")
        
        # 打印性能总结
        perf.print_summary()
    
    # 最终测试
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