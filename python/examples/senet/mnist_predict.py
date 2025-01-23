import torch
from torchvision import datasets, transforms
from neural_networks.models.senet import SENet
from safetensors.torch import load_model
import os
import random

def predict_single_image(model, device, dataset, index=None):
    """预测单张图片"""
    if index is None:
        index = random.randint(0, len(dataset) - 1)
    
    image, label = dataset[index]
    image = image.unsqueeze(0).to(device)  # 添加 batch 维度
    
    model.eval()
    with torch.no_grad():
        output = model(image)
        pred = output.argmax(dim=1, keepdim=True)
    
    return pred.item(), label, index

def main():
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 加载数据集
    data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data'))
    transform = transforms.Compose([
        transforms.Resize(224),  # SENet 期望 224x224 输入
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # 单通道归一化
    ])
    
    test_dataset = datasets.MNIST(data_path, train=False, transform=transform)
    
    # 加载模型
    model = SENet(
        layers=[2, 2, 2, 2],
        num_classes=10,
        in_channels=1
    ).to(device)
    
    # 加载模型权重
    model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../models/mnist'))
    model_path_safetensors = os.path.join(model_dir, 'senet_mnist.safetensors')
    model_path_pth = os.path.join(model_dir, 'senet_mnist.pth')
    
    if os.path.exists(model_path_safetensors):
        print(f"加载模型 (safetensors): {model_path_safetensors}")
        load_model(model, model_path_safetensors)
    elif os.path.exists(model_path_pth):
        print(f"加载模型 (PyTorch): {model_path_pth}")
        model.load_state_dict(torch.load(model_path_pth))
    else:
        raise FileNotFoundError("未找到训练好的模型文件！")
    
    # 预测多张图片
    num_predictions = 5
    print(f"\n预测 {num_predictions} 张随机图片:")
    
    for i in range(num_predictions):
        prediction, actual, index = predict_single_image(model, device, test_dataset)
        print(f"图片 {index}: 预测值 = {prediction}, 实际值 = {actual}")
        
        if prediction != actual:
            print(f"警告：预测错误！")

if __name__ == '__main__':
    main() 