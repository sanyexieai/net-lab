import torch
from torchvision import datasets, transforms
from neural_networks.models.senet import SENet
from safetensors.torch import load_model
import os
import random
import shutil
from PIL import Image
import torchvision.transforms.functional as F
from predict_utils import predict_single_image

def save_prediction_image(image, prediction, actual, save_dir, index):
    """保存预测图片"""
    # 创建文件名
    filename = f"{index}_pred{prediction}_true{actual}.png"
    save_path = os.path.join(save_dir, filename)
    
    # 如果是tensor，转换为PIL图像
    if isinstance(image, torch.Tensor):
        # 对于CIFAR-10，需要反归一化但不需要反色
        image = image * 0.5 + 0.5  # 反归一化 (x * std + mean)
        image = transforms.ToPILImage()(image)
    
    # 保存图片
    image.save(save_path)
    print(f"图片已保存: {save_path}")

def get_class_name(label):
    """获取CIFAR-10类别名称"""
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
              'dog', 'frog', 'horse', 'ship', 'truck']
    return classes[label]

def main():
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 加载数据集
    data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data'))
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    test_dataset = datasets.CIFAR10(data_path, train=False, download=True, transform=transform)
    
    # 加载模型
    model = SENet(
        layers=[2, 2, 2, 2],
        num_classes=10,
        in_channels=3
    ).to(device)
    
    # 加载模型权重
    model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../models/cifar'))
    model_path_safetensors = os.path.join(model_dir, 'senet_cifar.safetensors')
    model_path_pth = os.path.join(model_dir, 'senet_cifar.pth')
    
    if os.path.exists(model_path_safetensors):
        print(f"加载模型 (safetensors): {model_path_safetensors}")
        load_model(model, model_path_safetensors)
    elif os.path.exists(model_path_pth):
        print(f"加载模型 (PyTorch): {model_path_pth}")
        model.load_state_dict(torch.load(model_path_pth))
    else:
        raise FileNotFoundError("未找到训练好的模型文件！")
    
    # 创建临时目录
    temp_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'temp_cifar'))
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir)
    print(f"临时目录: {temp_dir}")
    
    # 预测
    num_predictions = 1
    print(f"\n预测 {num_predictions} 张随机图片:")
    
    for i in range(num_predictions):
        # 随机选择一张图片
        index = torch.randint(0, len(test_dataset), (1,)).item()
        image, actual = test_dataset[index]
        
        # 使用通用预测函数
        prediction, confidence, _ = predict_single_image(model, device, image, 'CIFAR-10')
        
        # 保存预测图片
        save_prediction_image(image, prediction, actual, temp_dir, index)
        
        print(f"图片 {index}:")
        print(f"预测类别 = {get_class_name(prediction)}")
        print(f"实际类别 = {get_class_name(actual)}")
        print(f"置信度: {confidence:.2%}")
        
        if prediction != actual:
            print("警告：预测错误！")

if __name__ == '__main__':
    main() 