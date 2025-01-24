import torch
from torchvision import datasets, transforms
from neural_networks.models.senet import SENet
from safetensors.torch import load_model
import os
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
        image = F.to_pil_image(image)
    
    # 保存图片
    image.save(save_path)
    print(f"图片已保存: {save_path}")

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
    
    # 创建模型
    model = SENet(
        layers=[2, 2, 2, 2],
        num_classes=10,
        in_channels=1
    ).to(device)
    
    # 加载模型权重
    model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../models/mnist/se'))
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
    
    # 创建临时目录
    temp_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'temp_mnist'))
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
        
        # 打印预处理信息
        print("\n" + "="*50)
        print("预处理信息:")
        
        # 获取原始图片
        raw_image = test_dataset.data[index]
        print(f"原始图片: shape={raw_image.shape}, dtype={raw_image.dtype}")
        print(f"值范围: [{raw_image.min()}, {raw_image.max()}]")
        print(f"均值: {raw_image.float().mean():.3f}")
        
        # 转换为tensor后
        # 先转换为PIL Image，再转回tensor
        pil_image = transforms.ToPILImage()(raw_image)
        temp_tensor = transforms.ToTensor()(pil_image)
        print(f"\n转换为tensor后:")
        print(f"Shape: {temp_tensor.shape}")
        print(f"值范围: [{temp_tensor.min():.3f}, {temp_tensor.max():.3f}]")
        print(f"均值: {temp_tensor.mean():.3f}")
        
        # 最终预处理后
        print(f"\n归一化后:")
        print(f"Shape: {image.shape}")
        print(f"值范围: [{image.min():.3f}, {image.max():.3f}]")
        print(f"均值: {image.mean():.3f}")
        print(f"标准差: {image.std():.3f}")
        print("="*50 + "\n")
        
        # 使用预测函数
        prediction, confidence, _ = predict_single_image(model, device, image, 'MNIST')
        
        # 保存预测图片
        save_prediction_image(pil_image, prediction, actual, temp_dir, index)
        
        print(f"图片 {index}: 预测值 = {prediction}, 实际值 = {actual}")
        print(f"置信度: {confidence:.2%}")
        
        if prediction != actual:
            print(f"警告：预测错误！")

if __name__ == '__main__':
    main() 