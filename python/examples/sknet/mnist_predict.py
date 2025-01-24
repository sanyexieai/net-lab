import torch
from torchvision import transforms
from PIL import Image
import os
from neural_networks.models.sknet import SKNet
from safetensors.torch import load_model

def predict_image(model_path: str, image_path: str, device: str = 'cuda') -> tuple:
    """使用SKNet模型预测单张MNIST图片
    
    参数:
        model_path: 模型文件路径
        image_path: 图片文件路径
        device: 使用的设备 ('cuda' 或 'cpu')
        
    返回:
        tuple: (预测数字, 预测概率)
    """
    # 数据预处理
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize(128),
        transforms.CenterCrop(128),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # 加载图片
    image = Image.open(image_path)
    image_tensor = transform(image).unsqueeze(0)
    
    # 创建模型
    model = SKNet(
        num_classes=10,
        input_channels=1,
        base_channels=64,
        layers=[2, 2, 2, 2],
        branches=2
    )
    
    # 加载模型权重
    if model_path.endswith('.safetensors'):
        load_model(model, model_path)
    else:
        model.load_state_dict(torch.load(model_path))
    
    # 设置设备和评估模式
    device = torch.device(device if torch.cuda.is_available() and device == 'cuda' else 'cpu')
    model = model.to(device)
    model.eval()
    
    # 预测
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        predicted_digit = torch.argmax(probabilities, dim=1).item()
        probability = probabilities[0][predicted_digit].item()
    
    return predicted_digit, probability

def predict_directory(model_path: str, directory: str, device: str = 'cuda'):
    """递归预测目录下的所有图片
    
    参数:
        model_path: 模型文件路径
        directory: 图片目录路径
        device: 使用的设备
    """
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(root, file)
                # 获取相对路径用于显示
                rel_path = os.path.relpath(image_path, directory)
                try:
                    predicted_digit, probability = predict_image(model_path, image_path, device)
                    print(f"\n图片: {rel_path}")
                    print(f"预测数字: {predicted_digit}")
                    print(f"预测概率: {probability:.2%}")
                except Exception as e:
                    print(f"\n处理图片 {rel_path} 时出错: {str(e)}")

def main():
    # 设置模型路径
    model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../models/mnist/sk'))
    model_path = os.path.join(model_dir, 'sknet_mnist.safetensors')
    
    # 检查模型文件
    if not os.path.exists(model_path):
        model_path = model_path.replace('.safetensors', '.pth')
        if not os.path.exists(model_path):
            print("错误：未找到训练好的模型文件")
            return
    
    # 获取测试图片路径
    image_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/test_images/mnist'))
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)
        print(f"请将测试图片放在此目录下：{image_dir}")
        return
    
    print(f"开始预测目录: {image_dir}")
    print("=" * 50)
    
    # 预测目录下的所有图片
    predict_directory(model_path, image_dir)

if __name__ == '__main__':
    main() 