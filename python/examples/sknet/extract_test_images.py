import torch
from torchvision import datasets, transforms
from PIL import Image
import os
import random

def extract_cifar10_images(num_images=10):
    """从CIFAR-10数据集提取测试图片
    
    参数:
        num_images: 每个类别提取的图片数量
    """
    # 设置数据集路径
    data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data'))
    save_path = os.path.join(data_path, 'test_images', 'cifar10')
    os.makedirs(save_path, exist_ok=True)
    
    # 加载CIFAR-10测试集
    test_dataset = datasets.CIFAR10(data_path, train=False, download=True)
    
    # 按类别组织图片
    class_images = [[] for _ in range(10)]
    for image, label in test_dataset:
        class_images[label].append(image)
    
    # CIFAR-10类别名称
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
              'dog', 'frog', 'horse', 'ship', 'truck']
    
    # 从每个类别随机选择并保存图片
    for label, images in enumerate(class_images):
        class_name = classes[label]
        class_dir = os.path.join(save_path, class_name)
        os.makedirs(class_dir, exist_ok=True)
        
        # 随机选择图片
        selected_images = random.sample(images, min(num_images, len(images)))
        
        # 保存图片
        for i, image in enumerate(selected_images):
            image_path = os.path.join(class_dir, f'{class_name}_{i+1}.png')
            image.save(image_path)
            print(f"保存CIFAR-10图片: {image_path}")

def extract_mnist_images(num_images=10):
    """从MNIST数据集提取测试图片
    
    参数:
        num_images: 每个数字提取的图片数量
    """
    # 设置数据集路径
    data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data'))
    save_path = os.path.join(data_path, 'test_images', 'mnist')
    os.makedirs(save_path, exist_ok=True)
    
    # 加载MNIST测试集
    test_dataset = datasets.MNIST(data_path, train=False, download=True)
    
    # 按数字组织图片
    digit_images = [[] for _ in range(10)]
    for image, label in test_dataset:
        digit_images[label].append(image)
    
    # 从每个数字类别随机选择并保存图片
    for digit, images in enumerate(digit_images):
        digit_dir = os.path.join(save_path, str(digit))
        os.makedirs(digit_dir, exist_ok=True)
        
        # 随机选择图片
        selected_images = random.sample(images, min(num_images, len(images)))
        
        # 保存图片
        for i, image in enumerate(selected_images):
            image_path = os.path.join(digit_dir, f'digit_{digit}_{i+1}.png')
            image.save(image_path)
            print(f"保存MNIST图片: {image_path}")

def main():
    # 设置随机种子以确保可重复性
    random.seed(42)
    
    print("开始提取CIFAR-10测试图片...")
    extract_cifar10_images(num_images=5)  # 每个类别提取5张图片
    
    print("\n开始提取MNIST测试图片...")
    extract_mnist_images(num_images=5)  # 每个数字提取5张图片
    
    print("\n图片提取完成！")
    print("CIFAR-10图片保存在: data/test_images/cifar10/")
    print("MNIST图片保存在: data/test_images/mnist/")

if __name__ == '__main__':
    main() 