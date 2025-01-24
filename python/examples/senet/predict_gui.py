import sys
import os
import torch
from torchvision import transforms
from PIL import Image
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QHBoxLayout, QPushButton, QLabel, QFileDialog, QComboBox)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap, QImage
import numpy as np
from neural_networks.models.senet import SENet
from safetensors.torch import load_model
from predict_utils import predict_single_image  # 直接从当前目录导入

class PredictWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.image = None
        self.image_path = None  # 添加图片路径存储
        self.model_paths = {
            'MNIST': {
                'safetensors': os.path.abspath(os.path.join(os.path.dirname(__file__), 
                    '../../models/mnist/se/senet_mnist.safetensors')),
                'pth': os.path.abspath(os.path.join(os.path.dirname(__file__), 
                    '../../models/mnist/se/senet_mnist.pth'))
            },
            'CIFAR-10': {
                'safetensors': os.path.abspath(os.path.join(os.path.dirname(__file__), 
                    '../../models/cifar/se/senet_cifar.safetensors')),
                'pth': os.path.abspath(os.path.join(os.path.dirname(__file__), 
                    '../../models/cifar/se/senet_cifar.pth'))
            }
        }
        self.loadDefaultModel()
    
    def initUI(self):
        self.setWindowTitle('SENet 图像分类预测')
        self.setGeometry(100, 100, 800, 600)
        
        # 创建中心部件和布局
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # 顶部控制栏
        control_layout = QHBoxLayout()
        
        # 模型选择下拉框
        self.model_combo = QComboBox()
        self.model_combo.addItems(['MNIST', 'CIFAR-10'])
        self.model_combo.currentTextChanged.connect(self.onModelChange)
        control_layout.addWidget(QLabel('选择模型:'))
        control_layout.addWidget(self.model_combo)
        
        # 加载图片按钮
        load_btn = QPushButton('加载图片', self)
        load_btn.clicked.connect(self.loadImage)
        control_layout.addWidget(load_btn)
        
        # 预测按钮
        predict_btn = QPushButton('预测', self)
        predict_btn.clicked.connect(self.predict)
        control_layout.addWidget(predict_btn)
        
        # 模型路径显示和选择
        path_layout = QHBoxLayout()
        path_layout.addWidget(QLabel('模型路径:'))
        self.path_label = QLabel()
        self.path_label.setStyleSheet("color: blue; text-decoration: underline; cursor: pointer;")
        self.path_label.mousePressEvent = self.selectModelPath
        path_layout.addWidget(self.path_label)
        
        layout.addLayout(control_layout)
        layout.addLayout(path_layout)
        
        # 图片显示区域
        image_layout = QHBoxLayout()
        
        # 原始图片
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMinimumSize(300, 300)
        self.image_label.setStyleSheet("border: 1px solid black")
        image_layout.addWidget(self.image_label)
        
        # 预测结果
        self.result_label = QLabel()
        self.result_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.result_label.setStyleSheet("font-size: 16px")
        image_layout.addWidget(self.result_label)
        
        layout.addLayout(image_layout)
    
    def loadDefaultModel(self):
        """加载默认模型（MNIST）"""
        self.loadModel('MNIST')
    
    def selectModelPath(self, event):
        """选择新的模型文件"""
        file_name, _ = QFileDialog.getOpenFileName(
            self, "选择模型文件", "", 
            "模型文件 (*.safetensors *.pth)")
        
        if file_name:
            model_type = self.model_combo.currentText()
            if file_name.endswith('.safetensors'):
                self.model_paths[model_type]['safetensors'] = file_name
            else:
                self.model_paths[model_type]['pth'] = file_name
            self.loadModel(model_type)
    
    def loadModel(self, model_type):
        """加载指定类型的模型"""
        if model_type == 'MNIST':
            in_channels = 1
        else:  # CIFAR-10
            in_channels = 3
        
        # 创建模型
        self.model = SENet(
            layers=[2, 2, 2, 2],
            num_classes=10,
            in_channels=in_channels
        ).to(self.device)
        
        # 加载模型权重
        model_paths = self.model_paths[model_type]
        loaded = False
        
        try:
            if os.path.exists(model_paths['safetensors']):
                load_model(self.model, model_paths['safetensors'])
                self.path_label.setText(model_paths['safetensors'])
                loaded = True
            elif os.path.exists(model_paths['pth']):
                self.model.load_state_dict(torch.load(model_paths['pth']))
                self.path_label.setText(model_paths['pth'])
                loaded = True
            
            if loaded:
                self.result_label.setText(f"已加载{model_type}模型")
                print(f"模型 {model_type} 已加载: {self.path_label.text()}")
            else:
                self.result_label.setText(f"错误：未找到{model_type}模型文件")
                self.path_label.setText("未找到模型文件")
        except Exception as e:
            self.result_label.setText(f"加载模型失败：{str(e)}")
            self.path_label.setText("加载失败")
            print(f"加载模型失败：{str(e)}")
    
    def onModelChange(self, model_type):
        """当模型选择改变时"""
        self.loadModel(model_type)
        self.image = None
        self.image_label.clear()
        self.result_label.setText(f"已切换到{model_type}模型")
    
    def loadImage(self):
        """加载图片并自动预测"""
        file_name, _ = QFileDialog.getOpenFileName(
            self, "选择图片", "", "图片文件 (*.png *.jpg *.jpeg *.bmp)")
        
        if file_name:
            # 保存图片路径
            self.image_path = file_name
            
            # 加载图片用于显示
            image = Image.open(file_name)
            
            # 转换为合适的格式用于显示
            if self.model_combo.currentText() == 'MNIST':
                if image.mode != 'L':
                    image = image.convert('L')
            else:
                if image.mode != 'RGB':
                    image = image.convert('RGB')
            
            # 显示图片
            if image.mode == 'L':
                qim = QImage(image.tobytes(), image.width, image.height, image.width, QImage.Format.Format_Grayscale8)
            else:
                qim = QImage(image.tobytes(), image.width, image.height, image.width * 3, QImage.Format.Format_RGB888)
            
            pixmap = QPixmap.fromImage(qim)
            scaled_pixmap = pixmap.scaled(300, 300, Qt.AspectRatioMode.KeepAspectRatio)
            self.image_label.setPixmap(scaled_pixmap)
            
            # 自动进行预测
            self.predict()
    
    def predict(self):
        """预测图片"""
        if self.image_path is None:
            self.result_label.setText("请先加载图片")
            return
        
        if self.model is None:
            self.result_label.setText("模型未加载")
            return
        
        try:
            model_type = self.model_combo.currentText()
            print("\n" + "="*50)
            print("调试信息:")
            
            # 创建debug目录（使用绝对路径）
            debug_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'debug_images'))
            print(f"调试目录: {debug_dir}")
            os.makedirs(debug_dir, exist_ok=True)
            
            # 从原始路径重新加载图片
            image = Image.open(self.image_path)
            print(f"1. 原始图片信息: mode={image.mode}, size={image.size}")
            
            # 预处理图像
            if model_type == 'MNIST':
                # 1. 转换为灰度图
                if image.mode != 'L':
                    image = image.convert('L')
                print(f"1. 转换为灰度图: mode={image.mode}, size={image.size}")
                
                # 2. 调整大小
                image = transforms.Resize(224)(image)
                print(f"2. 调整大小: size={image.size}")
                
                # 3. 转换为tensor
                img_tensor = transforms.ToTensor()(image)
                print(f"\n转换为tensor后:")
                print(f"值范围: [{img_tensor.min():.3f}, {img_tensor.max():.3f}]")
                print(f"均值: {img_tensor.mean():.3f}")
                
                # 4. 归一化
                img_tensor = transforms.Normalize((0.1307,), (0.3081,))(img_tensor)
                print(f"归一化后:")
                print(f"值范围: [{img_tensor.min():.3f}, {img_tensor.max():.3f}]")
                print(f"均值: {img_tensor.mean():.3f}")
                
                # 保存调试图片
                debug_tensor = img_tensor.clone()
                debug_tensor = debug_tensor * 0.3081 + 0.1307  # 反归一化
                debug_image = transforms.ToPILImage()(debug_tensor)
                debug_path = os.path.join(debug_dir, "final_processed.png")
                debug_image.save(debug_path)
                print(f"保存最终处理图片: {debug_path}")
                
            else:  # CIFAR-10
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                    print(f"2. 转换为RGB: mode={image.mode}, size={image.size}")
                
                transform = transforms.Compose([
                    transforms.Resize(224),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ])
            
                # 应用预处理
                img_tensor = transform(image)
                
            print(f"预处理结果: shape={img_tensor.shape}, range=[{img_tensor.min():.3f}, {img_tensor.max():.3f}]")
        
            print("="*50 + "\n")
            
            # 使用预测函数
            prediction, confidence, _ = predict_single_image(
                self.model, self.device, img_tensor, model_type)
            
            # 获取预测结果
            if model_type == 'CIFAR-10':
                classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                          'dog', 'frog', 'horse', 'ship', 'truck']
                result = f"{classes[prediction]} (置信度: {confidence:.2%})"
            else:
                result = f"{prediction} (置信度: {confidence:.2%})"
            
            self.result_label.setText(f"预测结果：{result}")
            
        except Exception as e:
            self.result_label.setText(f"预测失败：{str(e)}")
            print(f"预测错误: {str(e)}")
            import traceback
            traceback.print_exc()

    def compare_preprocessing(self):
        """比较预处理流程"""
        if self.image_path is None:
            return
        
        print("\n" + "="*50)
        print("预处理流程对比:")
        
        # GUI预处理流程
        image_gui = Image.open(self.image_path)
        if image_gui.mode != 'L':
            image_gui = image_gui.convert('L')
        image_gui = transforms.functional.invert(image_gui)
        transform_gui = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        tensor_gui = transform_gui(image_gui)
        
        # MNIST数据集预处理流程
        transform_mnist = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        print("\nGUI处理结果:")
        print(f"Tensor shape: {tensor_gui.shape}")
        print(f"值范围: [{tensor_gui.min():.3f}, {tensor_gui.max():.3f}]")
        print(f"均值: {tensor_gui.mean():.3f}")
        print(f"标准差: {tensor_gui.std():.3f}")

def main():
    app = QApplication(sys.argv)
    window = PredictWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main() 