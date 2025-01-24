import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import torch
import os
from cifar_predict import predict_image as predict_cifar
from mnist_predict import predict_image as predict_mnist

class SKNetGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("SKNet图像分类器")
        
        # 设置窗口大小和位置
        window_width = 800
        window_height = 600
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        x = (screen_width - window_width) // 2
        y = (screen_height - window_height) // 2
        self.root.geometry(f"{window_width}x{window_height}+{x}+{y}")
        
        # 创建主框架
        self.main_frame = ttk.Frame(root, padding="10")
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 创建模型选择下拉框
        self.model_var = tk.StringVar(value="CIFAR-10")
        model_label = ttk.Label(self.main_frame, text="选择模型:")
        model_label.grid(row=0, column=0, padx=5, pady=5)
        model_combo = ttk.Combobox(self.main_frame, textvariable=self.model_var)
        model_combo['values'] = ('CIFAR-10', 'MNIST')
        model_combo.grid(row=0, column=1, padx=5, pady=5)
        
        # 创建图片显示区域
        self.image_label = ttk.Label(self.main_frame)
        self.image_label.grid(row=1, column=0, columnspan=2, padx=5, pady=5)
        
        # 创建按钮
        button_frame = ttk.Frame(self.main_frame)
        button_frame.grid(row=2, column=0, columnspan=2, pady=10)
        
        select_button = ttk.Button(button_frame, text="选择图片", command=self.select_image)
        select_button.pack(side=tk.LEFT, padx=5)
        
        predict_button = ttk.Button(button_frame, text="开始预测", command=self.predict)
        predict_button.pack(side=tk.LEFT, padx=5)
        
        # 创建结果显示区域
        self.result_text = tk.Text(self.main_frame, height=10, width=50)
        self.result_text.grid(row=3, column=0, columnspan=2, padx=5, pady=5)
        
        # 初始化变量
        self.current_image_path = None
        self.photo = None
        
    def select_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("图片文件", "*.png *.jpg *.jpeg")]
        )
        if file_path:
            self.current_image_path = file_path
            self.display_image(file_path)
            self.result_text.delete(1.0, tk.END)
            
    def display_image(self, image_path):
        image = Image.open(image_path)
        # 调整图片大小以适应显示区域
        image.thumbnail((300, 300))
        self.photo = ImageTk.PhotoImage(image)
        self.image_label.configure(image=self.photo)
        
    def predict(self):
        if not self.current_image_path:
            messagebox.showwarning("警告", "请先选择一张图片！")
            return
            
        try:
            # 获取模型路径
            model_type = self.model_var.get()
            base_dir = os.path.dirname(os.path.abspath(__file__))
            if model_type == "CIFAR-10":
                model_path = os.path.join(base_dir, '../../models/cifar/sk/sknet_cifar.safetensors')
                if not os.path.exists(model_path):
                    model_path = model_path.replace('.safetensors', '.pth')
                result = predict_cifar(model_path, self.current_image_path)
            else:  # MNIST
                model_path = os.path.join(base_dir, '../../models/mnist/sk/sknet_mnist.safetensors')
                if not os.path.exists(model_path):
                    model_path = model_path.replace('.safetensors', '.pth')
                result = predict_mnist(model_path, self.current_image_path)
            
            # 显示结果
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, f"预测结果: {result[0]}\n")
            self.result_text.insert(tk.END, f"预测概率: {result[1]:.2%}\n")
            
        except Exception as e:
            messagebox.showerror("错误", f"预测过程中出现错误：{str(e)}")

def main():
    root = tk.Tk()
    app = SKNetGUI(root)
    root.mainloop()

if __name__ == '__main__':
    main() 