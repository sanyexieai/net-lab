import torch
import torch.nn.functional as F

def predict_single_image(model, device, tensor, model_type='MNIST'):
    """通用的单图预测函数
    
    Args:
        model: 加载好的模型
        device: 计算设备
        tensor: 预处理后的图像张量
        model_type: 'MNIST' 或 'CIFAR-10'
    
    Returns:
        prediction: 预测的类别
        confidence: 预测的置信度
        probabilities: 所有类别的预测概率
    """
    print("\n" + "="*50)
    print(f"预测信息:")
    print(f"模型类型: {model_type}")
    print(f"设备: {device}")
    print(f"输入张量形状: {tensor.shape}")
    
    # 确保输入是4D张量 [batch_size, channels, height, width]
    if len(tensor.shape) == 3:
        tensor = tensor.unsqueeze(0)
        print("添加batch维度")
    
    # 移动到指定设备
    tensor = tensor.to(device)
    print(f"最终输入: shape={tensor.shape}, range=[{tensor.min():.3f}, {tensor.max():.3f}]")
    
    # 预测
    model.eval()
    with torch.no_grad():
        print("\n模型预测:")
        output = model(tensor)
        print(f"原始输出: {output}")
        print(f"输出形状: {output.shape}")
        
        # 获取预测概率
        probs = F.softmax(output, dim=1)
        print("\n预测概率:")
        if model_type == 'CIFAR-10':
            classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                      'dog', 'frog', 'horse', 'ship', 'truck']
            for i, p in enumerate(probs[0]):
                print(f"{classes[i]}: {p:.2%}")
        else:
            for i, p in enumerate(probs[0]):
                print(f"数字 {i}: {p:.2%}")
        
        pred = output.argmax(dim=1, keepdim=True)
        print(f"\n最终预测: {pred.item()}")
    
    print("="*50 + "\n")
    return pred.item(), probs[0][pred.item()].item(), probs[0].cpu().numpy() 