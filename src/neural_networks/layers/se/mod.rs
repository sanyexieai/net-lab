use tch::{Tensor, nn};
use crate::nn::ModuleT;

/// SE (Squeeze-and-Excitation) 模块
#[derive(Debug)]
pub struct SE {
    // 全连接层1: C -> C/r
    fc1: nn::Linear,
    // 全连接层2: C/r -> C
    fc2: nn::Linear,
    // 设备
    device: tch::Device,
}

impl SE {
    pub fn new(vs: &nn::Path, in_channels: i64, reduction: i64) -> Self {
        // 计算中间通道数
        let hidden_channels = in_channels / reduction;
        
        // 创建两个全连接层
        let fc1 = nn::linear(
            vs, 
            in_channels, 
            hidden_channels, 
            Default::default()
        );
        
        let fc2 = nn::linear(
            vs,
            hidden_channels,
            in_channels,
            Default::default()
        );
        
        // 获取设备
        let device = vs.device();
        
        Self { fc1, fc2, device }
    }
}

impl ModuleT for SE {
    fn forward_t(&self, x: &Tensor, train: bool) -> Tensor {
        let batch_size = x.size()[0];
        let channels = x.size()[1];
        
        // Squeeze操作：全局平均池化
        let mut y = x.adaptive_avg_pool2d(&[1, 1]);
        
        // 重塑形状以适应全连接层
        y = y.view([batch_size, channels]);
        
        // Excitation操作
        y = self.fc1.forward(&y).relu();  // FC1 + ReLU
        y = self.fc2.forward(&y).sigmoid();  // FC2 + Sigmoid
        
        // 重塑形状以进行广播
        y = y.view([batch_size, channels, 1, 1]);
        
        // 特征重标定
        x * y
    }
} 