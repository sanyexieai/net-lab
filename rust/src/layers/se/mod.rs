use candle_core::{Module, Result, Tensor, Device};
use candle_nn::{AdaptiveAvgPool2d, Linear, Sequential};

pub mod layer;
pub mod net;
pub mod examples;

pub use layer::SELayer;
pub use net::SENet;

pub struct SELayer {
    avg_pool: AdaptiveAvgPool2d,
    fc: Sequential,
}

impl SELayer {
    pub fn new(
        channels: usize,
        reduction: usize,
        device: &Device,
    ) -> Result<Self> {
        let avg_pool = AdaptiveAvgPool2d::new(1, 1);
        
        // 创建两个全连接层
        let fc = Sequential::new(vec![
            Linear::new(channels, channels / reduction, device)?,
            candle_nn::ReLU::new(),
            Linear::new(channels / reduction, channels, device)?,
            candle_nn::Sigmoid::new(),
        ]);

        Ok(Self { avg_pool, fc })
    }
}

impl Module for SELayer {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let batch_size = x.dim(0)?;
        let channels = x.dim(1)?;
        
        // Squeeze操作
        let y = self.avg_pool.forward(x)?;
        
        // 重塑张量以便进行全连接层操作
        let y = y.reshape((batch_size, channels))?;
        
        // Excitation操作
        let y = self.fc.forward(&y)?;
        
        // 重塑回原来的维度并进行广播
        let y = y.reshape((batch_size, channels, 1, 1))?;
        
        // 应用注意力权重
        x.mul(&y)
    }
} 