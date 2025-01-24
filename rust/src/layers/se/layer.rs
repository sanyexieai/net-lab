use candle_core::{Module, Result, Tensor, Device};
use candle_nn::{AdaptiveAvgPool2d, Linear, Sequential};

#[derive(Debug)]
pub struct SELayer {
    avg_pool: AdaptiveAvgPool2d,
    fc: Sequential,
}

impl SELayer {
    pub fn new(
        channels: usize,
        reduction_ratio: usize,
        device: &Device,
    ) -> Result<Self> {
        let avg_pool = AdaptiveAvgPool2d::new(1, 1);
        
        let reduced_channels = channels / reduction_ratio;
        let fc = Sequential::new(vec![
            Linear::new(channels, reduced_channels, device)?,
            candle_nn::ReLU::new(),
            Linear::new(reduced_channels, channels, device)?,
            candle_nn::Sigmoid::new(),
        ]);

        Ok(Self { avg_pool, fc })
    }
}

impl Module for SELayer {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let batch_size = x.dim(0)?;
        let channels = x.dim(1)?;
        
        // Squeeze: Global Average Pooling
        let y = self.avg_pool.forward(x)?;
        let y = y.reshape((batch_size, channels))?;
        
        // Excitation: FC -> ReLU -> FC -> Sigmoid
        let y = self.fc.forward(&y)?;
        let y = y.reshape((batch_size, channels, 1, 1))?;
        
        // Scale: 特征重标定
        x.mul(&y)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn test_se_layer() -> Result<()> {
        let device = Device::Cpu;
        let batch_size = 2;
        let channels = 64;
        let height = 32;
        let width = 32;
        
        let se = SELayer::new(channels, 16, &device)?;
        let x = Tensor::randn(0.0, 1.0, (batch_size, channels, height, width), &device)?;
        let y = se.forward(&x)?;
        
        assert_eq!(y.dims(), &[batch_size, channels, height, width]);
        Ok(())
    }
} 