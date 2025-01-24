use candle_core::{Module, Result, Tensor, Device};
use candle_nn::{Conv2d, Sequential, BatchNorm, init};

use super::layer::SELayer;

#[derive(Debug)]
pub struct SEBlock {
    conv1: Conv2d,
    bn1: BatchNorm,
    conv2: Conv2d,
    bn2: BatchNorm,
    se: SELayer,
    downsample: Option<Sequential>,
    relu: candle_nn::ReLU,
}

impl SEBlock {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        stride: usize,
        reduction_ratio: usize,
        device: &Device,
    ) -> Result<Self> {
        let conv1 = Conv2d::new(
            in_channels,
            out_channels,
            3,
            Default::default()
                .with_stride(stride)
                .with_padding(1),
            device,
        )?;
        
        let bn1 = BatchNorm::new(out_channels, 1e-5, 0.1, device)?;
        
        let conv2 = Conv2d::new(
            out_channels,
            out_channels,
            3,
            Default::default().with_padding(1),
            device,
        )?;
        
        let bn2 = BatchNorm::new(out_channels, 1e-5, 0.1, device)?;
        
        let se = SELayer::new(out_channels, reduction_ratio, device)?;
        
        let downsample = if stride != 1 || in_channels != out_channels {
            Some(Sequential::new(vec![
                Conv2d::new(
                    in_channels,
                    out_channels,
                    1,
                    Default::default().with_stride(stride),
                    device,
                )?,
                BatchNorm::new(out_channels, 1e-5, 0.1, device)?,
            ]))
        } else {
            None
        };

        Ok(Self {
            conv1,
            bn1,
            conv2,
            bn2,
            se,
            downsample,
            relu: candle_nn::ReLU::new(),
        })
    }
}

impl Module for SEBlock {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let identity = x.clone();
        
        let mut out = self.conv1.forward(x)?;
        out = self.bn1.forward(&out)?;
        out = self.relu.forward(&out)?;
        
        out = self.conv2.forward(&out)?;
        out = self.bn2.forward(&out)?;
        out = self.se.forward(&out)?;
        
        if let Some(ref downsample) = self.downsample {
            let identity = downsample.forward(&identity)?;
        }
        
        out = out.add(&identity)?;
        self.relu.forward(&out)
    }
} 