use candle_core::{Result, Tensor, D};
use candle_nn::{Conv2d, Conv2dConfig, BatchNorm, Linear, VarBuilder};

pub struct SELayer {
    avg_pool: AdaptiveAvgPool2d,
    fc1: Linear,
    fc2: Linear,
}

impl SELayer {
    pub fn new(vb: VarBuilder, channel: usize, reduction: usize) -> Result<Self> {
        let avg_pool = AdaptiveAvgPool2d::new(1);
        let fc1 = Linear::new(
            vb.pp("fc1"), 
            channel, 
            channel / reduction, 
            false
        )?;
        let fc2 = Linear::new(
            vb.pp("fc2"), 
            channel / reduction, 
            channel, 
            false
        )?;
        
        Ok(Self { avg_pool, fc1, fc2 })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b, c, _, _) = x.dims4()?;
        
        // Squeeze
        let y = self.avg_pool.forward(x)?
            .reshape((b, c))?;
        
        // Excitation
        let y = self.fc1.forward(&y)?
            .relu()?;
        let y = self.fc2.forward(&y)?
            .sigmoid()?
            .reshape((b, c, 1, 1))?;
        
        // Scale
        x.broadcast_mul(&y)
    }
}

pub struct SEBasicBlock {
    conv1: Conv2d,
    bn1: BatchNorm,
    conv2: Conv2d,
    bn2: BatchNorm,
    se: SELayer,
    downsample: Option<(Conv2d, BatchNorm)>,
}

impl SEBasicBlock {
    pub fn new(
        vb: VarBuilder,
        inplanes: usize,
        planes: usize,
        stride: usize,
        reduction: usize,
    ) -> Result<Self> {
        let conv1 = Conv2d::new(
            vb.pp("conv1"),
            inplanes as i64,
            planes as i64,
            3,
            Conv2dConfig {
                stride: stride as i64,
                padding: 1,
                ..Default::default()
            },
        )?;
        
        let bn1 = BatchNorm::new(vb.pp("bn1"), planes as i64, Default::default())?;
        
        let conv2 = Conv2d::new(
            vb.pp("conv2"),
            planes as i64,
            planes as i64,
            3,
            Conv2dConfig {
                padding: 1,
                ..Default::default()
            },
        )?;
        
        let bn2 = BatchNorm::new(vb.pp("bn2"), planes as i64, Default::default())?;
        let se = SELayer::new(vb.pp("se"), planes, reduction)?;

        let downsample = if stride != 1 || inplanes != planes {
            let conv = Conv2d::new(
                vb.pp("downsample.0"),
                inplanes as i64,
                planes as i64,
                1,
                Conv2dConfig {
                    stride: stride as i64,
                    ..Default::default()
                },
            )?;
            let bn = BatchNorm::new(
                vb.pp("downsample.1"),
                planes as i64,
                Default::default(),
            )?;
            Some((conv, bn))
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
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let identity = x.clone();

        let out = self.conv1.forward(x)?;
        let out = self.bn1.forward(&out)?;
        let out = out.relu()?;

        let out = self.conv2.forward(&out)?;
        let out = self.bn2.forward(&out)?;
        let out = self.se.forward(&out)?;

        let identity = if let Some((conv, bn)) = &self.downsample {
            bn.forward(&conv.forward(&identity)?)?
        } else {
            identity
        };

        let out = out.add(&identity)?;
        out.relu()
    }
} 