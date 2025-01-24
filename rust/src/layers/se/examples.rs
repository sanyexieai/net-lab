use candle_core::{Device, Result, Tensor};
use super::{layer::SELayer, net::SEBlock};

pub fn se_layer_example() -> Result<()> {
    let device = Device::Cpu;
    
    // 创建一个SE层
    let se_layer = SELayer::new(64, 16, &device)?;
    
    // 创建输入张量
    let input = Tensor::randn(0.0, 1.0, (1, 64, 32, 32), &device)?;
    
    // 前向传播
    let output = se_layer.forward(&input)?;
    println!("SE Layer output shape: {:?}", output.dims());
    
    Ok(())
}

pub fn se_block_example() -> Result<()> {
    let device = Device::Cpu;
    
    // 创建一个SE Block
    let se_block = SEBlock::new(64, 128, 2, 16, &device)?;
    
    // 创建输入张量
    let input = Tensor::randn(0.0, 1.0, (1, 64, 32, 32), &device)?;
    
    // 前向传播
    let output = se_block.forward(&input)?;
    println!("SE Block output shape: {:?}", output.dims());
    
    Ok(())
} 