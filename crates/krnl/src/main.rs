use anyhow::Result;
use krnl::{buffer::Buffer, device::Device};
use std::time::Instant;

fn main() -> Result<()> {
    let device = Device::builder().build()?;
    let buffer = Buffer::from(vec![1f32; 256_000_000]).to_device(device.clone())?;
    device.wait()?;
    for _ in 0..10 {
        let _vec = buffer.to_vec()?;
    }
    let mut elapsed = std::time::Duration::default();
    for _ in 0..10 {
        let start = Instant::now();
        let _vec = buffer.to_vec()?;
        elapsed += start.elapsed() / 10;
    }
    println!("{elapsed:?}");
    Ok(())
}
