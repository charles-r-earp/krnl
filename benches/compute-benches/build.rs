#[cfg(feature = "cuda")]
use anyhow::bail;
use anyhow::Result;
#[cfg(feature = "cuda")]
use std::{env::var, path::PathBuf, process::Command};

fn main() -> Result<()> {
    #[cfg(feature = "cuda")]
    {
        let out_dir = PathBuf::from(var("OUT_DIR").unwrap());
        let output_path = out_dir.join("kernels.ptx");
        let status = Command::new("nvcc")
            .args([
                "src/kernels.cu",
                "--ptx",
                "--output-file",
                output_path.to_string_lossy().as_ref(),
            ])
            .status()?;
        if !status.success() {
            bail!("Failed to compile kernels.cu!");
        }
    }
    Ok(())
}
