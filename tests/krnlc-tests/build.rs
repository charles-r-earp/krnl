fn main() {
    #[cfg(feature = "krnlc")]
    run_krnlc();
}

#[cfg(feature = "krnlc")]
fn run_krnlc() {
    use std::{env::var, path::PathBuf, process::Command};

    let manifest_dir = PathBuf::from(var("CARGO_MANIFEST_DIR").unwrap());
    let status = Command::new("cargo")
        .args([
            "run",
            "--",
            "--manifest-path",
            "../../tests/krnlc-tests/Cargo.toml",
            "--target-dir",
            "target/krnlc-tests",
        ])
        .current_dir(manifest_dir.join("../../crates/krnlc"))
        .env_remove("RUSTUP_TOOLCHAIN")
        .status()
        .unwrap();
    if !status.success() {
        panic!("running krnlc failed!");
    }
}
