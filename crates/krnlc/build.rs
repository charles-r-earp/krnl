/*
use std::{
    env::{
        consts::{DLL_PREFIX, DLL_SUFFIX},
        var,
    },
    fs,
    path::PathBuf,
    process::Command,
};
*/

//use include_dir::include_dir;

fn main() {
    {
        let short = true;
        vergen::EmitBuilder::builder()
            .git_sha(short)
            .emit()
            .unwrap();
    }
    /*
    let out_dir = PathBuf::from(var("OUT_DIR").unwrap());

    let codegen_wrapper = include_dir!("$CARGO_MANIFEST_DIR/codegen-wrapper");
    let codegen_wrapper_dir = out_dir.join("codegen-wrapper");
    if !codegen_wrapper_dir.exists() {
        std::fs::create_dir(&codegen_wrapper_dir).unwrap();
        codegen_wrapper.extract(&codegen_wrapper_dir).unwrap();
    }
    let rust_toolchain_toml =
        std::fs::read_to_string(codegen_wrapper_dir.join("rust-toolchain.toml")).unwrap();
    let toolchain_line = rust_toolchain_toml.lines().skip(1).next().unwrap();
    let toolchain = toolchain_line.strip_prefix("channel = \"").unwrap();
    let toolchain = toolchain.strip_suffix('\"').unwrap();
    println!("cargo:rustc-env=KRNLC_RUST_TOOLCHAIN={toolchain}");
    let plus_toolchain = format!("+{toolchain}");
    let mut features = String::new();
    if cfg!(feature = "use-installed-tools") {
        features.push_str("use-installed-tools");
    } else if cfg!(feature = "use-compiled-tools") {
        if !features.is_empty() {
            features.push(',');
        }
        features.push_str("use-compiled-tools");
    }
    let target_dir = out_dir.join("target");
    let status = Command::new("cargo")
        .args([
            &plus_toolchain,
            "build",
            "--target-dir",
            &target_dir.to_string_lossy(),
            "--features",
            &features,
        ])
        .current_dir(&codegen_wrapper_dir)
        .env_remove("RUSTC")
        .env_remove("RUSTC_SRC_PATH")
        .env_remove("RUSTUP_TOOLCHAIN")
        .status()
        .unwrap();
    assert!(status.success());
    let profile = std::env::var("PROFILE").unwrap();
    let target_profile_dir = target_dir.join(&profile);
    let rustc_codegen_spirv_path =
        target_profile_dir.join(format!("{DLL_PREFIX}rustc_codegen_spirv{DLL_SUFFIX}"));
    println!(
        "cargo:rustc-env=KRNLC_LIBRUSTC_CODEGEN_SPIRV={}",
        rustc_codegen_spirv_path.display(),
    );
    if !rustc_codegen_spirv_path.exists() {
        fs::create_dir_all(target_profile_dir).unwrap();
        fs::write(rustc_codegen_spirv_path, []).unwrap();
    }
    let output = Command::new("cargo")
        .args([
            &plus_toolchain,
            "rustc",
            "-Z",
            "unstable-options",
            "--print",
            "sysroot",
        ])
        .env_remove("RUSTC")
        .env_remove("RUSTC_SRC_PATH")
        .env_remove("RUSTUP_TOOLCHAIN")
        .output()
        .unwrap();
    if !output.status.success() {
        panic!("{}", String::from_utf8(output.stderr).unwrap());
    }
    let sysroot = String::from_utf8(output.stdout).unwrap();
    let sysroot = sysroot.trim();
    let toolchain_lib = PathBuf::from(sysroot).join("lib");
    println!(
        "cargo:rustc-env=KRNLC_TOOLCHAIN_LIB={}",
        toolchain_lib.display()
    );
    for entry in fs::read_dir(&toolchain_lib).unwrap().map(Result::unwrap) {
        let file_name = entry.file_name();
        let file_name = file_name.to_str().unwrap();
        if file_name.starts_with(&format!("{DLL_PREFIX}LLVM-")) {
            println!("cargo:rustc-env=KRNLC_LIBLLVM={file_name}");
        } else if file_name.starts_with(&format!("{DLL_PREFIX}rustc_driver-")) {
            println!("cargo:rustc-env=KRNLC_LIBRUSTC_DRIVER={file_name}");
        } else if file_name.starts_with(&format!("{DLL_PREFIX}std-")) {
            println!("cargo:rustc-env=KRNLC_LIBSTD={file_name}");
        }
    }
    */
}
