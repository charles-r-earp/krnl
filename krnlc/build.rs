use std::{
    env::{
        consts::{DLL_PREFIX, DLL_SUFFIX},
        var,
    },
    fs,
    path::PathBuf,
    process::Command,
};

#[rustversion::not(nightly(2023-05-26))]
compile_error!(
    "krnlc requires nightly-2023-05-27, install with rustup:
rustup toolchain install nightly-2023-05-27
rustup component add --toolchain nightly-2023-05-27 rust-src rustc-dev llvm-tools-preview"
);

fn main() {
    let output = Command::new(var("RUSTC").unwrap())
        .args(["--print", "sysroot"])
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
    let out_dir = PathBuf::from(var("OUT_DIR").unwrap());
    let target_dir = out_dir.ancestors().nth(3).unwrap();
    let rustc_codegen_spirv_path =
        target_dir.join(format!("{DLL_PREFIX}rustc_codegen_spirv{DLL_SUFFIX}"));
    println!(
        "cargo:rustc-env=KRNLC_LIBRUSTC_CODEGEN_SPIRV={}",
        rustc_codegen_spirv_path.display(),
    );
    if !rustc_codegen_spirv_path.exists() {
        fs::create_dir_all(target_dir).unwrap();
        fs::write(rustc_codegen_spirv_path, []).unwrap();
    }
    {
        let short = true;
        vergen::EmitBuilder::builder()
            .git_sha(short)
            .emit()
            .unwrap();
    }
}
