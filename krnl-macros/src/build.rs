use std::{env::{var, vars}, fmt::Write};

fn main() {
    let target = var("TARGET").unwrap();
    let target_prefix = "spirv-unknown-";
    if target.starts_with(target_prefix) {
        let vulkan = target.split_at(target_prefix.len()).1.replace('.', "_");
        let mut cfg = format!("krnl_device_crate_{vulkan}");
        let target_feature_prefix = "CARGO_CFG_TARGET_FEATURE_";
        let mut target_features = vars().map(|(x, _)| x)
            .filter(|x| x.starts_with(target_feature_prefix))
            .map(|x| x.split_at(target_feature_prefix.len()).1.to_string())
            .collect::<Vec<_>>();
        target_features.sort();
        for target_feature in target_features.iter() {
            write!(&mut cfg, "_{target_feature}").unwrap();
        }
        println!("cargo:rustc-cfg={cfg}");
    }
}
