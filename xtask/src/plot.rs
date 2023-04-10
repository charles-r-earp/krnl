use std::{env::var, path::{Path, PathBuf}};
use serde_json::Value;

pub fn plot() {
    let manifest_dir = PathBuf::from(var("CARGO_MANIFEST_DIR").unwrap());
    let workspace_dir = manifest_dir.parent().unwrap();
    let plot_dir = workspace_dir.join("benches/compute-benches/plots/nv_gtx1060");
    std::fs::create_dir_all(&plot_dir).unwrap();
    let compute_dir = workspace_dir.join("target/criterion/compute");
    for op in ["upload", "download", "saxpy"] {
        let scale = if op == "saxpy" {
            2
        } else {
            4
        };
        let mut bar_data = Vec::new();
        for (s, n) in [("a_1M", 1_000_000), ("b_10M", 10_000_000), ("c_64M", 64_000_000)] {
            let mut data = Vec::new();
            for lib in ["ocl", "cuda", "autograph", "krnl"] {
                let name = format!("{op}_{s}_{lib}");
                let mean = read_mean_estimate(&compute_dir.join(&name).join("base/estimates.json"));
                let metric = (scale * n) as f64 / mean;
                data.push((metric, lib));
            }
            bar_data.push(data);
        }
        let mut bar_data = bar_data.into_iter();
        let data_a = bar_data.next().unwrap();
        let data_b = bar_data.next().unwrap();
        let data_c = bar_data.next().unwrap();
        let (a, yticks) = poloto::build::bar::gen_bar("1M", data_a, [0f64]);
        let b = poloto::build::bar::gen_bar("10M", data_b, [0f64]).0;
        let c = poloto::build::bar::gen_bar("64M", data_c, [0f64]).0;
        let data = poloto::plots!(c, b, a);
        let label = if op == "saxpy" {
            "GFlops"
        } else {
            "GB/s"
        };
        let plot = poloto::frame_build()
            .data(data)
            .map_yticks(|_| yticks)
            .build_and_label((&op.to_uppercase(), label, ""))
            .append_to(poloto::header().dark_theme())
            .render_string()
            .unwrap();
        std::fs::write(plot_dir.join(op).with_extension("svg"), plot.as_bytes()).unwrap();
    }
}

fn read_mean_estimate(path: &Path) -> f64 {
    let string = std::fs::read_to_string(path).unwrap();
    let value: Value = serde_json::from_str(&string).unwrap();
    let mean = value.as_object()
        .unwrap()
        .get("mean")
        .unwrap()
        .as_object()
        .unwrap()
        .get("point_estimate")
        .unwrap()
        .as_f64()
        .unwrap();
    mean
}

