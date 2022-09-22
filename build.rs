use std::{env::var, fmt::Write as FmtWrite, fs, path::PathBuf};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let out_dir = PathBuf::from(var("OUT_DIR")?);
    let scalars = ["u8", "u32", "i32", "f32"];
    let mut buffer_cast_kernels = String::new();
    let mut buffer_cast_tests = String::new();
    fn get_version(ty: &str) -> u32 {
        if ty.contains("8") {
            2
        } else {
            1
        }
    }
    fn get_caps(ty: &str) -> &'static [&'static str] {
        if ty.contains("8") {
            &["Int8", "StorageBuffer8BitAccess"]
        } else if ty.contains("16") {
            &["Int8", "Int16", "StorageBuffer16BitAccess"]
        } else if ty == "f64" {
            &["Int64", "Float64"]
        } else if ty.contains("64") {
            &["Int64"]
        } else {
            &[]
        }
    }
    for x in scalars {
        let x_version = get_version(x);
        let x_caps = get_caps(x);
        for y in scalars {
            let y_version = get_version(y);
            let y_caps = get_caps(y);
            let version = x_version.max(y_version);
            write!(
                &mut buffer_cast_kernels,
                "impl_cast!((version=\"1.{version}\", capabilities="
            )?;
            for cap in x_caps.iter().chain(y_caps) {
                write!(&mut buffer_cast_kernels, " {cap:?},")?;
            }
            writeln!(&mut buffer_cast_kernels, ") => ({x}, {y}));")?;
            writeln!(&mut buffer_cast_tests, "impl_buffer_cast_tests!({x}, {y});")?;
        }
    }
    let buffer_dir = out_dir.join("buffer");
    fs::create_dir_all(&buffer_dir)?;
    fs::write(
        buffer_dir.join("buffer_cast_kernels.in"),
        buffer_cast_kernels,
    )?;
    fs::write(buffer_dir.join("buffer_cast_tests.in"), buffer_cast_tests)?;
    Ok(())
}
