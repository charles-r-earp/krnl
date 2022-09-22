use std::{env::var, fmt::Write as FmtWrite, fs, path::PathBuf};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let target_arch = var("CARGO_CFG_TARGET_ARCH")?;
    let mut scalars = vec!["u8", "i8", "u16", "i16"];
    if cfg!(feature = "half") {
        scalars.extend(["f16", "bf16"]);
    }
    scalars.extend(["u32", "i32", "f32", "u64", "i64", "f64"]);
    let scalar_types = scalars
        .iter()
        .copied()
        .map(str::to_uppercase)
        .collect::<Vec<_>>();
    let mut scalar_trait = String::from(
        r#"/// Base trait for numerical types.
pub trait Scalar: Default + Copy + 'static + NumCast + FromPrimitive + NumAssign + PartialEq"#,
    );
    if target_arch != "spirv" {
        write!(
            &mut scalar_trait,
            "+ Into<ScalarElem> + Pod + Debug + Display + Serialize + for<'de> Deserialize<'de>"
        )?;
    }
    write!(
        &mut scalar_trait,
        "{}",
        "+ Sealed {
    /// The [`ScalarType`] of the scalar. 
    fn scalar_type() -> ScalarType; 
    fn cast<T: Scalar>(self) -> T;
}
",
    )?;
    for (t, s) in scalars.iter().copied().zip(scalar_types.iter()) {
        writeln!(
            &mut scalar_trait,
            "impl Scalar for {t} {{ 
    fn scalar_type() -> ScalarType {{ ScalarType::{s} }}
    fn cast<T: Scalar>(self) -> T {{
        use ScalarType::*;
        match T::scalar_type() {{"
        )?;
        for (t2, s2) in scalars.iter().copied().zip(scalar_types.iter()) {
            if target_arch == "spirv" {
                for b in ["8", "16", "64"] {
                    if t.contains(b) || t2.contains(b) {
                        writeln!(
                            &mut scalar_trait,
                            "#[cfg(not(target_feature = \"Int{b}\"))] {s2} => unreachable!(),"
                        )?;
                        writeln!(&mut scalar_trait, "#[cfg(target_feature = \"Int{b}\")]")?;
                        break;
                    }
                }
            }
            let t1 = if matches!((t, t2), ("f16", "bf16") | ("bf16", "f16")) {
                "f32"
            } else {
                t
            };
            writeln!(
                &mut scalar_trait,
                "           {s2} => {{
                    let x: {t1} = self.as_();
                    let y: {t2} = x.as_();
                    let y = T::from(y);
                    unsafe {{ y.unwrap_unchecked() }}
}}"
            )?;
        }
        writeln!(
            &mut scalar_trait,
            "       }}
    }}
}}"
        )?;
    }
    let scalar_trait_path = PathBuf::from(var("OUT_DIR")?).join("scalar_trait.in");
    fs::write(&scalar_trait_path, scalar_trait)?;
    Ok(())
}
