use anyhow::Result;
use dry::macro_for;
use half::{bf16, f16};
use krnl::{
    buffer::Buffer,
    device::{Device, Features},
    future::BlockableFuture,
    scalar::{Scalar, ScalarType},
};
use libtest_mimic::{Arguments, Trial};
use paste::paste;

fn main() -> Result<()> {
    let mut args = Arguments::from_args();
    args.test_threads.replace(1);
    let tests = if cfg!(feature = "device") {
        let device = Device::builder().build()?;
        tests(&Device::host())
            .into_iter()
            .chain(tests(&device))
            .collect()
    } else {
        tests(&Device::host()).into_iter().collect()
    };
    libtest_mimic::run(&args, tests).exit()
}

fn device_test(
    device: &Device,
    name: &str,
    f: impl Fn(Device) -> Result<()> + Send + Sync + 'static,
) -> Trial {
    let name = format!(
        "{name}_{}",
        if device.is_host() { "host" } else { "device" }
    );
    let device = device.clone();
    Trial::test(name, move || f(device).map_err(Into::into))
}

fn tests(device: &Device) -> impl IntoIterator<Item = Trial> {
    buffer_tests(device)
}

fn buffer_tests(device: &Device) -> impl IntoIterator<Item = Trial> {
    fn buffer_test_lengths() -> impl IntoIterator<Item = usize> {
        [0, 1, 3, 4, 16, 67, 300, 1011]
    }
    let features = device.features().copied().unwrap_or_default();
    let mut tests = Vec::new();

    fn buffer_from_vec(device: Device) -> Result<()> {
        for n in buffer_test_lengths() {
            let x_vec = (1..=n as u32).into_iter().collect::<Vec<_>>();
            let buffer = Buffer::from_vec(x_vec.clone())
                .into_device(device.clone())?
                .block()?;
            let y_vec = buffer.into_vec()?.block()?;
            assert_eq!(x_vec, y_vec);
        }
        Ok(())
    }
    tests.push(device_test(device, "buffer_from_vec", buffer_from_vec));

    fn buffer_fill<T: Scalar>(device: Device) -> Result<()> {
        let elem = T::one();
        for n in buffer_test_lengths() {
            let x_vec = (10..20)
                .cycle()
                .map(|x| T::from_u32(x).unwrap())
                .take(n)
                .collect::<Vec<_>>();
            let y_vec_true = vec![elem; n];
            let mut buffer = Buffer::from_vec(x_vec.clone())
                .into_device(device.clone())?
                .block()?;
            buffer.fill(elem)?;
            let y_vec: Vec<T> = buffer.into_vec()?.block()?;
            assert_eq!(y_vec, y_vec_true);
        }
        Ok(())
    }

    fn scalar_buffer_fill<T: Scalar>(device: Device) -> Result<()> {
        let elem = T::one();
        for n in buffer_test_lengths() {
            let x_vec = (10..20)
                .cycle()
                .map(|x| T::from_u32(x).unwrap())
                .take(n)
                .collect::<Vec<_>>();
            let y_vec_true = vec![elem; n];
            let mut buffer = Buffer::from_vec(x_vec)
                .into_device(device.clone())?
                .block()?;
            buffer.as_scalar_slice_mut().fill(elem.into())?;
            let y_vec: Vec<T> = buffer.into_vec()?.block()?;
            assert_eq!(y_vec, y_vec_true);
        }
        Ok(())
    }

    macro_for!($buffer in [buffer, scalar_buffer] {
        macro_for!($T in [u8, i8, u16, i16, f16, bf16, u32, i32, f32, u64, i64, f64] {
            paste! {
                let include = if device.is_host() {
                    true
                } else {
                    match $T::scalar_type().size() {
                        1 => features.shader_int8(),
                        2 => features.shader_int16(),
                        4 => true,
                        8 => features.shader_int64(),
                        _ => false,
                    }
                };
                if include {
                    tests.push(device_test(device, stringify!([<$buffer _fill_ $T>]), |device| [<$buffer _fill>]::<$T>(device)));
                }
            }
        });
    });

    fn buffer_cast<X: Scalar, Y: Scalar>(device: Device) -> Result<()> {
        for n in buffer_test_lengths() {
            let x_vec = (10u32..20)
                .cycle()
                .map(|x| X::from_u32(x).unwrap())
                .take(n)
                .collect::<Vec<X>>();
            let y_vec_true = x_vec.iter().map(|x| x.cast()).collect::<Vec<Y>>();
            let buffer = Buffer::from_vec(x_vec)
                .into_device(device.clone())?
                .block()?;
            let buffer = buffer.cast()?;
            let y_vec: Vec<Y> = buffer.into_vec()?.block()?;
            assert_eq!(&y_vec, &y_vec_true);
        }
        Ok(())
    }

    fn scalar_buffer_cast<X: Scalar, Y: Scalar>(device: Device) -> Result<()> {
        for n in buffer_test_lengths() {
            let x_vec = (10u32..20)
                .cycle()
                .map(|x| X::from_u32(x).unwrap())
                .take(n)
                .collect::<Vec<X>>();
            let y_vec_true = x_vec.iter().map(|x| x.cast()).collect::<Vec<Y>>();
            let buffer = Buffer::from_vec(x_vec)
                .into_device(device.clone())?
                .block()?;
            let buffer = buffer
                .as_scalar_slice()
                .cast(Y::scalar_type())?
                .into_scalar_buffer()?;
            let y_vec: Vec<Y> = buffer.try_as_slice().unwrap().into_vec()?.block()?;
            assert_eq!(&y_vec, &y_vec_true);
        }
        Ok(())
    }

    fn buffer_cast_features(x: ScalarType, y: ScalarType) -> Features {
        fn features(ty: ScalarType) -> Features {
            use ScalarType::*;
            match ty {
                U8 | I8 => Features::default().with_shader_int8(true),
                U16 | I16 => Features::default().with_shader_int16(true),
                F16 | BF16 => Features::default()
                    .with_shader_int8(true)
                    .with_shader_int16(true),
                U32 | I32 | F32 => Features::default(),
                U64 | I64 => Features::default().with_shader_int64(true),
                F64 => Features::default()
                    .with_shader_int64(true)
                    .with_shader_float64(true),
                _ => unreachable!(),
            }
        }
        features(x).union(&features(y))
    }

    macro_for!($buffer in [buffer, scalar_buffer] {
        macro_for!($X in [u8, i8, u16, i16, f16, bf16, u32, i32, f32, u64, i64, f64] {
            macro_for!($Y in [u8, i8, u16, i16, f16, bf16, u32, i32, f32, u64, i64, f64] {
                if device.is_host() || features.contains(&buffer_cast_features($X::scalar_type(), $Y::scalar_type())) {
                    paste! {
                        tests.push(device_test(device, stringify!([<$buffer _cast_ $X _ $Y>]), |device| [<$buffer _cast>]::<$X, $Y>(device)));
                    }
                }
            });
        });
    });

    tests
}
