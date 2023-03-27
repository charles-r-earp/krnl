use dry::macro_for;
use half::{bf16, f16};
use krnl::{
    buffer::{Buffer, Slice},
    device::{Device, Features},
    scalar::{Scalar, ScalarType},
};
use libtest_mimic::{Arguments, Trial};
use paste::paste;

fn main() {
    let mut args = Arguments::from_args();
    args.test_threads.replace(1);
    let tests = if cfg!(feature = "device") {
        let device = Device::builder().build().unwrap();
        tests(&Device::host())
            .into_iter()
            .chain(tests(&device))
            .collect()
    } else {
        tests(&Device::host()).into_iter().collect()
    };
    libtest_mimic::run(&args, tests).exit()
}

fn device_test(device: &Device, name: &str, f: impl Fn(Device) + Send + Sync + 'static) -> Trial {
    let name = format!(
        "{name}_{}",
        if device.is_host() { "host" } else { "device" }
    );
    let device = device.clone();
    Trial::test(name, move || Ok(f(device)))
}

fn tests(device: &Device) -> impl IntoIterator<Item = Trial> {
    buffer_tests(device)
}

fn buffer_tests(device: &Device) -> impl IntoIterator<Item = Trial> {
    fn buffer_test_lengths() -> impl IntoIterator<Item = usize> {
        [0, 1, 3, 4, 16, 67, 300, 1_011, 16_179]
    }
    let features = device
        .info()
        .map(|x| x.features())
        .unwrap_or(Features::empty());
    let mut tests = Vec::new();

    fn buffer_from_vec(device: Device) {
        for n in [
            0,
            1,
            3,
            4,
            16,
            67,
            300,
            1_011,
            16_179,
            247_341,
            5_437_921,
            48_532_978,
            112_789_546,
        ] {
            let x_vec = (1..=n as u32).into_iter().collect::<Vec<_>>();
            let buffer = Buffer::from(x_vec.clone())
                .to_device(device.clone())
                .unwrap();
            let y_vec = buffer.to_vec().unwrap();
            assert_eq!(x_vec, y_vec);
        }
    }
    tests.push(device_test(device, "buffer_from_vec", buffer_from_vec));

    fn buffer_fill<T: Scalar>(device: Device) {
        let elem = T::one();
        for n in buffer_test_lengths() {
            let x_vec = (10..20)
                .cycle()
                .map(|x| T::from_u32(x).unwrap())
                .take(n)
                .collect::<Vec<_>>();
            let y_vec_true = vec![elem; n];
            let mut buffer = Slice::from(x_vec.as_slice())
                .to_device(device.clone())
                .unwrap();
            buffer.fill(elem).unwrap();
            let y_vec: Vec<T> = buffer.to_vec().unwrap();
            assert_eq!(y_vec, y_vec_true);
        }
    }

    macro_for!($T in [u8, i8, u16, i16, f16, bf16, u32, i32, f32, u64, i64, f64] {
        paste! {
            {
                let ignore = if device.is_host() {
                    false
                } else {
                    match $T::scalar_type().size() {
                        1 => !features.shader_int8(),
                        2 => !features.shader_int16(),
                        4 => false,
                        8 => !features.shader_int64(),
                        _ => unreachable!(),
                    }
                };
                paste! {
                    let trial = device_test(device, stringify!([<buffer_fill_ $T>]), |device| [<buffer_fill>]::<$T>(device));
                }
                tests.push(trial.with_ignored_flag(ignore));
            }
        }
    });

    fn buffer_cast<X: Scalar, Y: Scalar>(device: Device) {
        for n in buffer_test_lengths() {
            let x_vec = (10u32..20)
                .cycle()
                .map(|x| X::from_u32(x).unwrap())
                .take(n)
                .collect::<Vec<X>>();
            let y_vec_true = x_vec.iter().map(|x| x.cast()).collect::<Vec<Y>>();
            let buffer = Buffer::from(x_vec).to_device(device.clone()).unwrap();
            let buffer = buffer.cast().unwrap();
            let y_vec: Vec<Y> = buffer.to_vec().unwrap();
            assert_eq!(&y_vec, &y_vec_true);
        }
    }

    fn buffer_cast_features(x: ScalarType, y: ScalarType) -> Features {
        fn features(ty: ScalarType) -> Features {
            use ScalarType::*;
            match ty {
                U8 | I8 => Features::empty().with_shader_int8(true),
                U16 | I16 => Features::empty().with_shader_int16(true),
                F16 | BF16 => Features::empty()
                    .with_shader_int8(true)
                    .with_shader_int16(true),
                U32 | I32 | F32 => Features::empty(),
                U64 | I64 => Features::empty().with_shader_int64(true),
                F64 => Features::empty()
                    .with_shader_int64(true)
                    .with_shader_float64(true),
                _ => unreachable!(),
            }
        }
        features(x).union(&features(y))
    }

    macro_for!($X in [u8, i8, u16, i16, f16, bf16, u32, i32, f32, u64, i64, f64] {
        macro_for!($Y in [u8, i8, u16, i16, f16, bf16, u32, i32, f32, u64, i64, f64] {
            {
                let ignore = !device.is_host() && !features.contains(&buffer_cast_features($X::scalar_type(), $Y::scalar_type()));
                paste! {
                    let trial = device_test(device, stringify!([<buffer_cast_ $X _ $Y>]), |device| [<buffer_cast>]::<$X, $Y>(device));
                    tests.push(trial.with_ignored_flag(ignore));
                }
            }
        });
    });

    tests
}
