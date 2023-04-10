use dry::macro_for;
use half::{bf16, f16};
use krnl::{buffer::Slice, device::Device, scalar::Scalar};
#[cfg(not(target_arch = "wasm32"))]
use krnl::{device::Features, scalar::ScalarType};
#[cfg(not(target_arch = "wasm32"))]
use libtest_mimic::{Arguments, Trial};
use paste::paste;
#[cfg(not(target_arch = "wasm32"))]
use std::str::FromStr;
#[cfg(target_arch = "wasm32")]
use wasm_bindgen_test::wasm_bindgen_test as test;

#[cfg(all(target_arch = "wasm32", run_in_browser))]
wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);

#[cfg(target_arch = "wasm32")]
fn main() {}

#[cfg(not(target_arch = "wasm32"))]
fn main() {
    let args = Arguments::from_args();
    let tests = if cfg!(feature = "device") && !cfg!(miri) {
        let devices: Vec<_> = [Device::builder().build().unwrap()]
            .into_iter()
            .chain((1..).map_while(|i| Device::builder().index(i).build().ok()))
            .collect();
        if devices.is_empty() {
            panic!("No device!");
        }
        let device_infos: Vec<_> = devices.iter().map(|x| x.info().unwrap()).collect();
        println!("devices: {device_infos:#?}");
        let krnl_device = std::env::var("KRNL_DEVICE");
        let device_index = if let Ok(krnl_device) = krnl_device.as_ref() {
            usize::from_str(krnl_device).unwrap()
        } else {
            0
        };
        let device_index2 = usize::from(device_index == 0);
        println!("KRNL_DEVICE = {krnl_device:?}");
        println!("testing device {device_index}");
        let device = devices.get(0).unwrap();
        let device2 = devices.get(1);
        if device2.is_some() {
            println!("using device {device_index2} for `buffer_device_to_device`");
        }
        tests(&Device::host(), None)
            .into_iter()
            .chain(tests(device, device2))
            .collect()
    } else {
        tests(&Device::host(), None).into_iter().collect()
    };
    libtest_mimic::run(&args, tests).exit()
}

#[cfg(not(target_arch = "wasm32"))]
fn device_test(device: &Device, name: &str, f: impl Fn(Device) + Send + Sync + 'static) -> Trial {
    let name = format!(
        "{name}_{}",
        if device.is_host() { "host" } else { "device" }
    );
    let device = device.clone();
    Trial::test(name, move || {
        f(device);
        Ok(())
    })
}

#[cfg(not(target_arch = "wasm32"))]
fn tests(device: &Device, device2: Option<&Device>) -> impl IntoIterator<Item = Trial> {
    buffer_tests(device, device2)
}

#[cfg(not(target_arch = "wasm32"))]
fn buffer_tests(device: &Device, device2: Option<&Device>) -> impl IntoIterator<Item = Trial> {
    let features = device
        .info()
        .map(|x| x.features())
        .unwrap_or(Features::empty());
    let mut tests = Vec::new();

    tests.push(device_test(device, "buffer_from_vec", buffer_from_vec));

    if device.is_device() {
        tests.push(
            Trial::test("buffer_device_to_device", {
                let device = device.clone();
                let device2 = device2.cloned();
                move || {
                    buffer_transfer(device, device2.unwrap());
                    Ok(())
                }
            })
            .with_ignored_flag(device2.is_none()),
        );
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
                let trial = paste! {
                    device_test(device, stringify!([<buffer_fill_ $T>]), [<buffer_fill>]::<$T>)
                };
                tests.push(trial.with_ignored_flag(ignore));
            }
        }
    });

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
                    let trial = device_test(device, stringify!([<buffer_cast_ $X _ $Y>]), [<buffer_cast>]::<$X, $Y>);
                    tests.push(trial.with_ignored_flag(ignore));
                    let trial = device_test(device, stringify!([<buffer_bitcast_ $X _ $Y>]), [<buffer_bitcast>]::<$X, $Y>);
                    tests.push(trial.with_ignored_flag(ignore));
                }
            }
        });
    });

    tests
}

fn buffer_test_lengths() -> impl ExactSizeIterator<Item = usize> {
    [0, 1, 3, 4, 16, 67, 157].into_iter()
}
fn buffer_transfer_test_lengths() -> impl ExactSizeIterator<Item = usize> {
    #[cfg(not(miri))]
    {
        [0, 1, 3, 4, 16, 345, 9_337_791].into_iter()
    }
    #[cfg(miri)]
    {
        [0, 1, 3, 4, 16, 345].into_iter()
    }
}

fn buffer_from_vec(device: Device) {
    let n = buffer_transfer_test_lengths().last().unwrap();
    let x = (10..20).cycle().take(n).collect::<Vec<_>>();
    for n in buffer_transfer_test_lengths() {
        let x = &x[..n];
        let y = Slice::from(x)
            .to_device(device.clone())
            .unwrap()
            .into_vec()
            .unwrap();
        assert_eq!(y.len(), n);
        if x != y.as_slice() {
            for (x, y) in x.iter().zip(y) {
                assert_eq!(&y, x);
            }
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn buffer_transfer(device: Device, device2: Device) {
    let n = buffer_transfer_test_lengths().last().unwrap();
    let x = (10..20).cycle().take(n).collect::<Vec<_>>();
    for n in buffer_transfer_test_lengths() {
        let x = &x[..n];
        let y = Slice::from(x)
            .to_device(device.clone())
            .unwrap()
            .to_device(device2.clone())
            .unwrap()
            .into_vec()
            .unwrap();
        if x != y.as_slice() {
            for (x, y) in x.iter().zip(y) {
                assert_eq!(&y, x);
            }
        }
    }
}

fn buffer_fill<T: Scalar>(device: Device) {
    let elem = T::one();
    let n = buffer_test_lengths().last().unwrap();
    let x = (10..20)
        .cycle()
        .map(|x| T::from_u32(x).unwrap())
        .take(n)
        .collect::<Vec<_>>();
    for n in buffer_test_lengths() {
        let x = &x[..n];
        let mut y = Slice::from(x).to_device(device.clone()).unwrap();
        y.fill(elem).unwrap();
        let y: Vec<T> = y.into_vec().unwrap();
        for y in y.into_iter() {
            assert_eq!(y, elem);
        }
    }
}

fn buffer_cast<X: Scalar, Y: Scalar>(device: Device) {
    let n = buffer_test_lengths().last().unwrap();
    let x = (10..20)
        .cycle()
        .map(|x| X::from_u32(x).unwrap())
        .take(n)
        .collect::<Vec<_>>();
    for n in buffer_test_lengths() {
        let x = &x[..n];
        let y = Slice::<X>::from(x)
            .into_device(device.clone())
            .unwrap()
            .cast_into::<Y>()
            .unwrap()
            .into_vec()
            .unwrap();
        for (x, y) in x.iter().zip(y.iter()) {
            assert_eq!(*y, x.cast::<Y>());
        }
    }
}

fn buffer_bitcast<X: Scalar, Y: Scalar>(device: Device) {
    let x_host = &[X::default(); 16];
    let x = Slice::from(x_host.as_ref()).to_device(device).unwrap();
    for i in 0..=16 {
        for range in [i..16, 0..i] {
            let bytemuck_result =
                bytemuck::try_cast_slice::<X, Y>(&x_host[range.clone()]).map(|_| ());
            let result = x.slice(range).unwrap().bitcast::<Y>().map(|_| ());
            #[cfg(miri)]
            let _ = (bytemuck_result, result);
            #[cfg(not(miri))]
            assert_eq!(result, bytemuck_result);
        }
    }
}

#[test]
fn buffer_from_vec_host() {
    buffer_from_vec(Device::host());
}

#[cfg(target_arch = "wasm32")]
macro_for!($T in [u8, i8, u16, i16, f16, bf16, u32, i32, f32, u64, i64, f64] {
    paste! {
        #[test]
        fn [<buffer_fill_ $T _host>]() {
            buffer_fill::<$T>(Device::host());
        }
    }
});

macro_for!($X in [u8, i8, u16, i16, f16, bf16, u32, i32, f32, u64, i64, f64] {
    macro_for!($Y in [u8, i8, u16, i16, f16, bf16, u32, i32, f32, u64, i64, f64] {
        paste! {
            #[test]
            fn [<buffer_cast_ $X _ $Y _host>]() {
                buffer_cast::<$X, $Y>(Device::host());
            }
            #[test]
            fn [<buffer_bitcast_ $X _ $Y _host>]() {
                buffer_bitcast::<$X, $Y>(Device::host());
            }
        }
    });
});
