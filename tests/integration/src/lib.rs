use dry::macro_for;
#[cfg(all(test, feature = "device"))]
use krnl::context::Device;
#[cfg(test)]
use krnl::{
    buffer::{Buffer, Slice},
    context::Context,
    scalar::{bf16, f16},
};
#[cfg(feature = "device")]
use maybe_async::maybe_async;
#[cfg(test)]
use num_traits::{AsPrimitive, Bounded};
use paste::paste;
#[cfg(all(test, feature = "device"))]
use std::sync::OnceLock;

#[cfg(target_family = "wasm")]
use wasm_bindgen_test::{wasm_bindgen_test as test, wasm_bindgen_test_configure};

#[cfg(target_family = "wasm")]
wasm_bindgen_test_configure!(run_in_browser);

#[cfg(all(test, feature = "device"))]
#[maybe_async]
async fn test_device() -> Device {
    static DEVICE: OnceLock<Device> = OnceLock::new();

    #[cfg(not(target_family = "wasm"))]
    {
        DEVICE
            .get_or_init(|| Device::builder().build().unwrap())
            .clone()
    }
    #[cfg(target_family = "wasm")]
    {
        if let Some(device) = DEVICE.get().cloned() {
            return device;
        }
        let device = Device::builder().build_async().await.unwrap();
        DEVICE.set(device.clone()).ok().unwrap();
        device
    }
}

#[cfg(feature = "device")]
#[maybe_async]
#[test]
async fn create_device() {
    test_device().await;
}

#[test]
fn buffer_uninit_host() {
    unsafe {
        Buffer::<u32>::uninit(Context::Host, 1).unwrap();
    }
}

#[cfg(feature = "device")]
#[maybe_async]
#[test]
async fn buffer_uninit_device() {
    let device = test_device().await;
    unsafe {
        Buffer::<u32>::uninit(device.into(), 1).unwrap();
    }
}

#[cfg(feature = "device")]
#[maybe_async]
#[test]
async fn device_event() {
    let device = test_device().await;
    let _y = Buffer::from(vec![1u32])
        .into_context(device.clone().into())
        .unwrap();
    let event = device.event();
    #[cfg(not(target_family = "wasm"))]
    event.wait().unwrap();
    #[cfg(target_family = "wasm")]
    event.wait_async().await.unwrap();
}

macro_for!($n in [1, 10, 100, 1_000, 1_000_000, 32_000_000] {
    paste! {
        #[test]
        fn [<buffer_into_context_u32_ $n _host>]() {
            let x: Vec<u32> = gen_fill_vec($n);
            let y = Slice::from(x.as_slice())
                .into_context(Context::Host)
                .unwrap();
            let y = y.into_vec().unwrap();
            assert_eq!(x, y);
        }

        #[cfg(feature = "device")]
        #[maybe_async]
        #[test]
        async fn [<buffer_into_context_u32_ $n _device>]() {
            let device = test_device().await;
            let x: Vec<u32> = gen_fill_vec($n);
            let y = Slice::from(x.as_slice())
                .into_context(device.clone().into())
                .unwrap();
            #[cfg(not(target_family = "wasm"))]
            let y = y.into_vec().unwrap();
            #[cfg(target_family = "wasm")]
            let y = y.into_vec_async().await.unwrap();
            assert_eq!(x, y);
        }
    }
});

#[cfg(test)]
fn gen_fill_vec<T: Bounded + AsPrimitive<f64>>(n: usize) -> Vec<T>
where
    usize: AsPrimitive<T> + Copy,
{
    let y_max: f64 = T::max_value().as_();
    let n_max = y_max.round() as usize;
    (1..n_max).cycle().take(n).map(|x| x.as_()).collect()
}

macro_for!($n in [1, 10, 100, 1000] {
    macro_for!($T in [u8, i8, u16, i16, f16, bf16, u32, i32, f32, u64, i64, f64] {
        paste! {
            #[test]
            fn [<fill_ $T _ $n _host>]() {
                let mut y = Buffer::from(gen_fill_vec($n));
                y.fill($T::MAX).unwrap();
                let y = y.into_vec().unwrap();
                assert_eq!(y, vec![$T::MAX; $n]);
            }

            #[cfg(all(feature = "device", not(target_family = "wasm")))]
            #[test]
            fn [<fill_ $T _ $n _device>]() {
                let device = test_device();
                let mut y = Buffer::from(gen_fill_vec($n))
                    .into_context(device.into())
                    .unwrap();
                y.fill($T::MAX).unwrap();
                let y = y.into_vec().unwrap();
                assert_eq!(y, vec![$T::MAX; $n]);
            }

            #[test]
            fn [<copy_ $T _ $n _host>]() {
                let x_vec = gen_fill_vec($n);
                let x = Slice::from(x_vec.as_slice());
                let mut y = Buffer::from(vec![$T::default(); x.len()]);
                y.copy_from_slice(x.as_slice()).unwrap();
                let y = y.into_vec().unwrap();
                assert_eq!(x_vec, y);
            }

            #[cfg(all(feature = "device", not(target_family = "wasm")))]
            #[test]
            fn [<copy_ $T _ $n _host_to_device>]() {
                let device = test_device();
                let x_vec = gen_fill_vec($n);
                let x = Slice::from(x_vec.as_slice());
                let mut y = Buffer::from(vec![$T::default(); x.len()])
                    .into_context(device.clone().into())
                    .unwrap();
                y.copy_from_slice(x.as_slice()).unwrap();
                let y = y.into_vec().unwrap();
                assert_eq!(x_vec, y);
            }

            #[cfg(all(feature = "device", not(target_family = "wasm")))]
            #[test]
            fn [<copy_ $T _ $n _device_to_host>]() {
                let device = test_device();
                let x_vec = gen_fill_vec($n);
                let x = Slice::from(x_vec.as_slice())
                    .into_context(device.clone().into())
                    .unwrap();
                let mut y = Buffer::from(vec![$T::default(); x.len()]);
                y.copy_from_slice(x.as_slice()).unwrap();
                let y = y.into_vec().unwrap();
                assert_eq!(x_vec, y);
            }

            #[cfg(all(feature = "device", not(target_family = "wasm")))]
            #[test]
            fn [<copy_ $T _ $n _device_to_device>]() {
                let device = test_device();
                let x_vec = gen_fill_vec($n);
                let x = Slice::from(x_vec.as_slice())
                    .into_context(device.clone().into())
                    .unwrap();
                let mut y = Buffer::from(vec![$T::default(); x.len()])
                    .into_context(device.clone().into())
                    .unwrap();
                y.copy_from_slice(x.as_slice()).unwrap();
                let y = y.into_vec().unwrap();
                assert_eq!(x_vec, y);
            }
        }
    });

    #[cfg(all(feature = "device", target_family = "wasm"))]
    macro_for!($T in [u32, i32, f32] {
        paste! {
            #[test]
            async fn [<fill_ $T _ $n _device>]() {
                let device = test_device().await;
                let mut y = Buffer::from(gen_fill_vec($n))
                    .into_context(device.into())
                    .unwrap();
                y.fill($T::MAX).unwrap();
                let y = y.into_vec_async().await.unwrap();
                assert_eq!(y, vec![$T::MAX; $n]);
            }

            #[test]
            async fn [<copy_ $T _ $n _host_to_device>]() {
                let device = test_device().await;
                let x_vec = gen_fill_vec($n);
                let x = Slice::from(x_vec.as_slice());
                let mut y = Buffer::from(vec![$T::default(); x.len()])
                    .into_context(device.clone().into())
                    .unwrap();
                y.copy_from_slice(x.as_slice()).unwrap();
                let y = y.into_vec_async().await.unwrap();
                assert_eq!(x_vec, y);
            }

            #[test]
            async fn [<copy_ $T _ $n _device_to_device>]() {
                let device = test_device().await;
                let x_vec = gen_fill_vec($n);
                let x = Slice::from(x_vec.as_slice())
                    .into_context(device.clone().into())
                    .unwrap();
                let mut y = Buffer::from(vec![$T::default(); x.len()])
                    .into_context(device.clone().into())
                    .unwrap();
                y.copy_from_slice(x.as_slice()).unwrap();
                let y = y.into_vec_async().await.unwrap();
                assert_eq!(x_vec, y);
            }
        }
    });
});

macro_for!($o in [1, 2, 5] {
    macro_for!($T in [u8, i8, u16, i16, f16, bf16, u32, i32, f32, u64, i64, f64] {
        paste! {
            #[test]
            fn [<fill_10_offset_ $o _ $T _host>]() {
                let n = 10;
                let mut y = Buffer::from(gen_fill_vec(n));
                y.as_slice_mut().slice_mut($o..).fill($T::MAX).unwrap();
                let y = y.into_vec().unwrap();
                let mut y_true = gen_fill_vec(n);
                for y in y_true[$o..].iter_mut() {
                    *y = $T::MAX;
                }
                assert_eq!(y, y_true);
            }

            #[cfg(all(feature = "device", not(target_family = "wasm")))]
            #[test]
            fn [<fill_10_offset_ $o _ $T _device>]() {
                let device = test_device();
                let n = 10;
                let mut y = Buffer::from(gen_fill_vec(n))
                    .into_context(device.into())
                    .unwrap();
                y.as_slice_mut().slice_mut($o..).fill($T::MAX).unwrap();
                let y = y.into_vec().unwrap();
                let mut y_true = gen_fill_vec(n);
                for y in y_true[$o..].iter_mut() {
                    *y = $T::MAX;
                }
                assert_eq!(y, y_true);
            }
        }
    });

    macro_for!($T in [u32, i32, f32] {
        paste! {
            #[cfg(all(feature = "device", target_family = "wasm"))]
            #[test]
            async fn [<fill_10_offset_ $o _ $T _device>]() {
                let device = test_device().await;
                let n = 10;
                let mut y = Buffer::from(gen_fill_vec(n))
                    .into_context(device.into())
                    .unwrap();
                y.as_slice_mut().slice_mut($o..).fill($T::MAX).unwrap();
                let y = y.into_vec_async().await.unwrap();
                let mut y_true = gen_fill_vec(n);
                for y in y_true[$o..].iter_mut() {
                    *y = $T::MAX;
                }
                assert_eq!(y, y_true);
            }
        }
    });
});

#[cfg(test)]
fn gen_cast_vec<X: Bounded + AsPrimitive<f64>, Y: Bounded + AsPrimitive<f64>>(n: usize) -> Vec<X>
where
    usize: AsPrimitive<X> + Copy,
{
    let x_max: f64 = X::max_value().as_();
    let y_max: f64 = Y::max_value().as_();
    let n_max = n.min(x_max.round() as usize).min(y_max.round() as usize);
    (1..=n_max).cycle().take(n).map(|x| x.as_()).collect()
}

macro_for!($n in [1, 10, 100, 1000] {
    macro_for!($X in [u8, i8, u16, i16, f16, bf16, u32, i32, f32, u64, i64, f64] {
        macro_for!($Y in [u8, i8, u16, i16, f16, bf16, u32, i32, f32, u64, i64, f64] {
            paste! {
                #[test]
                fn [<cast_ $X _ $Y _ $n _host>]() {
                    let x_vec: Vec<$X> = gen_cast_vec::<$X, $Y>($n);
                    let y_vec: Vec<$Y> = x_vec.iter().copied().map(|x| x.as_()).collect();
                    let x = Slice::from(x_vec.as_slice());
                    let y = x.cast::<$Y>().unwrap();
                    let y = y.into_vec().unwrap();
                    assert_eq!(y, y_vec);
                }

                #[cfg(all(feature = "device", not(target_family = "wasm")))]
                #[test]
                fn [<cast_ $X _ $Y _ $n _device>]() {
                    let device = test_device();
                    let x_vec: Vec<$X> = gen_cast_vec::<$X, $Y>($n);
                    let y_vec: Vec<$Y> = x_vec.iter().copied().map(|x| x.as_()).collect();
                    let x = Slice::from(x_vec.as_slice())
                        .into_context(device.clone().into())
                        .unwrap();
                    let y = x.cast::<$Y>().unwrap();
                    let y = y.into_vec().unwrap();
                    assert_eq!(y, y_vec);
                }
            }
        });
    });

    #[cfg(all(feature = "device", target_family = "wasm"))]
    macro_for!($X in [u32, i32, f32] {
        macro_for!($Y in [u32, i32, f32] {
            paste! {
                #[test]
                async fn [<cast_ $X _ $Y _ $n _device>]() {
                    let device = test_device().await;
                    let x_vec: Vec<$X> = gen_cast_vec::<$X, $Y>($n);
                    let y_vec: Vec<$Y> = x_vec.iter().copied().map(|x| x as $Y).collect();
                    let x = Slice::from(x_vec.as_slice())
                        .into_context(device.clone().into())
                        .unwrap();
                    let y = x.cast::<$Y>().unwrap();
                    let y = y.into_vec_async().await.unwrap();
                    assert_eq!(y, y_vec);
                }
            }
        });
    });
});

#[cfg(feature = "device")]
#[maybe_async]
#[test]
async fn group_buffer_const_64() {
    let device = test_device().await;
    let n = 64;
    let x_vec: Vec<u32> = (1..=n as u32).collect();
    let y_vec = vec![x_vec.iter().copied().sum()];
    let x = Buffer::from(x_vec)
        .into_context(device.clone().into())
        .unwrap();
    let mut y = Buffer::zeros(device.into(), 1).unwrap();
    unsafe {
        compile_tests::_group_buffer_const_64(x.as_slice(), y.as_slice_mut());
    }
    #[cfg(not(target_family = "wasm"))]
    let y = y.into_vec().unwrap();
    #[cfg(target_family = "wasm")]
    let y = y.into_vec_async().await.unwrap();
    assert_eq!(y, y_vec);
}

// TODO spec constant array lengths don't work on web
// need to be resolved via rspirv
#[cfg(not(target_family = "wasm"))]
macro_for!($n in [1, 7, 32, 64, 128, 256] {
    paste! {
        #[cfg(feature = "device")]
        #[maybe_async]
        #[test]
        async fn [<group_buffer_spec_ $n>]() {
            let device = test_device().await;
            let x_vec: Vec<u32> = (1..=$n).collect();
            let y_vec = vec![x_vec.iter().copied().sum()];
            let x = Buffer::from(x_vec)
                .into_context(device.clone().into())
                .unwrap();
            let mut y = Buffer::zeros(device.into(), 1).unwrap();
            unsafe {
                compile_tests::_group_buffer_spec(x.as_slice(), y.as_slice_mut());
            }
            #[cfg(not(target_family = "wasm"))]
            let y = y.into_vec().unwrap();
            #[cfg(target_family = "wasm")]
            let y = y.into_vec_async().await.unwrap();
            assert_eq!(y, y_vec);
        }
    }
});
