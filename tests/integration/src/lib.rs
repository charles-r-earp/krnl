use dry::macro_for;
#[cfg(feature = "device")]
use krnl::context::Device;
#[cfg(test)]
use krnl::{
    buffer::{Buffer, Slice},
    context::Context,
};
use paste::paste;
#[cfg(feature = "device")]
use {maybe_async::maybe_async, std::sync::OnceLock};

#[cfg(target_family = "wasm")]
use wasm_bindgen_test::{wasm_bindgen_test as test, wasm_bindgen_test_configure};

#[cfg(all(feature = "run_in_browser", target_family = "wasm"))]
wasm_bindgen_test_configure!(run_in_browser);

#[cfg(feature = "device")]
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

macro_for!($n in [1, 10, 100, 1_000, 1_000_000, 100_000_000] {
    paste! {
        #[test]
        fn [<buffer_into_context_u32_ $n _host>]() {
            let x: Vec<u32> = (1..=21).take($n).collect();
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
            let x: Vec<u32> = (1..=21).take($n).collect();
            let y = Slice::from(x.as_slice())
                .into_context(device.into())
                .unwrap();
            #[cfg(not(target_family = "wasm"))]
            let y = y.into_vec().unwrap();
            #[cfg(target_family = "wasm")]
            let y = y.into_vec_async().await.unwrap();
            assert_eq!(x, y);
        }
    }
});
