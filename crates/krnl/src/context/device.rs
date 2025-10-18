use crate::Result;
#[cfg(feature = "device")]
use crate::kernel::{KernelCreateInfo, KernelDesc, KernelKey};
use bytemuck::{Pod, cast_slice, cast_slice_mut};
use parking_lot::Mutex;
use std::{
    marker::PhantomData,
    sync::{Arc, Weak},
};

#[cfg(feature = "device")]
mod backend;
#[cfg(feature = "device")]
use backend::{
    Backend as _, Buffer as _, Device as _, DeviceOwned, DeviceSpecifier, Event as _, Kernel as _,
    Slice as _,
    backend_impl::{
        Backend, Buffer as RawBuffer, Device as RawDevice, Event as RawEvent, Kernel as RawKernel,
        Slice as RawSlice,
    },
};

pub const ENABLED: bool = cfg!(feature = "device");

#[cfg(feature = "device")]
fn backend() -> Result<Arc<Backend>> {
    #[cfg(not(all(target_family = "wasm", target_feature = "atomics")))]
    {
        static BACKEND: Mutex<Weak<Backend>> = Mutex::new(Weak::new());
        let mut guard = BACKEND.lock();
        if let Some(backend) = Weak::upgrade(&guard) {
            return Ok(backend);
        }
        let backend = Backend::create()?;
        *guard = Arc::downgrade(&backend);
        Ok(backend)
    }
    #[cfg(all(target_family = "wasm", target_feature = "atomics"))]
    {
        use std::cell::RefCell;
        thread_local! {
            static BACKEND: RefCell<Weak<Backend>> = RefCell::new(Weak::new());
        }
        BACKEND.with_borrow_mut(|guard| {
            if let Some(backend) = Weak::upgrade(&guard) {
                return Ok(backend);
            }
            let backend = Backend::create();
            if let Ok(backend) = backend.as_ref() {
                *guard = Arc::downgrade(backend);
            }
            backend
        })
    }
}

#[derive(Default)]
pub struct DeviceBuilder {
    #[cfg(all(feature = "device", not(target_family = "wasm")))]
    specifier: Option<DeviceSpecifier>,
}

impl DeviceBuilder {
    #[cfg(not(target_family = "wasm"))]
    pub fn index(self, index: usize) -> Self {
        #[cfg(not(feature = "device"))]
        let _ = index;
        Self {
            #[cfg(feature = "device")]
            specifier: Some(DeviceSpecifier::Index(index)),
            ..self
        }
    }
    #[cfg(not(target_family = "wasm"))]
    pub fn build(self) -> Result<Device> {
        #[cfg(feature = "device")]
        {
            let specifier = if let Some(specifier) = self.specifier {
                specifier
            } else {
                if let Ok(var) = std::env::var("KRNL_DEVICE") {
                    let index = var.parse().unwrap();
                    DeviceSpecifier::Index(index)
                } else {
                    DeviceSpecifier::Index(0)
                }
            };
            let raw = RawDevice::create(backend()?, specifier)?;
            Ok(Device { raw })
        }
        #[cfg(not(feature = "device"))]
        {
            todo!()
        }
    }
    #[cfg(target_family = "wasm")]
    pub async fn build_async(self) -> Result<Device> {
        #[cfg(feature = "device")]
        {
            let raw = RawDevice::create_async(backend()?).await?;
            Ok(Device { raw })
        }
        #[cfg(not(feature = "device"))]
        {
            todo!()
        }
    }
}

#[derive(Clone)]
pub struct Device {
    #[cfg(feature = "device")]
    raw: Arc<RawDevice>,
}

impl PartialEq for Device {
    fn eq(&self, other: &Self) -> bool {
        #[cfg(feature = "device")]
        {
            Arc::ptr_eq(&self.raw, &other.raw)
        }
        #[cfg(not(feature = "device"))]
        {
            unreachable!()
        }
    }
}

impl Eq for Device {}

impl Device {
    pub fn builder() -> DeviceBuilder {
        DeviceBuilder::default()
    }
    #[cfg(not(target_family = "wasm"))]
    pub fn wait(&self) -> Result<()> {
        self.event().wait()
    }
    pub fn event(&self) -> Event {
        Event {
            #[cfg(feature = "device")]
            raw: self.raw.event(),
        }
    }
    pub fn min_subgroup_threads(&self) -> usize {
        #[cfg(feature = "device")]
        {
            self.raw.properties().min_subgroup_threads()
        }
        #[cfg(not(feature = "device"))]
        {
            unreachable!()
        }
    }
    pub fn max_subgroup_threads(&self) -> usize {
        #[cfg(feature = "device")]
        {
            self.raw.properties().max_subgroup_threads()
        }
        #[cfg(not(feature = "device"))]
        {
            unreachable!()
        }
    }
    pub fn default_subgroup_threads(&self) -> usize {
        self.max_subgroup_threads()
    }
}

#[derive(Clone)]
pub struct Event {
    #[cfg(feature = "device")]
    raw: Arc<RawEvent>,
}

impl Event {
    #[cfg(not(target_family = "wasm"))]
    pub fn wait(&self) -> Result<()> {
        #[cfg(feature = "device")]
        {
            self.raw.wait()?;
        }
        Ok(())
    }
    #[cfg(target_family = "wasm")]
    pub async fn wait_async(&self) -> Result<()> {
        #[cfg(feature = "device")]
        {
            self.raw.wait_async().await?;
        }
        Ok(())
    }
}

#[cfg(feature = "device")]
pub(crate) struct Buffer<T> {
    raw: Arc<RawBuffer>,
    _m: PhantomData<T>,
}

#[cfg(feature = "device")]
impl<T> Buffer<T> {
    pub(super) fn device(&self) -> Device {
        Device {
            raw: self.raw.device().clone(),
        }
    }
    pub(super) fn len(&self) -> usize {
        self.raw.len() / size_of::<T>()
    }
    pub(super) unsafe fn uninit(device: Device, len: usize) -> Result<Self> {
        Ok(Self {
            raw: unsafe { RawBuffer::uninit(device.raw, len * size_of::<T>())? },
            _m: PhantomData,
        })
    }
    pub(super) fn as_slice(&self) -> Slice<'_, T> {
        Slice {
            raw: self.raw.clone().into_slice(),
            _m: PhantomData,
        }
    }
    pub(super) fn as_slice_mut(&mut self) -> SliceMut<'_, T> {
        SliceMut {
            raw: self.raw.clone().into_slice(),
            _m: PhantomData,
        }
    }
}

#[cfg(feature = "device")]
pub(crate) struct Slice<'a, T> {
    raw: Arc<RawSlice>,
    _m: PhantomData<&'a T>,
}

#[cfg(feature = "device")]
impl<T> Clone for Slice<'_, T> {
    fn clone(&self) -> Self {
        Self {
            raw: self.raw.clone(),
            _m: PhantomData,
        }
    }
}

#[cfg(feature = "device")]
impl<T> Slice<'_, T> {
    pub(super) fn device(&self) -> Device {
        Device {
            raw: self.raw.device().clone(),
        }
    }
    pub(super) fn len(&self) -> usize {
        self.raw.len() / size_of::<T>()
    }
    pub(super) fn as_slice(&self) -> Slice<'_, T> {
        Slice {
            #[cfg(feature = "device")]
            raw: self.raw.clone(),
            _m: PhantomData,
        }
    }
}

#[cfg(feature = "device")]
impl<T: Pod> Slice<'_, T> {
    #[cfg(not(target_family = "wasm"))]
    pub(super) fn download(&self, slice: &mut [T]) -> Result<()> {
        unsafe { self.raw.download(cast_slice_mut(slice)) }
    }
    #[cfg(target_family = "wasm")]
    pub(super) async fn download_async(&self, slice: &mut [T]) -> Result<()> {
        unsafe { self.raw.download_async(cast_slice_mut(slice)).await }
    }
}

#[cfg(feature = "device")]
pub(crate) struct SliceMut<'a, T> {
    raw: Arc<RawSlice>,
    _m: PhantomData<&'a mut T>,
}

#[cfg(feature = "device")]
impl<T> SliceMut<'_, T> {
    pub(super) fn device(&self) -> Device {
        Device {
            raw: self.raw.device().clone(),
        }
    }
    pub(super) fn len(&self) -> usize {
        self.raw.len() / size_of::<T>()
    }
    pub(super) fn as_slice(&self) -> Slice<'_, T> {
        Slice {
            raw: self.raw.clone(),
            _m: PhantomData,
        }
    }
    pub(super) fn as_slice_mut(&mut self) -> SliceMut<'_, T> {
        SliceMut {
            raw: self.raw.clone(),
            _m: PhantomData,
        }
    }
}

#[cfg(feature = "device")]
impl<'a, T: Pod> SliceMut<'a, T> {
    pub(super) fn upload(&mut self, slice: &[T]) -> Result<()> {
        unsafe { self.raw.upload(cast_slice(slice)) }
    }
    pub(super) fn try_bitcast_mut<Y>(self) -> Option<SliceMut<'a, Y>> {
        if self.raw.range().is_aligned_to(size_of::<Y>()) {
            Some(SliceMut {
                raw: self.raw,
                _m: PhantomData,
            })
        } else {
            None
        }
    }
}

#[cfg(feature = "device")]
pub(crate) struct Kernel {
    raw: Arc<RawKernel>,
}

#[cfg(feature = "device")]
impl Kernel {
    pub(crate) fn device(&self) -> Device {
        Device {
            raw: self.raw.device().clone(),
        }
    }

    pub(crate) fn get_or_create(
        device: Device,
        key: KernelKey,
        f: impl FnOnce() -> KernelCreateInfo,
    ) -> Result<Self> {
        let raw = RawKernel::get_or_create(device.raw, key, f)?;
        Ok(Self { raw })
    }
    pub(crate) fn desc(&self) -> &KernelDesc {
        self.raw.desc()
    }
    pub(crate) unsafe fn exec(
        &self,
        groups: usize,
        buffers: &BufferBindingVec,
        push_constants: &[u8],
    ) -> Result<()> {
        unsafe {
            self.raw
                .exec(groups.try_into().unwrap(), &buffers.0, push_constants)
        }
    }
}

#[cfg(feature = "device")]
type RawBufferBinding = backend::BufferBinding<RawSlice>;

#[cfg(feature = "device")]
pub(crate) struct BufferBindingVec(Vec<RawBufferBinding>);

#[cfg(feature = "device")]
impl BufferBindingVec {
    pub(crate) fn set_slice<T: Pod>(&mut self, index: usize, slice: &Slice<T>) {
        self.0[index] = RawBufferBinding {
            slice: slice.raw.clone(),
            mutable: false,
        };
    }

    pub(crate) fn with_capacity(capacity: usize) -> Self {
        Self(Vec::with_capacity(capacity))
    }
    pub(crate) fn push_slice<T: Pod>(&mut self, slice: &Slice<T>) {
        self.0.push(RawBufferBinding {
            slice: slice.raw.clone(),
            mutable: false,
        });
    }
    pub(crate) fn push_slice_mut<T: Pod>(&mut self, slice: &mut SliceMut<T>) {
        self.0.push(RawBufferBinding {
            slice: slice.raw.clone(),
            mutable: true,
        });
    }
}
