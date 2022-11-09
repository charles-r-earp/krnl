
# compile steps
- Parse module attrs
- check if module already built
- build
    - init .krnl dir
        - compile builder
    - generate device crate
    - compile device crate, multimodule
        - kernel macro writes to descriptors
- load kernel descriptors -> store to thread_local
- kernel macro expansion

# run steps
- Create KernelDesc from spirv
- Modify KernelDesc as necessary
    - remap types
    - mark push constants as padding
- Create Kernel


```rust
#[module(
    dependency("krnl-core", version = "0.0.1"), // krnl-core is auto included if not specified
)]
mod axpy {
    use krnl_core::kernel;

    #[kernel(threads(256)]
    pub fn saxpy(global_id: u32, global_threads: u32, #[global] x: &[f32], #[push] alpha: T, #[global] y: &mut UnsafeMut<[f32]>) {
        use krnl_core::spirv_std::arch::IndexUnchecked;

        let idx = global_id as usize;
        let global_threads = global_threads as usize;
        let y = unsafe {
            y.unsafe_mut()
        };
        let items = min(x.len(), y.len());
        while idx < items {
            unsafe {
                *y.index_unchecked_mut(idx) += alpha * *x.index_unchecked(idx);
            }
            idx += global_threads;
        }
    }

    #[kernel(threads(256), for_each]
    pub fn saxpy_for_each(#[item] x: &T, #[push] alpha: T, #[item] y: &mut T) {
        *y += alpha * *x;
    }

    /* generates */
    pub mod saxpy {
        pub struct KernelProto {
            inner: KernelProtoInner,
        }

        impl KernelProto {
            pub fn new() -> Result<KernelProto> {
                let spirv = todo!();
                let inner = KernelProtoInner::builder()
                    .spirv_bytes(&spirv)
                    .build()?;
                Ok(KernelProto {
                    inner,
                })
            }
            pub fn kernel(&self, device: Device) -> Result<Kernel> {
                todo!()
            }s
        }

        pub struct Kernel {

        }

        impl Kernel {
            pub fn dispatch(dim: impl Into<DispatchDim<[u32; 1]>>, args: DispatchArgs) -> Result<()> {
                todo!()
            }
        }

        pub struct DispatchArgs<'a> {
            x: Slice<'a, f32>,
            alpha: f32,
            y: SliceMut<'a, f32>,
        }
    }

    pub mod saxpy_for_each {
        pub struct KernelProto {
            inner: KernelProtoInner,
        }

        impl KernelProto {
            pub fn new() -> Result<KernelProto> {
                let spirv = todo!();
                let inner = KernelProtoInner::builder()
                    .spirv_bytes(&spirv)
                    .build()?;
                Ok(KernelProto {
                    inner,
                })
            }
            pub fn kernel(&self, device: Device) -> Result<Kernel> {
                todo!()
            }
        }

        pub struct Kernel {
            inner: KernelInner,
        }

        impl Kernel {
            pub fn dispatch(&self, dim: impl Into<DispatchDim<[u32; 1]>>, args: DispatchArgs) -> Result<()> {
                let items = x.len();
                for len in [y.len()] {
                    if len != items {
                        bail!("");
                    }
                }
                todo!()
            }
            pub fn inner(&self) -> &KernelInner {
                todo!()
            }
        }

        pub struct DispatchArgs<'a> {
            x: Slice<'a, f32>,
            alpha: f32,
            y: SliceMut<'a, f32>,
        }
    }
}

fn main() -> Result<()> {
    let device = Device::new(0)?;
    let x = Buffer::from_vec(vec![1f32]).into_device(device.clone())?.block()?;
    let alpha = 2f32;
    let mut y = Buffer::from_vec(vec![0f32]).into_device(device.clone())?.block()?;
    static PROTO: OnceCell<saxpy_for_each::KernelProto> = OnceCell::new();
    let proto = PROTO.get_or_try_init(|| saxpy_for_each::proto)?;
    let kernel = proto.kernel(device.clone())?;
    let dim = if let Some(groups) = kernel.inner().concurrent_groups() {
        DispatchDim::Groups(groups)
    } else {
        DispatchDim::GlobalThreads(x.len() as u32)
    }
    kernel.dispatch(dim, saxpy_for_each::DispatchArgs {
        x: x.as_slice(),
        alpha,
        y: y.as_slice_mut(),
    })?;
    kernel.inner().cache(true);
}

```

```rust

pub enum ThreadDesc {
    Value(u32),
    SpecId(u32),
}

pub struct SpecDesc {
    id: u32,
    name: Option<Cow<'static, str>>,
    value: ScalarType,
}

pub struct SliceDesc {
    binding: u32,
    name: Option<Cow<'static, str>>,
    scalar_type: ScalarType,
}

pub struct PushDesc {
    offset: u32,
    name: Option<Cow<'static, str>>,
    scalar_type: ScalarType,
    padding: bool,
}

pub struct Spirv {
    words: Vec<u32>,
}

pub struct KernelDesc {
    spirv: Spirv,
    name: Cow<'static, str>,
    path: Option<Cow<'static, str>>,
    thread_descs: [Option<ThreadDesc>; 3],
    spec_descs: Vec<SpecDesc>,
    slice_descs: Vec<SliceDesc>,
    push_descs: Vec<PushDesc>,
}

pub enum SpecBinding {
    SpecId(u32),
    Name(Cow<'static, str>),
}

pub enum SliceBinding {
    Binding(u32),
    Name(Cow<'static, str>),
}

pub enum PushBinding {
    Index(usize),
    Name(Cow<'static, str>),
}

pub struct KernelProto {

}

impl KernelProto {
    pub fn builder_from_spirv_bytes(&[u8]) -> Result<KernelProtoBuilder> {
        todo!()
    }
    pub fn kernel_builder(&self) -> KernelBuilder {
        todo!()
    }
}

pub struct KernelProtoBuilder {
    builder: Builder,
    desc: KernelDesc,
}

impl KernelProtoBuilder {
    pub fn build(self) -> Result<KernelProto> {
        todo!()
    }
}

pub struct KernelBuilder {

}

impl KernelBuilder {
    pub fn build(self) -> Result<Kernel> {
        todo!()
    }
}

pub struct Kernel {
    cache: Arc<KernelCache>,
}

impl Kernel {
    pub fn dispatch_builder(&self) -> DispatchBuilder {
        todo!()
    }
}

pub struct DispatchBuilder<'a> {
    _m: PhantomData<&'a ()>,
}

impl<'a> DispatchBuilder<'a> {
    pub fn global_threads(mut self, global_threads: impl AsRef<[u32]>) -> Self {
        todo!()
    }
    pub fn groups(mut self, groups: impl AsRef<[u32]>) -> Self {
        todo!()
    }
    pub fn slice<'s: 'a>(mut self, binding: impl Into<SliceBinding>, slice: ScalarSlice<'s>) -> DispatchBuilder<'s> {
        todo!()
    }
    pub fn slice_mut<'s: 'a>(mut self, binding: impl Into<SliceBinding>, slice: ScalarSliceMut<'s>) -> DispatchBuilder<'s> {
        todo!()
    }
    pub fn push(mut self, binding: impl Into<PushBinding>, push: ScalarElem) -> Self {
        todo!()
    }
    pub fn dispatch(self) -> Result<()> {
        todo!("check safe");
        unsafe {
            self.unsafe_dispatch()
        }

    }
    pub unsafe fn unsafe_dispatch(self) -> Result<()> {
        todo!()
    }
}

```

```rust

#[module]
mod foo {
    #[kernel(
        foreach,
        threads(TS),
        generics(
            X(u8, i8, u32, i32, f32, f16, bf16, u32, i32, f32, u64, i64, f64),
            Y(u8, i8, u32, i32, f32, f16, bf16, u32, i32, f32, u64, i64, f64),
        ),
    )]
    pub fn cast<X, Y, #[spec] TS: u32>(
        #[item] x: &X,
        #[item] y: &mut Y,
    ) {
        *y = x as _;
    }

    #[kernel(group_threads(1)]
    pub fn foo<spec: N: u32>(
        /* attrs */
        #[global] x: &[f32],
        #[global] x: &mut UnsafeMut<[f32]>,
        #[group] x: &mut UnsafeMut<[u32; 10]>,
        #[subgroup] x: &mut [i32; 10],
        #[thread] x: &mut [u32; n],
        #[push] alpha: f32,
        #[builtin] global_threads: UVec2,
        /* global types */
        #[global] x: &[u32],
        #[global] y: &mut UnsafeMut<[i32]>,
        /* builtins */
        // Vector impl From<u32 or [u32; 2] or [u32; 3]> based on Dimensionality
        // set by threads(..)
        global_threads: Vector,
        global_index: u32,
        global_id: Vector,
        groups: Vector,
        group_index: u32,
        group_id: Vecor,
        subgroups: Vector,
        subgroup_index: u32,
        subgroup_id: Vector,
        threads: Vector,
        thread_index: Vector,
        thread_id: Vector,
        /* for_each */
        items: u32,
        item_id: Vector,
        item_index: u32,
        #[item] x: &u32,
        #[item] y: &mut f32,
    ) {
        w.load();
        w.store();
    }
}

```

# scalar_fn

```rust

#[scalar_fn(
    generics(T(F32, f64))
)]
pub fn scale_mut<T>(#[elem] x: T, #[slice] y: &mut SliceMut<T>) -> Result<()> {
    y.as_host_slice_mut()
        .ok_or_else(|| todo!())?
        .iter_mut()
        .for_each(|y| *y *= x);
}

/* generates */
pub fn scale_mut(x: ScalarElem, y: &mut ScalarSliceMut) -> Result<()> {
    if let Some((x, y)) = f32::try_from(x).ok().zip(y.try_as_slice_mut()) {
        y.as_host_slice_mut()
            .ok_or_else(|| todo!())?
            .iter_mut()
            .for_each(|y| *y *= x);
    } else if let Some((x, y)) = f64::try_from(x).ok().zip(y.try_as_slice_mut()) {
        y.as_host_slice_mut()
            .ok_or_else(|| todo!())?
            .iter_mut()
            .for_each(|y| *y *= x);
    }
}

```

```rust

#[module]
pub mod foo {
    // etc

    /* generated */
    pub mod module {
        pub fn proto(name: &str) -> Result<KernelProto> {
            todo!()
        }
    }
}

```

# krnlc

Compiler for krnl

Creates cache at $CARGO_MANIFEST_DIR .krnl/cache and tag .krnl/CASHDIRTAG

module macro
```rust
#[module]
pub mod kernels {

    /* generated */
    mod module {
        _ = {
            let bytes = module(module_path!());
            let success =
                bytes.len() == len
                && bytes[0] == other[0]
                && bytes[0] == other[1]
                ..;
            if !success {
                panic!("module modified!");
            }
        };

        include!(concat!(env!("CARGO_MANIFEST_DIR"), "/.krnl/cache"));
        /* cache */
        const fn __module(path: &'static str) -> Option<&'static [u8]> {
            todo!()
        }
        pub(super) const fn __kernel(path: &'static str) -> Option<&'static [u8]> {
            todo!()
        }
    }
}
