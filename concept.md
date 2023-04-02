```rust

#[module]
mod foo {    
    #[kernel]
    pub fn saxpy(
        #[item] x: f32,
        alpha: f32,
        #[item] y: &mut f32,
    ) {
        *y = alpha * x;
    }
    
    /* generates */
    fn saxpy(
        
    ) {
        let __krnl_items = __krnl_push_consts.__krnl_len_x.max(__krnl_push_consts.__krnl_len_y);
        let mut kernel = KernelBase::from(KernelInner {
            groups,
            group_id,
            threads,
            thread_id,
            items: __krnl_items,
            item_id,
        });
        while {
            saxpy(
                &kernel, 
                x[kernel.item_id() as usize + x_offset], 
                unsafe { 
                    y.unsafe_index_mut(kernel.item_id() as usize)
                }
            );
            kernel.next_item();
        }
    }
    
    
    #[kernel]
    pub fn saxpy(
        #[global] x: &[f32],
        alpha: f32,
        #[global] y: &mut [f32],
    ) {
        let idx = kernel.global_id() as usize;
        if idx < x.len().min(y.len()) {
            unsafe {
                *y.unsafe_index_mut(idx) = alpha * x[idx];
            }
        }
    }
    
    #[kernel]
    pub fn saxpy(
        #[global] x: Slice<f32>,
        alpha: f32,
        #[global] y: UnsafeSliceMut<f32>,
        #[group] y_group: UnsafeArrayMut<f32>,
        #[group] z_group: UnsafeArrayMut<f32, 100>,
    ) {
        let idx = kernel.global_id() as usize;
        if idx < x.len().min(y.len()) {
            unsafe {
                *y.unsafe_index_mut(idx) = alpha * x[idx];
            }
        }
    }
    
    #[kernel]
    pub fn foo<
        #[spec] const M: u32, 
        #[spec] const N: u32,
        #[spec] const (MxN, Q) = { (M * N, 2 * N) },
    >(
        #[group]
        x_group: UnsafeSliceMut<f32, {}>,
    ) {
       const MxN: u32 = M * N;
       
       __krnl_kernel_data[0] = MxN;
    }
    
    
    
    struct KernelBase<T> {
        
    }
    
    impl<T> KernelBase<T> {
        fn global_threads(&self) -> T;
        fn global_id(&self) -> T;
        fn global_index(&self) -> u32;
        fn groups(&self) -> T;
        fn group_id(&self) -> T;
        fn group_index(&self) -> u32;
        fn threads(&self) -> T;
        fn thread_id(&self) -> T;
        fn thread_index(&self) -> u32;
        fn items(&self) -> u32;
        fn item_id(&self) -> u32;
    }
    
    #[kernel]
    pub fn bias(
        kernel: &Kernel1,
        #[global] x: &[f32],
        #[item] y: &mut f32,
    ) {
        let x_idx = kernel.item_id() as usize % x.len();
        *y += x[x_idx];
    }
    
    
    #[kernel]
    pub fn transpose_naive(
        kernel: &Kernel2,   
        #[global] input: &[f32],
        #[global] output: &mut [f32],
        output_rows: u32,
        output_cols: u32,
    ) {
        let global_id = kernel.global_id();
        let [output_col, output_row] = global_id.to_array();
        let [input_row, input_col] = [output_col, output_row];
        let [input_rows, input_cols] = [output_cols, output_rows];
        if output_row < output_rows && output_col < output_cols {
            let input_idx = (input_row * input_cols + input_col) as usize;
            let output_idx = (output_row + output_rows * output_col) as usize;
            unsafe {
                *output.unsafe_index_mut(output_idx) = input[input_idx];
            }
        } 
    }
    
    #[kernel(threads(128))]
    pub fn sum(
        #[global] x: &[u32],
        #[group] x_group: &mut [u32; 128],
        #[global] y: &mut [u32],
    ) {
        use krnl_core::spirv_std::workgroup_memory_barrier;
        
        let global_id = kernel.global_id() as usize;
        let group_id = kernel.group_id() as usize;
        let thread_id = kernel.thread_id() as usize;
        if global_id < x.len() {
            unsafe { 
                *x_group.unsafe_index_mut(group_id) = x[global_id];
            }
        }
        unsafe {
            workgroup_memory_barrier();
        }
        if thread_id == 0 {
            for i in 0 .. x.len() % x_group.len() {
                unsafe {
                    *y.unsafe_index_mut(group_id) += *x_group.unsafe_index_mut(i); 
                }
            }
        }
    }
    
    
    trait UnsafeSliceMutLike<T>: UnsafeIndexMut<usize, Output=T> + Length {}
    
    pub fn axpy<T>(
        x: T,
        alpha: T,
        y: &mut T,
    ) {
        *y += alpha * x;
    }
    
    #[kernel]
    pub fn saxpy(
        kernel: &Kernel<1>,
        #[global] x: &Slice<f32>,
        alpha: f32,
        #[global] y: &UnsafeSliceMut<u32>,
    ) {
        let idx = kernel.global_index();
        if idx < x.len().min(y.len()) {
            axpy(x[idx], alpha, unsafe { y.unsafe_index_mut(idx) });
        }
    }
    
    impl Kernel<'a> {
        pub fn dispatch(self) -> Result<()> {
            
        }
    }
    
    mod saxpy {
        pub struct KernelBuilder {
            
        }
        
        pub fn builder() -> KernelBuilder<'static> {
            todo!()
        }
        
        impl<'a> KernelBuilder<'a> {
            pub fn device(mut self, device: Device) {
                todo!()
            }
            pub fn args<'b>(mut self, x: Slice<'b, f32>, alpha: f32, y: KernelScopedMut<SliceMut<'b, f32>>) -> KernelBuilder<'b> {
                todo!()
            }
            pub fn global_threads(mut self, global_threads: [u32; 1]) -> Self {
                todo!()
            }
            pub fn groups(mut self, groups: [u32; 1]) -> Self {
                todo!()
            }
            pub fn threads(mut self, threads: [u32; 1]) -> Self {
                todo!()
            }
            pub fn build(self) -> Result<Kernel<'a>> {
                todo!()
            }
        }
    }
    
    
    
    
    
    
    #[kernel]
    pub fn sum<#[spec] const X_GROUP: u32>(
        #[builtin] global_id: u32,
        #[builtin] group_id: u32,
        #[builtin] threads: u32,
        #[builtin] thread_id: u32,
        #[global] x: &Slice<f32>,
        #[group] x_group: &UnsafeArrayMut<f32, X_GROUP>, // -> &UnsafeSliceMutLike<f32>
        #[global] y: &mut ItemMut<f32>,
    ) {
        let idx = global_id as usize;
        let thread_id = thread_id as usize;
        if idx < x.len() {
            x_group[thread_id] = x[idx];
        }
        unsafe {
            group_barrier();
        } 
        if thread_id == 0 {
            *y = 0f32;
            for i in 0 .. threads as usize {
                if idx + i < x.len() {
                    *y += x_group[i];
                }
            }
        }
    }
    
    pub mod saxpy {
        pub fn call(x: f32, alpha: f32, y: &mut f32) {
            *y = alpha * x;
        }
        
        impl KernelBuilder {
            pub fn threads(mut self, threads: [u32; 1]) -> Self {
                todo!()
            }
            pub fn build(device: Device) -> Result<Kernel> {
                todo!()
            }
        }
        
        impl Kernel {
            fn global_threads(mut self, global_threads: [u32; 1]) -> Self {
                todo!()
            }
            fn groups(mut self, groups: [u32; 1]) -> Self {
                todo!()
            }
            fn parallel(mut self, parallel: bool) -> Self {
                todo!()
            }
            fn dispatch(mut self, x: Slice<f32>, alpha: f32, y: SliceMut<f32>) -> Result<()> {
                let x = x.as_host_slice().unwrap();
                let y = y.as_host_slice_mut().unwrap();
                y.par_iter_mut().zip(x.iter().copied()).for_each(|(x, y)| call(x, alpha, y));
            }
        }
    }
    
    
    #[kernel(foreach, threads(TS))]
    pub fn cast_u32_f32<#[spec] const TS: u32>(
        #[item] x: u32,
        #[item] y: &mut f32,
    ) {
        *y = x as _;
    }

    #[kernel]
    pub fn saxpy(#[global] x: &Slice<f32>, alpha: f32, #[global] y: &UnsafeSliceMut<f32>) {
        let idx = global_id.x as usize;
        if idx < x.len() && idx < y.len() {
            unsafe {
                *y.unsafe_index_mut(idx) += alpha * x[idx];
            }
        }
    }
    
    /* generated */
    #[cfg(feature = "spirv")]
    #[repr(C)]
    #[allow(non_camel_case_types)]
    pub struct __krnl_saxpyPushConsts {
        alpha: f32,
        #[allow(non_snake_case_idents)]
        __krnl_offset_x: u32,
        #[allow(non_snake_case_idents)]
        __krnl_len_x: u32,
         #[allow(non_snake_case_idents)]
        __krnl_offset_y: u32,
        #[allow(non_snake_case_idents)]
        __krnl_len_y: u32,
    }
    #[cfg(feature = "spirv")]
    #[::krnl_core::spirv_std::spirv(compute(threads(256, 1, 1)))]
    pub fn saxpy(
        #[spirv(num_workgroups)] __krnl_groups: UVec3,
        #[spirv(workgroup_id)] __krnl_group_id: UVec3,
        #[spirv(local_invocation_id)] __krnl_thread_id: UVec3,
        #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] x: &[f32],
        #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] y: &mut [f32],
        #[spirv(push_constant)] __krnl_push_consts: &__krnl_saxpyPushConsts,
    ) {
        let __krnl_threads = ::krnl_core::glam::UVec3::new(256, 1, 1);
        let __krnl_global_id = __krnl_group_id * __krnl_threads + __krnl_thread_id;
        let ref x = unsafe { ::krnl_core::arch::Slice::from_raw_parts(x, __krnl_push_consts.__krnl_offset_x, __krnl_push_consts.__krnl_len_x) };
        let ref y = unsafe { ::krnl_core::arch::UnsafeSliceMut::from_raw_parts(x, __krnl_push_consts.__krnl_offset_y, __krnl_push_consts.__krnl_len_y) };
        fn saxpy(
            x: &impl (::core::ops::Index<usize, Output=f32> + ::krnl_core::arch::Length),
            alpha: f32,
            y: &impl (::krnl_core::arch::UnsafeIndexMut<usize, Output=f32> + ::krnl_core::arch::Length),
            #[allow(unused)]
            global_id: UVec3, 
            #[allow(unused)]
            groups: UVec3,
            #[allow(unused)]
            group_id: UVec3, 
            #[allow(unused)]
            threads: UVec3,
            #[allow(unused)]
            thread_id: UVec3,
        ) {
            let idx = global_id.x as usize;
            if idx < x.len() && idx < y.len() {
                unsafe {
                    *y.unsafe_index_mut(idx) += alpha * x[idx];
                }   
            }   
        }
        saxpy(
            x,
            __krnl_push_consts.alpha,
            y,
            __krnl_global_id,
            __krnl_groups,
            __krnl_group_id,
            __krnl_threads,
            __krnl_thread_id,
        );
    }
    
    #[kernel(for_each, threads(256))]
    pub fn saxpy_for_each(
        #[item] x: f32,
        alpha: f32,
        #[item] y: &mut f32,
    ) {
        *y += alpha * x;
    }

    #[kernel(threads(1)]
    pub fn foo<#[spec] const N: u32>(
        /* attrs */
        #[global] x: &[f32],
        #[global] x: &mut [f32],
        #[group] x: &mut [f32; N],
        /* push constants */
        alpha: f32,
        /* global types */
        #[global] x: &[u32], // => &impl (Index<usize, Output> + Length)
        #[global] y: &mut [i32], // => &mut impl (UnsafeIndexMut<usize, Output> + Length)
        /* builtins */
        global_id: UVec3,
        group_id: UVec3,
        groups: UVec3,
        thread_id: UVec3,
        threads: UVec3,
        /* for_each */
        #[item] x: u32,
        #[item] y: &mut f32,
    ) {
        /* .. */
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

Activates config `krnl_krnlc_compile`

## module macro

```rust
#[module]
pub mod kernels { 
    include!(concat!(env!"CARGO_MANIFEST_DIR"),"/.krnl/krnlc.in")
}

/* options */
#[module]
#[krnl(crate=krnl)]
#[krnl(no_build)]

```


  

## package layout

- my-crate
    - .krnl
        - foo-mod-with-hash
            - bar 
    - target 
        - krnl 
            - lib 
            - my-crate
                - Cargo.toml
                - config.toml
                - src
                    - lib.rs
                - target 

    

## encode data in entry_point
__krnl_ #(#word)_* // <- words of serialized KernelDesc as bincode  

## dependencies 

```toml

[metadata.krnlc]
features = ["device"]

[metadata.krnlc.dependencies]
"krnl-core" = { path = "" } 
```