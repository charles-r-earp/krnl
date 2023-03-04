```rust

#[module]
mod foo {
    #[kernel(foreach, threads(TS))]
    pub fn cast_u32_f32<#[spec] const TS: u32>(
        #[item] x: u32,
        #[item] y: &mut f32,
    ) {
        *y = x as _;
    }

    #[kernel(threads(256))]
    pub fn saxpy(#[global] x: &[f32], alpha: f32, #[global] y: &mut [f32]) {
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