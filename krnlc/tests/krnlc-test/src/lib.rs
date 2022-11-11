mod bar {
    use krnl_macros::module;

    #[module(
        dependencies(
            "spirv-std = { git = \"https://github.com/EmbarkStudios/rust-gpu\", rev = \"9e2e66729d440fdbf53ce21d173b683240aa2805\" }"
        ),
    )]
    mod foo {
        #[cfg(target_arch = "spirv")]
        #[allow(dead_code)]
        #[spirv(compute(threads(1)))]
        pub fn baz(#[spirv(storage_buffer, descriptor_set = 0, binding = 0)] y: &mut [u32]) {
            y[0] = 1;
        }
    }
}
