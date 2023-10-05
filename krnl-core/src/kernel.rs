#[doc(hidden)]
#[cfg(target_arch = "spirv")]
pub mod __private {
    #[repr(simd)]
    pub struct UVec3 {
        pub x: u32,
        pub y: u32,
        pub z: u32,
    }
}

pub struct Kernel {
    /// The number of global threads.
    ///
    /// `global_threads = groups * threads`
    pub global_threads: u32,
    /// The global thread id.
    ///
    /// `global_id = group_id * threads + thread_id`
    pub global_id: u32,
    /// The number of thread groups.
    pub groups: u32,
    /// The group id.
    pub group_id: u32,
    /// The number of subgroups per group.
    pub subgroups: u32,
    /// The subgroup id.
    pub subgroup_id: u32,
    /// The number of threads per subgroup.
    pub subgroup_threads: u32,
    /// The subgroup thread id.
    pub subgroup_thread_id: u32,
    /// The number of threads per group.
    pub threads: u32,
    /// The thread id.
    pub thread_id: u32,
}

pub struct ItemKernel {
    /// The id of the item.
    pub item_id: u32,
    /// The number of items.
    ///
    /// This will be the minimum length of buffers with `#[item]`.
    pub items: u32,
}
