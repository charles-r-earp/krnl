use krnl::{anyhow::Result, buffer::Buffer, device::Device, macros::module};

#[module]
pub mod kernels {
    #[cfg(not(target_arch = "spirv"))]
    use krnl::krnl_core;
    use krnl_core::macros::kernel;

    #[cfg_attr(not(target_arch = "spirv"), derive(PartialEq, Eq, Debug))]
    pub struct SubgroupInfo {
        pub subgroups: u32,
        pub subgroup_id: u32,
        pub subgroup_threads: u32,
        pub subgroup_thread_id: u32,
    }

    impl SubgroupInfo {
        pub const N: usize = 4;
        #[cfg(target_arch = "spirv")]
        fn to_array(self) -> [u32; 4] {
            let Self {
                subgroups,
                subgroup_id,
                subgroup_threads,
                subgroup_thread_id,
            } = self;
            [subgroups, subgroup_id, subgroup_threads, subgroup_thread_id]
        }
    }

    #[kernel]
    fn subgroup_info(#[global] y: UnsafeSlice<u32>) {
        use krnl_core::buffer::UnsafeIndex;

        let info = SubgroupInfo {
            subgroups: kernel.subgroups() as u32,
            subgroup_id: kernel.subgroup_id() as u32,
            subgroup_threads: kernel.subgroup_threads() as u32,
            subgroup_thread_id: kernel.subgroup_thread_id() as u32,
        }
        .to_array();
        for i in 0..info.len() {
            unsafe {
                *y.unsafe_index_mut(kernel.global_id() * info.len() + i) = info[i];
            }
        }
    }
}
pub use kernels::SubgroupInfo;

impl SubgroupInfo {
    fn from_array(input: [u32; Self::N]) -> Self {
        let [subgroups, subgroup_id, subgroup_threads, subgroup_thread_id] = input;
        Self {
            subgroups,
            subgroup_id,
            subgroup_threads,
            subgroup_thread_id,
        }
    }
}

pub fn subgroup_info(device: Device, threads: u32) -> Result<Vec<SubgroupInfo>> {
    let mut y = Buffer::zeros(device, threads as usize * SubgroupInfo::N)?;
    kernels::subgroup_info::builder()?
        .with_threads(threads)
        .build(y.device())?
        .with_global_threads(threads)
        .dispatch(y.as_slice_mut())?;
    let output = y
        .to_vec()?
        .chunks_exact(4)
        .map(|x| SubgroupInfo::from_array(x.try_into().unwrap()))
        .collect();
    Ok(output)
}
