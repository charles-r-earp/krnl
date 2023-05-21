#[cfg(feature = "autograph")]
use compute_benches::autograph_backend::AutographBackend;
#[cfg(feature = "cuda")]
use compute_benches::cuda_backend::CudaBackend;
use compute_benches::krnl_backend::KrnlBackend;
#[cfg(feature = "ocl")]
use compute_benches::ocl_backend::OclBackend;
use criterion::{criterion_group, criterion_main, Criterion};
use rand::{distributions::OpenClosed01, thread_rng, Rng};
use std::{
    str::FromStr,
    time::{Duration, Instant},
};

pub fn criterion_benchmark(c: &mut Criterion) {
    let krnl_device = std::env::var("KRNL_DEVICE");
    println!("KRNL_DEVICE = {krnl_device:?}");
    let device_index = if let Ok(krnl_device) = krnl_device.as_ref() {
        usize::from_str(krnl_device).unwrap()
    } else {
        0
    };
    println!("testing device {device_index}");
    #[cfg_attr(not(feature = "cuda"), allow(unused))]
    let cuda_device_index = if cfg!(feature = "cuda") {
        let cuda_device = std::env::var("CUDA_DEVICE");
        println!("CUDA_DEVICE = {cuda_device:?}");
        let cuda_device_index = if let Ok(cuda_device) = cuda_device.as_ref() {
            usize::from_str(cuda_device).unwrap()
        } else {
            0
        };
        println!("testing cuda device {cuda_device_index}");
        device_index
    } else {
        0
    };
    #[cfg_attr(not(feature = "ocl"), allow(unused))]
    let (ocl_platform_index, ocl_device_index) = if cfg!(feature = "ocl") {
        let ocl_platform = std::env::var("OCL_PLATFORM");
        let ocl_device = std::env::var("OCL_DEVICE");
        println!("OCL_PLATFORM = {ocl_platform:?} OCL_DEVICE = {ocl_device:?}");
        let ocl_platform_index = if let Ok(ocl_platform) = ocl_platform.as_ref() {
            usize::from_str(ocl_platform).unwrap()
        } else {
            0
        };
        let ocl_device_index = if let Ok(ocl_device) = ocl_device.as_ref() {
            usize::from_str(ocl_device).unwrap()
        } else {
            0
        };
        println!("testing ocl platform {ocl_platform_index} device {ocl_device_index}");
        (ocl_platform_index, ocl_device_index)
    } else {
        (0, 0)
    };
    #[cfg(debug_assertions)]
    let lens = [("a_1K", 1000)];
    #[cfg(not(debug_assertions))]
    let lens = [
        ("a_1M", 1_000_000),
        ("b_10M", 10_000_000),
        ("c_64M", 64_000_000),
    ];
    let n_max = lens.iter().last().unwrap().1;
    let x: Vec<f32> = thread_rng().sample_iter(OpenClosed01).take(n_max).collect();
    let alpha = 0.5;
    let y: Vec<f32> = thread_rng().sample_iter(OpenClosed01).take(n_max).collect();

    let mut g = c.benchmark_group("compute");
    {
        let krnl = KrnlBackend::new(device_index).unwrap();
        for (s, n) in lens {
            g.bench_function(&format!("alloc_{s}_krnl"), |b| {
                let krnl = krnl.clone();
                b.iter_custom(move |i| {
                    let mut duration = Duration::default();
                    for _ in 0..i {
                        let start = Instant::now();
                        let _alloc = krnl.alloc(n).unwrap();
                        duration += start.elapsed();
                    }
                    duration
                });
            });
        }
        for (s, n) in lens {
            let mut upload = krnl.upload(&x[..n]).unwrap();
            g.bench_function(&format!("upload_{s}_krnl"), move |b| {
                b.iter(|| upload.run().unwrap());
            });
        }
        for (s, n) in lens {
            let mut download = krnl.download(&x[..n]).unwrap();
            g.bench_function(&format!("download_{s}_krnl"), move |b| {
                b.iter(|| download.run().unwrap());
            });
        }
        for (s, n) in lens {
            let mut zero = krnl.zero(n).unwrap();
            g.bench_function(&format!("zero_{s}_krnl"), move |b| {
                b.iter(|| zero.run().unwrap());
            });
        }
        for (s, n) in lens {
            let mut saxpy = krnl.saxpy(&x[..n], alpha, &y[..n]).unwrap();
            g.bench_function(&format!("saxpy_{s}_krnl"), move |b| {
                b.iter(|| saxpy.run().unwrap());
            });
        }
    }
    #[cfg(feature = "autograph")]
    {
        let autograph = AutographBackend::new(device_index).unwrap();
        for (s, n) in lens {
            if n > n_max {
                break;
            }
            g.bench_function(&format!("alloc_{s}_autograph"), |b| {
                let autograph = autograph.clone();
                b.iter_custom(move |i| {
                    let mut duration = Duration::default();
                    for _ in 0..i {
                        let start = Instant::now();
                        let _alloc = autograph.alloc(n).unwrap();
                        duration += start.elapsed();
                    }
                    duration
                });
            });
        }
        for (s, n) in lens {
            let mut upload = autograph.upload(&x[..n]).unwrap();
            g.bench_function(&format!("upload_{s}_autograph"), |b| {
                b.iter_custom(|i| {
                    let mut duration = Duration::default();
                    for _ in 0..i {
                        let start = Instant::now();
                        upload.run().unwrap();
                        duration += start.elapsed();
                    }
                    duration
                });
            });
        }
        for (s, n) in lens {
            let download = autograph.download(&x[..n]).unwrap();
            g.bench_function(&format!("download_{s}_autograph"), move |b| {
                b.iter_custom(|i| {
                    let mut duration = Duration::default();
                    for _ in 0..i {
                        let start = Instant::now();
                        download.run().unwrap();
                        duration += start.elapsed();
                    }
                    duration
                });
            });
        }
        for (s, n) in lens {
            let mut saxpy = autograph.saxpy(&x[..n], alpha, &y[..n]).unwrap();
            g.bench_function(&format!("saxpy_{s}_autograph"), move |b| {
                b.iter_custom(|i| {
                    let mut duration = Duration::default();
                    for _ in 0..i {
                        let start = Instant::now();
                        saxpy.run().unwrap();
                        duration += start.elapsed();
                    }
                    duration
                });
            });
        }
    }
    #[cfg(feature = "cuda")]
    {
        let cuda = CudaBackend::new(cuda_device_index).unwrap();
        for (s, n) in lens {
            g.bench_function(&format!("alloc_{s}_cuda"), |b| {
                let cuda = cuda.clone();
                b.iter_custom(move |i| {
                    let mut duration = Duration::default();
                    for _ in 0..i {
                        let start = Instant::now();
                        let _alloc = cuda.alloc(n).unwrap();
                        duration += start.elapsed();
                    }
                    duration
                });
            });
        }
        for (s, n) in lens {
            let mut upload = cuda.upload(&x[..n]).unwrap();
            g.bench_function(&format!("upload_{s}_cuda"), move |b| {
                b.iter(|| upload.run().unwrap());
            });
        }
        for (s, n) in lens {
            let mut download = cuda.download(&x[..n]).unwrap();
            g.bench_function(&format!("download_{s}_cuda"), move |b| {
                b.iter(|| download.run().unwrap());
            });
        }
        for (s, n) in lens {
            let mut zero = cuda.zero(n).unwrap();
            g.bench_function(&format!("zero_{s}_cuda"), move |b| {
                b.iter(|| zero.run().unwrap());
            });
        }
        for (s, n) in lens {
            let mut saxpy = cuda.saxpy(&x[..n], alpha, &y[..n]).unwrap();
            g.bench_function(&format!("saxpy_{s}_cuda"), move |b| {
                b.iter(|| saxpy.run().unwrap());
            });
        }
    }
    #[cfg(feature = "ocl")]
    {
        let ocl = OclBackend::new(ocl_platform_index, ocl_device_index).unwrap();
        for (s, n) in lens {
            g.bench_function(&format!("alloc_{s}_ocl"), |b| {
                let ocl = ocl.clone();
                b.iter_custom(move |i| {
                    let mut duration = Duration::default();
                    for _ in 0..i {
                        let start = Instant::now();
                        let _alloc = ocl.alloc(n).unwrap();
                        duration += start.elapsed();
                    }
                    duration
                });
            });
        }
        for (s, n) in lens {
            g.bench_function(&format!("upload_{s}_ocl"), |b| {
                let mut upload = ocl.upload(&x[..n]).unwrap();
                b.iter_custom(move |i| {
                    let mut duration = Duration::default();
                    for _ in 0..i {
                        let start = Instant::now();
                        upload.run().unwrap();
                        duration += start.elapsed();
                    }
                    duration
                });
            });
        }
        for (s, n) in lens {
            let mut download = ocl.download(&x[..n]).unwrap();
            g.bench_function(&format!("download_{s}_ocl"), move |b| {
                b.iter_custom(|i| {
                    let mut duration = Duration::default();
                    for _ in 0..i {
                        let start = Instant::now();
                        download.run().unwrap();
                        duration += start.elapsed();
                    }
                    duration
                });
            });
        }
        for (s, n) in lens {
            let mut saxpy = ocl.saxpy(&x[..n], alpha, &y[..n]).unwrap();
            g.bench_function(&format!("saxpy_{s}_ocl"), move |b| {
                b.iter(|| saxpy.run().unwrap());
            });
        }
    }
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
