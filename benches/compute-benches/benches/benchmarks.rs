#[cfg(feature = "autograph")]
use compute_benches::autograph_backend::AutographBackend;
#[cfg(feature = "cuda")]
use compute_benches::cuda_backend::CudaBackend;
use compute_benches::krnl_backend::KrnlBackend;
#[cfg(feature = "ocl")]
use compute_benches::ocl_backend::OclBackend;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use num_format::{Locale, ToFormattedString};
use rand::{distributions::OpenClosed01, thread_rng, Rng};
use std::{
    str::FromStr,
    time::{Duration, Instant},
};

pub fn criterion_benchmark(c: &mut Criterion) {
    let device_index = {
        let krnl_device = std::env::var("KRNL_DEVICE");
        println!("KRNL_DEVICE = {krnl_device:?}");
        let device_index = if let Ok(krnl_device) = krnl_device.as_ref() {
            usize::from_str(krnl_device).unwrap()
        } else {
            0
        };
        println!("testing device {device_index}");
        device_index
    };

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
        cuda_device_index
    } else {
        0
    };

    #[cfg(feature = "ocl")]
    let (ocl_platform_index, ocl_device_index) = {
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
    };

    let lens = [1_000_000, 10_000_000, 64_000_000];
    let n_max = lens.last().copied().unwrap();
    {
        let mut g = c.benchmark_group("alloc");
        {
            let krnl = KrnlBackend::new(device_index).unwrap();
            for n in lens {
                let id = BenchmarkId::new("krnl", n.to_formatted_string(&Locale::en));
                g.bench_function(id, |b| {
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
        }
        #[cfg(feature = "cuda")]
        {
            let cuda = CudaBackend::new(cuda_device_index).unwrap();
            for n in lens {
                let id = BenchmarkId::new("cuda", n.to_formatted_string(&Locale::en));
                g.bench_function(id, |b| {
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
        }
        #[cfg(feature = "ocl")]
        {
            let ocl = OclBackend::new(ocl_platform_index, ocl_device_index).unwrap();
            for n in lens {
                let id = BenchmarkId::new("ocl", n.to_formatted_string(&Locale::en));
                g.bench_function(id, |b| {
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
        }
    }

    let x: Vec<f32> = thread_rng().sample_iter(OpenClosed01).take(n_max).collect();
    let y: Vec<f32> = thread_rng().sample_iter(OpenClosed01).take(n_max).collect();

    {
        let mut g = c.benchmark_group("upload");
        {
            let krnl = KrnlBackend::new(device_index).unwrap();
            for n in lens {
                let id = BenchmarkId::new("krnl", n.to_formatted_string(&Locale::en));
                let mut upload = krnl.upload(&x[..n]).unwrap();
                g.bench_function(id, move |b| {
                    b.iter(|| upload.run().unwrap());
                });
            }
        }
        #[cfg(feature = "cuda")]
        {
            let cuda = CudaBackend::new(cuda_device_index).unwrap();
            for n in lens {
                let id = BenchmarkId::new("cuda", n.to_formatted_string(&Locale::en));
                let mut upload = cuda.upload(&x[..n]).unwrap();
                g.bench_function(id, move |b| {
                    b.iter(|| upload.run().unwrap());
                });
            }
        }
        #[cfg(feature = "ocl")]
        {
            let ocl = OclBackend::new(ocl_platform_index, ocl_device_index).unwrap();
            for n in lens {
                let id = BenchmarkId::new("ocl", n.to_formatted_string(&Locale::en));
                let mut upload = ocl.upload(&x[..n]).unwrap();
                g.bench_function(id, move |b| {
                    b.iter(|| upload.run().unwrap());
                });
            }
        }
    }

    {
        let mut g = c.benchmark_group("download");
        {
            let krnl = KrnlBackend::new(device_index).unwrap();
            for n in lens {
                let id = BenchmarkId::new("krnl", n.to_formatted_string(&Locale::en));
                let mut download = krnl.download(&x[..n]).unwrap();
                g.bench_function(id, move |b| {
                    b.iter(|| download.run().unwrap());
                });
            }
        }
        #[cfg(feature = "cuda")]
        {
            let cuda = CudaBackend::new(cuda_device_index).unwrap();
            for n in lens {
                let id = BenchmarkId::new("cuda", n.to_formatted_string(&Locale::en));
                let mut download = cuda.download(&x[..n]).unwrap();
                g.bench_function(id, move |b| {
                    b.iter(|| download.run().unwrap());
                });
            }
        }
        #[cfg(feature = "ocl")]
        {
            let ocl = OclBackend::new(ocl_platform_index, ocl_device_index).unwrap();
            for n in lens {
                let id = BenchmarkId::new("ocl", n.to_formatted_string(&Locale::en));
                let mut download = ocl.download(&x[..n]).unwrap();
                g.bench_function(id, move |b| {
                    b.iter(|| download.run().unwrap());
                });
            }
        }
    }

    {
        let mut g = c.benchmark_group("zero");
        {
            let krnl = KrnlBackend::new(device_index).unwrap();
            for n in lens {
                let id = BenchmarkId::new("krnl", n.to_formatted_string(&Locale::en));
                let mut zero = krnl.zero(n).unwrap();
                g.bench_function(id, move |b| {
                    b.iter(|| zero.run().unwrap());
                });
            }
        }
        #[cfg(feature = "cuda")]
        {
            let cuda = CudaBackend::new(cuda_device_index).unwrap();
            for n in lens {
                let id = BenchmarkId::new("cuda", n.to_formatted_string(&Locale::en));
                let mut zero = cuda.zero(n).unwrap();
                g.bench_function(id, move |b| {
                    b.iter(|| zero.run().unwrap());
                });
            }
        }
        #[cfg(feature = "ocl")]
        {
            let ocl = OclBackend::new(ocl_platform_index, ocl_device_index).unwrap();
            for n in lens {
                let id = BenchmarkId::new("ocl", n.to_formatted_string(&Locale::en));
                let mut zero = ocl.zero(n).unwrap();
                g.bench_function(id, move |b| {
                    b.iter(|| zero.run().unwrap());
                });
            }
        }
    }

    {
        let mut g = c.benchmark_group("saxpy");
        let alpha = 0.5;
        {
            let krnl = KrnlBackend::new(device_index).unwrap();
            for n in lens {
                let id = BenchmarkId::new("krnl", n.to_formatted_string(&Locale::en));
                let mut saxpy = krnl.saxpy(&x[..n], alpha, &y[..n]).unwrap();
                g.bench_function(id, move |b| {
                    b.iter(|| saxpy.run().unwrap());
                });
            }
        }
        #[cfg(feature = "cuda")]
        {
            let cuda = CudaBackend::new(cuda_device_index).unwrap();
            for n in lens {
                let id = BenchmarkId::new("cuda", n.to_formatted_string(&Locale::en));
                let mut saxpy = cuda.saxpy(&x[..n], alpha, &y[..n]).unwrap();
                g.bench_function(id, move |b| {
                    b.iter(|| saxpy.run().unwrap());
                });
            }
        }
        #[cfg(feature = "ocl")]
        {
            let ocl = OclBackend::new(ocl_platform_index, ocl_device_index).unwrap();
            for n in lens {
                let id = BenchmarkId::new("ocl", n.to_formatted_string(&Locale::en));
                let mut saxpy = ocl.saxpy(&x[..n], alpha, &y[..n]).unwrap();
                g.bench_function(id, move |b| {
                    b.iter(|| saxpy.run().unwrap());
                });
            }
        }
    }
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
