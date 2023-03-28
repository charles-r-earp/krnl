#[cfg(feature = "cuda")]
use compute_benchmarks::cuda_backend::CudaBackend;
use compute_benchmarks::krnl_backend::KrnlBackend;
#[cfg(feature = "ocl")]
use compute_benchmarks::ocl_backend::OclBackend;
use criterion::{criterion_group, criterion_main, Criterion};
use rand::{distributions::OpenClosed01, thread_rng, Rng};
use std::{
    cell::RefCell,
    env::var,
    rc::Rc,
    str::FromStr,
    time::{Duration, Instant},
};

fn index_from_env(name: &str) -> usize {
    if let Ok(value) = var(name) {
        usize::from_str(&value).unwrap()
    } else {
        0
    }
}

pub fn criterion_benchmark(c: &mut Criterion) {
    let saxpy_n = if cfg!(debug_assertions) {
        1024
    } else {
        256_000_000
    };
    let alloc_n = 256_000;
    let saxpy_x: Rc<Vec<f32>> = Rc::new(
        thread_rng()
            .sample_iter(OpenClosed01)
            .take(saxpy_n)
            .collect(),
    );
    let saxpy_alpha = 0.5;
    let saxpy_y: Rc<Vec<f32>> = Rc::new(
        thread_rng()
            .sample_iter(OpenClosed01)
            .take(saxpy_n)
            .collect(),
    );
    {
        let index = index_from_env("KRNL_DEVICE");
        let krnl = KrnlBackend::new(index).unwrap();
        {
            c.bench_function("alloc_krnl", |b| {
                let krnl = krnl.clone();
                b.iter_custom(move |i| {
                    let mut duration = Duration::default();
                    for _ in 0..i {
                        let start = Instant::now();
                        let _alloc = krnl.alloc(alloc_n).unwrap();
                        duration += start.elapsed();
                    }
                    duration
                });
            });
        }
        {
            let krnl = krnl.clone();
            let x = saxpy_x.clone();
            c.bench_function("upload_krnl", move |b| {
                let krnl = krnl.clone();
                let x = x.clone();
                b.iter_custom(move |i| {
                    let mut duration = Duration::default();
                    for _ in 0..i {
                        let start = Instant::now();
                        let _upload = krnl.upload(&x).unwrap();
                        duration += start.elapsed();
                    }
                    duration
                });
            });
        }
        {
            let krnl = krnl.clone();
            let x = saxpy_x.clone();
            c.bench_function("download_krnl", move |b| {
                let download = krnl.download(&x).unwrap();
                b.iter_custom(move |i| {
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
        {
            let saxpy = Rc::new(RefCell::new(
                krnl.saxpy(&saxpy_x, saxpy_alpha, &saxpy_y).unwrap(),
            ));
            c.bench_function("saxpy_krnl", move |b| {
                let saxpy = saxpy.clone();
                b.iter_custom(move |i| {
                    let mut duration = Duration::default();
                    for _ in 0..i {
                        let start = Instant::now();
                        saxpy.borrow_mut().run().unwrap();
                        duration += start.elapsed();
                    }
                    duration
                });
            });
        }
    }
    #[cfg(feature = "cuda")]
    {
        let index = index_from_env("CUDA_DEVICE");
        let cuda = CudaBackend::new(index).unwrap();
        {
            c.bench_function("alloc_cuda", |b| {
                let cuda = cuda.clone();
                b.iter_custom(move |i| {
                    let mut duration = Duration::default();
                    for _ in 0..i {
                        let start = Instant::now();
                        let _alloc = cuda.alloc(alloc_n).unwrap();
                        duration += start.elapsed();
                    }
                    duration
                });
            });
        }
        {
            let cuda = cuda.clone();
            let x = saxpy_x.clone();
            c.bench_function("upload_cuda", move |b| {
                let cuda = cuda.clone();
                let x = x.clone();
                b.iter_custom(move |i| {
                    let mut duration = Duration::default();
                    for _ in 0..i {
                        let start = Instant::now();
                        let _upload = cuda.upload(&x).unwrap();
                        duration += start.elapsed();
                    }
                    duration
                });
            });
        }
        {
            let cuda = cuda.clone();
            let x = saxpy_x.clone();
            c.bench_function("download_cuda", move |b| {
                let download = cuda.download(&x).unwrap();
                b.iter_custom(move |i| {
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
        {
            let saxpy = Rc::new(RefCell::new(
                cuda.saxpy(&saxpy_x, saxpy_alpha, &saxpy_y).unwrap(),
            ));
            c.bench_function("saxpy_cuda", move |b| {
                let saxpy = saxpy.clone();
                b.iter_custom(move |i| {
                    let mut duration = Duration::default();
                    for _ in 0..i {
                        let start = Instant::now();
                        saxpy.borrow_mut().run().unwrap();
                        duration += start.elapsed();
                    }
                    duration
                });
            });
        }
    }
    #[cfg(feature = "ocl")]
    {
        let platform_index = index_from_env("OCL_PLATFORM");
        let device_index = index_from_env("OCL_DEVICE");
        let ocl = OclBackend::new(platform_index, device_index).unwrap();
        {
            c.bench_function("alloc_ocl", |b| {
                let ocl = ocl.clone();
                b.iter_custom(move |i| {
                    let mut duration = Duration::default();
                    for _ in 0..i {
                        let start = Instant::now();
                        let _alloc = ocl.alloc(alloc_n).unwrap();
                        duration += start.elapsed();
                    }
                    duration
                });
            });
        }
        {
            let ocl = ocl.clone();
            let x = saxpy_x.clone();
            c.bench_function("upload_ocl", move |b| {
                let ocl = ocl.clone();
                let x = x.clone();
                b.iter_custom(move |i| {
                    let mut duration = Duration::default();
                    for _ in 0..i {
                        let start = Instant::now();
                        let _upload = ocl.upload(&x).unwrap();
                        duration += start.elapsed();
                    }
                    duration
                });
            });
        }
        {
            let ocl = ocl.clone();
            let x = saxpy_x.clone();
            c.bench_function("download_ocl", move |b| {
                let download = ocl.download(&x).unwrap();
                b.iter_custom(move |i| {
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
        {
            let saxpy = Rc::new(RefCell::new(
                ocl.saxpy(&saxpy_x, saxpy_alpha, &saxpy_y).unwrap(),
            ));
            c.bench_function("saxpy_ocl", move |b| {
                let saxpy = saxpy.clone();
                b.iter_custom(move |i| {
                    let mut duration = Duration::default();
                    for _ in 0..i {
                        let start = Instant::now();
                        saxpy.borrow_mut().run().unwrap();
                        duration += start.elapsed();
                    }
                    duration
                });
            });
        }
    }
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
