

# Install [wasm-pack](https://rustwasm.github.io/wasm-pack)
```
cargo install wasm-pack
```

# Host tests
```
RUSTFLAGS='--cfg run_in_browser' wasm-pack test --headless --firefox -- --no-default-features
```

# Device tests 

WebGPU is supported on nightly browsers:
- [firefox nightly](https://www.mozilla.org/en-US/firefox/channel/desktop/#nightly)

Extra flags need to be provided to enable threads, see [wasm_thread](https://github.com/chemicstry/wasm_thread).
```
RUSTFLAGS='--cfg run_in_browser --cfg web_sys_unstable_apis -C target-feature=+atomics,+bulk-memory,+mutable-globals' wasm-pack test --firefox -- -Z build-std=panic_abort,std
```







