// Compile the CUDA kernels with nvcc into a fat binary embedded in the executable:
//   CUDA_ARCH  virtual architecture of the embedded PTX, JIT-compiled by the driver for the GPU it runs on
//              (default compute_80: Ampere and newer)
//   CUDA_SASS  optional comma-separated real architectures to also embed ready-made code for, e.g. "86,89,121";
//              skips the JIT at start-up where no driver cache is kept (containers)
use std::{env, path::PathBuf, process::Command};

fn nvcc() -> PathBuf {
    if let Ok(p) = env::var("NVCC") {
        return p.into();
    }
    for var in ["CUDA_HOME", "CUDA_PATH", "CUDA_ROOT"] {
        if let Ok(root) = env::var(var) {
            let p = PathBuf::from(root).join("bin/nvcc");
            if p.exists() {
                return p;
            }
        }
    }
    let on_path = Command::new("nvcc").arg("--version").output().is_ok_and(|o| o.status.success());
    if on_path { "nvcc".into() } else { "/usr/local/cuda/bin/nvcc".into() }
}

fn main() {
    let src = "kernels/whisper.cu";
    println!("cargo:rerun-if-changed={src}");
    for var in ["NVCC", "CUDA_ARCH", "CUDA_SASS", "CUDA_HOME", "CUDA_PATH", "CUDA_ROOT"] {
        println!("cargo:rerun-if-env-changed={var}");
    }
    let out = PathBuf::from(env::var("OUT_DIR").unwrap()).join("whisper.fatbin");
    let arch = env::var("CUDA_ARCH").unwrap_or_else(|_| "compute_80".into());
    let mut args = vec!["-fatbin".to_string(), "-O3".into(), format!("-gencode=arch={arch},code={arch}")];
    for sm in env::var("CUDA_SASS").unwrap_or_default().split(',').map(str::trim).filter(|s| !s.is_empty()) {
        assert!(sm.chars().all(|c| c.is_ascii_alphanumeric()), "CUDA_SASS: bad architecture {sm:?}");
        args.push(format!("-gencode=arch=compute_{sm},code=sm_{sm}"));
    }
    let nvcc = nvcc();
    let st = Command::new(&nvcc)
        .args(&args)
        .arg(src)
        .arg("-o")
        .arg(&out)
        .status()
        .unwrap_or_else(|e| panic!("failed to run {}: {e} (set NVCC or CUDA_HOME)", nvcc.display()));
    assert!(st.success(), "nvcc failed");
}
