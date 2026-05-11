/*!
 * build.rs: when the `cuda` feature is enabled, compile cuda/kernel.cu
 * into a static library via nvcc and link it plus libcudart.so. The
 * static library exposes the extern "C" surface declared in cuda/miner.h.
 */

use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=cuda/kernel.cu");
    println!("cargo:rerun-if-changed=cuda/miner.h");
    println!("cargo:rerun-if-changed=build.rs");

    if env::var("CARGO_FEATURE_CUDA").is_err() {
        return;
    }

    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR"));
    let manifest = PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"));
    let cuda_root = env::var("CUDA_PATH")
        .or_else(|_| env::var("CUDA_HOME"))
        .unwrap_or_else(|_| "/usr/local/cuda".to_string());
    let cuda_root = PathBuf::from(cuda_root);
    let cuda_inc = cuda_root.join("include");
    let cuda_lib = cuda_root.join("lib64");

    let nvcc = which("nvcc").unwrap_or_else(|| cuda_root.join("bin").join("nvcc"));

    let arches = env::var("CUDA_ARCH").unwrap_or_else(|_| "75;86;89;90".to_string());

    let obj = out_dir.join("kernel.o");
    let mut cmd = Command::new(&nvcc);
    cmd.arg("-O3")
        .arg("-Xcompiler").arg("-O3")
        .arg("-Xcompiler").arg("-fPIC")
        .arg("--use_fast_math")
        .arg("-std=c++17")
        /* Cap the register file at 80 per thread. nvcc otherwise picks
         * 91 on the round-1 precomputed kernel which drops sm_75 from 26
         * warps/SM to 22. At 80 regs there are no spills (verified via
         * -Xptxas -v) and we keep 24 warps/SM = 75% occupancy. */
        .arg("-maxrregcount=80")
        .arg("-Xptxas").arg("-v")
        .arg("-I").arg(manifest.join("cuda"));

    for sm in arches.split([';', ',']) {
        let sm = sm.trim();
        if sm.is_empty() {
            continue;
        }
        cmd.arg("-gencode")
            .arg(format!("arch=compute_{sm},code=sm_{sm}"));
    }

    cmd.arg("-c")
        .arg(manifest.join("cuda").join("kernel.cu"))
        .arg("-o")
        .arg(&obj);

    let status = cmd.status().expect("failed to run nvcc");
    if !status.success() {
        panic!("nvcc failed: {status}");
    }

    let lib = out_dir.join("libhash_miner_cuda.a");
    let _ = std::fs::remove_file(&lib);
    let ar = which("ar").unwrap_or_else(|| PathBuf::from("ar"));
    let status = Command::new(ar)
        .arg("rcs")
        .arg(&lib)
        .arg(&obj)
        .status()
        .expect("failed to run ar");
    if !status.success() {
        panic!("ar failed: {status}");
    }

    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-search=native={}", cuda_lib.display());
    println!("cargo:rustc-link-lib=static=hash_miner_cuda");
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=stdc++");
    let _ = cuda_inc;
}

fn which(prog: &str) -> Option<PathBuf> {
    let path = env::var_os("PATH")?;
    for p in env::split_paths(&path) {
        let candidate = p.join(prog);
        if candidate.is_file() {
            return Some(candidate);
        }
    }
    None
}
