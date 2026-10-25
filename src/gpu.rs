//! Thin layer over the CUDA driver and cuBLAS: raw device buffers, kernel launches, fp16 GEMM.
//! Buffers live for the whole process, so plain device pointers are used instead of owned slices.
use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::Arc;

use cudarc::cublas::{CudaBlas, result as blas, sys as bsys};
use cudarc::driver::{CudaContext, CudaFunction, CudaStream, result as drv, sys};
use cudarc::nvrtc::Ptx;

pub type R<T> = Result<T, Box<dyn std::error::Error>>;
pub type Ptr = u64;

/// The kernels as a fat binary: PTX (JIT-compiled by the driver for any newer GPU) plus any SASS chosen at build time.
const KERNEL_IMAGE: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/whisper.fatbin"));
const KERNELS: &[&str] = &[
    "layernorm",
    "add_layernorm",
    "add_inplace",
    "fill_bias",
    "fill_bias_heads",
    "gelu_inplace",
    "add_pos",
    "im2col1",
    "im2col2",
    "softmax_rows",
    "enc_flash",
    "embed",
    "kv_write",
    "self_attn",
    "cross_scores",
    "cross_pv",
    "cross_combine",
    "argmax_tok",
    "advance",
    "gemv_1",
    "gemv_2",
    "gemv_4",
    "gemv_8",
];
const WS_BYTES: usize = 64 << 20;
const ALIGN: usize = 256;

#[derive(Clone, Copy)]
pub enum A {
    P(Ptr),
    I(i32),
    L(i64),
    F(f32),
}

/// Column-major strided batched GEMM (cuBLAS conventions): C = alpha op(A) op(B) + beta C, batch strides in elements.
#[derive(Clone, Copy)]
pub struct Mm {
    pub ta: bool,
    pub tb: bool,
    pub m: usize,
    pub n: usize,
    pub k: usize,
    pub lda: usize,
    pub ldb: usize,
    pub ldc: usize,
    pub sa: i64,
    pub sb: i64,
    pub sc: i64,
    pub batch: usize,
}

pub struct Gpu {
    _ctx: Arc<CudaContext>,
    pub stream: Arc<CudaStream>,
    blas: CudaBlas,
    funcs: HashMap<&'static str, CudaFunction>,
    pub allocated: usize,
    /// use cuBLAS for every GEMM (comparison); otherwise M <= 8 uses the gemv kernels
    pub cublas_only: bool,
}

pub fn h2f(h: u16) -> f32 {
    let s = ((h >> 15) as u32) << 31;
    let e = ((h >> 10) & 0x1f) as u32;
    let m = (h & 0x3ff) as u32;
    match e {
        0 => {
            let v = m as f32 / 16_777_216.0; // subnormal: m * 2^-24
            if s != 0 { -v } else { v }
        }
        31 => f32::from_bits(s | 0x7f80_0000 | (m << 13)),
        _ => f32::from_bits(s | ((e + 112) << 23) | (m << 13)),
    }
}

/// f32 -> fp16 bits, round to nearest even (as __float2half_rn).
pub fn f2h(x: f32) -> u16 {
    let u = x.to_bits();
    let s = ((u >> 16) & 0x8000) as u16;
    let a = u & 0x7fff_ffff;
    if a >= 0x7f80_0000 {
        return s | if a > 0x7f80_0000 { 0x7e00 } else { 0x7c00 };
    }
    if a >= 0x477f_f000 {
        return s | 0x7c00; // >= 65520 rounds to infinity
    }
    if a < 0x3880_0000 {
        return s | (f32::from_bits(a) * 16_777_216.0).round_ties_even() as u16; // subnormal
    }
    let m = a & 0x7f_ffff;
    let mut hb = ((((a >> 23) as i32 - 112) as u32) << 10) | (m >> 13);
    let rem = m & 0x1fff;
    if rem > 0x1000 || (rem == 0x1000 && hb & 1 == 1) {
        hb += 1;
    }
    s | hb as u16
}

/// Round an f32 through fp16, like a PyTorch fp16 op.
pub fn rh(x: f32) -> f32 {
    h2f(f2h(x))
}

impl Gpu {
    pub fn new() -> R<Self> {
        let ctx = CudaContext::new(0)?;
        unsafe { ctx.disable_event_tracking() };
        let stream = ctx.new_stream()?; // graph capture is not allowed on the legacy default stream
        let blas = CudaBlas::new(stream.clone())?;
        let module = ctx.load_module(Ptx::from_binary(KERNEL_IMAGE.to_vec()))?;
        let mut funcs = HashMap::new();
        for &k in KERNELS {
            funcs.insert(k, module.load_function(k)?);
        }
        let mut g = Gpu { _ctx: ctx, stream, blas, funcs, allocated: 0, cublas_only: false };
        // fixed cuBLAS workspace so GEMMs never allocate while a graph is being captured
        let ws = g.alloc(WS_BYTES)?;
        unsafe { bsys::cublasSetWorkspace_v2(*g.blas.handle(), ws as *mut c_void, WS_BYTES).result()? };
        Ok(g)
    }

    /// Zero-filled device buffer.
    pub fn alloc(&mut self, bytes: usize) -> R<Ptr> {
        let p = unsafe { drv::malloc_sync(bytes.max(16))? };
        unsafe { drv::memset_d8_async(p, 0, bytes.max(16), self.cu())? };
        self.allocated += bytes;
        Ok(p)
    }

    pub fn alloc_h(&mut self, n: usize) -> R<Ptr> {
        self.alloc(n * 2)
    }

    /// Device memory (free, total) in bytes.
    pub fn mem_info(&self) -> R<(usize, usize)> {
        Ok(drv::mem_get_info()?)
    }

    pub fn cu(&self) -> sys::CUstream {
        self.stream.cu_stream()
    }

    pub fn upload<T>(&self, dst: Ptr, src: &[T]) -> R<()> {
        unsafe { drv::memcpy_htod_async(dst, src, self.cu())? };
        self.sync()
    }

    pub fn upload_new<T>(&mut self, src: &[T]) -> R<Ptr> {
        let p = self.alloc(std::mem::size_of_val(src))?;
        self.upload(p, src)?;
        Ok(p)
    }

    pub fn download<T: Default + Clone>(&self, src: Ptr, n: usize) -> R<Vec<T>> {
        let mut v = vec![T::default(); n];
        unsafe { drv::memcpy_dtoh_async(&mut v[..], src, self.cu())? };
        self.sync()?;
        Ok(v)
    }

    pub fn download_h(&self, src: Ptr, n: usize) -> R<Vec<f32>> {
        Ok(self.download::<u16>(src, n)?.into_iter().map(h2f).collect())
    }

    pub fn sync(&self) -> R<()> {
        self.stream.synchronize()?;
        Ok(())
    }

    pub fn k(&self, name: &str, grid: (u32, u32, u32), block: u32, smem: u32, args: &[A]) -> R<()> {
        let mut vals = args.to_vec();
        let mut ptrs: Vec<*mut c_void> = vals
            .iter_mut()
            .map(|a| match a {
                A::P(v) => v as *mut u64 as *mut c_void,
                A::I(v) => v as *mut i32 as *mut c_void,
                A::L(v) => v as *mut i64 as *mut c_void,
                A::F(v) => v as *mut f32 as *mut c_void,
            })
            .collect();
        let f = self.funcs.get(name).ok_or_else(|| format!("no kernel {name}"))?;
        unsafe { drv::launch_kernel(f.cu_function(), grid, (block, 1, 1), smem, self.cu(), &mut ptrs)? };
        Ok(())
    }

    /// Elementwise launch over n items.
    pub fn k1(&self, name: &str, n: usize, args: &[A]) -> R<()> {
        self.k(name, (n.div_ceil(256) as u32, 1, 1), 256, 0, args)
    }

    /// y[m, n] (row stride ldy) = x[m, k] (row stride ldx) · W[n, k]^T + bias[n], all fp16, fp32 accumulation;
    /// the bias is added in fp32 before the single rounding to fp16, as torch's addmm does.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm(&self, x: Ptr, ldx: usize, w: Ptr, y: Ptr, ldy: usize, m: usize, n: usize, k: usize, bias: Option<Ptr>) -> R<()> {
        // measured: the gemv kernels beat cuBLAS up to M = 8, cuBLAS wins above
        if m <= 8 && !self.cublas_only {
            let (name, mr, cols, kc) = match m {
                1 => ("gemv_1", 1, 4, 2048),
                2 => ("gemv_2", 2, 4, 2048),
                3..=4 => ("gemv_4", 4, 4, 1024),
                _ => ("gemv_8", 8, 4, 1024),
            };
            let per_block = 8 * cols;
            return self.k(
                name,
                (n.div_ceil(per_block) as u32, 1, 1),
                256,
                (mr * kc * 2) as u32,
                &[A::P(x), A::I(ldx as i32), A::P(w), A::P(y), A::I(ldy as i32), A::I(m as i32), A::I(n as i32), A::I(k as i32), A::P(bias.unwrap_or(0))],
            );
        }
        // the bias goes into y first and the GEMM accumulates onto it (beta = 1): one rounding of acc + bias
        if let Some(b) = bias {
            self.k1("fill_bias", m * n / 8, &[A::P(y), A::P(b), A::L(m as i64), A::I(n as i32), A::L(ldy as i64)])?;
        }
        // column-major: Y^T [n, m] = W [n, k] (stored k x n, transposed) · X^T [k, m]
        let mm = Mm { ta: true, tb: false, m: n, n: m, k, lda: k, ldb: ldx, ldc: ldy, sa: 0, sb: 0, sc: 0, batch: 1 };
        self.matmul(&mm, 1.0, w, x, if bias.is_some() { 1.0 } else { 0.0 }, y)
    }

    /// y = x W^T + bias like `gemm`, for a y that already holds the bias in every row (the cuBLAS path only
    /// accumulates onto it; the gemv kernels add the bias themselves).
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_onto_bias(&self, x: Ptr, ldx: usize, w: Ptr, y: Ptr, ldy: usize, m: usize, n: usize, k: usize, bias: Ptr) -> R<()> {
        if m <= 8 && !self.cublas_only {
            return self.gemm(x, ldx, w, y, ldy, m, n, k, Some(bias));
        }
        let mm = Mm { ta: true, tb: false, m: n, n: m, k, lda: k, ldb: ldx, ldc: ldy, sa: 0, sb: 0, sc: 0, batch: 1 };
        self.matmul(&mm, 1.0, w, x, 1.0, y)
    }

    pub fn copy(&self, dst: Ptr, src: Ptr, bytes: usize) -> R<()> {
        unsafe { drv::memcpy_dtod_async(dst, src, bytes, self.cu())? };
        Ok(())
    }

    /// Strided batched fp16 GEMM with fp32 accumulation (cuBLAS chooses the kernel).
    pub fn matmul(&self, mm: &Mm, alpha: f32, a: Ptr, b: Ptr, beta: f32, c: Ptr) -> R<()> {
        let op = |t: bool| if t { bsys::cublasOperation_t::CUBLAS_OP_T } else { bsys::cublasOperation_t::CUBLAS_OP_N };
        let f16 = bsys::cudaDataType::CUDA_R_16F;
        unsafe {
            blas::gemm_strided_batched_ex(
                *self.blas.handle(),
                op(mm.ta),
                op(mm.tb),
                mm.m as i32,
                mm.n as i32,
                mm.k as i32,
                &alpha as *const f32 as *const c_void,
                a as *const c_void,
                f16,
                mm.lda as i32,
                mm.sa,
                b as *const c_void,
                f16,
                mm.ldb as i32,
                mm.sb,
                &beta as *const f32 as *const c_void,
                c as *mut c_void,
                f16,
                mm.ldc as i32,
                mm.sc,
                mm.batch as i32,
                bsys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
                bsys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
            )?
        };
        Ok(())
    }
}

/// Device memory for many read-only tensors in one allocation, each placed at a 256-byte boundary. Bytes go
/// through two pinned staging buffers, so copying into one overlaps the transfer of the other.
pub struct Arena {
    base: Ptr,
    used: usize,
    cap: usize,
    stage: [*mut u8; 2],
    ev: [sys::CUevent; 2],
    cur: usize,
    /// arena offset where the current staging buffer starts, and the bytes in it
    win: usize,
    fill: usize,
}

const STAGE: usize = 32 << 20;

impl Arena {
    pub fn new(g: &mut Gpu, cap: usize) -> R<Self> {
        let base = unsafe { drv::malloc_sync(cap.max(16))? };
        g.allocated += cap;
        let mut stage = [std::ptr::null_mut(); 2];
        for s in stage.iter_mut() {
            *s = unsafe { drv::malloc_host(STAGE, 0)? } as *mut u8;
        }
        let ev = [drv::event::create(sys::CUevent_flags::CU_EVENT_DISABLE_TIMING)?, drv::event::create(sys::CUevent_flags::CU_EVENT_DISABLE_TIMING)?];
        Ok(Arena { base, used: 0, cap, stage, ev, cur: 0, win: 0, fill: 0 })
    }

    /// Place the parts back to back (one tensor) and return its device address.
    pub fn put(&mut self, g: &Gpu, parts: &[&[u8]]) -> R<Ptr> {
        let p = self.base + self.used as u64;
        let n: usize = parts.iter().map(|b| b.len()).sum();
        if self.used + n > self.cap {
            return Err("weight arena too small".into());
        }
        for b in parts {
            self.write(g, b)?;
        }
        let pad = self.used.next_multiple_of(ALIGN) - self.used;
        self.write(g, &[0u8; ALIGN][..pad])?;
        Ok(p)
    }

    fn write(&mut self, g: &Gpu, mut bytes: &[u8]) -> R<()> {
        while !bytes.is_empty() {
            if self.fill == STAGE {
                self.flush(g)?;
            }
            let n = (STAGE - self.fill).min(bytes.len());
            unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), self.stage[self.cur].add(self.fill), n) };
            self.fill += n;
            self.used += n;
            bytes = &bytes[n..];
        }
        Ok(())
    }

    /// Send the current staging buffer and switch to the other one once its previous transfer is done.
    fn flush(&mut self, g: &Gpu) -> R<()> {
        if self.fill > 0 {
            let src = unsafe { std::slice::from_raw_parts(self.stage[self.cur], self.fill) };
            unsafe {
                drv::memcpy_htod_async(self.base + self.win as u64, src, g.cu())?;
                drv::event::record(self.ev[self.cur], g.cu())?;
            }
        }
        self.cur ^= 1;
        unsafe { sys::cuEventSynchronize(self.ev[self.cur]).result()? };
        self.win = self.used;
        self.fill = 0;
        Ok(())
    }

    /// Wait for the last transfers and release the staging buffers.
    pub fn finish(mut self, g: &Gpu) -> R<()> {
        self.flush(g)?;
        g.sync()?;
        for i in 0..2 {
            unsafe {
                drv::free_host(self.stage[i] as *mut c_void)?;
                sys::cuEventDestroy_v2(self.ev[i]).result()?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fp16_round_trip_and_rounding() {
        for h in 0..=u16::MAX {
            let x = h2f(h);
            if x.is_nan() {
                assert!(h2f(f2h(x)).is_nan());
            } else {
                assert_eq!(f2h(x), h, "{h:#06x}");
            }
        }
        assert_eq!(f2h(1.0 + 1.0 / 2048.0), 0x3c00); // tie rounds to even
        assert_eq!(f2h(1.0 + 3.0 / 2048.0), 0x3c02);
        assert_eq!(f2h(65520.0), 0x7c00); // overflow
        assert_eq!(f2h(65519.0), 0x7bff);
        assert_eq!(f2h(2f32.powi(-25)), 0); // half of the smallest subnormal, tie to even
        assert_eq!(f2h(3.0 * 2f32.powi(-25)), 2);
        assert_eq!(rh(0.1), 1638.0 / 16384.0); // 0.1 is between fp16 steps of 2^-14
    }
}
