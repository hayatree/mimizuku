// CUDA kernels for the Whisper engine. Storage is fp16, arithmetic fp32.
// Where PyTorch rounds an intermediate to fp16 (linear outputs, GELU, residual adds, and the decoder's attention
// scores and probabilities), the kernels round at the same point (rh). Encoder attention is fused like PyTorch's
// SDPA (fp32 scores, online softmax), so it agrees to accumulation-order noise rather than bit for bit.
// Model width d (a multiple of HD) is a launch argument, so one build serves every Whisper size.
//
// Decoder kernels read the position from device memory (ctr[0]), never from launch arguments,
// so one captured CUDA graph is replayed for every decoding step.
#include <cuda_fp16.h>

typedef __half h16;

#define HD 64      // head size of every Whisper model
#define TENC 1500  // encoder positions (30 s)
#define NFR 3000   // mel frames (30 s)

__device__ __forceinline__ float f(h16 x) { return __half2float(x); }
__device__ __forceinline__ h16 h(float x) { return __float2half_rn(x); }
__device__ __forceinline__ float rh(float x) { return __half2float(__float2half_rn(x)); }

// Sum over the block; blockDim.x must be a multiple of 32 and at most 1024.
__device__ float block_sum(float v) {
    __shared__ float part[32];
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
    int w = threadIdx.x >> 5, l = threadIdx.x & 31;
    __syncthreads();
    if (l == 0) part[w] = v;
    __syncthreads();
    int nw = blockDim.x >> 5;
    v = l < nw ? part[l] : 0.f;
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
    return v;
}

__device__ float block_max(float v) {
    __shared__ float part[32];
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, o));
    int w = threadIdx.x >> 5, l = threadIdx.x & 31;
    __syncthreads();
    if (l == 0) part[w] = v;
    __syncthreads();
    int nw = blockDim.x >> 5;
    v = l < nw ? part[l] : -INFINITY;
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, o));
    return v;
}

// y = h((x - mean) * rstd * w + b) over one row of n2 half2 pairs; x is read again from memory per pass.
__device__ __forceinline__ void ln_row(const __half2* x, const __half2* w, const __half2* b, __half2* y, int n2, float eps) {
    float s = 0.f;
    for (int i = threadIdx.x; i < n2; i += blockDim.x) { float2 v = __half22float2(x[i]); s += v.x + v.y; }
    float mean = block_sum(s) / (2 * n2);
    float var = 0.f;
    for (int i = threadIdx.x; i < n2; i += blockDim.x) {
        float2 v = __half22float2(x[i]);
        var += (v.x - mean) * (v.x - mean) + (v.y - mean) * (v.y - mean);
    }
    float rstd = rsqrtf(block_sum(var) / (2 * n2) + eps);
    for (int i = threadIdx.x; i < n2; i += blockDim.x) {
        float2 v = __half22float2(x[i]), wv = __half22float2(w[i]), bv = __half22float2(b[i]);
        y[i] = __floats2half2_rn((v.x - mean) * rstd * wv.x + bv.x, (v.y - mean) * rstd * wv.y + bv.y);
    }
}

// One block per row of dim (even) elements.
extern "C" __global__ void layernorm(const h16* x, const h16* w, const h16* bias, h16* out, int dim, float eps) {
    long r = (long)blockIdx.x * dim / 2;
    ln_row((const __half2*)x + r, (const __half2*)w, (const __half2*)bias, (__half2*)out + r, dim / 2, eps);
}

// Residual add fused with the next layer norm: x = h(x + d), then out = layernorm(x). One block per row.
extern "C" __global__ void add_layernorm(h16* x, const h16* d, const h16* w, const h16* bias, h16* out, int dim, float eps) {
    long r = (long)blockIdx.x * dim / 2;
    __half2* xr = (__half2*)x + r;
    const __half2* dr = (const __half2*)d + r;
    for (int i = threadIdx.x; i < dim / 2; i += blockDim.x) {
        float2 a = __half22float2(xr[i]), b = __half22float2(dr[i]);
        xr[i] = __floats2half2_rn(a.x + b.x, a.y + b.y);
    }
    // each thread reads back only the elements it wrote, so no barrier is needed before the statistics
    ln_row(xr, (const __half2*)w, (const __half2*)bias, (__half2*)out + r, dim / 2, eps);
}

// res += d elementwise over n2 half2 pairs.
extern "C" __global__ void add_inplace(h16* res, const h16* d, long n2) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n2) return;
    float2 a = __half22float2(((__half2*)res)[i]), b = __half22float2(((const __half2*)d)[i]);
    ((__half2*)res)[i] = __floats2half2_rn(a.x + b.x, a.y + b.y);
}

// y[r, c] = bias[c] for an output whose rows are ld apart (n and ld multiples of 8; 8 halves per thread);
// the GEMM then accumulates with beta = 1, which rounds acc + bias once, like torch's addmm.
extern "C" __global__ void fill_bias(h16* y, const h16* bias, long rows, int n, long ld) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x, n8 = n / 8;
    if (i < rows * n8) { long r = i / n8; int c = i % n8; ((uint4*)(y + r * ld))[c] = ((const uint4*)bias)[c]; }
}

// Bias of the head-contiguous cross k/v buffer: y[blk, j, c] = bias[blk * HD + c] (blk = layer, k|v, head),
// 8 halves per thread over n8 = n / 8 groups.
extern "C" __global__ void fill_bias_heads(h16* y, const h16* bias, long n8) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n8) ((uint4*)y)[i] = ((const uint4*)bias)[(i / ((long)TENC * HD / 8)) * (HD / 8) + i % (HD / 8)];
}

// Exact (erf) GELU, as nn.GELU(), over n2 half2 pairs.
__device__ __forceinline__ float gelu(float v) { return 0.5f * v * (1.f + erff(v * 0.70710678118654752f)); }
extern "C" __global__ void gelu_inplace(h16* x, long n2) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n2) return;
    float2 v = __half22float2(((__half2*)x)[i]);
    ((__half2*)x)[i] = __floats2half2_rn(gelu(v.x), gelu(v.y));
}

// x[r, c] += pos[r % period, c]
extern "C" __global__ void add_pos(h16* x, const h16* pos, long rows, int n, int period) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < rows * n) { long r = i / n; int c = i % n; x[i] = h(f(x[i]) + f(pos[(long)(r % period) * n + c])); }
}

// conv1 (nm -> d, kernel 3, padding 1) as a GEMM: X[g, t, ci * 3 + k] = mel[g, ci, t + k - 1].
extern "C" __global__ void im2col1(const h16* mel, h16* X, long n, int nm) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int j = i % (3 * nm);
    long r = i / (3 * nm);
    int t = r % NFR, g = r / NFR, ci = j / 3, tt = t + j % 3 - 1;
    X[i] = (tt >= 0 && tt < NFR) ? mel[((long)g * nm + ci) * NFR + tt] : h(0.f);
}

// conv2 (d -> d, kernel 3, stride 2, padding 1): X[g, t, ci * 3 + k] = h1[g, 2t + k - 1, ci].
extern "C" __global__ void im2col2(const h16* h1, h16* X, long n, int d) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int j = i % (3 * d);
    long r = i / (3 * d);
    int t = r % TENC, g = r / TENC, ci = j / 3, tt = 2 * t + j % 3 - 1;
    X[i] = (tt >= 0 && tt < NFR) ? h1[((long)g * NFR + tt) * d + ci] : h(0.f);
}

// Row softmax in place (fp16 in and out, fp32 inside), rows of n <= 2048. Block 256.
extern "C" __global__ void softmax_rows(h16* s, int n) {
    h16* r = s + (long)blockIdx.x * n;
    float v[8];
    float m = -INFINITY;
#pragma unroll
    for (int k = 0; k < 8; k++) {
        int i = threadIdx.x + k * 256;
        v[k] = i < n ? f(r[i]) : -INFINITY;
        m = fmaxf(m, v[k]);
    }
    m = block_max(m);
    float sum = 0.f;
#pragma unroll
    for (int k = 0; k < 8; k++) {
        int i = threadIdx.x + k * 256;
        v[k] = i < n ? expf(v[k] - m) : 0.f;
        sum += v[k];
    }
    sum = block_sum(sum);
#pragma unroll
    for (int k = 0; k < 8; k++) {
        int i = threadIdx.x + k * 256;
        if (i < n) r[i] = h(v[k] / sum);
    }
}

// Encoder self-attention, flash-attention style (as PyTorch's SDPA runs it): one block per (64 queries, head,
// clip), 4 warps x 16 query rows, K/V tiles of 64 keys double-buffered in shared memory with cp.async.
// Scores stay in fp32 registers (mma.m16n8k16), online softmax, P rounded to fp16 for P.V, O / l rounded to fp16.
#define AT 64
#define ALD 72 // shared row pitch in halves (144 B: 16-byte aligned rows, no ldmatrix bank conflicts)

__device__ __forceinline__ unsigned smem_u32(const void* p) { return (unsigned)__cvta_generic_to_shared(p); }

__device__ __forceinline__ void cp16(void* dst, const void* src, bool valid) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(smem_u32(dst)), "l"(src), "r"(valid ? 16 : 0));
}

__device__ __forceinline__ void ldsm4(unsigned* r, const void* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(smem_u32(p)));
}

__device__ __forceinline__ void ldsm4t(unsigned* r, const void* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(smem_u32(p)));
}

__device__ __forceinline__ void mma16816(float* c, const unsigned* a, unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ unsigned pack_h2(float lo, float hi) {
    __half2 v = __floats2half2_rn(lo, hi);
    return *(unsigned*)&v;
}

// 64 rows x 64 halves starting at row `row0` of a [TENC, 3d] buffer (column offset already in src) -> shared
__device__ __forceinline__ void flash_tile(h16* dst, const h16* src, int row0, int d) {
    for (int i = threadIdx.x; i < AT * 8; i += blockDim.x) {
        int r = i >> 3, c = (i & 7) * 8;
        bool ok = row0 + r < TENC;
        cp16(dst + r * ALD + c, src + (long)(ok ? row0 + r : 0) * 3 * d + c, ok);
    }
}

extern "C" __global__ void __launch_bounds__(128) enc_flash(const h16* qkv, h16* out, int d) {
    __shared__ __align__(128) h16 Qs[AT * ALD];
    __shared__ __align__(128) h16 Ks[2][AT * ALD];
    __shared__ __align__(128) h16 Vs[2][AT * ALD];
    int q0 = blockIdx.x * AT, hh = blockIdx.y, g = blockIdx.z;
    int w = threadIdx.x >> 5, lane = threadIdx.x & 31, gr = lane >> 2, tq = lane & 3;
    const h16* base = qkv + (long)g * TENC * 3 * d + hh * HD;
    flash_tile(Qs, base, q0, d);
    flash_tile(Ks[0], base + d, 0, d);
    flash_tile(Vs[0], base + 2 * d, 0, d);
    asm volatile("cp.async.commit_group;\n");
    const int ntiles = (TENC + AT - 1) / AT;
    unsigned qa[4][4];
    float o[8][4], m[2] = {-INFINITY, -INFINITY}, l[2] = {0.f, 0.f};
    for (int j = 0; j < 8; j++) o[j][0] = o[j][1] = o[j][2] = o[j][3] = 0.f;
    for (int kt = 0; kt < ntiles; kt++) {
        int cur = kt & 1;
        if (kt + 1 < ntiles) {
            flash_tile(Ks[cur ^ 1], base + d, (kt + 1) * AT, d);
            flash_tile(Vs[cur ^ 1], base + 2 * d, (kt + 1) * AT, d);
        }
        asm volatile("cp.async.commit_group;\n");
        asm volatile("cp.async.wait_group 1;\n");
        __syncthreads();
        if (kt == 0)
            for (int kk = 0; kk < 4; kk++)
                ldsm4(qa[kk], Qs + (w * 16 + (lane & 7) + ((lane >> 3) & 1) * 8) * ALD + kk * 16 + (lane >> 4) * 8);
        // S = Q K^T for this warp's 16 rows and 64 keys: 8 n-tiles of 8 keys
        float sc[8][4];
        for (int j = 0; j < 8; j++) sc[j][0] = sc[j][1] = sc[j][2] = sc[j][3] = 0.f;
        const h16* K = Ks[cur];
        for (int kk = 0; kk < 4; kk++)
            for (int j = 0; j < 8; j += 2) {
                unsigned b[4];
                ldsm4(b, K + (j * 8 + (lane & 7) + (lane >> 4) * 8) * ALD + kk * 16 + ((lane >> 3) & 1) * 8);
                mma16816(sc[j], qa[kk], b[0], b[1]);
                mma16816(sc[j + 1], qa[kk], b[2], b[3]);
            }
        // online softmax; this thread holds rows gr (c0, c1) and gr + 8 (c2, c3), columns j * 8 + 2 tq + {0, 1}
        float tmax[2] = {-INFINITY, -INFINITY};
        for (int j = 0; j < 8; j++)
            for (int e = 0; e < 4; e++) {
                int key = kt * AT + j * 8 + 2 * tq + (e & 1);
                float v = key < TENC ? sc[j][e] * 0.125f : -INFINITY;
                sc[j][e] = v;
                tmax[e >> 1] = fmaxf(tmax[e >> 1], v);
            }
        float corr[2];
        for (int r = 0; r < 2; r++) {
            tmax[r] = fmaxf(tmax[r], __shfl_xor_sync(0xffffffff, tmax[r], 1));
            tmax[r] = fmaxf(tmax[r], __shfl_xor_sync(0xffffffff, tmax[r], 2));
            float mn = fmaxf(m[r], tmax[r]);
            corr[r] = expf(m[r] - mn);
            m[r] = mn;
            l[r] *= corr[r];
        }
        for (int j = 0; j < 8; j++) {
            o[j][0] *= corr[0]; o[j][1] *= corr[0]; o[j][2] *= corr[1]; o[j][3] *= corr[1];
        }
        unsigned pa[4][4];
        for (int j = 0; j < 8; j++) {
            float p0 = expf(sc[j][0] - m[0]), p1 = expf(sc[j][1] - m[0]);
            float p2 = expf(sc[j][2] - m[1]), p3 = expf(sc[j][3] - m[1]);
            l[0] += p0 + p1;
            l[1] += p2 + p3;
            // the S accumulator layout of n-tiles 2kk, 2kk+1 is the A layout of k-chunk kk
            pa[j >> 1][(j & 1) * 2] = pack_h2(p0, p1);
            pa[j >> 1][(j & 1) * 2 + 1] = pack_h2(p2, p3);
        }
        // O += P V
        const h16* V = Vs[cur];
        for (int kk = 0; kk < 4; kk++)
            for (int j = 0; j < 8; j += 2) {
                unsigned b[4];
                ldsm4t(b, V + (kk * 16 + (lane & 7) + ((lane >> 3) & 1) * 8) * ALD + j * 8 + (lane >> 4) * 8);
                mma16816(o[j], pa[kk], b[0], b[1]);
                mma16816(o[j + 1], pa[kk], b[2], b[3]);
            }
        __syncthreads();
    }
    for (int r = 0; r < 2; r++) {
        l[r] += __shfl_xor_sync(0xffffffff, l[r], 1);
        l[r] += __shfl_xor_sync(0xffffffff, l[r], 2);
    }
    for (int r = 0; r < 2; r++) {
        int row = q0 + w * 16 + gr + r * 8;
        if (row >= TENC) continue;
        h16* dst = out + ((long)g * TENC + row) * d + hh * HD + 2 * tq;
        for (int j = 0; j < 8; j++) *(__half2*)(dst + j * 8) = __floats2half2_rn(o[j][2 * r] / l[r], o[j][2 * r + 1] / l[r]);
    }
}

// ---- decoder (one row per sequence; position p = ctr[0]) ----

// x[b] = tok_emb[tok[b, p]] + pos_emb[p]
extern "C" __global__ void embed(const int* tok, const int* ctr, const h16* emb, const h16* pe, h16* x, int maxT, int d) {
    int b = blockIdx.x, p = ctr[0], t = tok[b * maxT + p];
    for (int c = threadIdx.x; c < d; c += blockDim.x) x[(long)b * d + c] = h(f(emb[(long)t * d + c]) + f(pe[(long)p * d + c]));
}

// k and v of the current position into the per-layer self-attention cache [B, maxT, d].
extern "C" __global__ void kv_write(const h16* qkv, h16* kc, h16* vc, const int* ctr, int maxT, int d) {
    int b = blockIdx.x, p = ctr[0];
    long dst = ((long)b * maxT + p) * d;
    for (int c = threadIdx.x; c < d; c += blockDim.x) {
        kc[dst + c] = qkv[(long)b * 3 * d + d + c];
        vc[dst + c] = qkv[(long)b * 3 * d + 2 * d + c];
    }
}

// Causal self-attention over positions 0..p. grid (B, heads), block 128, maxT <= 448.
// q is scaled by 1/8 before the product (exact in fp16), scores and probabilities are rounded to fp16.
extern "C" __global__ void self_attn(const h16* qkv, const h16* kc, const h16* vc, h16* out, const int* ctr, const int* done, int maxT, int d) {
    __shared__ float q[HD];
    __shared__ float sc[448];
    __shared__ float red[HD];
    int b = blockIdx.x, hh = blockIdx.y, L = ctr[0] + 1;
    if (done[b]) return;
    if (threadIdx.x < HD) q[threadIdx.x] = rh(f(qkv[(long)b * 3 * d + hh * HD + threadIdx.x]) * 0.125f);
    __syncthreads();
    int w = threadIdx.x >> 5, l = threadIdx.x & 31, nw = blockDim.x >> 5;
    const h16* K = kc + (long)b * maxT * d + hh * HD;
    for (int j = w; j < L; j += nw) {
        float2 k2 = __half22float2(((const __half2*)(K + (long)j * d))[l]);
        float a = q[2 * l] * k2.x + q[2 * l + 1] * k2.y;
        for (int o = 16; o > 0; o >>= 1) a += __shfl_xor_sync(0xffffffff, a, o);
        if (l == 0) sc[j] = rh(a);
    }
    __syncthreads();
    float m = -INFINITY;
    for (int j = threadIdx.x; j < L; j += blockDim.x) m = fmaxf(m, sc[j]);
    m = block_max(m);
    float s = 0.f;
    for (int j = threadIdx.x; j < L; j += blockDim.x) s += expf(sc[j] - m);
    s = block_sum(s);
    for (int j = threadIdx.x; j < L; j += blockDim.x) sc[j] = rh(expf(sc[j] - m) / s);
    __syncthreads();
    int c = threadIdx.x & (HD - 1), half = threadIdx.x >> 6;
    const h16* V = vc + (long)b * maxT * d + hh * HD + c;
    float a = 0.f;
    for (int j = half; j < L; j += 2) a += sc[j] * f(V[(long)j * d]);
    if (half == 1) red[c] = a;
    __syncthreads();
    if (half == 0) out[(long)b * d + hh * HD + c] = h(a + red[c]);
}

// Cross-attention over the 1500 encoder positions, in three kernels so every launch has enough blocks:
// scores (rounded to fp16) and per-chunk softmax statistics -> per-chunk weighted sums with the global softmax
// -> sum of the chunks. kv holds each sequence's cross k and v head-contiguous: [b][layer][k | v][head][position][HD],
// so a block reads one contiguous run of positions with 16-byte loads. Finished sequences (done[b]) are skipped:
// their tokens are fixed to eos.
#define XCH 64
#define NCH ((TENC + XCH - 1) / XCH)
__device__ __forceinline__ long xkv_off(int b, int layer, int kv, int hh, int nl, int nh) {
    return ((((long)b * nl + layer) * 2 + kv) * nh + hh) * (long)TENC * HD;
}

// grid (B * heads, NCH), block 128: each group of 8 lanes takes one position (8 x 16-byte loads = 64 halves).
// Also writes the chunk's max score and sum of exp(score - max) to stat[bh, chunk].
extern "C" __global__ void cross_scores(const h16* q, const h16* kv, float* S, float2* stat, const int* done, int layer, int nl, int d) {
    __shared__ float qs[HD];
    __shared__ float cs[XCH];
    int nh = d / HD, bh = blockIdx.x, b = bh / nh, hh = bh % nh, j0 = blockIdx.y * XCH;
    if (done[b]) return;
    if (threadIdx.x < HD) qs[threadIdx.x] = rh(f(q[(long)b * d + hh * HD + threadIdx.x]) * 0.125f);
    __syncthreads();
    const h16* K = kv + xkv_off(b, layer, 0, hh, nl, nh);
    int grp = threadIdx.x >> 3, sub = threadIdx.x & 7, ngrp = blockDim.x >> 3;
    for (int jj = grp; jj < XCH; jj += ngrp) {
        int j = j0 + jj;
        float a = 0.f;
        if (j < TENC) {
            uint4 u = *(const uint4*)(K + (long)j * HD + sub * 8);
            const __half2* p2 = (const __half2*)&u;
#pragma unroll
            for (int t = 0; t < 4; t++) { float2 v = __half22float2(p2[t]); a += qs[sub * 8 + 2 * t] * v.x + qs[sub * 8 + 2 * t + 1] * v.y; }
        }
        for (int o = 4; o > 0; o >>= 1) a += __shfl_xor_sync(0xffffffff, a, o);
        if (sub == 0) {
            a = j < TENC ? rh(a) : -INFINITY;
            cs[jj] = a;
            if (j < TENC) S[(long)bh * TENC + j] = a;
        }
    }
    __syncthreads();
    if (threadIdx.x < 32) {
        float a = cs[threadIdx.x], b2 = cs[threadIdx.x + 32];
        float m = fmaxf(a, b2);
        for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, o));
        float s = (a == -INFINITY ? 0.f : expf(a - m)) + (b2 == -INFINITY ? 0.f : expf(b2 - m));
        for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffff, s, o);
        if (threadIdx.x == 0) stat[(long)bh * NCH + blockIdx.y] = make_float2(m, s);
    }
}

// grid (B * heads, NCH), block 128 (16 groups of 8 lanes; a lane owns 8 of the 64 output columns).
// Also stores the probabilities of the alignment heads (aslot[layer * heads + head] >= 0) into
// align[b, p, slot, :] for the token timestamps.
extern "C" __global__ void cross_pv(const float* S, const float2* stat, const h16* kv, float* part, h16* align, const int* aslot,
                                    const int* ctr, const int* done, int layer, int nl, int d, int na, int maxT) {
    __shared__ float p[XCH];
    __shared__ float red[16][HD + 1];
    int nh = d / HD, bh = blockIdx.x, b = bh / nh, hh = bh % nh, ch = blockIdx.y, j0 = ch * XCH, t = threadIdx.x;
    if (done[b]) return;
    // global max and sum from the chunk statistics
    float m = -INFINITY;
    for (int c = 0; c < NCH; c++) m = fmaxf(m, stat[(long)bh * NCH + c].x);
    float sum = 0.f;
    for (int c = 0; c < NCH; c++) { float2 st = stat[(long)bh * NCH + c]; sum += st.y * expf(st.x - m); }
    int n = min(XCH, TENC - j0);
    if (t < n) p[t] = rh(expf(S[(long)bh * TENC + j0 + t] - m) / sum);
    __syncthreads();
    int grp = t >> 3, sub = t & 7;
    const h16* V = kv + xkv_off(b, layer, 1, hh, nl, nh) + (long)j0 * HD + sub * 8;
    float a[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    for (int jj = grp; jj < n; jj += 16) {
        uint4 u = *(const uint4*)(V + (long)jj * HD);
        const __half2* p2 = (const __half2*)&u;
        float pj = p[jj];
#pragma unroll
        for (int k = 0; k < 4; k++) { float2 v = __half22float2(p2[k]); a[2 * k] += pj * v.x; a[2 * k + 1] += pj * v.y; }
    }
#pragma unroll
    for (int k = 0; k < 8; k++) red[grp][sub * 8 + k] = a[k];
    __syncthreads();
    if (t < HD) {
        float s = 0.f;
        for (int g = 0; g < 16; g++) s += red[g][t];
        part[((long)bh * NCH + ch) * HD + t] = s;
    }
    int slot = aslot[layer * nh + hh];
    if (slot >= 0 && t < n) align[(((long)b * maxT + ctr[0]) * na + slot) * TENC + j0 + t] = h(p[t]);
}

extern "C" __global__ void cross_combine(const float* part, h16* out, const int* done, int d) {
    int nh = d / HD, bh = blockIdx.x, c = threadIdx.x;
    if (done[bh / nh]) return;
    float a = 0.f;
    for (int k = 0; k < NCH; k++) a += part[((long)bh * NCH + k) * HD + c];
    out[(long)(bh / nh) * d + (bh % nh) * HD + c] = h(a);
}

// Greedy choice of token p + 1 with the suppressed tokens (sup[i] != 0; and the begin-suppressed ones
// at the first generated position); the prompt positions are left as they are. Finished rows get eos, and so
// does a row whose every token is suppressed.
// grid B, block 1024.
extern "C" __global__ void argmax_tok(const h16* logits, int V, const unsigned char* sup, int* tok, int* done, const int* ctr,
                                      int maxT, int prompt_len, int eos, int bs1, int bs2) {
    __shared__ float sv[32];
    __shared__ int si[32];
    int b = blockIdx.x, p = ctr[0];
    if (p + 1 < prompt_len || p + 1 >= maxT) return;
    bool first = p + 1 == prompt_len;
    const h16* lg = logits + (long)b * V;
    float best = -INFINITY;
    int bi = 0x7fffffff;
    for (int i = threadIdx.x; i < V; i += blockDim.x) {
        if (sup[i] || (first && (i == bs1 || i == bs2))) continue;
        float v = f(lg[i]);
        if (v > best || (v == best && i < bi)) { best = v; bi = i; }
    }
    for (int o = 16; o > 0; o >>= 1) {
        float ov = __shfl_xor_sync(0xffffffff, best, o);
        int oi = __shfl_xor_sync(0xffffffff, bi, o);
        if (ov > best || (ov == best && oi < bi)) { best = ov; bi = oi; }
    }
    int w = threadIdx.x >> 5, l = threadIdx.x & 31;
    if (l == 0) { sv[w] = best; si[w] = bi; }
    __syncthreads();
    if (w == 0) {
        int nw = blockDim.x >> 5;
        best = l < nw ? sv[l] : -INFINITY;
        bi = l < nw ? si[l] : 0x7fffffff;
        for (int o = 16; o > 0; o >>= 1) {
            float ov = __shfl_xor_sync(0xffffffff, best, o);
            int oi = __shfl_xor_sync(0xffffffff, bi, o);
            if (ov > best || (ov == best && oi < bi)) { best = ov; bi = oi; }
        }
        if (l == 0) {
            int t = done[b] || bi == 0x7fffffff ? eos : bi;
            tok[b * maxT + p + 1] = t;
            if (t == eos) done[b] = 1;
        }
    }
}

extern "C" __global__ void advance(int* ctr) { ctr[0] += 1; }

// Skinny GEMM for decode-sized M: y[m, n] = x[m, :] . W[n, :] (+ bias[n], added in fp32 before the rounding,
// like torch's addmm), M <= MR. Each warp owns COLS output columns and streams their weight rows once
// with 16-byte loads; x is staged in shared memory in K chunks and reused by every column of the block.
template <int MR, int COLS, int KC>
__device__ __forceinline__ void gemv_body(const h16* __restrict__ x, int ldx, const h16* __restrict__ W, h16* y,
                                          int ldy, int M, int N, int K, const h16* bias) {
    extern __shared__ __align__(16) unsigned char smraw[];
    h16* xs = (h16*)smraw;
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int n0 = (blockIdx.x * (blockDim.x >> 5) + warp) * COLS;
    float acc[COLS][MR];
#pragma unroll
    for (int c = 0; c < COLS; c++)
#pragma unroll
        for (int m = 0; m < MR; m++) acc[c][m] = 0.f;
    for (int k0 = 0; k0 < K; k0 += KC) {
        int kc = min(KC, K - k0);
        __syncthreads();
        for (int i = threadIdx.x; i < M * (kc / 8); i += blockDim.x) {
            int m = i / (kc / 8), j = i % (kc / 8);
            ((uint4*)(xs + m * KC))[j] = *(const uint4*)(x + (long)m * ldx + k0 + j * 8);
        }
        __syncthreads();
        for (int kk = lane * 8; kk < kc; kk += 256) {
            float w[COLS][8];
#pragma unroll
            for (int c = 0; c < COLS; c++) {
                if (n0 + c < N) {
                    uint4 u = __ldg((const uint4*)(W + (long)(n0 + c) * K + k0 + kk));
                    const __half2* p = (const __half2*)&u;
#pragma unroll
                    for (int j = 0; j < 4; j++) { float2 v = __half22float2(p[j]); w[c][2 * j] = v.x; w[c][2 * j + 1] = v.y; }
                } else {
#pragma unroll
                    for (int j = 0; j < 8; j++) w[c][j] = 0.f;
                }
            }
#pragma unroll
            for (int m = 0; m < MR; m++) {
                if (m < M) {
                    uint4 u = *(const uint4*)(xs + m * KC + kk);
                    const __half2* p = (const __half2*)&u;
                    float xf[8];
#pragma unroll
                    for (int j = 0; j < 4; j++) { float2 v = __half22float2(p[j]); xf[2 * j] = v.x; xf[2 * j + 1] = v.y; }
#pragma unroll
                    for (int c = 0; c < COLS; c++)
#pragma unroll
                        for (int j = 0; j < 8; j++) acc[c][m] += w[c][j] * xf[j];
                }
            }
        }
    }
#pragma unroll
    for (int c = 0; c < COLS; c++) {
#pragma unroll
        for (int m = 0; m < MR; m++) {
            float v = acc[c][m];
            for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
            if (lane == (m & 31) && m < M && n0 + c < N) y[(long)m * ldy + n0 + c] = h(bias ? v + f(bias[n0 + c]) : v);
        }
    }
}

#define GEMV(MR, COLS, KC) \
    extern "C" __global__ void gemv_##MR(const h16* x, int ldx, const h16* W, h16* y, int ldy, int M, int N, int K, const h16* bias) { \
        gemv_body<MR, COLS, KC>(x, ldx, W, y, ldy, M, N, K, bias); }
GEMV(1, 4, 2048)
GEMV(2, 4, 2048)
GEMV(4, 4, 1024)
GEMV(8, 4, 1024)
