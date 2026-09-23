// CUDA kernels for the Laya decision model (ModernBERT encoder + decision head).
//
// Activations are [B, T, d] row-major, padded to a common T; lens[b] gives the
// real length of sequence b and only attention needs it. Every reduction that
// laya.c performs in double is done in double here too, so the two agree to the
// same ~1e-4 the C port claims against PyTorch.

__device__ __forceinline__ float gelu_erf(float x)
{
    return 0.5f * x * (1.0f + erff(x * 0.70710678118654752f));
}

// Row-wise LayerNorm over [rows, d] with optional bias, matching laya.c's
// double-precision mean/variance accumulation.
extern "C" __global__ void LayerNorm(
    const float *__restrict__ x,
    const float *__restrict__ w,
    const float *__restrict__ b,
    float *__restrict__ out,
    int rows, int d, float eps)
{
    int row = blockIdx.x;
    if (row >= rows) return;

    const float *xr = x + (size_t)row * d;
    float *o = out + (size_t)row * d;

    __shared__ double s_mu[32];
    __shared__ double s_var[32];
    __shared__ double s_mean;
    __shared__ float s_inv;

    int tid = threadIdx.x, lane = tid & 31, wid = tid >> 5, nwarp = blockDim.x >> 5;

    double mu = 0.0;
    for (int i = tid; i < d; i += blockDim.x) mu += (double)xr[i];
    for (int off = 16; off > 0; off >>= 1) mu += __shfl_down_sync(0xffffffff, mu, off);
    if (lane == 0) s_mu[wid] = mu;
    __syncthreads();
    if (tid == 0) {
        double t = 0.0;
        for (int i = 0; i < nwarp; i++) t += s_mu[i];
        s_mean = t / (double)d;
    }
    __syncthreads();

    double mean = s_mean, var = 0.0;
    for (int i = tid; i < d; i += blockDim.x) {
        double c = (double)xr[i] - mean;
        var += c * c;
    }
    for (int off = 16; off > 0; off >>= 1) var += __shfl_down_sync(0xffffffff, var, off);
    if (lane == 0) s_var[wid] = var;
    __syncthreads();
    if (tid == 0) {
        double t = 0.0;
        for (int i = 0; i < nwarp; i++) t += s_var[i];
        s_inv = (float)(1.0 / sqrt(t / (double)d + (double)eps));
    }
    __syncthreads();

    float m = (float)mean, inv = s_inv;
    for (int i = tid; i < d; i += blockDim.x)
        o[i] = (xr[i] - m) * inv * w[i] + (b ? b[i] : 0.0f);
}

// Gather token embeddings: x[b,t,:] = emb[ids[b,t],:].
extern "C" __global__ void Embed(
    const float *__restrict__ emb,
    const int *__restrict__ ids,
    float *__restrict__ x,
    int rows, int d)
{
    int row = blockIdx.x;
    if (row >= rows) return;
    const float *src = emb + (size_t)ids[row] * d;
    float *dst = x + (size_t)row * d;
    for (int i = threadIdx.x; i < d; i += blockDim.x) dst[i] = src[i];
}

// Split a fused [B,T,3d] qkv projection into head-major Q/K/V [B,nh,T,hd],
// optionally applying RoPE to Q and K with rotate_half pairing (j, j+hd/2).
extern "C" __global__ void SplitQKVRope(
    const float *__restrict__ qkv,
    const float *__restrict__ cs,
    const float *__restrict__ sn,
    float *__restrict__ Q,
    float *__restrict__ K,
    float *__restrict__ V,
    int B, int T, int d, int nh, int hd, int use_rope)
{
    int t = blockIdx.x, h = blockIdx.y, b = blockIdx.z;
    if (t >= T || h >= nh || b >= B) return;

    int half = hd >> 1;
    const float *q = qkv + ((size_t)b * T + t) * 3 * d + (size_t)h * hd;
    const float *k = q + d;
    const float *v = q + 2 * d;

    size_t base = (((size_t)b * nh + h) * T + t) * hd;
    float *qo = Q + base, *ko = K + base, *vo = V + base;

    if (use_rope) {
        const float *c = cs + (size_t)t * half;
        const float *s = sn + (size_t)t * half;
        for (int j = threadIdx.x; j < half; j += blockDim.x) {
            float qj = q[j], qh = q[j + half];
            float kj = k[j], kh = k[j + half];
            qo[j] = qj * c[j] - qh * s[j];
            qo[j + half] = qh * c[j] + qj * s[j];
            ko[j] = kj * c[j] - kh * s[j];
            ko[j + half] = kh * c[j] + kj * s[j];
        }
    } else {
        for (int j = threadIdx.x; j < hd; j += blockDim.x) {
            qo[j] = q[j];
            ko[j] = k[j];
        }
    }
    for (int j = threadIdx.x; j < hd; j += blockDim.x) vo[j] = v[j];
}

// Softmax over one row of the attention scores, in place, restricted to the
// symmetric window |i-j| <= win (win < 0 means full). Columns outside the
// window and rows past the sequence length are zeroed, so the PV GEMM that
// follows sees them as absent. One block per (query position, sequence-head).
extern "C" __global__ void AttnSoftmax(
    float *__restrict__ S,
    const int *__restrict__ lens,
    int T, int nh, int win)
{
    int i = blockIdx.x, bh = blockIdx.y;
    if (i >= T) return;

    float *row = S + ((size_t)bh * T + i) * T;
    int len = lens[bh / nh];

    int lo = 0, hi = -1; // empty for a padded query position
    if (i < len) {
        lo = (win >= 0 && i - win > 0) ? i - win : 0;
        hi = (win >= 0 && i + win < len - 1) ? i + win : len - 1;
    }

    __shared__ float s_red[32];
    __shared__ float s_mx, s_sum;

    int tid = threadIdx.x, lane = tid & 31, wid = tid >> 5, nwarp = blockDim.x >> 5;

    float mx = -__int_as_float(0x7f800000); // -inf; NVRTC has no <math.h>
    for (int j = lo + tid; j <= hi; j += blockDim.x) mx = fmaxf(mx, row[j]);
    for (int off = 16; off > 0; off >>= 1) mx = fmaxf(mx, __shfl_down_sync(0xffffffff, mx, off));
    if (lane == 0) s_red[wid] = mx;
    __syncthreads();
    if (tid == 0) {
        float t = s_red[0];
        for (int u = 1; u < nwarp; u++) t = fmaxf(t, s_red[u]);
        s_mx = t;
    }
    __syncthreads();

    float m = s_mx, sum = 0.0f;
    for (int j = lo + tid; j <= hi; j += blockDim.x) {
        float e = __expf(row[j] - m);
        row[j] = e;
        sum += e;
    }
    for (int off = 16; off > 0; off >>= 1) sum += __shfl_down_sync(0xffffffff, sum, off);
    if (lane == 0) s_red[wid] = sum;
    __syncthreads();
    if (tid == 0) {
        float t = 0.0f;
        for (int u = 0; u < nwarp; u++) t += s_red[u];
        s_sum = t;
    }
    __syncthreads();

    float inv = hi >= lo ? 1.0f / s_sum : 0.0f;
    for (int j = tid; j < T; j += blockDim.x)
        row[j] = (j >= lo && j <= hi) ? row[j] * inv : 0.0f;
}

// x += y over n elements.
extern "C" __global__ void AddResidual(float *__restrict__ x, const float *__restrict__ y, int n)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
        x[i] += y[i];
}

// Add a per-column bias to every row of a [rows, n] matrix.
extern "C" __global__ void AddBias(float *__restrict__ x, const float *__restrict__ b, int rows, int n)
{
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < rows * n; idx += blockDim.x * gridDim.x)
        x[idx] += b[idx % n];
}

// Add each sequence's question-type embedding to all of its token states.
extern "C" __global__ void AddTypeEmb(
    float *__restrict__ x,
    const float *__restrict__ emb,
    const int *__restrict__ types,
    int B, int T, int d)
{
    int n = B * T * d;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n; idx += blockDim.x * gridDim.x)
        x[idx] += emb[(size_t)types[idx / (T * d)] * d + idx % d];
}

// GeGLU over a fused [rows, 2*dff] projection: out[j] = gelu(in[j]) * in[dff+j].
extern "C" __global__ void GeGLU(
    const float *__restrict__ in, float *__restrict__ out, int rows, int dff)
{
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < rows * dff; idx += blockDim.x * gridDim.x) {
        int r = idx / dff, j = idx - r * dff;
        const float *src = in + (size_t)r * 2 * dff;
        out[idx] = gelu_erf(src[j]) * src[dff + j];
    }
}

extern "C" __global__ void ReLU(float *__restrict__ x, int n)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
        x[i] = x[i] > 0.0f ? x[i] : 0.0f;
}

extern "C" __global__ void GELU(float *__restrict__ x, int n)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
        x[i] = gelu_erf(x[i]);
}

// Gather arbitrary rows: out[r,:] = x[rowIdx[r],:].
extern "C" __global__ void GatherRows(
    const float *__restrict__ x,
    const int *__restrict__ rowIdx,
    float *__restrict__ out,
    int rows, int d)
{
    int r = blockIdx.x;
    if (r >= rows) return;
    const float *src = x + (size_t)rowIdx[r] * d;
    float *dst = out + (size_t)r * d;
    for (int i = threadIdx.x; i < d; i += blockDim.x) dst[i] = src[i];
}

// Fused attention.
//
// FlashAttention computes softmax(QK^T/sqrt(hd)) V for one tile of 64 query
// positions of one (sequence, head) without materializing the T by T score
// matrix: keys and values stream through shared memory a tile at a time and the
// softmax is kept online, with a running maximum and normalizer per query row.
// Attention is bidirectional, so every tile in the (windowed) key range is
// visited; a query position past the sequence length sees no key at all and its
// output row is exactly zero.
//
// The two products run on the tensor cores as mma.sync.aligned.m16n8k8 with
// tf32 operands and float32 accumulation. sm_121 has neither wgmma nor tcgen05,
// so mma.sync is the widest instruction available. The block is four warps, one
// per group of 16 query rows; the 16x8 accumulator fragment puts each row in
// one quad of lanes, which is what makes the row-wise softmax reductions four
// __shfl_xor_sync steps instead of a shared-memory round trip.

#define FA_TILE 64 // query rows per block, and key rows per iteration
#define FA_HD 64   // head dimension the kernel is specialized for
#define FA_KPAD 68 // row stride of the K/P tile: 68 % 32 == 4 spreads a quad's
#define FA_VPAD 72 // row stride of the V tile: 72 % 32 == 8, likewise
#define FA_NEG 1e30f // finite stand-in for -inf, so an all-masked row stays 0

__device__ __forceinline__ unsigned fa_tf32(float x)
{
    unsigned r;
    asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(r) : "f"(x));
    return r;
}

// fa_mma accumulates one 16x8x8 tf32 product into acc.
__device__ __forceinline__ void fa_mma(float *acc, const unsigned *a, const unsigned *b)
{
    asm volatile(
        "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

extern "C" __global__ __launch_bounds__(128) void FlashAttention(
    const float *__restrict__ Q,
    const float *__restrict__ K,
    const float *__restrict__ V,
    const int *__restrict__ lens,
    float *__restrict__ O,
    int B, int T, int nh, int win, float scale)
{
    const int qt = blockIdx.x, h = blockIdx.y, b = blockIdx.z;
    const int q0 = qt * FA_TILE;
    if (q0 >= T || h >= nh || b >= B) return;

    const int d = nh * FA_HD;
    const int len = lens[b];
    float *Oh = O + ((size_t)b * T) * d + (size_t)h * FA_HD;

    if (q0 >= len) { // every query in this tile is padding
        for (int i = threadIdx.x; i < FA_TILE * FA_HD; i += blockDim.x) {
            int t = q0 + (i >> 6);
            if (t < T) Oh[(size_t)t * d + (i & 63)] = 0.0f;
        }
        return;
    }

    __shared__ float sKP[FA_TILE * FA_KPAD]; // the K tile, then the P tile
    __shared__ float sV[FA_TILE * FA_VPAD];

    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int gid = lane >> 2, tig = lane & 3;
    const int r0 = warp * 16; // first query row this warp owns

    const size_t head = ((size_t)b * nh + h) * T * FA_HD;
    const float *Qh = Q + head, *Kh = K + head, *Vh = V + head;

    // Q stays in registers as the A fragments of all eight k blocks.
    unsigned qa[8][4];
    {
        int ra = q0 + r0 + gid, rb = ra + 8;
        const float *qra = ra < T ? Qh + (size_t)ra * FA_HD : nullptr;
        const float *qrb = rb < T ? Qh + (size_t)rb * FA_HD : nullptr;
#pragma unroll
        for (int kb = 0; kb < 8; kb++) {
            int c = 8 * kb + tig;
            qa[kb][0] = fa_tf32(qra ? qra[c] : 0.0f);
            qa[kb][1] = fa_tf32(qrb ? qrb[c] : 0.0f);
            qa[kb][2] = fa_tf32(qra ? qra[c + 4] : 0.0f);
            qa[kb][3] = fa_tf32(qrb ? qrb[c + 4] : 0.0f);
        }
    }

    // Online softmax state and output accumulator. Each thread owns query rows
    // r0+gid and r0+gid+8, which is what the [2] indexing below tracks.
    float mrun[2] = {-FA_NEG, -FA_NEG};
    float lrun[2] = {0.0f, 0.0f};
    float acc[8][4];
#pragma unroll
    for (int nt = 0; nt < 8; nt++)
#pragma unroll
        for (int u = 0; u < 4; u++) acc[nt][u] = 0.0f;

    int qhi = q0 + FA_TILE - 1;
    if (qhi > len - 1) qhi = len - 1;
    int klo = 0, khi = len - 1;
    if (win >= 0) {
        klo = q0 - win > 0 ? q0 - win : 0;
        khi = qhi + win < len - 1 ? qhi + win : len - 1;
    }

    for (int jb = (klo / FA_TILE) * FA_TILE; jb <= khi; jb += FA_TILE) {
        __syncthreads(); // the previous iteration has finished with sKP and sV
        for (int i = threadIdx.x; i < FA_TILE * FA_HD; i += blockDim.x) {
            int r = i >> 6, c = i & 63, j = jb + r;
            float kv = 0.0f, vv = 0.0f;
            if (j < T) {
                kv = Kh[(size_t)j * FA_HD + c];
                vv = Vh[(size_t)j * FA_HD + c];
            }
            sKP[r * FA_KPAD + c] = kv;
            sV[r * FA_VPAD + c] = vv;
        }
        __syncthreads();

        float s[8][4];
#pragma unroll
        for (int nt = 0; nt < 8; nt++) {
            float c[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
            for (int kb = 0; kb < 8; kb++) {
                const float *kr = &sKP[(8 * nt + gid) * FA_KPAD + 8 * kb + tig];
                unsigned bf[2] = {fa_tf32(kr[0]), fa_tf32(kr[4])};
                fa_mma(c, qa[kb], bf);
            }
#pragma unroll
            for (int u = 0; u < 4; u++) s[nt][u] = c[u] * scale;
        }

        // Mask, then the per-row maximum of this tile. A masked entry is set to
        // -FA_NEG rather than -inf so that an all-masked row never forms inf-inf.
        float tmax[2] = {-FA_NEG, -FA_NEG};
#pragma unroll
        for (int nt = 0; nt < 8; nt++) {
#pragma unroll
            for (int u = 0; u < 4; u++) {
                int rr = u >> 1;
                int i = q0 + r0 + gid + 8 * rr;
                int j = jb + 8 * nt + 2 * tig + (u & 1);
                int lo = 0, hi = -1; // an empty range for a padded query
                if (i < len) {
                    lo = (win >= 0 && i - win > 0) ? i - win : 0;
                    hi = (win >= 0 && i + win < len - 1) ? i + win : len - 1;
                }
                if (j < lo || j > hi) s[nt][u] = -FA_NEG;
                tmax[rr] = fmaxf(tmax[rr], s[nt][u]);
            }
        }
#pragma unroll
        for (int off = 1; off < 4; off <<= 1) {
            tmax[0] = fmaxf(tmax[0], __shfl_xor_sync(0xffffffff, tmax[0], off, 4));
            tmax[1] = fmaxf(tmax[1], __shfl_xor_sync(0xffffffff, tmax[1], off, 4));
        }

        float alpha[2], sum[2] = {0.0f, 0.0f};
#pragma unroll
        for (int u = 0; u < 2; u++) {
            float mn = fmaxf(mrun[u], tmax[u]);
            alpha[u] = __expf(mrun[u] - mn); // 0 on the first valid tile
            mrun[u] = mn;
        }
#pragma unroll
        for (int nt = 0; nt < 8; nt++)
#pragma unroll
            for (int u = 0; u < 4; u++) {
                int rr = u >> 1;
                float p = s[nt][u] <= -FA_NEG ? 0.0f : __expf(s[nt][u] - mrun[rr]);
                s[nt][u] = p;
                sum[rr] += p;
            }
#pragma unroll
        for (int off = 1; off < 4; off <<= 1) {
            sum[0] += __shfl_xor_sync(0xffffffff, sum[0], off, 4);
            sum[1] += __shfl_xor_sync(0xffffffff, sum[1], off, 4);
        }
        lrun[0] = alpha[0] * lrun[0] + sum[0];
        lrun[1] = alpha[1] * lrun[1] + sum[1];
#pragma unroll
        for (int nt = 0; nt < 8; nt++)
#pragma unroll
            for (int u = 0; u < 4; u++) acc[nt][u] *= alpha[u >> 1];

        // P replaces K in shared memory: the A fragments of P K-blocks read the
        // probabilities back in the layout the second product wants.
        __syncthreads();
#pragma unroll
        for (int nt = 0; nt < 8; nt++) {
            int col = 8 * nt + 2 * tig;
            sKP[(r0 + gid) * FA_KPAD + col] = s[nt][0];
            sKP[(r0 + gid) * FA_KPAD + col + 1] = s[nt][1];
            sKP[(r0 + gid + 8) * FA_KPAD + col] = s[nt][2];
            sKP[(r0 + gid + 8) * FA_KPAD + col + 1] = s[nt][3];
        }
        __syncthreads();

#pragma unroll
        for (int kb = 0; kb < 8; kb++) {
            const float *pa0 = &sKP[(r0 + gid) * FA_KPAD + 8 * kb + tig];
            const float *pa1 = &sKP[(r0 + gid + 8) * FA_KPAD + 8 * kb + tig];
            unsigned pa[4] = {fa_tf32(pa0[0]), fa_tf32(pa1[0]), fa_tf32(pa0[4]), fa_tf32(pa1[4])};
#pragma unroll
            for (int nt = 0; nt < 8; nt++) {
                const float *vr = &sV[(8 * kb + tig) * FA_VPAD + 8 * nt + gid];
                unsigned bf[2] = {fa_tf32(vr[0]), fa_tf32(vr[4 * FA_VPAD])};
                fa_mma(acc[nt], pa, bf);
            }
        }
    }

    float inv[2];
#pragma unroll
    for (int u = 0; u < 2; u++) inv[u] = lrun[u] > 0.0f ? 1.0f / lrun[u] : 0.0f;
#pragma unroll
    for (int nt = 0; nt < 8; nt++)
#pragma unroll
        for (int u = 0; u < 4; u++) {
            int t = q0 + r0 + gid + 8 * (u >> 1);
            if (t < T) Oh[(size_t)t * d + 8 * nt + 2 * tig + (u & 1)] = acc[nt][u] * inv[u >> 1];
        }
}
