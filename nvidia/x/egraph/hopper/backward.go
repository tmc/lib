package hopper

import (
	"fmt"
	"strings"
)

const (
	// F16 is IEEE binary16 (half precision), rounded to nearest even.
	F16 DType = I64 + 1

	// FP16 and FP32 are aliases for [F16] and [F32].
	FP16 = F16
	FP32 = F32
)

// backwardPrelude extends [Prelude] with type-dispatched FP32 conversions
// (eg_to_f32 and eg_from_f32) over __nv_bfloat16, __half, and float.
const backwardPrelude = `#ifndef EG_BACKWARD_PRELUDE
#define EG_BACKWARD_PRELUDE

#if defined(EG_HOST_EMU)
struct __half { uint16_t b; };
#else
#if !defined(__CUDA_BF16_TYPES_EXIST__)
struct __nv_bfloat16 { uint16_t b; };
#endif
#if !defined(__CUDA_FP16_TYPES_EXIST__)
struct __half { uint16_t b; };
#endif
#endif

EG_DEV float eg_f16_to_f32(uint32_t h) {
    const uint32_t s = (h & 0x8000u) << 16;
    const uint32_t e = (h >> 10) & 31u, m = h & 0x3ffu;
    if (e == 0) {
        const float v = (float)m * (1.0f / 16777216.0f);
        return s ? -v : v;
    }
    if (e == 31) return __uint_as_float(s | 0x7f800000u | (m << 13));
    return __uint_as_float(s | ((e + 112u) << 23) | (m << 13));
}

EG_DEV uint16_t eg_f32_to_f16(float f) {
    return (uint16_t)eg_pack_f16x2(f, 0.0f);
}

EG_DEV float eg_to_f32(float v) { return v; }
EG_DEV float eg_to_f32(uint16_t v) { return eg_bf16_to_f32(v); }
EG_DEV float eg_to_f32(__nv_bfloat16 v) {
    uint16_t u;
    __builtin_memcpy(&u, &v, 2);
    return eg_bf16_to_f32(u);
}
EG_DEV float eg_to_f32(__half v) {
    uint16_t u;
    __builtin_memcpy(&u, &v, 2);
    return eg_f16_to_f32(u);
}

template <typename T> EG_DEV T eg_from_f32(float v);
template <> EG_DEV float eg_from_f32<float>(float v) { return v; }
template <> EG_DEV uint16_t eg_from_f32<uint16_t>(float v) { return eg_f32_to_bf16(v); }
template <> EG_DEV __nv_bfloat16 eg_from_f32<__nv_bfloat16>(float v) {
    const uint16_t u = eg_f32_to_bf16(v);
    __nv_bfloat16 r;
    __builtin_memcpy(&r, &u, 2);
    return r;
}
template <> EG_DEV __half eg_from_f32<__half>(float v) {
    const uint16_t u = eg_f32_to_f16(v);
    __half r;
    __builtin_memcpy(&r, &u, 2);
    return r;
}

#endif // EG_BACKWARD_PRELUDE
`

func bwdCType(d DType) (string, error) {
	switch d {
	case BF16:
		return "__nv_bfloat16", nil
	case FP16:
		return "__half", nil
	case FP32:
		return "float", nil
	default:
		return "", fmt.Errorf("unsupported dtype %v (want BF16, FP16, or FP32)", d)
	}
}

func normThreads(threads int) (int, error) {
	if threads == 0 {
		return 256, nil
	}
	if threads < 32 || threads > 1024 || threads%32 != 0 {
		return 0, fmt.Errorf("threads %d must be a multiple of 32 in 32..1024", threads)
	}
	return threads, nil
}

// RMSNormBackward emits a fused CUDA C++ backward kernel for root-mean-square
// layer normalization:
//
//	y[r, i] = x[r, i] * rsqrt(mean_j(x[r, j]^2) + Eps) * w[i]
//
// One CTA per row r = blockIdx.x computes dX[r, i] and atomically accumulates
// the weight gradient into dW_accum[i] in FP32.
type RMSNormBackward struct {
	Name    string
	Cols    int
	Eps     float32
	DType   DType // BF16, FP16, FP32
	Threads int
}

func (k RMSNormBackward) norm() (RMSNormBackward, string, error) {
	if k.Name == "" {
		k.Name = "rmsnorm_bwd"
	}
	if !cIdent(k.Name) {
		return k, "", fmt.Errorf("rmsnorm backward: bad name %q", k.Name)
	}
	if k.Cols < 1 {
		return k, "", fmt.Errorf("rmsnorm backward %s: cols %d must be positive", k.Name, k.Cols)
	}
	if k.Eps < 0 {
		return k, "", fmt.Errorf("rmsnorm backward %s: negative eps %g", k.Name, k.Eps)
	}
	if k.Eps == 0 {
		k.Eps = 1e-5
	}
	if k.DType == 0 {
		k.DType = BF16
	}
	ct, err := bwdCType(k.DType)
	if err != nil {
		return k, "", fmt.Errorf("rmsnorm backward %s: %w", k.Name, err)
	}
	k.Threads, err = normThreads(k.Threads)
	if err != nil {
		return k, "", fmt.Errorf("rmsnorm backward %s: %w", k.Name, err)
	}
	return k, ct, nil
}

func (k RMSNormBackward) params(ct string) []string {
	return []string{
		ct + "* __restrict__ dX",
		"float* __restrict__ dW_accum",
		"const " + ct + "* __restrict__ dY",
		"const " + ct + "* __restrict__ X",
		"const " + ct + "* __restrict__ W",
		"int rows",
	}
}

// Grid returns the launch grid and block for rows rows.
func (k RMSNormBackward) Grid(rows int) (grid, block [3]int) {
	threads := k.Threads
	if threads == 0 {
		threads = 256
	}
	return [3]int{rows, 1, 1}, [3]int{threads, 1, 1}
}

// CUDA returns the kernel source (with [backwardPrelude], without [Prelude]).
func (k RMSNormBackward) CUDA() (string, error) {
	k, ct, err := k.norm()
	if err != nil {
		return "", err
	}
	var b strings.Builder
	b.WriteString(backwardPrelude)
	b.WriteString("\n")
	fmt.Fprintf(&b, "// %s: fused RMSNorm backward over rows of %d values.\n", k.Name, k.Cols)
	fmt.Fprintf(&b, "__global__ void __launch_bounds__(%d) %s(\n    %s) {\n",
		k.Threads, k.Name, strings.Join(k.params(ct), ",\n    "))
	fmt.Fprintf(&b, "    constexpr int COLS = %d, THREADS = %d;\n", k.Cols, k.Threads)
	fmt.Fprintf(&b, "    constexpr float EPS = %s, INV_COLS = 1.0f / (float)COLS;\n", floatLiteral(k.Eps))
	fmt.Fprintf(&b, "    using T = %s;\n", ct)
	b.WriteString(`    const int r = blockIdx.x;
    if (r >= rows) return;
    __shared__ float eg_red[32];
    const T* x_row = X + (size_t)r * COLS;
    const T* dy_row = dY + (size_t)r * COLS;
    T* dx_row = dX + (size_t)r * COLS;
    float ss = 0.0f, dot = 0.0f;
    for (int i = threadIdx.x; i < COLS; i += THREADS) {
        const float x = eg_to_f32(x_row[i]);
        const float dy = eg_to_f32(dy_row[i]);
        const float w = eg_to_f32(W[i]);
        ss += x * x;
        dot += dy * w * x;
    }
    ss = eg_block_combine<0>(eg_warp_sum(ss), eg_red);
    dot = eg_block_combine<0>(eg_warp_sum(dot), eg_red);
    const float inv_rms = rsqrtf(ss * INV_COLS + EPS);
    const float c = (dot * INV_COLS) * (inv_rms * inv_rms * inv_rms);
    for (int i = threadIdx.x; i < COLS; i += THREADS) {
        const float x = eg_to_f32(x_row[i]);
        const float dy = eg_to_f32(dy_row[i]);
        const float w = eg_to_f32(W[i]);
        dx_row[i] = eg_from_f32<T>(inv_rms * (dy * w) - x * c);
        atomicAdd(&dW_accum[i], dy * x * inv_rms);
    }
}
`)
	return b.String(), nil
}

// Launcher returns void Name_launch(<params...>, cudaStream_t stream).
func (k RMSNormBackward) Launcher() (string, error) {
	k, ct, err := k.norm()
	if err != nil {
		return "", err
	}
	return launcher(k.Name, k.params(ct), func(b *strings.Builder) {
		fmt.Fprintf(b, "    const dim3 grid(rows > 0 ? rows : 0), block(%d);\n", k.Threads)
	}), nil
}

// Emit returns the complete CUDA C++ translation unit ([Prelude], kernel, and launcher).
func (k RMSNormBackward) Emit() (string, error) {
	src, err := k.CUDA()
	if err != nil {
		return "", err
	}
	l, err := k.Launcher()
	if err != nil {
		return "", err
	}
	return Prelude + "\n" + src + "\n" + l, nil
}

// SwiGLUBackward emits a fused CUDA C++ backward kernel for SwiGLU:
//
//	y[r, i] = silu(g[r, i]) * u[r, i],  silu(g) = g * sigmoid(g)
//
// where GateUp has 2*Cols columns per row (g in 0..Cols-1, u in Cols..2*Cols-1).
type SwiGLUBackward struct {
	Name    string
	Cols    int // output hidden cols (GateUp has 2*Cols)
	DType   DType
	Threads int
}

func (k SwiGLUBackward) norm() (SwiGLUBackward, string, error) {
	if k.Name == "" {
		k.Name = "swiglu_bwd"
	}
	if !cIdent(k.Name) {
		return k, "", fmt.Errorf("swiglu backward: bad name %q", k.Name)
	}
	if k.Cols < 1 {
		return k, "", fmt.Errorf("swiglu backward %s: cols %d must be positive", k.Name, k.Cols)
	}
	if k.DType == 0 {
		k.DType = BF16
	}
	ct, err := bwdCType(k.DType)
	if err != nil {
		return k, "", fmt.Errorf("swiglu backward %s: %w", k.Name, err)
	}
	k.Threads, err = normThreads(k.Threads)
	if err != nil {
		return k, "", fmt.Errorf("swiglu backward %s: %w", k.Name, err)
	}
	return k, ct, nil
}

func (k SwiGLUBackward) params(ct string) []string {
	return []string{
		ct + "* __restrict__ dGateUp",
		"const " + ct + "* __restrict__ dY",
		"const " + ct + "* __restrict__ GateUp",
		"int rows",
	}
}

// Grid returns the launch grid and block for rows rows.
func (k SwiGLUBackward) Grid(rows int) (grid, block [3]int) {
	threads := k.Threads
	if threads == 0 {
		threads = 256
	}
	return [3]int{rows, 1, 1}, [3]int{threads, 1, 1}
}

// CUDA returns the kernel source (with [backwardPrelude], without [Prelude]).
func (k SwiGLUBackward) CUDA() (string, error) {
	k, ct, err := k.norm()
	if err != nil {
		return "", err
	}
	var b strings.Builder
	b.WriteString(backwardPrelude)
	b.WriteString("\n")
	fmt.Fprintf(&b, "// %s: fused SwiGLU backward over rows of %d values (GateUp width %d).\n", k.Name, k.Cols, 2*k.Cols)
	fmt.Fprintf(&b, "__global__ void __launch_bounds__(%d) %s(\n    %s) {\n",
		k.Threads, k.Name, strings.Join(k.params(ct), ",\n    "))
	fmt.Fprintf(&b, "    constexpr int COLS = %d, THREADS = %d;\n", k.Cols, k.Threads)
	fmt.Fprintf(&b, "    using T = %s;\n", ct)
	b.WriteString(`    const int r = blockIdx.x;
    if (r >= rows) return;
    const T* gu_row = GateUp + (size_t)r * (2 * COLS);
    const T* dy_row = dY + (size_t)r * COLS;
    T* dgu_row = dGateUp + (size_t)r * (2 * COLS);
    for (int i = threadIdx.x; i < COLS; i += THREADS) {
        const float dy = eg_to_f32(dy_row[i]);
        const float g = eg_to_f32(gu_row[i]);
        const float u = eg_to_f32(gu_row[COLS + i]);
        const float sig = 1.0f / (1.0f + expf(-g));
        const float silu = g * sig;
        const float dsilu = sig * (1.0f + g * (1.0f - sig));
        dgu_row[i] = eg_from_f32<T>(dy * u * dsilu);
        dgu_row[COLS + i] = eg_from_f32<T>(dy * silu);
    }
}
`)
	return b.String(), nil
}

// Launcher returns void Name_launch(<params...>, cudaStream_t stream).
func (k SwiGLUBackward) Launcher() (string, error) {
	k, ct, err := k.norm()
	if err != nil {
		return "", err
	}
	return launcher(k.Name, k.params(ct), func(b *strings.Builder) {
		fmt.Fprintf(b, "    const dim3 grid(rows > 0 ? rows : 0), block(%d);\n", k.Threads)
	}), nil
}

// Emit returns the complete CUDA C++ translation unit ([Prelude], kernel, and launcher).
func (k SwiGLUBackward) Emit() (string, error) {
	src, err := k.CUDA()
	if err != nil {
		return "", err
	}
	l, err := k.Launcher()
	if err != nil {
		return "", err
	}
	return Prelude + "\n" + src + "\n" + l, nil
}

// CrossEntropySoftcapBackward emits a fused CUDA C++ forward-loss + backward
// kernel for cross-entropy with optional logit softcapping:
//
//	z_cap = Softcap * tanh(z / Softcap)   (when Softcap > 0, else z)
//	loss[r] = -log(softmax(z_cap[r])[targets[r]])
//	dLogits[r, i] = (softmax(z_cap[r])[i] - 1[i == targets[r]]) * dLoss[r] * dz_cap
//
// Targets outside 0..Vocab-1 are treated as masked/ignored tokens with zero
// loss and zero logit gradients.
type CrossEntropySoftcapBackward struct {
	Name    string
	Vocab   int
	Softcap float32 // if > 0, applies cap * tanh(z / cap) before softmax
	DType   DType
	Threads int
}

func (k CrossEntropySoftcapBackward) norm() (CrossEntropySoftcapBackward, string, error) {
	if k.Name == "" {
		k.Name = "cross_entropy_softcap_bwd"
	}
	if !cIdent(k.Name) {
		return k, "", fmt.Errorf("cross entropy backward: bad name %q", k.Name)
	}
	if k.Vocab < 1 {
		return k, "", fmt.Errorf("cross entropy backward %s: vocab %d must be positive", k.Name, k.Vocab)
	}
	if k.DType == 0 {
		k.DType = BF16
	}
	ct, err := bwdCType(k.DType)
	if err != nil {
		return k, "", fmt.Errorf("cross entropy backward %s: %w", k.Name, err)
	}
	k.Threads, err = normThreads(k.Threads)
	if err != nil {
		return k, "", fmt.Errorf("cross entropy backward %s: %w", k.Name, err)
	}
	return k, ct, nil
}

func (k CrossEntropySoftcapBackward) params(ct string) []string {
	return []string{
		ct + "* __restrict__ dLogits",
		"float* __restrict__ losses",
		"const float* __restrict__ dLoss",
		"const " + ct + "* __restrict__ logits",
		"const int* __restrict__ targets",
		"int rows",
	}
}

// Grid returns the launch grid and block for rows rows.
func (k CrossEntropySoftcapBackward) Grid(rows int) (grid, block [3]int) {
	threads := k.Threads
	if threads == 0 {
		threads = 256
	}
	return [3]int{rows, 1, 1}, [3]int{threads, 1, 1}
}

// CUDA returns the kernel source (with [backwardPrelude], without [Prelude]).
func (k CrossEntropySoftcapBackward) CUDA() (string, error) {
	k, ct, err := k.norm()
	if err != nil {
		return "", err
	}
	var b strings.Builder
	b.WriteString(backwardPrelude)
	b.WriteString("\n")
	fmt.Fprintf(&b, "// %s: fused cross-entropy + softcap backward over vocab %d.\n", k.Name, k.Vocab)
	fmt.Fprintf(&b, "__global__ void __launch_bounds__(%d) %s(\n    %s) {\n",
		k.Threads, k.Name, strings.Join(k.params(ct), ",\n    "))
	fmt.Fprintf(&b, "    constexpr int VOCAB = %d, THREADS = %d;\n", k.Vocab, k.Threads)
	fmt.Fprintf(&b, "    constexpr float CAP = %s;\n", floatLiteral(k.Softcap))
	fmt.Fprintf(&b, "    using T = %s;\n", ct)
	b.WriteString(`    const int r = blockIdx.x;
    if (r >= rows) return;
    __shared__ float eg_red[32];
    const T* z_row = logits + (size_t)r * VOCAB;
    T* dz_row = dLogits + (size_t)r * VOCAB;
    const int target = targets[r];
    if (target < 0 || target >= VOCAB) {
        if (threadIdx.x == 0) losses[r] = 0.0f;
        for (int i = threadIdx.x; i < VOCAB; i += THREADS) {
            dz_row[i] = eg_from_f32<T>(0.0f);
        }
        return;
    }
    auto cap_val = [&](float z) -> float {
        return CAP > 0.0f ? CAP * tanhf(z / CAP) : z;
    };
    float row_max = -__uint_as_float(0x7f800000u);
    for (int i = threadIdx.x; i < VOCAB; i += THREADS) {
        const float z_cap = cap_val(eg_to_f32(z_row[i]));
        row_max = fmaxf(row_max, z_cap);
    }
    row_max = eg_block_combine<1>(eg_warp_max(row_max), eg_red);
    float sum_exp = 0.0f;
    for (int i = threadIdx.x; i < VOCAB; i += THREADS) {
        const float z_cap = cap_val(eg_to_f32(z_row[i]));
        sum_exp += expf(z_cap - row_max);
    }
    sum_exp = eg_block_combine<0>(eg_warp_sum(sum_exp), eg_red);
    if (threadIdx.x == 0) {
        const float z_t = cap_val(eg_to_f32(z_row[target]));
        losses[r] = -(z_t - row_max - logf(sum_exp));
    }
    const float inv_sum = 1.0f / sum_exp;
    const float dloss = dLoss[r];
    for (int i = threadIdx.x; i < VOCAB; i += THREADS) {
        const float z = eg_to_f32(z_row[i]);
        float z_cap = z;
        float dz_cap = 1.0f;
        if (CAP > 0.0f) {
            z_cap = CAP * tanhf(z / CAP);
            const float u = z_cap / CAP;
            dz_cap = 1.0f - u * u;
        }
        const float prob_i = expf(z_cap - row_max) * inv_sum;
        const float grad = ((prob_i - (i == target ? 1.0f : 0.0f)) * dloss) * dz_cap;
        dz_row[i] = eg_from_f32<T>(grad);
    }
}
`)
	return b.String(), nil
}

// Launcher returns void Name_launch(<params...>, cudaStream_t stream).
func (k CrossEntropySoftcapBackward) Launcher() (string, error) {
	k, ct, err := k.norm()
	if err != nil {
		return "", err
	}
	return launcher(k.Name, k.params(ct), func(b *strings.Builder) {
		fmt.Fprintf(b, "    const dim3 grid(rows > 0 ? rows : 0), block(%d);\n", k.Threads)
	}), nil
}

// Emit returns the complete CUDA C++ translation unit ([Prelude], kernel, and launcher).
func (k CrossEntropySoftcapBackward) Emit() (string, error) {
	src, err := k.CUDA()
	if err != nil {
		return "", err
	}
	l, err := k.Launcher()
	if err != nil {
		return "", err
	}
	return Prelude + "\n" + src + "\n" + l, nil
}
