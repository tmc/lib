package train

import (
	"fmt"
	"strings"
)

// FusedAdamW emits a fused CUDA C++ AdamW optimizer kernel with FP32 master
// weights and moments and a low-precision (BF16 or FP16, or FP32) parameter
// and gradient buffer:
//
//	g = grad[i] * grad_scale
//	w = master_w[i] * (1 - lr * weight_decay)
//	m = beta1 * exp_avg[i] + (1 - beta1) * g
//	v = beta2 * exp_avg_sq[i] + (1 - beta2) * g * g
//	w -= lr * (m / bias_corr1) / (sqrt(v / bias_corr2) + eps)
//	master_w[i] = w; param[i] = DType(w)
type FusedAdamW struct {
	Name    string
	DType   DType // parameter dtype (BF16 or FP16, or FP32) alongside FP32 master weights & moments
	Threads int
}

func (k FusedAdamW) norm() (FusedAdamW, string, error) {
	if k.Name == "" {
		k.Name = "fused_adamw"
	}
	if !cIdent(k.Name) {
		return k, "", fmt.Errorf("fused adamw: bad name %q", k.Name)
	}
	if k.DType == 0 {
		k.DType = BF16
	}
	ct, err := bwdCType(k.DType)
	if err != nil {
		return k, "", fmt.Errorf("fused adamw %s: %w", k.Name, err)
	}
	k.Threads, err = normThreads(k.Threads)
	if err != nil {
		return k, "", fmt.Errorf("fused adamw %s: %w", k.Name, err)
	}
	return k, ct, nil
}

func (k FusedAdamW) params(ct string) []string {
	return []string{
		ct + "* __restrict__ param",
		"float* __restrict__ master_w",
		"float* __restrict__ exp_avg",
		"float* __restrict__ exp_avg_sq",
		"const " + ct + "* __restrict__ grad",
		"int n",
		"float lr",
		"float beta1",
		"float beta2",
		"float eps",
		"float weight_decay",
		"float bias_corr1",
		"float bias_corr2",
		"float grad_scale",
	}
}

// Grid returns the launch grid and block for n elements.
func (k FusedAdamW) Grid(n int) (grid, block [3]int) {
	threads := k.Threads
	if threads == 0 {
		threads = 256
	}
	blocks := 0
	if n > 0 {
		blocks = (n + threads - 1) / threads
	}
	return [3]int{blocks, 1, 1}, [3]int{threads, 1, 1}
}

// CUDA returns the kernel source (with [backwardPrelude], without [Prelude]).
func (k FusedAdamW) CUDA() (string, error) {
	k, ct, err := k.norm()
	if err != nil {
		return "", err
	}
	var b strings.Builder
	b.WriteString(backwardPrelude)
	b.WriteString("\n")
	fmt.Fprintf(&b, "// %s: fused AdamW update with FP32 master weights and %s parameters.\n", k.Name, ct)
	fmt.Fprintf(&b, "__global__ void __launch_bounds__(%d) %s(\n    %s) {\n",
		k.Threads, k.Name, strings.Join(k.params(ct), ",\n    "))
	fmt.Fprintf(&b, "    constexpr int THREADS = %d;\n", k.Threads)
	fmt.Fprintf(&b, "    using T = %s;\n", ct)
	b.WriteString(`    const int stride = (int)gridDim.x * THREADS;
    for (int i = (int)blockIdx.x * THREADS + (int)threadIdx.x; i < n; i += stride) {
        const float g = eg_to_f32(grad[i]) * grad_scale;
        float w = master_w[i];
        float m = exp_avg[i];
        float v = exp_avg_sq[i];
        w *= (1.0f - lr * weight_decay);
        m = beta1 * m + (1.0f - beta1) * g;
        v = beta2 * v + (1.0f - beta2) * g * g;
        w -= lr * (m / bias_corr1) / (sqrtf(v / bias_corr2) + eps);
        master_w[i] = w;
        exp_avg[i] = m;
        exp_avg_sq[i] = v;
        param[i] = eg_from_f32<T>(w);
    }
}
`)
	return b.String(), nil
}

// Launcher returns void Name_launch(<params...>, cudaStream_t stream).
func (k FusedAdamW) Launcher() (string, error) {
	k, ct, err := k.norm()
	if err != nil {
		return "", err
	}
	return launcher(k.Name, k.params(ct), func(b *strings.Builder) {
		fmt.Fprintf(b, "    const dim3 grid(n > 0 ? (n + %d - 1) / %d : 0), block(%d);\n", k.Threads, k.Threads, k.Threads)
	}), nil
}

// Emit returns the complete CUDA C++ translation unit ([Prelude], kernel, and launcher).
func (k FusedAdamW) Emit() (string, error) {
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
