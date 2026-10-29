package hopper

import "github.com/tmc/lib/nvidia/x/egraph/train"

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

func (k FusedAdamW) toTrain() train.FusedAdamW {
	return train.FusedAdamW{
		Name:    k.Name,
		DType:   train.DType(k.DType),
		Threads: k.Threads,
	}
}

// Grid returns the launch grid and block for n elements.
func (k FusedAdamW) Grid(n int) (grid, block [3]int) { return k.toTrain().Grid(n) }

// CUDA returns the kernel source (without [Prelude]).
func (k FusedAdamW) CUDA() (string, error) { return k.toTrain().CUDA() }

// Launcher returns void Name_launch(<params...>, cudaStream_t stream).
func (k FusedAdamW) Launcher() (string, error) { return k.toTrain().Launcher() }

// Emit returns the complete CUDA C++ translation unit ([Prelude], kernel, and launcher).
func (k FusedAdamW) Emit() (string, error) { return k.toTrain().Emit() }
