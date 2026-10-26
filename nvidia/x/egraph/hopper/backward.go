package hopper

import "github.com/tmc/lib/nvidia/x/egraph/train"

const (
	// F16 is IEEE binary16 (half precision), rounded to nearest even.
	F16 DType = I64 + 1

	// FP16 and FP32 are aliases for [F16] and [F32].
	FP16 = F16
	FP32 = F32
)

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

func (k RMSNormBackward) toTrain() train.RMSNormBackward {
	return train.RMSNormBackward{
		Name:    k.Name,
		Cols:    k.Cols,
		Eps:     k.Eps,
		DType:   train.DType(k.DType),
		Threads: k.Threads,
	}
}

// Grid returns the launch grid and block for rows rows.
func (k RMSNormBackward) Grid(rows int) (grid, block [3]int) { return k.toTrain().Grid(rows) }

// CUDA returns the kernel source (without [Prelude]).
func (k RMSNormBackward) CUDA() (string, error) { return k.toTrain().CUDA() }

// Launcher returns void Name_launch(<params...>, cudaStream_t stream).
func (k RMSNormBackward) Launcher() (string, error) { return k.toTrain().Launcher() }

// Emit returns the complete CUDA C++ translation unit ([Prelude], kernel, and launcher).
func (k RMSNormBackward) Emit() (string, error) { return k.toTrain().Emit() }

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

func (k SwiGLUBackward) toTrain() train.SwiGLUBackward {
	return train.SwiGLUBackward{
		Name:    k.Name,
		Cols:    k.Cols,
		DType:   train.DType(k.DType),
		Threads: k.Threads,
	}
}

// Grid returns the launch grid and block for rows rows.
func (k SwiGLUBackward) Grid(rows int) (grid, block [3]int) { return k.toTrain().Grid(rows) }

// CUDA returns the kernel source (without [Prelude]).
func (k SwiGLUBackward) CUDA() (string, error) { return k.toTrain().CUDA() }

// Launcher returns void Name_launch(<params...>, cudaStream_t stream).
func (k SwiGLUBackward) Launcher() (string, error) { return k.toTrain().Launcher() }

// Emit returns the complete CUDA C++ translation unit ([Prelude], kernel, and launcher).
func (k SwiGLUBackward) Emit() (string, error) { return k.toTrain().Emit() }

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

func (k CrossEntropySoftcapBackward) toTrain() train.CrossEntropySoftcapBackward {
	return train.CrossEntropySoftcapBackward{
		Name:    k.Name,
		Vocab:   k.Vocab,
		Softcap: k.Softcap,
		DType:   train.DType(k.DType),
		Threads: k.Threads,
	}
}

// Grid returns the launch grid and block for rows rows.
func (k CrossEntropySoftcapBackward) Grid(rows int) (grid, block [3]int) {
	return k.toTrain().Grid(rows)
}

// CUDA returns the kernel source (without [Prelude]).
func (k CrossEntropySoftcapBackward) CUDA() (string, error) { return k.toTrain().CUDA() }

// Launcher returns void Name_launch(<params...>, cudaStream_t stream).
func (k CrossEntropySoftcapBackward) Launcher() (string, error) { return k.toTrain().Launcher() }

// Emit returns the complete CUDA C++ translation unit ([Prelude], kernel, and launcher).
func (k CrossEntropySoftcapBackward) Emit() (string, error) { return k.toTrain().Emit() }
