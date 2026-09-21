package cublasltx

import (
	"errors"
	"fmt"

	"github.com/tmc/lib/nvidia/cublaslt"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudamem"
)

// MatMulWithGELUBias computes D = GELU(A*B+bias) for column-major
// float32 matrices. Bias contains one value for each row of D.
func (e *Engine) MatMulWithGELUBias(
	s *cuda.Stream,
	a cudamem.Slice[float32], m, k int,
	b cudamem.Slice[float32], n int,
	bias cudamem.Slice[float32],
	d cudamem.Slice[float32],
) error {
	if e == nil {
		return errors.New("cublasltx: nil engine")
	}
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.handle == 0 {
		return errors.New("cublasltx: closed engine")
	}
	if err := validateMatMulLengths(a.Len(), m, k, b.Len(), n, d.Len()); err != nil {
		return err
	}
	if bias.Len() < m {
		return fmt.Errorf("cublasltx: bias length %d is smaller than %d", bias.Len(), m)
	}
	if s != nil && s.Device() != e.dev {
		return errors.New("cublasltx: stream belongs to another device")
	}
	return e.execute(s, MatMulConfig{
		ComputeType: cublaslt.Compute32F,
		ScaleType:   cublaslt.R32F,
		AType:       cublaslt.R32F,
		BType:       cublaslt.R32F,
		DType:       cublaslt.R32F,
		Alpha:       1.0,
		Epilogue:    cublaslt.EpilogueGeluBias,
		Bias:        bias.Ptr(),
	}, a.Ptr(), m, k, m, b.Ptr(), k, n, k, d.Ptr(), m, n, m)
}
