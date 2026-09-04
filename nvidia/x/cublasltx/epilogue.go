package cublasltx

import (
	"errors"
	"fmt"
	"unsafe"

	"github.com/tmc/lib/nvidia/cublaslt"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudamem"
)

// MatMulWithGELUBias executes fused Matrix Multiplication + Bias addition + GELU activation in a single pass:
// D = GELU( (A * B) + Bias )
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
	if bias.Len() < n {
		return fmt.Errorf("cublasltx: bias length %d is smaller than %d", bias.Len(), n)
	}
	if s != nil && s.Device() != e.dev {
		return errors.New("cublasltx: stream belongs to another device")
	}

	return e.dev.WithContext(func() error {
		opDesc, err := cublaslt.NewMatmulDesc(cublaslt.Compute32F, cublaslt.R32F)
		if err != nil {
			return fmt.Errorf("create matmul desc: %w", err)
		}
		defer opDesc.Destroy()

		// Configure Epilogue: GELU + Bias Fusion
		epilogue := cublaslt.EpilogueGeluBias
		if err := opDesc.SetAttribute(cublaslt.MatmulDescEpilogue, uintptr(unsafe.Pointer(&epilogue)), uint(unsafe.Sizeof(epilogue))); err != nil {
			return fmt.Errorf("set epilogue gelu+bias: %w", err)
		}

		biasPtr := bias.Ptr()
		if err := opDesc.SetAttribute(cublaslt.MatmulDescBiasPointer, uintptr(unsafe.Pointer(&biasPtr)), uint(unsafe.Sizeof(biasPtr))); err != nil {
			return fmt.Errorf("set bias pointer: %w", err)
		}

		layoutA, err := cublaslt.NewMatrixLayout(cublaslt.R32F, uint64(m), uint64(k), int64(m))
		if err != nil {
			return fmt.Errorf("create layout A: %w", err)
		}
		defer layoutA.Destroy()

		layoutB, err := cublaslt.NewMatrixLayout(cublaslt.R32F, uint64(k), uint64(n), int64(k))
		if err != nil {
			return fmt.Errorf("create layout B: %w", err)
		}
		defer layoutB.Destroy()

		layoutD, err := cublaslt.NewMatrixLayout(cublaslt.R32F, uint64(m), uint64(n), int64(m))
		if err != nil {
			return fmt.Errorf("create layout D: %w", err)
		}
		defer layoutD.Destroy()

		alpha := float32(1.0)
		beta := float32(0.0)

		var streamHandle cublaslt.Stream
		if s != nil {
			streamHandle = cublaslt.Stream(s.Raw())
		}

		return e.handle.Matmul(
			opDesc,
			uintptr(unsafe.Pointer(&alpha)),
			uintptr(a.Ptr()),
			layoutA,
			uintptr(b.Ptr()),
			layoutB,
			uintptr(unsafe.Pointer(&beta)),
			uintptr(d.Ptr()),
			layoutD,
			uintptr(d.Ptr()),
			layoutD,
			nil,
			0,
			0,
			streamHandle,
		)
	})
}
