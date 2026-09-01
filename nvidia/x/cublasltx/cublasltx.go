package cublasltx

import (
	"fmt"
	"sync"
	"unsafe"

	"github.com/tmc/lib/nvidia/cublaslt"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudamem"
)

// Engine manages cuBLASLt contexts, matrix layouts, and execution heuristics.
type Engine struct {
	mu     sync.Mutex
	dev    *cuda.Device
	handle cublaslt.Handle
}

// NewEngine initializes a cuBLASLt engine for dev.
func NewEngine(dev *cuda.Device) (*Engine, error) {
	if dev == nil {
		return nil, fmt.Errorf("cublasltx: nil device")
	}

	var h cublaslt.Handle
	err := dev.WithContext(func() error {
		var err error
		h, err = cublaslt.Create()
		return err
	})
	if err != nil {
		return nil, fmt.Errorf("cublasltx: create handle: %w", err)
	}

	return &Engine{
		dev:    dev,
		handle: h,
	}, nil
}

// Close destroys the underlying cuBLASLt handle.
func (e *Engine) Close() error {
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.handle != 0 {
		_ = e.dev.WithContext(func() error {
			return e.handle.Destroy()
		})
		e.handle = 0
	}
	return nil
}

// MatMul executes high-throughput 2D Tensor Core GEMM: D = alpha * (A * B) + beta * C
// A is [M, K], B is [K, N], D is [M, N]
func (e *Engine) MatMul(
	s *cuda.Stream,
	a cudamem.Slice[float32], m, k int,
	b cudamem.Slice[float32], n int,
	d cudamem.Slice[float32],
) error {
	e.mu.Lock()
	defer e.mu.Unlock()

	return e.dev.WithContext(func() error {
		// Create operation descriptor (FP32 compute)
		opDesc, err := cublaslt.NewMatmulDesc(cublaslt.Compute32F, cublaslt.R32F)
		if err != nil {
			return fmt.Errorf("create matmul desc: %w", err)
		}
		defer opDesc.Destroy()

		// Set transpose options: C = A * B
		// cuBLAS is column-major: A[m, k] -> rows=m, cols=k, ld=m
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

		// Execute cuBLASLt GEMM directly
		err = e.handle.Matmul(
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
			nil, // heuristic default algorithm
			0,
			0,
			streamHandle,
		)
		if err != nil {
			return fmt.Errorf("cublaslt matmul failed: %w", err)
		}
		return nil
	})
}
