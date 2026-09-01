package cublasltx

import (
	"fmt"
	"sync"
	"unsafe"

	"github.com/tmc/lib/nvidia/cublaslt"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudamem"
)

// Engine manages cuBLASLt contexts, matrix layouts, workspace buffers, and execution heuristics.
type Engine struct {
	mu            sync.Mutex
	dev           *cuda.Device
	handle        cublaslt.Handle
	workspace     cudamem.Slice[byte]
	workspaceSize uint
}

// NewEngine initializes a high-performance cuBLASLt engine with a 32MB workspace buffer.
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

	// 32MB workspace enables cuBLASLt split-K and high-occupancy tile algorithms
	wsSize := 32 * 1024 * 1024
	ws, err := cudamem.Alloc[byte](dev, wsSize)
	if err != nil {
		_ = dev.WithContext(func() error { return h.Destroy() })
		return nil, fmt.Errorf("cublasltx: alloc workspace: %w", err)
	}

	return &Engine{
		dev:           dev,
		handle:        h,
		workspace:     ws,
		workspaceSize: uint(wsSize),
	}, nil
}

// Close destroys the underlying cuBLASLt handle and frees workspace memory.
func (e *Engine) Close() error {
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.handle != 0 {
		_ = e.dev.WithContext(func() error {
			return e.handle.Destroy()
		})
		e.handle = 0
	}
	if e.workspace.Len() > 0 {
		_ = e.workspace.Free()
	}
	return nil
}

// MatMul executes high-throughput 2D Tensor Core GEMM using hardware heuristic algorithm selection:
// D = alpha * (A * B) + beta * C
func (e *Engine) MatMul(
	s *cuda.Stream,
	a cudamem.Slice[float32], m, k int,
	b cudamem.Slice[float32], n int,
	d cudamem.Slice[float32],
) error {
	e.mu.Lock()
	defer e.mu.Unlock()

	return e.dev.WithContext(func() error {
		// Use FastAccum for peak Tensor Core arithmetic utilization
		opDesc, err := cublaslt.NewMatmulDesc(cublaslt.Compute32FFast16F, cublaslt.R32F)
		if err != nil {
			// Fallback to standard 32F if FastAccum is unsupported
			opDesc, err = cublaslt.NewMatmulDesc(cublaslt.Compute32F, cublaslt.R32F)
			if err != nil {
				return fmt.Errorf("create matmul desc: %w", err)
			}
		}
		defer opDesc.Destroy()

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

		// Query Heuristic algorithm with 32MB workspace preference
		pref, err := cublaslt.NewMatmulPreference()
		if err == nil {
			defer pref.Destroy()
			wsBytes := uint64(e.workspaceSize)
			_ = pref.SetAttribute(cublaslt.MatmulPrefMaxWorkspaceBytes, uintptr(unsafe.Pointer(&wsBytes)), uint(unsafe.Sizeof(wsBytes)))
		}

		var heuristicResult cublaslt.MatmulHeuristicResult
		var algoPtr *cublaslt.MatmulAlgo

		returnedCount, err := e.handle.MatmulAlgoGetHeuristic(opDesc, layoutA, layoutB, layoutD, layoutD, pref, 1, &heuristicResult)
		if err == nil && returnedCount > 0 && heuristicResult.State == cublaslt.Success {
			algoPtr = &heuristicResult.Algo
		}

		alpha := float32(1.0)
		beta := float32(0.0)

		var streamHandle cublaslt.Stream
		if s != nil {
			streamHandle = cublaslt.Stream(s.Raw())
		}

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
			algoPtr,
			uintptr(e.workspace.Ptr()),
			e.workspaceSize,
			streamHandle,
		)
		if err != nil {
			return fmt.Errorf("cublaslt matmul: %w", err)
		}
		return nil
	})
}
