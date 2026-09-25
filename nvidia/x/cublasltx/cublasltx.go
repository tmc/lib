package cublasltx

import (
	"errors"
	"fmt"
	"math"
	"runtime"
	"sync"
	"unsafe"

	"github.com/tmc/lib/nvidia/cublaslt"
	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudamem"
)

// Option configures an Engine.
type Option func(*engineConfig) error

type engineConfig struct {
	workspaceBytes *int
	customWS       cudamem.Slice[byte]
	hasCustomWS    bool
}

// WithWorkspaceBytes configures the size of the engine-owned workspace buffer in bytes.
// Setting bytes to 0 disables engine workspace allocation.
// Negative sizes return an error.
func WithWorkspaceBytes(bytes int) Option {
	return func(c *engineConfig) error {
		if bytes < 0 {
			return fmt.Errorf("cublasltx: negative workspace size: %d", bytes)
		}
		c.workspaceBytes = &bytes
		return nil
	}
}

// WithWorkspace configures the engine to use a caller-provided workspace buffer.
// The engine does not free the provided workspace upon Close.
// The workspace must belong to the same device as the engine and its address
// must be 256-byte aligned.
func WithWorkspace(ws cudamem.Slice[byte]) Option {
	return func(c *engineConfig) error {
		if ws.Len() > 0 && uintptr(ws.Ptr())%256 != 0 {
			return fmt.Errorf("cublasltx: workspace pointer %v not 256-byte aligned", ws.Ptr())
		}
		c.customWS = ws
		c.hasCustomWS = true
		return nil
	}
}

// Engine manages cuBLASLt contexts, matrix layouts, workspace buffers, and execution heuristics.
type Engine struct {
	mu             sync.Mutex
	dev            *cuda.Device
	handle         cublaslt.Handle
	workspace      cudamem.Slice[byte]
	workspaceSize  uint
	ownedWorkspace bool
	sharedHandle   bool
}

// NewEngine initializes a cuBLASLt engine.
// By default, it allocates a 32MB workspace buffer owned by the engine.
// Optional functional options can customize workspace size or provide a caller-owned buffer.
// A non-nil engine must be closed even when an error is returned. It may retain
// a created handle or workspace after a later allocation or context error.
func NewEngine(dev *cuda.Device, opts ...Option) (*Engine, error) {
	cfg := engineConfig{}
	for _, opt := range opts {
		if opt != nil {
			if err := opt(&cfg); err != nil {
				return nil, err
			}
		}
	}

	if cfg.hasCustomWS && cfg.workspaceBytes != nil {
		return nil, fmt.Errorf("cublasltx: cannot specify both WithWorkspace and WithWorkspaceBytes")
	}

	if dev == nil {
		return nil, fmt.Errorf("cublasltx: nil device")
	}

	return newEngine(dev, cfg, dev.WithContext, cublaslt.Create, func(size int) (cudamem.Slice[byte], error) { return cudamem.Alloc[byte](dev, size) })
}

func newEngine(dev *cuda.Device, cfg engineConfig, run func(func() error) error, create func() (cublaslt.Handle, error), allocate func(int) (cudamem.Slice[byte], error)) (*Engine, error) {
	var h cublaslt.Handle
	err := run(func() error {
		var err error
		h, err = create()
		return err
	})
	var e *Engine
	if h != 0 {
		e = &Engine{dev: dev, handle: h}
	}
	if err != nil {
		return e, fmt.Errorf("cublasltx: create handle: %w", err)
	}
	if e == nil {
		return nil, errors.New("cublasltx: create returned no handle")
	}

	if cfg.hasCustomWS {
		e.workspace = cfg.customWS
		e.workspaceSize = uint(e.workspace.Len())
	} else {
		size := 32 * 1024 * 1024
		if cfg.workspaceBytes != nil {
			size = *cfg.workspaceBytes
		}
		if size > 0 {
			var allocErr error
			e.workspace, allocErr = allocate(size)
			e.ownedWorkspace = e.workspace.Ptr() != 0
			e.workspaceSize = uint(e.workspace.Len())
			if allocErr != nil {
				return e, fmt.Errorf("cublasltx: alloc workspace: %w", allocErr)
			}
		}
	}

	return e, nil
}

// Workspace returns the default workspace buffer configured for the engine.
func (e *Engine) Workspace() cudamem.Slice[byte] {
	if e == nil {
		return cudamem.Slice[byte]{}
	}
	e.mu.Lock()
	defer e.mu.Unlock()
	return e.workspace
}

// WorkspaceSize returns the size in bytes of the engine's default workspace.
func (e *Engine) WorkspaceSize() uint {
	if e == nil {
		return 0
	}
	e.mu.Lock()
	defer e.mu.Unlock()
	return e.workspaceSize
}

// WithWorkspace returns a child Engine sharing the cuBLASLt handle and device
// of e, but using ws as its workspace. This enables per-stream or per-operation
// execution without workspace collision hazards across concurrent streams.
//
// The returned child Engine does not own ws and does not own the cuBLASLt handle.
// Closing the child Engine does not destroy the parent handle or free ws.
// The parent Engine must remain open while the child Engine is in use.
func (e *Engine) WithWorkspace(ws cudamem.Slice[byte]) (*Engine, error) {
	if e == nil {
		return nil, errors.New("cublasltx: nil engine")
	}
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.handle == 0 {
		return nil, errors.New("cublasltx: closed engine")
	}
	if ws.Len() > 0 && uintptr(ws.Ptr())%256 != 0 {
		return nil, fmt.Errorf("cublasltx: workspace pointer %v not 256-byte aligned", ws.Ptr())
	}
	return &Engine{
		dev:            e.dev,
		handle:         e.handle,
		workspace:      ws,
		workspaceSize:  uint(ws.Len()),
		ownedWorkspace: false,
		sharedHandle:   true,
	}, nil
}

// Close destroys the underlying cuBLASLt handle (unless shared) and frees
// any workspace memory owned by the engine.
func (e *Engine) Close() error {
	if e == nil {
		return nil
	}
	return e.closeWith(e.dev.WithContext, cublaslt.Handle.Destroy, func() error { return e.workspace.Free() })
}

func (e *Engine) closeWith(run func(func() error) error, destroy func(cublaslt.Handle) error, free func() error) error {
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.handle != 0 && !e.sharedHandle {
		err := run(func() error {
			if err := destroy(e.handle); err != nil {
				return err
			}
			e.handle = 0
			return nil
		})
		if err != nil {
			return fmt.Errorf("cublasltx: destroy handle: %w", err)
		}
	} else if e.sharedHandle {
		e.handle = 0
	}
	if e.ownedWorkspace && e.workspace.Len() > 0 {
		if err := free(); err != nil {
			return fmt.Errorf("cublasltx: free workspace: %w", err)
		}
	}
	e.workspace = cudamem.Slice[byte]{}
	e.workspaceSize = 0
	return nil
}

// MatMulConfig specifies options for raw matrix multiplication.
type MatMulConfig struct {
	ComputeType   cublaslt.ComputeType // e.g. Compute32F, Compute32FFast16F, Compute16F
	ScaleType     cublaslt.DataType    // e.g. R32F, R16F
	AType         cublaslt.DataType    // e.g. R32F, R16F, R16BF
	BType         cublaslt.DataType    // e.g. R32F, R16F, R16BF
	DType         cublaslt.DataType    // e.g. R32F, R16F, R16BF
	TransA        bool                 // transpose op(A) = A^T
	TransB        bool                 // transpose op(B) = B^T
	Alpha         float32              // scalar multiplier for A*B (defaults to 1.0 if Alpha and Beta are 0)
	Beta          float32              // scalar multiplier for D (defaults to 0.0)
	Workspace     cudamem.Slice[byte]  // Optional per-operation workspace override.
	WorkspaceSize uint                 // Optional workspace size limit in bytes (defaults to Workspace.Len() if Workspace is set).

	Epilogue cublaslt.Epilogue // postprocessing applied to D; zero means none
	Bias     driver.DevicePtr  // bias vector, one value per row of D, required by the bias and bgrad epilogues
	BiasType cublaslt.DataType // element type of Bias (defaults to DType)
	Aux      driver.DevicePtr  // auxiliary buffer the EpilogueGeluAux and EpilogueReluAux families write or read
	AuxLd    int               // leading dimension of Aux, in elements (defaults to the leading dimension of D)
}

// epilogueUsesBias reports whether ep reads or writes the bias vector, which
// cuBLASLt takes from MatmulDescBiasPointer.
func epilogueUsesBias(ep cublaslt.Epilogue) bool {
	switch ep {
	case cublaslt.EpilogueBias, cublaslt.EpilogueReluBias, cublaslt.EpilogueReluAuxBias,
		cublaslt.EpilogueGeluBias, cublaslt.EpilogueGeluAuxBias,
		cublaslt.EpilogueDreluBgrad, cublaslt.EpilogueDgeluBgrad,
		cublaslt.EpilogueBgrada, cublaslt.EpilogueBgradb:
		return true
	}
	return false
}

// epilogueUsesAux reports whether ep reads or writes the auxiliary buffer,
// which cuBLASLt takes from MatmulDescEpilogueAuxPointer.
func epilogueUsesAux(ep cublaslt.Epilogue) bool {
	switch ep {
	case cublaslt.EpilogueReluAux, cublaslt.EpilogueReluAuxBias,
		cublaslt.EpilogueGeluAux, cublaslt.EpilogueGeluAuxBias,
		cublaslt.EpilogueDrelu, cublaslt.EpilogueDreluBgrad,
		cublaslt.EpilogueDgelu, cublaslt.EpilogueDgeluBgrad:
		return true
	}
	return false
}

func (e *Engine) execute(
	s *cuda.Stream,
	cfg MatMulConfig,
	aPtr driver.DevicePtr, aRows, aCols, lda int,
	bPtr driver.DevicePtr, bRows, bCols, ldb int,
	dPtr driver.DevicePtr, dRows, dCols, ldd int,
) error {
	wsPtr := e.workspace.Ptr()
	wsSize := e.workspaceSize
	if cfg.Workspace.Len() > 0 || cfg.WorkspaceSize > 0 {
		wsPtr = cfg.Workspace.Ptr()
		if cfg.WorkspaceSize > 0 {
			wsSize = cfg.WorkspaceSize
		} else {
			wsSize = uint(cfg.Workspace.Len())
		}
	}
	if wsPtr != 0 && wsPtr%256 != 0 {
		return fmt.Errorf("cublasltx: workspace pointer 0x%x not 256-byte aligned", wsPtr)
	}

	if cfg.ComputeType == 0 {
		cfg.ComputeType = cublaslt.Compute32F
	}
	if cfg.ScaleType == 0 {
		if cfg.ComputeType == cublaslt.Compute16F {
			cfg.ScaleType = cublaslt.R16F
		} else {
			cfg.ScaleType = cublaslt.R32F
		}
	}
	if cfg.AType == 0 {
		cfg.AType = cublaslt.R32F
	}
	if cfg.BType == 0 {
		cfg.BType = cublaslt.R32F
	}
	if cfg.DType == 0 {
		cfg.DType = cublaslt.R32F
	}
	if cfg.Alpha == 0 && cfg.Beta == 0 {
		cfg.Alpha = 1.0
	}
	if cfg.BiasType == 0 {
		cfg.BiasType = cfg.DType
	}
	if epilogueUsesBias(cfg.Epilogue) && cfg.Bias == 0 {
		return fmt.Errorf("cublasltx: epilogue %v needs a bias pointer", cfg.Epilogue)
	}
	if epilogueUsesAux(cfg.Epilogue) && cfg.Aux == 0 {
		return fmt.Errorf("cublasltx: epilogue %v needs an auxiliary pointer", cfg.Epilogue)
	}
	if cfg.AuxLd == 0 {
		cfg.AuxLd = ldd
	}

	return e.dev.WithContext(func() error {
		var pin runtime.Pinner
		defer pin.Unpin()

		opDesc, err := cublaslt.NewMatmulDesc(cfg.ComputeType, cfg.ScaleType)
		if err != nil {
			return fmt.Errorf("create matmul desc: %w", err)
		}
		defer opDesc.Destroy()

		if cfg.TransA {
			op := cublaslt.CublasOpT
			pin.Pin(&op)
			if err := opDesc.SetAttribute(cublaslt.MatmulDescTransa, uintptr(unsafe.Pointer(&op)), uint(unsafe.Sizeof(op))); err != nil {
				return fmt.Errorf("set transa: %w", err)
			}
		}
		if cfg.TransB {
			op := cublaslt.CublasOpT
			pin.Pin(&op)
			if err := opDesc.SetAttribute(cublaslt.MatmulDescTransb, uintptr(unsafe.Pointer(&op)), uint(unsafe.Sizeof(op))); err != nil {
				return fmt.Errorf("set transb: %w", err)
			}
		}
		if cfg.Epilogue != 0 {
			ep := cfg.Epilogue
			pin.Pin(&ep)
			if err := opDesc.SetAttribute(cublaslt.MatmulDescEpilogue, uintptr(unsafe.Pointer(&ep)), uint(unsafe.Sizeof(ep))); err != nil {
				return fmt.Errorf("set epilogue: %w", err)
			}
		}
		if cfg.Bias != 0 {
			biasPtr := cfg.Bias
			pin.Pin(&biasPtr)
			if err := opDesc.SetAttribute(cublaslt.MatmulDescBiasPointer, uintptr(unsafe.Pointer(&biasPtr)), uint(unsafe.Sizeof(biasPtr))); err != nil {
				return fmt.Errorf("set bias pointer: %w", err)
			}
			biasType := cfg.BiasType
			pin.Pin(&biasType)
			if err := opDesc.SetAttribute(cublaslt.MatmulDescBiasDataType, uintptr(unsafe.Pointer(&biasType)), uint(unsafe.Sizeof(biasType))); err != nil {
				return fmt.Errorf("set bias data type: %w", err)
			}
		}
		if cfg.Aux != 0 {
			auxPtr := cfg.Aux
			pin.Pin(&auxPtr)
			if err := opDesc.SetAttribute(cublaslt.MatmulDescEpilogueAuxPointer, uintptr(unsafe.Pointer(&auxPtr)), uint(unsafe.Sizeof(auxPtr))); err != nil {
				return fmt.Errorf("set epilogue aux pointer: %w", err)
			}
			auxLd := int64(cfg.AuxLd)
			pin.Pin(&auxLd)
			if err := opDesc.SetAttribute(cublaslt.MatmulDescEpilogueAuxLd, uintptr(unsafe.Pointer(&auxLd)), uint(unsafe.Sizeof(auxLd))); err != nil {
				return fmt.Errorf("set epilogue aux ld: %w", err)
			}
		}

		layoutA, err := cublaslt.NewMatrixLayout(cfg.AType, uint64(aRows), uint64(aCols), int64(lda))
		if err != nil {
			return fmt.Errorf("create layout A: %w", err)
		}
		defer layoutA.Destroy()

		layoutB, err := cublaslt.NewMatrixLayout(cfg.BType, uint64(bRows), uint64(bCols), int64(ldb))
		if err != nil {
			return fmt.Errorf("create layout B: %w", err)
		}
		defer layoutB.Destroy()

		layoutD, err := cublaslt.NewMatrixLayout(cfg.DType, uint64(dRows), uint64(dCols), int64(ldd))
		if err != nil {
			return fmt.Errorf("create layout D: %w", err)
		}
		defer layoutD.Destroy()

		pref, err := cublaslt.NewMatmulPreference()
		if err == nil {
			defer pref.Destroy()
			wsBytes := uint64(wsSize)
			pin.Pin(&wsBytes)
			_ = pref.SetAttribute(cublaslt.MatmulPrefMaxWorkspaceBytes, uintptr(unsafe.Pointer(&wsBytes)), uint(unsafe.Sizeof(wsBytes)))
		}

		var heuristicResult cublaslt.MatmulHeuristicResult
		var algoPtr *cublaslt.MatmulAlgo

		returnedCount, err := e.handle.MatmulAlgoGetHeuristic(opDesc, layoutA, layoutB, layoutD, layoutD, pref, 1, &heuristicResult)
		if err == nil && returnedCount > 0 && heuristicResult.State == cublaslt.Success {
			algoPtr = &heuristicResult.Algo
		}

		var alphaPtr, betaPtr uintptr
		if cfg.ScaleType == cublaslt.R16F {
			alphaHalf := float32ToF16Bits(cfg.Alpha)
			betaHalf := float32ToF16Bits(cfg.Beta)
			pin.Pin(&alphaHalf)
			pin.Pin(&betaHalf)
			alphaPtr = uintptr(unsafe.Pointer(&alphaHalf))
			betaPtr = uintptr(unsafe.Pointer(&betaHalf))
		} else {
			alpha := cfg.Alpha
			beta := cfg.Beta
			pin.Pin(&alpha)
			pin.Pin(&beta)
			alphaPtr = uintptr(unsafe.Pointer(&alpha))
			betaPtr = uintptr(unsafe.Pointer(&beta))
		}

		err = withStream(s, func(streamHandle driver.Stream) error {
			return e.handle.Matmul(
				opDesc,
				alphaPtr,
				aPtr,
				layoutA,
				bPtr,
				layoutB,
				betaPtr,
				dPtr,
				layoutD,
				dPtr,
				layoutD,
				algoPtr,
				wsPtr,
				wsSize,
				streamHandle,
			)
		})
		if err != nil {
			return fmt.Errorf("cublaslt matmul: %w", err)
		}
		return nil
	})
}

// MatMulRaw computes D = alpha*op(A)*op(B) + beta*D using raw device pointers and strides.
func (e *Engine) MatMulRaw(
	s *cuda.Stream,
	cfg MatMulConfig,
	aPtr driver.DevicePtr, aRows, aCols, lda int,
	bPtr driver.DevicePtr, bRows, bCols, ldb int,
	dPtr driver.DevicePtr, dRows, dCols, ldd int,
) error {
	if e == nil {
		return errors.New("cublasltx: nil engine")
	}
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.handle == 0 {
		return errors.New("cublasltx: closed engine")
	}
	if aPtr == 0 || bPtr == 0 || dPtr == 0 {
		return errors.New("cublasltx: nil device pointer")
	}
	if aRows <= 0 || aCols <= 0 || bRows <= 0 || bCols <= 0 || dRows <= 0 || dCols <= 0 {
		return errors.New("cublasltx: dimensions must be positive")
	}
	if lda < aRows || ldb < bRows || ldd < dRows {
		return errors.New("cublasltx: leading dimension smaller than row count")
	}
	if s != nil && s.Device() != e.dev {
		return errors.New("cublasltx: stream belongs to another device")
	}
	return e.execute(s, cfg, aPtr, aRows, aCols, lda, bPtr, bRows, bCols, ldb, dPtr, dRows, dCols, ldd)
}

// MatMul computes D = A*B for column-major float32 matrices:
// A has shape m by k, B has shape k by n, and D has shape m by n.
func (e *Engine) MatMul(
	s *cuda.Stream,
	a cudamem.Slice[float32], m, k int,
	b cudamem.Slice[float32], n int,
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
	}, a.Ptr(), m, k, m, b.Ptr(), k, n, k, d.Ptr(), m, n, m)
}

// MatMulF16 computes D = A*B for column-major float16 matrices (encoded as uint16 bits)
// using float32 compute precision:
// A has shape m by k, B has shape k by n, and D has shape m by n.
func (e *Engine) MatMulF16(
	s *cuda.Stream,
	a cudamem.Slice[uint16], m, k int,
	b cudamem.Slice[uint16], n int,
	d cudamem.Slice[uint16],
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
	if s != nil && s.Device() != e.dev {
		return errors.New("cublasltx: stream belongs to another device")
	}
	return e.execute(s, MatMulConfig{
		ComputeType: cublaslt.Compute32F,
		ScaleType:   cublaslt.R32F,
		AType:       cublaslt.R16F,
		BType:       cublaslt.R16F,
		DType:       cublaslt.R16F,
		Alpha:       1.0,
	}, a.Ptr(), m, k, m, b.Ptr(), k, n, k, d.Ptr(), m, n, m)
}

// MatMulBF16 computes D = A*B for column-major bfloat16 matrices (encoded as uint16 bits)
// using float32 compute precision:
// A has shape m by k, B has shape k by n, and D has shape m by n.
func (e *Engine) MatMulBF16(
	s *cuda.Stream,
	a cudamem.Slice[uint16], m, k int,
	b cudamem.Slice[uint16], n int,
	d cudamem.Slice[uint16],
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
	if s != nil && s.Device() != e.dev {
		return errors.New("cublasltx: stream belongs to another device")
	}
	return e.execute(s, MatMulConfig{
		ComputeType: cublaslt.Compute32F,
		ScaleType:   cublaslt.R32F,
		AType:       cublaslt.R16BF,
		BType:       cublaslt.R16BF,
		DType:       cublaslt.R16BF,
		Alpha:       1.0,
	}, a.Ptr(), m, k, m, b.Ptr(), k, n, k, d.Ptr(), m, n, m)
}

// MatMulRowMajor computes D = A*B for row-major float32 matrices:
// A has shape m by k, B has shape k by n, and D has shape m by n.
func (e *Engine) MatMulRowMajor(
	s *cuda.Stream,
	a cudamem.Slice[float32], m, k int,
	b cudamem.Slice[float32], n int,
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
	if s != nil && s.Device() != e.dev {
		return errors.New("cublasltx: stream belongs to another device")
	}
	// D = A*B in row-major is equivalent to D^T = B^T * A^T in column-major.
	// B stored memory: column-major n by k, ld n.
	// A stored memory: column-major k by m, ld k.
	// D stored memory: column-major n by m, ld n.
	return e.execute(s, MatMulConfig{
		ComputeType: cublaslt.Compute32F,
		ScaleType:   cublaslt.R32F,
		AType:       cublaslt.R32F,
		BType:       cublaslt.R32F,
		DType:       cublaslt.R32F,
		Alpha:       1.0,
	}, b.Ptr(), n, k, n, a.Ptr(), k, m, k, d.Ptr(), n, m, n)
}

// MatMulRowMajorF16 computes D = A*B for row-major float16 matrices (encoded as uint16 bits)
// using float32 compute precision:
// A has shape m by k, B has shape k by n, and D has shape m by n.
func (e *Engine) MatMulRowMajorF16(
	s *cuda.Stream,
	a cudamem.Slice[uint16], m, k int,
	b cudamem.Slice[uint16], n int,
	d cudamem.Slice[uint16],
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
	if s != nil && s.Device() != e.dev {
		return errors.New("cublasltx: stream belongs to another device")
	}
	return e.execute(s, MatMulConfig{
		ComputeType: cublaslt.Compute32F,
		ScaleType:   cublaslt.R32F,
		AType:       cublaslt.R16F,
		BType:       cublaslt.R16F,
		DType:       cublaslt.R16F,
		Alpha:       1.0,
	}, b.Ptr(), n, k, n, a.Ptr(), k, m, k, d.Ptr(), n, m, n)
}

// MatMulRowMajorBF16 computes D = A*B for row-major bfloat16 matrices (encoded as uint16 bits)
// using float32 compute precision:
// A has shape m by k, B has shape k by n, and D has shape m by n.
func (e *Engine) MatMulRowMajorBF16(
	s *cuda.Stream,
	a cudamem.Slice[uint16], m, k int,
	b cudamem.Slice[uint16], n int,
	d cudamem.Slice[uint16],
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
	if s != nil && s.Device() != e.dev {
		return errors.New("cublasltx: stream belongs to another device")
	}
	return e.execute(s, MatMulConfig{
		ComputeType: cublaslt.Compute32F,
		ScaleType:   cublaslt.R32F,
		AType:       cublaslt.R16BF,
		BType:       cublaslt.R16BF,
		DType:       cublaslt.R16BF,
		Alpha:       1.0,
	}, b.Ptr(), n, k, n, a.Ptr(), k, m, k, d.Ptr(), n, m, n)
}

// LinearRowMajor computes Y = X * W^T for row-major float32 matrices:
// X has shape tokens by inFeatures, W has shape outFeatures by inFeatures,
// and Y has shape tokens by outFeatures.
func (e *Engine) LinearRowMajor(
	s *cuda.Stream,
	x cudamem.Slice[float32], tokens, inFeatures int,
	w cudamem.Slice[float32], outFeatures int,
	y cudamem.Slice[float32],
) error {
	if e == nil {
		return errors.New("cublasltx: nil engine")
	}
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.handle == 0 {
		return errors.New("cublasltx: closed engine")
	}
	if err := validateLinearLengths(x.Len(), tokens, inFeatures, w.Len(), outFeatures, y.Len()); err != nil {
		return err
	}
	if s != nil && s.Device() != e.dev {
		return errors.New("cublasltx: stream belongs to another device")
	}
	// Y^T = W * X^T where W is transposed from stored layout.
	// W stored memory: column-major inFeatures by outFeatures, ld inFeatures.
	// X stored memory: column-major inFeatures by tokens, ld inFeatures.
	// Y stored memory: column-major outFeatures by tokens, ld outFeatures.
	return e.execute(s, MatMulConfig{
		ComputeType: cublaslt.Compute32F,
		ScaleType:   cublaslt.R32F,
		AType:       cublaslt.R32F,
		BType:       cublaslt.R32F,
		DType:       cublaslt.R32F,
		TransA:      true,
		Alpha:       1.0,
	}, w.Ptr(), inFeatures, outFeatures, inFeatures, x.Ptr(), inFeatures, tokens, inFeatures, y.Ptr(), outFeatures, tokens, outFeatures)
}

// LinearRowMajorF16 computes Y = X * W^T for row-major float16 matrices (encoded as uint16 bits)
// using float32 compute precision:
// X has shape tokens by inFeatures, W has shape outFeatures by inFeatures,
// and Y has shape tokens by outFeatures.
func (e *Engine) LinearRowMajorF16(
	s *cuda.Stream,
	x cudamem.Slice[uint16], tokens, inFeatures int,
	w cudamem.Slice[uint16], outFeatures int,
	y cudamem.Slice[uint16],
) error {
	if e == nil {
		return errors.New("cublasltx: nil engine")
	}
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.handle == 0 {
		return errors.New("cublasltx: closed engine")
	}
	if err := validateLinearLengths(x.Len(), tokens, inFeatures, w.Len(), outFeatures, y.Len()); err != nil {
		return err
	}
	if s != nil && s.Device() != e.dev {
		return errors.New("cublasltx: stream belongs to another device")
	}
	return e.execute(s, MatMulConfig{
		ComputeType: cublaslt.Compute32F,
		ScaleType:   cublaslt.R32F,
		AType:       cublaslt.R16F,
		BType:       cublaslt.R16F,
		DType:       cublaslt.R16F,
		TransA:      true,
		Alpha:       1.0,
	}, w.Ptr(), inFeatures, outFeatures, inFeatures, x.Ptr(), inFeatures, tokens, inFeatures, y.Ptr(), outFeatures, tokens, outFeatures)
}

// LinearRowMajorBF16 computes Y = X * W^T for row-major bfloat16 matrices (encoded as uint16 bits)
// using float32 compute precision:
// X has shape tokens by inFeatures, W has shape outFeatures by inFeatures,
// and Y has shape tokens by outFeatures.
func (e *Engine) LinearRowMajorBF16(
	s *cuda.Stream,
	x cudamem.Slice[uint16], tokens, inFeatures int,
	w cudamem.Slice[uint16], outFeatures int,
	y cudamem.Slice[uint16],
) error {
	if e == nil {
		return errors.New("cublasltx: nil engine")
	}
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.handle == 0 {
		return errors.New("cublasltx: closed engine")
	}
	if err := validateLinearLengths(x.Len(), tokens, inFeatures, w.Len(), outFeatures, y.Len()); err != nil {
		return err
	}
	if s != nil && s.Device() != e.dev {
		return errors.New("cublasltx: stream belongs to another device")
	}
	return e.execute(s, MatMulConfig{
		ComputeType: cublaslt.Compute32F,
		ScaleType:   cublaslt.R32F,
		AType:       cublaslt.R16BF,
		BType:       cublaslt.R16BF,
		DType:       cublaslt.R16BF,
		TransA:      true,
		Alpha:       1.0,
	}, w.Ptr(), inFeatures, outFeatures, inFeatures, x.Ptr(), inFeatures, tokens, inFeatures, y.Ptr(), outFeatures, tokens, outFeatures)
}

// LinearRowMajorF16ToF32 computes Y = X * W^T where X and W are row-major float16 matrices
// (encoded as uint16 bits) and Y is a row-major float32 matrix.
func (e *Engine) LinearRowMajorF16ToF32(
	s *cuda.Stream,
	x cudamem.Slice[uint16], tokens, inFeatures int,
	w cudamem.Slice[uint16], outFeatures int,
	y cudamem.Slice[float32],
) error {
	if e == nil {
		return errors.New("cublasltx: nil engine")
	}
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.handle == 0 {
		return errors.New("cublasltx: closed engine")
	}
	if err := validateLinearLengths(x.Len(), tokens, inFeatures, w.Len(), outFeatures, y.Len()); err != nil {
		return err
	}
	if s != nil && s.Device() != e.dev {
		return errors.New("cublasltx: stream belongs to another device")
	}
	return e.execute(s, MatMulConfig{
		ComputeType: cublaslt.Compute32F,
		ScaleType:   cublaslt.R32F,
		AType:       cublaslt.R16F,
		BType:       cublaslt.R16F,
		DType:       cublaslt.R32F,
		TransA:      true,
		Alpha:       1.0,
	}, w.Ptr(), inFeatures, outFeatures, inFeatures, x.Ptr(), inFeatures, tokens, inFeatures, y.Ptr(), outFeatures, tokens, outFeatures)
}

func withStream(s *cuda.Stream, fn func(driver.Stream) error) error {
	if s == nil {
		return fn(0)
	}
	return s.WithHandle(fn)
}

func validateMatMulLengths(aLen, m, k, bLen, n, dLen int) error {
	if m <= 0 || k <= 0 || n <= 0 {
		return fmt.Errorf("cublasltx: matrix dimensions must be positive: m=%d k=%d n=%d", m, k, n)
	}
	if m > int(^uint(0)>>1)/k || k > int(^uint(0)>>1)/n || m > int(^uint(0)>>1)/n {
		return fmt.Errorf("cublasltx: matrix dimensions overflow: m=%d k=%d n=%d", m, k, n)
	}
	if aLen < m*k {
		return fmt.Errorf("cublasltx: matrix A length %d is smaller than %d", aLen, m*k)
	}
	if bLen < k*n {
		return fmt.Errorf("cublasltx: matrix B length %d is smaller than %d", bLen, k*n)
	}
	if dLen < m*n {
		return fmt.Errorf("cublasltx: matrix D length %d is smaller than %d", dLen, m*n)
	}
	return nil
}

func validateLinearLengths(xLen, tokens, inFeatures, wLen, outFeatures, yLen int) error {
	if tokens <= 0 || inFeatures <= 0 || outFeatures <= 0 {
		return fmt.Errorf("cublasltx: dimensions must be positive: tokens=%d in=%d out=%d", tokens, inFeatures, outFeatures)
	}
	if tokens > int(^uint(0)>>1)/inFeatures || outFeatures > int(^uint(0)>>1)/inFeatures || tokens > int(^uint(0)>>1)/outFeatures {
		return fmt.Errorf("cublasltx: dimensions overflow: tokens=%d in=%d out=%d", tokens, inFeatures, outFeatures)
	}
	if xLen < tokens*inFeatures {
		return fmt.Errorf("cublasltx: matrix X length %d is smaller than %d", xLen, tokens*inFeatures)
	}
	if wLen < outFeatures*inFeatures {
		return fmt.Errorf("cublasltx: matrix W length %d is smaller than %d", wLen, outFeatures*inFeatures)
	}
	if yLen < tokens*outFeatures {
		return fmt.Errorf("cublasltx: matrix Y length %d is smaller than %d", yLen, tokens*outFeatures)
	}
	return nil
}

func float32ToF16Bits(f float32) uint16 {
	bits := math.Float32bits(f)
	sign := uint16((bits >> 16) & 0x8000)
	exp := int((bits>>23)&0xff) - 127 + 15
	mant := bits & 0x007fffff
	if exp <= 0 {
		return sign
	} else if exp >= 31 {
		return sign | 0x7c00
	}
	return sign | uint16(exp<<10) | uint16(mant>>13)
}
