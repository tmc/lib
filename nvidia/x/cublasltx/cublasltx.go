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

// Engine manages cuBLASLt contexts, matrix layouts, workspace buffers, and execution heuristics.
type Engine struct {
	mu            sync.Mutex
	dev           *cuda.Device
	handle        cublaslt.Handle
	workspace     cudamem.Slice[byte]
	workspaceSize uint
}

// NewEngine initializes a cuBLASLt engine with a 32MB workspace buffer.
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
	if e == nil {
		return nil
	}
	e.mu.Lock()
	defer e.mu.Unlock()
	var errs []error
	if e.handle != 0 {
		err := e.dev.WithContext(func() error {
			return e.handle.Destroy()
		})
		if err != nil {
			errs = append(errs, fmt.Errorf("cublasltx: destroy handle: %w", err))
		} else {
			e.handle = 0
		}
	}
	if e.workspace.Len() > 0 {
		if err := e.workspace.Free(); err != nil {
			errs = append(errs, fmt.Errorf("cublasltx: free workspace: %w", err))
		} else {
			e.workspace = cudamem.Slice[byte]{}
		}
	}
	return errors.Join(errs...)
}

// MatMulConfig specifies options for raw matrix multiplication.
type MatMulConfig struct {
	ComputeType cublaslt.ComputeType // e.g. Compute32F, Compute32FFast16F, Compute16F
	ScaleType   cublaslt.DataType    // e.g. R32F, R16F
	AType       cublaslt.DataType    // e.g. R32F, R16F, R16BF
	BType       cublaslt.DataType    // e.g. R32F, R16F, R16BF
	DType       cublaslt.DataType    // e.g. R32F, R16F, R16BF
	TransA      bool                 // transpose op(A) = A^T
	TransB      bool                 // transpose op(B) = B^T
	Alpha       float32              // scalar multiplier for A*B (defaults to 1.0 if Alpha and Beta are 0)
	Beta        float32              // scalar multiplier for D (defaults to 0.0)
}

func (e *Engine) execute(
	s *cuda.Stream,
	cfg MatMulConfig,
	aPtr driver.DevicePtr, aRows, aCols, lda int,
	bPtr driver.DevicePtr, bRows, bCols, ldb int,
	dPtr driver.DevicePtr, dRows, dCols, ldd int,
) error {
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
			wsBytes := uint64(e.workspaceSize)
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

		err = withStream(s, func(streamHandle cublaslt.Stream) error {
			return e.handle.Matmul(
				opDesc,
				alphaPtr,
				uintptr(aPtr),
				layoutA,
				uintptr(bPtr),
				layoutB,
				betaPtr,
				uintptr(dPtr),
				layoutD,
				uintptr(dPtr),
				layoutD,
				algoPtr,
				uintptr(e.workspace.Ptr()),
				e.workspaceSize,
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

func withStream(s *cuda.Stream, fn func(cublaslt.Stream) error) error {
	if s == nil {
		return fn(0)
	}
	return s.WithHandle(func(stream driver.Stream) error {
		return fn(cublaslt.Stream(stream))
	})
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
	exp := int((bits >> 23) & 0xff) - 127 + 15
	mant := bits & 0x007fffff
	if exp <= 0 {
		return sign
	} else if exp >= 31 {
		return sign | 0x7c00
	}
	return sign | uint16(exp<<10) | uint16(mant>>13)
}

