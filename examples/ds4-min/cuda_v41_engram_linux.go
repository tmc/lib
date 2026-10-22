//go:build linux

package main

import (
	_ "embed"
	"errors"
	"fmt"
	"math"
	"runtime"
	"slices"
	"sync"

	"github.com/tmc/lib/examples/ds4-min/dsv41"
	"github.com/tmc/lib/nvidia/cublas"
	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cudalaunch"
	"github.com/tmc/lib/nvidia/x/cudamem"
	"github.com/tmc/lib/nvidia/x/nvrtcx"
)

//go:embed cuda_v41_engram.cu
var cudaV41EngramSource string

type cudaV41EngramWeight struct {
	Projection, Query, Key dsv41.TensorDescriptor
	F16                    cudamem.Slice[uint16]
	Q, K                   cudamem.Slice[float32]
}

// cudaV41Engram owns one request's staged hash/rows, device scratch, module
// and dedicated cuBLAS handle. Weights and foundation are borrowed and must
// outlive successful Close. Methods serialize request state. After a runtime
// error the whole graph must invalidate its residual/cache and call Reset;
// no failed token may publish history. This is not a complete V4.1 backend.
type cudaV41Engram struct {
	mu                             sync.Mutex
	r                              *cudaFoundation
	w                              [2]cudaV41EngramWeight
	rows                           *dsv41.EngramRows
	module                         driver.Module
	half, round, add               driver.Function
	blas                           cublas.Handle
	input                          [2]cudamem.Slice[float32]
	xhalf                          cudamem.Slice[uint16]
	kv                             cudamem.Slice[float32]
	pending                        cudamem.Slice[float32]
	prepared, text, valid, closing bool
	applied                        uint8
	cleanup                        *cudaV41EngramCleanup
}

type cudaV41EngramCleanup struct {
	run       func(func() error) error
	drain     func() error
	freeFloat func(cudamem.Slice[float32]) error
	freeHalf  func(cudamem.Slice[uint16]) error
	destroy   func(cublas.Handle) error
	unload    func(driver.Module) error
}

func checkCUDAV41EngramWeights(w [2]cudaV41EngramWeight) error {
	for i, layer := range []int{1, 14} {
		for _, v := range []struct {
			d        dsv41.TensorDescriptor
			suffix   string
			encoding uint32
			n        int
			dims     []uint64
			ptr      uint64
		}{
			{w[i].Projection, "kv", 1, 6144 * 25600, []uint64{6144, 25600}, uint64(w[i].F16.Ptr())},
			{w[i].Query, "q_norm", 0, 4 * 5120, []uint64{5120, 4}, uint64(w[i].Q.Ptr())},
			{w[i].Key, "k_norm", 0, 4 * 5120, []uint64{5120, 4}, uint64(w[i].K.Ptr())},
		} {
			bytes := uint64(v.n) * 4
			if v.encoding == 1 {
				bytes = uint64(v.n) * 2
			}
			if v.d.Name != fmt.Sprintf("blk.%d.engram_%s.weight", layer, v.suffix) || v.d.Encoding != v.encoding || v.d.Offset < 0 || uint64(v.d.Offset) > math.MaxInt64-bytes || v.d.Bytes != int64(bytes) || len(v.d.Dimensions) != 2 || v.ptr == 0 || v.ptr > math.MaxUint64-bytes {
				return errors.New("V4.1 Engram: invalid weight descriptor")
			}
			for j, n := range v.dims {
				if v.d.Dimensions[j] != n {
					return errors.New("V4.1 Engram: invalid weight shape")
				}
			}
		}
		if w[i].F16.Len() != 6144*25600 || w[i].Q.Len() != 20480 || w[i].K.Len() != 20480 {
			return errors.New("V4.1 Engram: invalid weight extent")
		}
	}
	return nil
}

func newCUDAV41Engram(r *cudaFoundation, w [2]cudaV41EngramWeight, m *dsv41.Model, context int) (*cudaV41Engram, error) {
	for _, weight := range w {
		for _, descriptor := range []dsv41.TensorDescriptor{weight.Projection, weight.Query, weight.Key} {
			want, ok := m.Tensor(descriptor.Name)
			if !ok || descriptor.Encoding != want.Encoding || descriptor.Offset != want.Offset || descriptor.Bytes != want.Bytes || !slices.Equal(descriptor.Dimensions, want.Dimensions) {
				return nil, errors.New("V4.1 Engram: weight does not belong to bound model")
			}
		}
	}
	rows, err := m.NewEngramRows(context)
	if err != nil {
		return nil, err
	}
	return compileCUDAV41Engram(r, w, rows, cudaV41EngramSource)
}

// compileCUDAV41Engram consumes rows, including on failure. A nonnil owner
// returned with an error retains resources whose cleanup must be retried.
func compileCUDAV41Engram(r *cudaFoundation, w [2]cudaV41EngramWeight, rows *dsv41.EngramRows, source string) (owner *cudaV41Engram, err error) {
	if rows == nil {
		return nil, errors.New("V4.1 Engram: missing row source")
	}
	if r == nil || r.device == nil {
		_ = rows.Close()
		return nil, errors.New("V4.1 Engram: missing foundation")
	}
	if err := checkCUDAV41EngramWeights(w); err != nil {
		_ = rows.Close()
		return nil, err
	}
	s := &cudaV41Engram{r: r, w: w, rows: rows}
	defer func() {
		if err != nil {
			if e := s.Close(); e != nil {
				owner = s
				err = errors.Join(err, e)
			}
		}
	}()
	p, err := compileCUDAV41Device(source, nvrtcx.WithName("cuda_v41_engram.cu"), nvrtcx.WithDeviceArch(r.device), nvrtcx.WithFastMath())
	if err != nil {
		return nil, err
	}
	s.module, _, err = p.LoadOutputWithLog(r.device, nvrtcx.OutputCUBIN)
	if err != nil {
		return nil, err
	}
	err = r.device.WithContext(func() error {
		for _, v := range []struct {
			name string
			dst  *driver.Function
		}{{"ds4_v41_engram_half", &s.half}, {"ds4_v41_engram_round", &s.round}, {"ds4_v41_engram_add", &s.add}} {
			f, e := s.module.Function(v.name)
			if e != nil {
				return e
			}
			*v.dst = f
		}
		s.blas, err = cublas.Create()
		if err != nil {
			return err
		}
		if err := s.blas.SetPointerMode(cublas.PointerModeHost); err != nil {
			return err
		}
		if err := s.blas.SetMathMode(cublas.TF32TensorOpMath); err != nil {
			return err
		}
		return r.stream().WithHandle(func(stream driver.Stream) error { return s.blas.SetStream(driver.Stream(stream)) })
	})
	if err != nil {
		return nil, err
	}
	for i := range s.input {
		s.input[i], err = cudamem.Alloc[float32](r.device, 6144)
		if err != nil {
			return nil, err
		}
	}
	s.xhalf, err = cudamem.Alloc[uint16](r.device, 6144)
	if err != nil {
		return nil, err
	}
	s.kv, err = cudamem.Alloc[float32](r.device, 25600)
	if err != nil {
		return nil, err
	}
	s.valid = true
	return s, nil
}

func (s *cudaV41Engram) operations() cudaV41EngramCleanup {
	if s.cleanup != nil {
		return *s.cleanup
	}
	return cudaV41EngramCleanup{
		run:       s.r.device.WithContext,
		drain:     func() error { return s.r.device.WithContext(func() error { return syncStream(s.r.stream()) }) },
		freeFloat: func(x cudamem.Slice[float32]) error { return x.Free() }, freeHalf: func(x cudamem.Slice[uint16]) error { return x.Free() },
		destroy: func(h cublas.Handle) error { return s.r.device.WithContext(h.Close) },
		unload:  func(m driver.Module) error { return s.r.device.WithContext(m.Close) },
	}
}

// Prepare reads both sparse rows before any token work, uploads their BF16
// values, and leaves the committed frontier unchanged. Image tokens stage DEAD
// history without reads/uploads. Caller must evaluate all model layers/logits
// before Commit; device writes are not a rollback mechanism.
func (s *cudaV41Engram) Prepare(token int32, text bool, position int) error {
	if s == nil {
		return errors.New("V4.1 Engram: closed")
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closing || s.module == 0 || !s.valid || s.prepared {
		return errors.New("V4.1 Engram: invalid prepare state")
	}
	if err := s.rows.Prepare(token, text, position); err != nil {
		return err
	}
	s.valid = false
	err := s.operations().drain()
	if err == nil && text {
		for i, layer := range []int{1, 14} {
			var x []float32
			x, _, err = s.rows.Rows(layer)
			if err == nil {
				err = s.input[i].CopyFromHost(x)
			}
			if err != nil {
				break
			}
		}
	}
	if err != nil {
		s.rows.Abort()
		return fmt.Errorf("prepare V4.1 Engram: %w", err)
	}
	s.prepared, s.text, s.valid = true, text, true
	s.applied = 0
	return nil
}

// Apply executes the cuBLAS F16-input/F16-weight default projection, rounds
// all five output rows to BF16, then performs the four-head residual gate.
// Only insertion layers1 and14 are admitted, once each in order. The residual
// is borrowed until completion; on failed drain retain it until Reset/Close.
func (s *cudaV41Engram) Apply(layer int, residual cudamem.Slice[float32]) error {
	if s == nil {
		return errors.New("V4.1 Engram: closed")
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closing || s.module == 0 || !s.valid || !s.prepared {
		return errors.New("V4.1 Engram: token not prepared")
	}
	index := 0
	if layer == 14 {
		index = 1
	} else if layer != 1 {
		return errors.New("V4.1 Engram: invalid insertion layer")
	}
	if s.applied != uint8((1<<index)-1) {
		return errors.New("V4.1 Engram: repeated or unordered insertion")
	}
	if residual.Len() != 20480 || residual.Ptr() == 0 {
		return errors.New("V4.1 Engram: invalid residual")
	}
	if err := s.disjoint(residual); err != nil {
		return err
	}
	if !s.text {
		s.applied |= 1 << index
		return nil
	}
	s.valid = false
	s.pending = residual
	err := s.operations().run(func() error {
		if err := cudalaunch.Kernel(s.half).Grid(24, 1, 1).Block(256, 1, 1).On(s.r.stream()).Launch(s.xhalf, s.input[index], uint32(6144)); err != nil {
			return err
		}
		alpha, beta := float32(1), float32(0)
		var pin runtime.Pinner
		pin.Pin(&alpha)
		pin.Pin(&beta)
		defer pin.Unpin()
		if err := s.blas.GemmEx(cublas.OpT, cublas.OpN, 25600, 1, 6144, cublas.Float32Scalar(alpha), driver.DevicePtr(s.w[index].F16.Ptr()), cublas.R16F, 6144, driver.DevicePtr(s.xhalf.Ptr()), cublas.R16F, 6144, cublas.Float32Scalar(beta), driver.DevicePtr(s.kv.Ptr()), cublas.R32F, 25600, cublas.Compute32F, cublas.GemmDefault); err != nil {
			return err
		}
		if err := cudalaunch.Kernel(s.round).Grid(100, 1, 1).Block(256, 1, 1).On(s.r.stream()).Launch(s.kv, uint32(25600)); err != nil {
			return err
		}
		return cudalaunch.Kernel(s.add).Grid(1, 1, 1).Block(128, 1, 1).On(s.r.stream()).Launch(residual, s.kv, s.w[index].Q, s.w[index].K, uint32(5120))
	})
	drain := s.operations().drain()
	if drain == nil {
		s.pending = cudamem.Slice[float32]{}
	}
	if err != nil || drain != nil {
		s.rows.Abort()
		s.prepared = false
		return fmt.Errorf("apply V4.1 Engram: %w", errors.Join(err, drain))
	}
	s.applied |= 1 << index
	s.valid = true
	return nil
}

func (s *cudaV41Engram) disjoint(residual cudamem.Slice[float32]) error {
	a, n := uint64(residual.Ptr()), uint64(residual.Len())*4
	if a > math.MaxUint64-n {
		return errors.New("V4.1 Engram: residual range overflow")
	}
	ranges := [][2]uint64{{uint64(s.xhalf.Ptr()), 6144 * 2}, {uint64(s.kv.Ptr()), 25600 * 4}, {uint64(s.input[0].Ptr()), 6144 * 4}, {uint64(s.input[1].Ptr()), 6144 * 4}}
	for _, w := range s.w {
		ranges = append(ranges, [2]uint64{uint64(w.F16.Ptr()), 6144 * 25600 * 2}, [2]uint64{uint64(w.Q.Ptr()), 20480 * 4}, [2]uint64{uint64(w.K.Ptr()), 20480 * 4})
	}
	for _, b := range ranges {
		if b[0] == 0 || b[0] > math.MaxUint64-b[1] || a < b[0]+b[1] && b[0] < a+n {
			return errors.New("V4.1 Engram: overlapping or invalid device buffers")
		}
	}
	return nil
}

// Commit is called only after the whole token's layers and logits succeed.
// Both insertion sites must have completed. No device work is enqueued here.
func (s *cudaV41Engram) Commit() error {
	if s == nil {
		return errors.New("V4.1 Engram: closed")
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closing || !s.valid || !s.prepared || s.applied != 3 {
		return errors.New("V4.1 Engram: incomplete token")
	}
	if err := s.rows.Commit(); err != nil {
		return err
	}
	s.prepared = false
	return nil
}

// Abort invalidates this component after any later layer/logit error, without
// advancing hash history. The graph must also invalidate all other state.
func (s *cudaV41Engram) Abort() {
	if s != nil {
		s.mu.Lock()
		defer s.mu.Unlock()
		s.valid = false
		s.prepared = false
		s.rows.Abort()
	}
}

func (s *cudaV41Engram) Reset() error {
	if s == nil {
		return errors.New("V4.1 Engram: closed")
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closing || s.module == 0 {
		return errors.New("V4.1 Engram: closed")
	}
	s.valid = false
	if err := s.operations().drain(); err != nil {
		return err
	}
	s.pending = cudamem.Slice[float32]{}
	s.prepared = false
	s.applied = 0
	if err := s.rows.Reset(); err != nil {
		return err
	}
	s.valid = true
	return nil
}

func (s *cudaV41Engram) Close() error {
	if s == nil {
		return nil
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	s.closing = true
	s.valid = false
	if s.rows == nil && s.module == 0 && s.blas == 0 && s.input[0].Ptr() == 0 && s.input[1].Ptr() == 0 && s.kv.Ptr() == 0 && s.xhalf.Ptr() == 0 {
		return nil
	}
	if s.rows != nil {
		s.rows.Abort()
	}
	ops := s.operations()
	if err := ops.drain(); err != nil {
		return err
	}
	s.pending = cudamem.Slice[float32]{}
	if s.blas != 0 {
		if err := ops.destroy(s.blas); err != nil {
			return err
		}
		s.blas = 0
	}
	for i := range s.input {
		if err := ops.freeFloat(s.input[i]); err != nil {
			return err
		}
		s.input[i] = cudamem.Slice[float32]{}
	}
	if err := ops.freeFloat(s.kv); err != nil {
		return err
	}
	s.kv = cudamem.Slice[float32]{}
	if err := ops.freeHalf(s.xhalf); err != nil {
		return err
	}
	s.xhalf = cudamem.Slice[uint16]{}
	if s.module != 0 {
		if err := ops.unload(s.module); err != nil {
			return err
		}
		s.module = 0
	}
	if s.rows != nil {
		if err := s.rows.Close(); err != nil {
			return err
		}
		s.rows = nil
	}
	return nil
}
