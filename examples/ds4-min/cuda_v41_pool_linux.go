//go:build linux

package main

import (
	_ "embed"
	"errors"
	"fmt"
	"math"
	"sync"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cudalaunch"
	"github.com/tmc/lib/nvidia/x/cudamem"
	"github.com/tmc/lib/nvidia/x/nvrtcx"
)

//go:embed cuda_v41_pool.cu
var cudaV41PoolSource string

// cudaV41Pool owns one request's device carry and module. The foundation is
// borrowed and must outlive Close. Successful Append synchronizes before return.
// After a runtime error callers must retain borrowed operands until Reset or
// Close successfully drains the stream, even though the frontier is uncommitted.
// A runtime failure invalidates the request until successful Reset; admission
// errors do not change it. Finite input values are a caller precondition.
type cudaV41Pool struct {
	mu              sync.Mutex
	r               *cudaFoundation
	module          driver.Module
	pool, carry     driver.Function
	kv, scores      cudamem.Slice[float32]
	pending         []cudamem.Slice[float32]
	width, position uint32
	valid, closing  bool
}

func newCUDAV41Pool(r *cudaFoundation, width uint32) (*cudaV41Pool, error) {
	return compileCUDAV41Pool(r, width, cudaV41PoolSource)
}

func compileCUDAV41Pool(r *cudaFoundation, width uint32, source string) (*cudaV41Pool, error) {
	if r == nil || r.device == nil || width == 0 || width > 1048576 || uint64(width) > uint64(math.MaxInt)/4 {
		return nil, errors.New("V4.1 CUDA pool: invalid foundation or width")
	}
	s := &cudaV41Pool{r: r, width: width}
	p, err := compileCUDAV41Device(source, nvrtcx.WithName("cuda_v41_pool.cu"), nvrtcx.WithDeviceArch(r.device))
	if err != nil {
		return nil, err
	}
	s.module, _, err = p.LoadOutputWithLog(r.device, nvrtcx.OutputCUBIN)
	if err == nil {
		err = r.device.WithContext(func() error {
			var e error
			s.pool, e = s.module.Function("ds4_v41_pool")
			if e != nil {
				return e
			}
			s.carry, e = s.module.Function("ds4_v41_pool_carry")
			return e
		})
	}
	if err == nil {
		s.kv, err = cudamem.Alloc[float32](r.device, int(width))
	}
	if err == nil {
		s.scores, err = cudamem.Alloc[float32](r.device, int(width))
	}
	if err == nil {
		err = s.Reset()
	}
	if err != nil {
		if e := s.Close(); e != nil {
			return s, errors.Join(err, e)
		}
		return nil, err
	}
	return s, nil
}

func (s *cudaV41Pool) Reset() error {
	if s == nil {
		return errors.New("V4.1 CUDA pool: closed")
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.module == 0 || s.closing {
		return errors.New("V4.1 CUDA pool: closed")
	}
	s.valid = false
	err := s.r.device.WithContext(func() error {
		if err := syncStream(s.r.stream()); err != nil {
			return err
		}
		if err := cudalaunch.Kernel(s.carry).Grid((s.width+255)/256, 1, 1).Block(256, 1, 1).On(s.r.stream()).Launch(s.kv, s.scores, s.kv, s.scores, s.width, uint32(0), uint32(1)); err != nil {
			return err
		}
		return syncStream(s.r.stream())
	})
	if err != nil {
		return fmt.Errorf("reset V4.1 CUDA pool: %w", err)
	}
	s.position = 0
	s.pending = nil
	s.valid = true
	return nil
}

func (s *cudaV41Pool) Append(out, kv, scores cudamem.Slice[float32], start uint32) error {
	if s == nil {
		return errors.New("V4.1 CUDA pool: closed")
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.module == 0 || s.closing {
		return errors.New("V4.1 CUDA pool: closed")
	}
	if !s.valid {
		return errors.New("V4.1 CUDA pool: reset required")
	}
	if start != s.position || kv.Len() == 0 || kv.Len()%int(s.width) != 0 || scores.Len() != kv.Len() {
		return errors.New("V4.1 CUDA pool: invalid input or frontier")
	}
	rows := uint64(kv.Len()) / uint64(s.width)
	if rows > 1048576 || uint64(start)+rows > 1048576 {
		return errors.New("V4.1 CUDA pool: context exceeded")
	}
	pairs := (rows + uint64(start&1)) / 2
	count := pairs * uint64(s.width)
	if uint64(out.Len()) != count || (count+255)/256 > math.MaxInt32 {
		return errors.New("V4.1 CUDA pool: invalid output")
	}
	// Carry is private. Distinct inputs/outputs avoid writes changing later reads.
	views := []cudamem.Slice[float32]{out, kv, scores, s.kv, s.scores}
	for i, a := range views {
		if a.Len() == 0 {
			continue
		}
		if uint64(a.Len()) > math.MaxUint64/4 {
			return errors.New("V4.1 CUDA pool: device range overflow")
		}
		lo, n := uint64(a.Ptr()), uint64(a.Len())*4
		if lo == 0 || lo > math.MaxUint64-n {
			return errors.New("V4.1 CUDA pool: invalid device range")
		}
		for _, b := range views[:i] {
			if b.Len() == 0 {
				continue
			}
			bl, bn := uint64(b.Ptr()), uint64(b.Len())*4
			if lo < bl+bn && bl < lo+n {
				return errors.New("V4.1 CUDA pool: overlapping operands")
			}
		}
	}
	s.valid = false
	s.pending = []cudamem.Slice[float32]{out, kv, scores}
	err := s.r.device.WithContext(func() error {
		if pairs != 0 {
			if err := cudalaunch.Kernel(s.pool).Grid(uint32((count+255)/256), 1, 1).Block(256, 1, 1).On(s.r.stream()).Launch(out, kv, scores, s.kv, s.scores, s.width, uint32(pairs), start&1); err != nil {
				return err
			}
		}
		last := (uint64(start) + rows - 1) &^ uint64(1)
		if last >= uint64(start) {
			if err := cudalaunch.Kernel(s.carry).Grid((s.width+255)/256, 1, 1).Block(256, 1, 1).On(s.r.stream()).Launch(s.kv, s.scores, kv, scores, s.width, uint32(last-uint64(start)), uint32(0)); err != nil {
				return err
			}
		}
		return syncStream(s.r.stream())
	})
	if err != nil {
		// A later launch may fail after the first kernel was submitted. Drain
		// even on that path. A failed drain retains the borrow obligation.
		drainErr := s.r.device.WithContext(func() error { return syncStream(s.r.stream()) })
		if drainErr == nil {
			s.pending = nil
		}
		return fmt.Errorf("append V4.1 CUDA pool: %w", errors.Join(err, drainErr))
	}
	s.pending = nil
	s.position += uint32(rows)
	s.valid = true
	return nil
}

func (s *cudaV41Pool) Close() error {
	if s == nil {
		return nil
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	s.closing = true
	s.valid = false
	if s.module == 0 && s.kv.Ptr() == 0 && s.scores.Ptr() == 0 {
		return nil
	}
	if s.r == nil {
		return nil
	}
	if err := s.r.device.WithContext(func() error { return syncStream(s.r.stream()) }); err != nil {
		return err
	}
	s.pending = nil
	var failures []error
	if err := s.kv.Free(); err != nil {
		failures = append(failures, err)
	} else {
		s.kv = cudamem.Slice[float32]{}
	}
	if err := s.scores.Free(); err != nil {
		failures = append(failures, err)
	} else {
		s.scores = cudamem.Slice[float32]{}
	}
	if len(failures) != 0 {
		return errors.Join(failures...)
	}
	if s.module != 0 {
		if err := s.r.device.WithContext(s.module.Close); err != nil {
			return err
		}
		s.module = 0
		s.pool = 0
		s.carry = 0
	}
	return nil
}
