//go:build linux

package main

import (
	_ "embed"
	"errors"
	"fmt"
	"math"
	"sync"
	"unsafe"

	"github.com/tmc/lib/examples/ds4-min/dsv41"
	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cudalaunch"
	"github.com/tmc/lib/nvidia/x/cudamem"
	"github.com/tmc/lib/nvidia/x/nvrtcx"
)

//go:embed cuda_v41_raw_attention.cu
var cudaV41RawAttentionSource string

// cudaV41RawAttention owns a single layer0/1 raw-only request evaluator. It
// borrows the foundation and owns its modules, scratch and 128-row ring. Query
// and KV inputs are copied on device before RoPE/quantization; they and sinks
// remain unchanged. Append success has completed its output; after runtime
// failure all borrowed buffers must survive until successful Reset/Close.
// Inputs are finite, pre-RoPE BF16 projected/normalized activations. This is
// not a projection, compressed attention, whole layer or model implementation.
type cudaV41RawAttention struct {
	mu                     sync.Mutex
	r                      *cudaFoundation
	ops                    *cudaV41Ops
	module                 driver.Module
	copy, commit, attend   driver.Function
	q, kv, ring, frequency cudamem.Slice[float32]
	scores, scoresStorage  cudamem.Slice[float32]
	pending                []cudamem.Slice[float32]
	position, context      uint32
	valid, closing         bool
	drainProbe             func() error // optional uncertainty seam; nil uses the owning stream drain
}

func newCUDAV41RawAttention(r *cudaFoundation, context uint32) (*cudaV41RawAttention, error) {
	return compileCUDAV41RawAttention(r, context, cudaV41RawAttentionSource)
}

func compileCUDAV41RawAttention(r *cudaFoundation, context uint32, source string) (*cudaV41RawAttention, error) {
	return compileCUDAV41RawAttentionCaptured(r, context, source, nil)
}

func compileCUDAV41RawAttentionCaptured(r *cudaFoundation, context uint32, source string, capture func(*nvrtcx.CompiledModule) error) (*cudaV41RawAttention, error) {
	if r == nil || r.device == nil || context == 0 || context > 1048576 {
		return nil, errors.New("V4.1 raw attention: invalid foundation or context")
	}
	s := &cudaV41RawAttention{r: r, context: context}
	var err error
	s.ops, err = newCUDAV41Ops(r)
	if err == nil {
		options := []nvrtcx.Option{nvrtcx.WithName("cuda_v41_raw_attention.cu"), nvrtcx.WithDeviceArch(r.device)}
		if capture != nil {
			options = append(options, nvrtcx.WithNoCache())
		}
		p, e := compileCUDAV41Device(source, options...)
		err = e
		if err == nil && capture != nil {
			err = capture(p)
		}
		if err == nil {
			s.module, _, err = p.LoadOutputWithLog(r.device, nvrtcx.OutputCUBIN)
		}
	}
	if err == nil {
		err = r.device.WithContext(func() error {
			var e error
			s.copy, e = s.module.Function("ds4_v41_raw_copy")
			if e != nil {
				return e
			}
			s.commit, e = s.module.Function("ds4_v41_raw_commit")
			if e != nil {
				return e
			}
			s.attend, e = s.module.Function("ds4_v41_raw_attend")
			return e
		})
	}
	for _, v := range []struct {
		dst *cudamem.Slice[float32]
		n   int
	}{{&s.q, 64 * 512}, {&s.kv, 512}, {&s.ring, 128 * 512}, {&s.frequency, 32}, {&s.scoresStorage, 64*128 + 64}} {
		if err == nil {
			*v.dst, err = cudamem.Alloc[float32](r.device, v.n)
		}
	}
	if err == nil {
		s.scores = s.scoresStorage.Slice(32, 32+64*128)
	}
	if err == nil {
		err = s.frequency.CopyFromHost(cudaV41Frequencies(false))
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

func (s *cudaV41RawAttention) Reset() error {
	if s == nil {
		return errors.New("V4.1 raw attention: closed")
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.module == 0 || s.closing {
		return errors.New("V4.1 raw attention: closed")
	}
	s.valid = false
	err := s.r.device.WithContext(func() error {
		if err := syncStream(s.r.stream()); err != nil {
			return err
		}
		if err := cudalaunch.Kernel(s.copy).Grid(256, 1, 1).Block(256, 1, 1).On(s.r.stream()).Launch(s.ring, s.ring, uint32(128*512), uint32(1)); err != nil {
			return err
		}
		if err := cudalaunch.Kernel(s.copy).Grid(33, 1, 1).Block(256, 1, 1).On(s.r.stream()).Launch(s.scoresStorage, s.scoresStorage, uint32(64*128+64), uint32(1)); err != nil {
			return err
		}
		return syncStream(s.r.stream())
	})
	if err != nil {
		return fmt.Errorf("reset V4.1 raw attention: %w", err)
	}
	s.pending = nil
	s.position = 0
	s.valid = true
	return nil
}

func (s *cudaV41RawAttention) Append(out, q, kv, sinks cudamem.Slice[float32], position uint32) error {
	if s == nil {
		return errors.New("V4.1 raw attention: closed")
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.module == 0 || s.closing {
		return errors.New("V4.1 raw attention: closed")
	}
	if !s.valid {
		return errors.New("V4.1 raw attention: reset required")
	}
	if position != s.position || position >= s.context || out.Len() != 64*512 || q.Len() != 64*512 || kv.Len() != 512 || sinks.Len() != 64 {
		return errors.New("V4.1 raw attention: invalid shape or frontier")
	}
	if s.scores.Len() != 64*128 || s.scoresStorage.Len() != 64*128+64 || s.scores.Ptr() != s.scoresStorage.Slice(32, 32+64*128).Ptr() {
		return errors.New("V4.1 raw attention: invalid score workspace")
	}
	views := []cudamem.Slice[float32]{out, q, kv, sinks, s.q, s.kv, s.ring, s.frequency, s.scoresStorage}
	for i, a := range views {
		lo, n := uint64(a.Ptr()), uint64(a.Len())*4
		if lo == 0 || lo > math.MaxUint64-n {
			return errors.New("V4.1 raw attention: invalid device range")
		}
		if err := cudaV41RawDeviceRange(s.r, a.Ptr(), n); err != nil {
			return err
		}
		for _, b := range views[:i] {
			bl, bn := uint64(b.Ptr()), uint64(b.Len())*4
			if lo < bl+bn && bl < lo+n {
				return errors.New("V4.1 raw attention: overlapping buffers")
			}
		}
	}
	s.valid = false
	s.pending = []cudamem.Slice[float32]{out, q, kv, sinks}
	err := s.r.device.WithContext(func() error {
		for _, v := range []struct{ dst, src cudamem.Slice[float32] }{{s.q, q}, {s.kv, kv}} {
			if err := cudalaunch.Kernel(s.copy).Grid(uint32((v.src.Len()+255)/256), 1, 1).Block(256, 1, 1).On(s.r.stream()).Launch(v.dst, v.src, uint32(v.src.Len()), uint32(0)); err != nil {
				return err
			}
		}
		return nil
	})
	if err == nil {
		err = s.ops.Rope(s.q, s.frequency, 512, 64, 1, position, 1, false)
	}
	if err == nil {
		err = s.ops.Rope(s.kv, s.frequency, 512, 1, 1, position, 1, false)
	}
	if err == nil {
		err = s.ops.Quantize(s.kv, 512, 1, dsv41.FP8E8M0)
	}
	if err == nil {
		err = s.r.device.WithContext(func() error {
			if e := cudalaunch.Kernel(s.commit).Grid(2, 1, 1).Block(256, 1, 1).On(s.r.stream()).Launch(s.ring, s.kv, position); e != nil {
				return e
			}
			return cudalaunch.Kernel(s.attend).Grid(2, 1, 1).Block(32, 1, 1).On(s.r.stream()).Launch(out, s.q, s.ring, sinks, s.scores, position)
		})
	}
	if err == nil {
		err = s.ops.Quantize(out, 512, 64, dsv41.BFloat16)
	}
	if err == nil {
		err = s.ops.Rope(out, s.frequency, 512, 64, 1, position, 1, true)
	}
	drain := s.drainRawStream()
	if drain == nil {
		s.pending = nil
	}
	if err != nil || drain != nil {
		return fmt.Errorf("append V4.1 raw attention: %w", errors.Join(err, drain))
	}
	s.position++
	s.valid = true
	return nil
}

func cudaV41RawDeviceRange(r *cudaFoundation, p driver.DevicePtr, n uint64) error {
	return r.device.WithContext(func() error {
		current, err := driver.CtxGetCurrent()
		if err != nil {
			return err
		}
		var owner driver.Context
		if err := p.PointerGetAttribute(unsafe.Pointer(&owner), driver.PointerAttributeContext); err != nil {
			return err
		}
		if owner != current {
			return errors.New("V4.1 raw attention: foreign device context")
		}
		base, size, err := p.MemGetAddressRange()
		if err != nil {
			return err
		}
		if p < base || uint64(p-base) > uint64(size) || n > uint64(size)-uint64(p-base) {
			return errors.New("V4.1 raw attention: allocation range exceeded")
		}
		return nil
	})
}

func (s *cudaV41RawAttention) Close() error {
	if s == nil {
		return nil
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	s.closing = true
	s.valid = false
	if s.module == 0 && s.ops == nil && s.q.Ptr() == 0 && s.kv.Ptr() == 0 && s.ring.Ptr() == 0 && s.frequency.Ptr() == 0 && s.scoresStorage.Ptr() == 0 {
		return nil
	}
	if err := s.drainRawStream(); err != nil {
		return err
	}
	s.pending = nil
	var failures []error
	for _, v := range []*cudamem.Slice[float32]{&s.q, &s.kv, &s.ring, &s.frequency, &s.scoresStorage} {
		if err := v.Free(); err != nil {
			failures = append(failures, err)
		} else {
			*v = cudamem.Slice[float32]{}
		}
	}
	if s.scoresStorage.Ptr() == 0 {
		s.scores = cudamem.Slice[float32]{}
	}
	if len(failures) != 0 {
		return errors.Join(failures...)
	}
	if s.ops != nil {
		if err := s.ops.Close(); err != nil {
			return err
		}
		s.ops = nil
	}
	if s.module != 0 {
		if err := s.r.device.WithContext(s.module.Close); err != nil {
			return err
		}
		s.module = 0
		s.copy = 0
		s.commit = 0
		s.attend = 0
	}
	return nil
}

// drainRawStream preserves the default synchronization route. The optional
// test probe must own its terminal observation and never implies GPU failure.
func (s *cudaV41RawAttention) drainRawStream() error {
	if s.drainProbe != nil {
		return s.drainProbe()
	}
	return s.r.device.WithContext(func() error { return syncStream(s.r.stream()) })
}
