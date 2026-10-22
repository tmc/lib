//go:build linux

package main

import (
	_ "embed"
	"errors"
	"fmt"
	"math"
	"sync"

	"github.com/tmc/lib/examples/ds4-min/dsv41"
	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cudalaunch"
	"github.com/tmc/lib/nvidia/x/cudamem"
	"github.com/tmc/lib/nvidia/x/nvrtcx"
)

//go:embed cuda_v41_aligned_q8.cu
var cudaV41AlignedQ8Source string

const cudaV41AlignedQ8Route = "aligned-q8_0/canonical-q8_1/single-column"

func cudaV41AlignedQ8Eligible(d dsv41.TensorDescriptor) bool {
	if d.Encoding != 8 || d.Offset < 0 || d.Bytes < 0 || d.Offset > math.MaxInt64-d.Bytes || len(d.Dimensions) != 2 || d.Dimensions[0]%1024 != 0 || d.Dimensions[1]%128 != 0 {
		return false
	}
	k, m, ok := cudaV41AlignedQ8NamedShape(d.Name)
	if !ok {
		return false
	}
	return d.Dimensions[0] == k && d.Dimensions[1] == m && d.Bytes == int64(k/32*34*m) && d.Bytes >= 2*1024*1024
}

// cudaV41AlignedQ8 owns an immutable one-time transformed weight and one
// request's canonical activation scratch. Raw source is borrowed only until
// constructor completion; success permits its release. Foundation remains
// borrowed through Close. This is not a producer-folded activation route.
type cudaV41AlignedQ8 struct {
	mu                                    sync.Mutex
	r                                     *cudaFoundation
	descriptor                            dsv41.TensorDescriptor
	module                                driver.Module
	clear, repack, quantize, matvec, dots driver.Function
	aligned, activation                   cudamem.Slice[byte]
	scaleBytes                            uint64
	warps                                 uint32
	valid, closing, ready                 bool
	pendingRaw                            cudamem.Slice[byte]
	pending                               []cudamem.Slice[float32]
}

func newCUDAV41AlignedQ8(r *cudaFoundation, w cudaV41Layer0Q8Weight) (*cudaV41AlignedQ8, error) {
	return compileCUDAV41AlignedQ8(r, w, cudaV41AlignedQ8Source)
}

func compileCUDAV41AlignedQ8(r *cudaFoundation, w cudaV41Layer0Q8Weight, source string) (*cudaV41AlignedQ8, error) {
	if r == nil || r.device == nil || !cudaV41AlignedQ8Eligible(w.Descriptor) || w.Data.Len() != int(w.Descriptor.Bytes) || w.Data.Ptr() == 0 || uint64(w.Data.Ptr()) > math.MaxUint64-uint64(w.Data.Len()) {
		return nil, errors.New("V4.1 aligned Q8: unsupported descriptor or source")
	}
	d := w.Descriptor
	d.Dimensions = append([]uint64(nil), d.Dimensions...)
	s := &cudaV41AlignedQ8{r: r, descriptor: d, warps: 1, pendingRaw: w.Data}
	major, minor, err := r.device.ComputeCapability()
	if err != nil {
		return nil, err
	}
	if major == 12 && minor == 1 {
		s.warps = 16
	}
	p, err := compileCUDAV41Device(source, nvrtcx.WithName("cuda_v41_aligned_q8.cu"), nvrtcx.WithDeviceArch(r.device), nvrtcx.WithFlag("--use_fast_math"))
	if err != nil {
		return nil, err
	}
	s.module, _, err = p.LoadOutputWithLog(r.device, nvrtcx.OutputCUBIN)
	if err == nil {
		err = r.device.WithContext(func() error {
			for _, v := range []struct {
				name string
				fn   *driver.Function
			}{{"ds4_v41_aligned_clear", &s.clear}, {"ds4_v41_aligned_repack", &s.repack}, {"ds4_v41_aligned_quantize", &s.quantize}, {"ds4_v41_aligned_matvec", &s.matvec}, {"ds4_v41_aligned_dots", &s.dots}} {
				f, e := s.module.Function(v.name)
				if e != nil {
					return e
				}
				*v.fn = f
			}
			return nil
		})
	}
	nblk := d.Dimensions[0] / 32 * d.Dimensions[1]
	s.scaleBytes = (nblk*2 + 63) &^ uint64(63)
	nbytes := s.scaleBytes + nblk*32
	if nbytes > uint64(math.MaxInt) {
		err = errors.Join(err, errors.New("V4.1 aligned Q8: artifact overflow"))
	}
	if err == nil {
		s.aligned, err = cudamem.Alloc[byte](r.device, int(nbytes))
	}
	if err == nil {
		s.activation, err = cudamem.Alloc[byte](r.device, int(d.Dimensions[0]/32*36))
	}
	if err == nil {
		err = r.device.WithContext(func() error {
			if e := cudalaunch.Kernel(s.clear).Grid(uint32((nbytes+255)/256), 1, 1).Block(256, 1, 1).On(r.stream()).Launch(s.aligned, nbytes); e != nil {
				return e
			}
			return cudalaunch.Kernel(s.repack).Grid(uint32((nblk*34+255)/256), 1, 1).Block(256, 1, 1).On(r.stream()).Launch(s.aligned, w.Data, nblk, s.scaleBytes)
		})
	}
	if err == nil {
		err = r.device.WithContext(func() error { return syncStream(r.stream()) })
	}
	if err != nil {
		if e := s.Close(); e != nil {
			return s, errors.Join(err, e)
		}
		return nil, err
	}
	s.pendingRaw = cudamem.Slice[byte]{}
	s.ready = true
	s.valid = true
	return s, nil
}

// Eval executes the named canonical activation route synchronously. The caller
// applies BF16 at the graph boundary. Runtime failure invalidates the owner;
// failed drain retains borrow obligations until Reset/Close succeeds.
func (s *cudaV41AlignedQ8) Eval(out, x cudamem.Slice[float32]) error {
	if s == nil {
		return errors.New("V4.1 aligned Q8: closed")
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.module == 0 || s.closing {
		return errors.New("V4.1 aligned Q8: closed")
	}
	if !s.valid {
		return errors.New("V4.1 aligned Q8: reset required")
	}
	k, m := s.descriptor.Dimensions[0], s.descriptor.Dimensions[1]
	if uint64(x.Len()) != k || uint64(out.Len()) != m {
		return errors.New("V4.1 aligned Q8: invalid operands")
	}
	type extent struct{ ptr, n uint64 }
	views := []extent{{uint64(x.Ptr()), k * 4}, {uint64(out.Ptr()), m * 4}, {uint64(s.aligned.Ptr()), uint64(s.aligned.Len())}, {uint64(s.activation.Ptr()), uint64(s.activation.Len())}}
	for i, a := range views {
		if a.ptr == 0 || a.ptr > math.MaxUint64-a.n {
			return errors.New("V4.1 aligned Q8: invalid range")
		}
		for _, b := range views[:i] {
			if a.ptr < b.ptr+b.n && b.ptr < a.ptr+a.n {
				return errors.New("V4.1 aligned Q8: overlapping operands")
			}
		}
	}
	s.valid = false
	s.pending = []cudamem.Slice[float32]{out, x}
	err := s.r.device.WithContext(func() error {
		if e := cudalaunch.Kernel(s.quantize).Grid(uint32((k+255)/256), 1, 1).Block(256, 1, 1).On(s.r.stream()).Launch(s.activation, x, uint32(k)); e != nil {
			return e
		}
		return cudalaunch.Kernel(s.matvec).Grid(uint32((m+uint64(s.warps)-1)/uint64(s.warps)), 1, 1).Block(s.warps*32, 1, 1).On(s.r.stream()).Launch(out, s.aligned, s.activation, uint32(m), uint32(k/32), s.scaleBytes)
	})
	drain := s.r.device.WithContext(func() error { return syncStream(s.r.stream()) })
	if drain == nil {
		s.pending = nil
	}
	if err != nil || drain != nil {
		return fmt.Errorf("evaluate V4.1 aligned Q8: %w", errors.Join(err, drain))
	}
	s.valid = true
	return nil
}

func (s *cudaV41AlignedQ8) Reset() error {
	if s == nil {
		return errors.New("V4.1 aligned Q8: closed")
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.module == 0 || s.closing {
		return errors.New("V4.1 aligned Q8: closed")
	}
	if !s.ready {
		return errors.New("V4.1 aligned Q8: initialization incomplete; close required")
	}
	s.valid = false
	if e := s.r.device.WithContext(func() error { return syncStream(s.r.stream()) }); e != nil {
		return e
	}
	s.pending = nil
	s.pendingRaw = cudamem.Slice[byte]{}
	s.valid = true
	return nil
}

func (s *cudaV41AlignedQ8) Close() error {
	if s == nil {
		return nil
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	s.closing = true
	s.valid = false
	if s.module == 0 && s.aligned.Ptr() == 0 && s.activation.Ptr() == 0 {
		return nil
	}
	if e := s.r.device.WithContext(func() error { return syncStream(s.r.stream()) }); e != nil {
		return e
	}
	s.pending = nil
	s.pendingRaw = cudamem.Slice[byte]{}
	var failures []error
	for _, v := range []*cudamem.Slice[byte]{&s.aligned, &s.activation} {
		if e := v.Free(); e != nil {
			failures = append(failures, e)
		} else {
			*v = cudamem.Slice[byte]{}
		}
	}
	if len(failures) != 0 {
		return errors.Join(failures...)
	}
	if s.module != 0 {
		if e := s.r.device.WithContext(s.module.Close); e != nil {
			return e
		}
		s.module = 0
	}
	return nil
}
