//go:build linux

package main

import (
	_ "embed"
	"errors"
	"fmt"
	"math"

	"github.com/tmc/lib/examples/ds4-min/dsv4shape"
	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cudalaunch"
	"github.com/tmc/lib/nvidia/x/cudamem"
	"github.com/tmc/lib/nvidia/x/nvrtcx"
)

//go:embed cuda_shape_rms.cu
var cudaShapeRMSSource string

// cudaShapeForward owns scratch and persistent cache, but borrows validated
// device norm weights and the foundation. This is not a complete forward pass.
// The foundation and every borrowed norm must outlive this state, including
// retained cleanup retries. Close never closes the foundation or norm owners.
type cudaShapeForward struct {
	r       *cudaFoundation
	plan    shapeForwardPlan
	cache   *cudaShapeCache
	scratch map[string]cudamem.Slice[float32]
	weights map[string]cudamem.Slice[float32]
	module  driver.Module
	rms     driver.Function
	closed  bool
	ops     shapeForwardOperations
}

type shapeForwardOperations struct {
	cache    func(uint64) (*cudaShapeCache, error)
	alloc    func(int) (cudamem.Slice[float32], error)
	load     func() (driver.Module, driver.Function, error)
	drain    func() error
	free     func(*cudamem.Slice[float32]) error
	closeRun func(func() error) error
	unload   func(driver.Module) error
}

func newCUDAShapeForward(r *cudaFoundation, c dsv4shape.Config, binding *dsv4shape.Binding, weights map[string]cudamem.Slice[float32], context, reserve uint64) (*cudaShapeForward, error) {
	if err := r.checkOpen(); err != nil {
		return nil, err
	}
	if r.device == nil {
		return nil, errors.New("cuda shape forward: missing device")
	}
	ops := shapeForwardOperations{
		cache: func(extra uint64) (*cudaShapeCache, error) { return newCUDAShapeCache(r, c, context, extra) },
		alloc: func(n int) (cudamem.Slice[float32], error) { return cudamem.Alloc[float32](r.device, n) },
		load: func() (driver.Module, driver.Function, error) {
			p, e := nvrtcx.Compile(cudaShapeRMSSource, nvrtcx.WithName("cuda_shape_rms.cu"), nvrtcx.WithDeviceArch(r.device))
			if e != nil {
				return 0, 0, e
			}
			m, _, e := p.LoadOutputWithLog(r.device, nvrtcx.OutputCUBIN)
			if e != nil {
				return m, 0, e
			}
			var f driver.Function
			e = r.withContext(func() error { var e error; f, e = m.Function("ds4_shape_rms"); return e })
			return m, f, e
		},
		drain: func() error { return r.device.WithContext(func() error { return syncStream(r.stream()) }) },
		free:  func(v *cudamem.Slice[float32]) error { return v.Free() }, closeRun: r.device.WithContext, unload: func(m driver.Module) error { return m.Close() },
	}
	return newCUDAShapeForwardWithOperations(r, c, binding, weights, context, reserve, ops)
}

func newCUDAShapeForwardWithOperations(r *cudaFoundation, c dsv4shape.Config, binding *dsv4shape.Binding, weights map[string]cudamem.Slice[float32], context, reserve uint64, ops shapeForwardOperations) (*cudaShapeForward, error) {
	if err := r.checkOpen(); err != nil {
		return nil, err
	}
	p, err := planShapeForward(c, binding, context)
	if err != nil {
		return nil, err
	}
	s := &cudaShapeForward{r: r, plan: p, ops: ops, scratch: make(map[string]cudamem.Slice[float32]), weights: make(map[string]cudamem.Slice[float32])}
	for name := range p.norms {
		w, ok := weights[name]
		if !ok || w.Ptr() == 0 || w.Len() != c.Embed {
			return nil, fmt.Errorf("cuda shape forward: missing device norm %s", name)
		}
		s.weights[name] = w
	}
	if p.bytes > ^uint64(0)-reserve {
		return nil, errors.New("cuda shape forward: reserve overflow")
	}
	fail := func(err error) (*cudaShapeForward, error) { return nil, finishBackendCleanup(err, s) }
	s.cache, err = ops.cache(reserve + p.bytes)
	if err != nil {
		return fail(err)
	}
	for _, b := range p.buffers {
		v, e := ops.alloc(b.elements)
		s.scratch[b.name] = v
		if e != nil {
			return fail(e)
		}
		if v.Ptr() == 0 || v.Len() != b.elements {
			return fail(fmt.Errorf("cuda shape forward: invalid allocation %s", b.name))
		}
	}
	s.module, s.rms, err = ops.load()
	if err != nil {
		return fail(err)
	}
	if s.module == 0 || s.rms == 0 {
		return fail(errors.New("cuda shape forward: missing compiled entry point"))
	}
	return s, nil
}

// Normalize consumes the actual owned attention/FFN input and output buffers.
// Callers must populate the input first; this does not synthesize activations.
func (s *cudaShapeForward) Normalize(layer int, phase string, epsilon float32) error {
	if s == nil || s.closed {
		return errors.New("cuda shape forward: closed")
	}
	if err := s.r.checkOpen(); err != nil {
		return err
	}
	if layer < 0 || layer >= s.plan.config.Layers || (phase != "attn" && phase != "ffn") || epsilon <= 0 || math.IsNaN(float64(epsilon)) || math.IsInf(float64(epsilon), 0) {
		return errors.New("cuda shape forward: invalid norm request")
	}
	in, out := s.scratch[phase+"_cur"], s.scratch[phase+"_norm"]
	w := s.weights[fmt.Sprintf("blk.%d.%s_norm.weight", layer, phase)]
	if err := checkMXFP4DeviceRanges(mxfp4DeviceRange{uint64(in.Ptr()), uint64(in.Len()) * 4}, mxfp4DeviceRange{uint64(out.Ptr()), uint64(out.Len()) * 4}, mxfp4DeviceRange{uint64(w.Ptr()), uint64(w.Len()) * 4}); err != nil {
		return err
	}
	return s.r.withContext(func() error {
		return cudalaunch.Kernel(s.rms).Grid(1, 1, 1).Block(1, 1, 1).On(s.r.stream()).Launch(in, w, out, int32(s.plan.config.Embed), int32(1), epsilon)
	})
}

func (s *cudaShapeForward) Close() error {
	if s == nil {
		return nil
	}
	s.closed = true
	if len(s.scratch) == 0 && s.module == 0 && s.cache == nil {
		return nil
	}
	if err := s.ops.drain(); err != nil {
		return err
	}
	var errs []error
	for name, v := range s.scratch {
		if v.Ptr() == 0 {
			delete(s.scratch, name)
			continue
		}
		if err := s.ops.free(&v); err != nil {
			s.scratch[name] = v
			if v.Ptr() == 0 {
				delete(s.scratch, name)
			}
			errs = append(errs, err)
		} else {
			delete(s.scratch, name)
		}
	}
	if s.module != 0 {
		errs = append(errs, mxfp4UnloadInContext(s.ops.closeRun, func() error { return s.ops.unload(s.module) }, func() { s.module = 0; s.rms = 0 }))
	}
	if s.cache != nil {
		if err := s.cache.Close(); err != nil {
			errs = append(errs, err)
		} else {
			s.cache = nil
		}
	}
	return errors.Join(errs...)
}
