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

//go:embed cuda_shape_hc_mix.cu
var cudaShapeHCMixSource string

type shapeHCWeights struct {
	fn          cudamem.Slice[uint16]
	scale, base cudamem.Slice[float32]
}

// cudaShapeHCMix owns two small scratch buffers and a module. Validated uploaded
// weights and the foundation are borrowed and must outlive cleanup retries.
type cudaShapeHCMix struct {
	r                       *cudaFoundation
	config                  dsv4shape.Config
	weights                 map[string]shapeHCWeights
	mix, split              cudamem.Slice[float32]
	module                  driver.Module
	norm, project, sinkhorn driver.Function
	closed                  bool
	prepared                bool
	layer                   int
	phase                   string
	ops                     shapeHCMixOperations
}

type shapeHCMixOperations struct {
	alloc    func(int) (cudamem.Slice[float32], error)
	load     func() (driver.Module, driver.Function, driver.Function, driver.Function, error)
	drain    func() error
	free     func(*cudamem.Slice[float32]) error
	closeRun func(func() error) error
	unload   func(driver.Module) error
}

func newCUDAShapeHCMix(r *cudaFoundation, c dsv4shape.Config, b *dsv4shape.Binding, weights map[string]shapeHCWeights) (*cudaShapeHCMix, error) {
	if err := r.checkOpen(); err != nil {
		return nil, err
	}
	if r.device == nil {
		return nil, errors.New("cuda shape HC mix: missing device")
	}
	ops := shapeHCMixOperations{
		alloc:    func(n int) (cudamem.Slice[float32], error) { return cudamem.Alloc[float32](r.device, n) },
		drain:    func() error { return r.device.WithContext(func() error { return syncStream(r.stream()) }) },
		free:     func(v *cudamem.Slice[float32]) error { return v.Free() },
		closeRun: r.device.WithContext,
		unload:   func(m driver.Module) error { return m.Close() },
		load: func() (driver.Module, driver.Function, driver.Function, driver.Function, error) {
			p, err := nvrtcx.Compile(cudaShapeHCMixSource, nvrtcx.WithName("cuda_shape_hc_mix.cu"), nvrtcx.WithDeviceArch(r.device))
			if err != nil {
				return 0, 0, 0, 0, err
			}
			m, _, err := p.LoadOutputWithLog(r.device, nvrtcx.OutputCUBIN)
			if err != nil {
				return m, 0, 0, 0, err
			}
			var norm, project, split driver.Function
			err = r.withContext(func() error {
				var err error
				for _, entry := range []struct {
					name string
					dst  *driver.Function
				}{
					{"ds4_shape_hc_plain_rms", &norm}, {"ds4_shape_hc_project", &project}, {"ds4_shape_hc_split", &split},
				} {
					*entry.dst, err = m.Function(entry.name)
					if err != nil {
						return err
					}
				}
				return nil
			})
			return m, norm, project, split, err
		},
	}
	return newCUDAShapeHCMixWithOperations(r, c, b, weights, ops)
}

func newCUDAShapeHCMixWithOperations(r *cudaFoundation, c dsv4shape.Config, b *dsv4shape.Binding, weights map[string]shapeHCWeights, ops shapeHCMixOperations) (*cudaShapeHCMix, error) {
	if err := r.checkOpen(); err != nil {
		return nil, err
	}
	if _, err := planShapeForward(c, b, 1); err != nil {
		return nil, err
	}
	s := &cudaShapeHCMix{r: r, config: c, weights: make(map[string]shapeHCWeights), ops: ops}
	for i := 0; i < c.Layers; i++ {
		for _, phase := range []string{"attn", "ffn"} {
			key := fmt.Sprintf("blk.%d.hc_%s", i, phase)
			w, ok := weights[key]
			if !ok || w.fn.Ptr() == 0 || w.scale.Ptr() == 0 || w.base.Ptr() == 0 || w.fn.Len() != c.HC*c.Embed*24 || w.scale.Len() != 3 || w.base.Len() != 24 {
				return nil, fmt.Errorf("cuda shape HC mix: missing uploaded weights %s", key)
			}
			s.weights[key] = w
		}
	}
	fail := func(err error) (*cudaShapeHCMix, error) { return nil, finishBackendCleanup(err, s) }
	var err error
	s.mix, err = ops.alloc(24)
	if err != nil {
		return fail(err)
	}
	if s.mix.Ptr() == 0 || s.mix.Len() != 24 {
		return fail(errors.New("cuda shape HC mix: invalid mix allocation"))
	}
	s.split, err = ops.alloc(24)
	if err != nil {
		return fail(err)
	}
	if s.split.Ptr() == 0 || s.split.Len() != 24 {
		return fail(errors.New("cuda shape HC mix: invalid split allocation"))
	}
	s.module, s.norm, s.project, s.sinkhorn, err = ops.load()
	if err != nil {
		return fail(err)
	}
	if s.module == 0 || s.norm == 0 || s.project == 0 || s.sinkhorn == 0 {
		return fail(errors.New("cuda shape HC mix: missing compiled entry point"))
	}
	return s, nil
}

// Prepare normalizes the actual HC current state, projects to the 24 HC mix
// values, then performs the pinned CUDA sigmoid/softmax/Sinkhorn split.
func (s *cudaShapeHCMix) Prepare(state *cudaShapeForward, layer int, phase string, rmsEpsilon, hcEpsilon float32) error {
	if s == nil || s.closed || s.prepared || state == nil || state.closed || s.module == 0 {
		return errors.New("cuda shape HC mix: closed state")
	}
	if err := s.r.checkOpen(); err != nil {
		return err
	}
	if state.r != s.r || state.plan.config != s.config || layer < 0 || layer >= s.config.Layers || (phase != "attn" && phase != "ffn") {
		return errors.New("cuda shape HC mix: mismatched state or request")
	}
	for _, epsilon := range []float32{rmsEpsilon, hcEpsilon} {
		if epsilon <= 0 || math.IsNaN(float64(epsilon)) || math.IsInf(float64(epsilon), 0) {
			return errors.New("cuda shape HC mix: invalid epsilon")
		}
	}
	w := s.weights[fmt.Sprintf("blk.%d.hc_%s", layer, phase)]
	input, normalized := state.scratch["cur"], state.scratch["hc_flat"]
	if input.Len() != s.config.HC*s.config.Embed || normalized.Len() != s.config.HC*s.config.Embed || s.mix.Len() != 24 || s.split.Len() != 24 {
		return errors.New("cuda shape HC mix: invalid scratch extents")
	}
	ranges := []mxfp4DeviceRange{{uint64(w.fn.Ptr()), uint64(w.fn.Len()) * 2}}
	for _, v := range []cudamem.Slice[float32]{input, normalized, s.mix, s.split, w.scale, w.base} {
		ranges = append(ranges, mxfp4DeviceRange{uint64(v.Ptr()), uint64(v.Len()) * 4})
	}
	if err := checkMXFP4DeviceRanges(ranges...); err != nil {
		return err
	}
	return s.r.withContext(func() error {
		if err := cudalaunch.Kernel(s.norm).Grid(1, 1, 1).Block(256, 1, 1).On(s.r.stream()).Launch(normalized, input, uint32(s.config.HC*s.config.Embed), uint32(1), rmsEpsilon); err != nil {
			return err
		}
		if err := cudalaunch.Kernel(s.project).Grid(24, 1, 1).Block(32, 1, 1).On(s.r.stream()).Launch(s.mix, w.fn, normalized, uint64(s.config.HC*s.config.Embed), uint64(24), uint64(1)); err != nil {
			return err
		}
		return cudalaunch.Kernel(s.sinkhorn).Grid(1, 1, 1).Block(1, 1, 1).On(s.r.stream()).Launch(s.split, s.mix, w.scale, w.base, uint32(1), uint32(s.config.SinkhornIters), hcEpsilon)
	})
}

// BeforePhase executes the actual HC mix/split/collapse and layer RMS consumers.
// The attention or FFN evaluator must then fill the corresponding block output.
func (s *cudaShapeHCMix) BeforePhase(hc *cudaShapeHC, state *cudaShapeForward, layer int, phase string, rmsEpsilon, hcEpsilon float32) error {
	if s == nil || s.prepared {
		return errors.New("cuda shape HC mix: unconsumed phase")
	}
	if err := hc.checkState(state); err != nil {
		return err
	}
	if err := s.Prepare(state, layer, phase, rmsEpsilon, hcEpsilon); err != nil {
		return err
	}
	if err := hc.Collapse(state, phase, s.split.Slice(0, 4)); err != nil {
		return err
	}
	if err := state.Normalize(layer, phase, rmsEpsilon); err != nil {
		return err
	}
	s.prepared, s.layer, s.phase = true, layer, phase
	return nil
}

// AfterPhase expands the computed block output into HC next and advances the
// actual ping-pong state. It does not replace the missing attention/FFN kernels.
func (s *cudaShapeHCMix) AfterPhase(hc *cudaShapeHC, state *cudaShapeForward, layer int, phase string) error {
	if s == nil || s.closed || !s.prepared || s.split.Len() != 24 || s.layer != layer || s.phase != phase {
		return errors.New("cuda shape HC mix: phase not prepared")
	}
	if err := hc.checkState(state); err != nil {
		return err
	}
	if err := hc.Expand(state, phase, s.split.Slice(4, 8), s.split.Slice(8, 24)); err != nil {
		return err
	}
	state.scratch["cur"], state.scratch["next"] = state.scratch["next"], state.scratch["cur"]
	s.prepared = false
	return nil
}

func (s *cudaShapeHCMix) Close() error {
	if s == nil {
		return nil
	}
	s.closed = true
	if s.module == 0 && s.mix.Ptr() == 0 && s.split.Ptr() == 0 {
		return nil
	}
	if err := s.ops.drain(); err != nil {
		return err
	}
	var errs []error
	for _, v := range []*cudamem.Slice[float32]{&s.mix, &s.split} {
		if v.Ptr() == 0 {
			continue
		}
		if err := s.ops.free(v); err != nil {
			errs = append(errs, err)
		} else {
			*v = cudamem.Slice[float32]{}
		}
	}
	if s.module != 0 {
		errs = append(errs, mxfp4UnloadInContext(s.ops.closeRun, func() error { return s.ops.unload(s.module) }, func() { s.module = 0; s.norm = 0; s.project = 0; s.sinkhorn = 0 }))
	}
	return errors.Join(errs...)
}
