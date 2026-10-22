//go:build linux

package main

import (
	_ "embed"
	"errors"

	"github.com/tmc/lib/examples/ds4-min/dsv4shape"
	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cudalaunch"
	"github.com/tmc/lib/nvidia/x/cudamem"
	"github.com/tmc/lib/nvidia/x/nvrtcx"
)

//go:embed cuda_shape_hc.cu
var cudaShapeHCSource string

// cudaShapeHC borrows its foundation and call operands; only its module is
// owned. It is a source-derived HC primitive, not a complete forward pass.
type cudaShapeHC struct {
	r                *cudaFoundation
	config           dsv4shape.Config
	module           driver.Module
	collapse, expand driver.Function
	closed           bool
	ops              shapeHCOperations
}

type shapeHCOperations struct {
	load     func() (driver.Module, driver.Function, driver.Function, error)
	drain    func() error
	closeRun func(func() error) error
	unload   func(driver.Module) error
}

func newCUDAShapeHC(r *cudaFoundation, c dsv4shape.Config) (*cudaShapeHC, error) {
	if err := r.checkOpen(); err != nil {
		return nil, err
	}
	if err := c.Validate(); err != nil {
		return nil, err
	}
	if r.device == nil {
		return nil, errors.New("cuda shape HC: missing device")
	}
	ops := shapeHCOperations{
		load: func() (driver.Module, driver.Function, driver.Function, error) {
			p, e := nvrtcx.Compile(cudaShapeHCSource, nvrtcx.WithName("cuda_shape_hc.cu"), nvrtcx.WithDeviceArch(r.device))
			if e != nil {
				return 0, 0, 0, e
			}
			m, _, e := p.LoadOutputWithLog(r.device, nvrtcx.OutputCUBIN)
			if e != nil {
				return m, 0, 0, e
			}
			var collapse, expand driver.Function
			e = r.withContext(func() error {
				var e error
				collapse, e = m.Function("ds4_shape_hc_weighted_sum")
				if e != nil {
					return e
				}
				expand, e = m.Function("ds4_shape_hc_expand")
				return e
			})
			return m, collapse, expand, e
		},
		drain: func() error { return r.device.WithContext(func() error { return syncStream(r.stream()) }) }, closeRun: r.device.WithContext, unload: func(m driver.Module) error { return m.Close() },
	}
	return newCUDAShapeHCWithOperations(r, c, ops)
}

func newCUDAShapeHCWithOperations(r *cudaFoundation, c dsv4shape.Config, ops shapeHCOperations) (*cudaShapeHC, error) {
	if err := r.checkOpen(); err != nil {
		return nil, err
	}
	if err := c.Validate(); err != nil {
		return nil, err
	}
	s := &cudaShapeHC{r: r, config: c, ops: ops}
	var err error
	s.module, s.collapse, s.expand, err = ops.load()
	if err != nil {
		return nil, finishBackendCleanup(err, s)
	}
	if s.module == 0 || s.collapse == 0 || s.expand == 0 {
		return nil, finishBackendCleanup(errors.New("cuda shape HC: missing compiled entry point"), s)
	}
	return s, nil
}

func (s *cudaShapeHC) checkState(state *cudaShapeForward) error {
	if s == nil || s.closed || state == nil || state.closed || s.module == 0 {
		return errors.New("cuda shape HC: closed state")
	}
	if err := s.r.checkOpen(); err != nil {
		return err
	}
	if state.r != s.r || state.plan.config != s.config {
		return errors.New("cuda shape HC: mismatched state")
	}
	return nil
}

// Collapse reads the actual HC current state into the attention/FFN input.
// Split weights are borrowed and must contain at least the four pre weights.
func (s *cudaShapeHC) Collapse(state *cudaShapeForward, phase string, pre cudamem.Slice[float32]) error {
	if err := s.checkState(state); err != nil {
		return err
	}
	if (phase != "attn" && phase != "ffn") || pre.Len() < s.config.HC {
		return errors.New("cuda shape HC: invalid collapse operands")
	}
	x, out := state.scratch["cur"], state.scratch[phase+"_cur"]
	if x.Len() != s.config.HC*s.config.Embed || out.Len() != s.config.Embed {
		return errors.New("cuda shape HC: invalid collapse extents")
	}
	if err := checkMXFP4DeviceRanges(mxfp4DeviceRange{uint64(x.Ptr()), uint64(x.Len()) * 4}, mxfp4DeviceRange{uint64(out.Ptr()), uint64(out.Len()) * 4}, mxfp4DeviceRange{uint64(pre.Ptr()), uint64(pre.Len()) * 4}); err != nil {
		return err
	}
	return s.r.withContext(func() error {
		return cudalaunch.Kernel(s.collapse).Grid(uint32((s.config.Embed+255)/256), 1, 1).Block(256, 1, 1).On(s.r.stream()).Launch(out, x, pre, uint32(s.config.Embed), uint32(s.config.HC), uint32(1), uint32(pre.Len()))
	})
}

// Expand combines the actual block output and current residual into next.
// post and comb are separate borrowed buffers produced by the HC split.
func (s *cudaShapeHC) Expand(state *cudaShapeForward, phase string, post, comb cudamem.Slice[float32]) error {
	if err := s.checkState(state); err != nil {
		return err
	}
	if (phase != "attn" && phase != "ffn") || post.Len() != s.config.HC || comb.Len() != s.config.HC*s.config.HC {
		return errors.New("cuda shape HC: invalid expand operands")
	}
	block, residual, out := state.scratch[phase+"_out"], state.scratch["cur"], state.scratch["next"]
	if block.Len() != s.config.Embed || residual.Len() != s.config.HC*s.config.Embed || out.Len() != s.config.HC*s.config.Embed {
		return errors.New("cuda shape HC: invalid expand extents")
	}
	ranges := []mxfp4DeviceRange{}
	for _, v := range []cudamem.Slice[float32]{block, residual, out, post, comb} {
		ranges = append(ranges, mxfp4DeviceRange{uint64(v.Ptr()), uint64(v.Len()) * 4})
	}
	if err := checkMXFP4DeviceRanges(ranges...); err != nil {
		return err
	}
	return s.r.withContext(func() error {
		return cudalaunch.Kernel(s.expand).Grid(uint32((s.config.HC*s.config.Embed+255)/256), 1, 1).Block(256, 1, 1).On(s.r.stream()).Launch(out, block, block, block, residual, post, comb, uint32(s.config.Embed), uint32(s.config.HC), uint32(1), uint32(post.Len()), uint32(comb.Len()), int32(0), int32(0))
	})
}

func (s *cudaShapeHC) Close() error {
	if s == nil {
		return nil
	}
	s.closed = true
	if s.module == 0 {
		return nil
	}
	if err := s.ops.drain(); err != nil {
		return err
	}
	return mxfp4UnloadInContext(s.ops.closeRun, func() error { return s.ops.unload(s.module) }, func() { s.module = 0; s.collapse = 0; s.expand = 0 })
}
