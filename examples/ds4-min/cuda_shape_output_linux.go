//go:build linux

package main

import (
	_ "embed"
	"errors"
	"fmt"
	"math"
	"slices"

	"github.com/tmc/lib/examples/ds4-min/dsv4shape"
	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cudalaunch"
	"github.com/tmc/lib/nvidia/x/cudamem"
)

//go:embed cuda_shape_output.cu
var cudaShapeOutputSource string

type shapeOutputWeights struct {
	fn, scale, base, norm, output cudaModelTensor
}

// encodeShapeOutput consumes the current four HC lanes and writes all logits.
// The module owner must keep weightEntry and the borrowed weights/buffers live
// until stream completion. No head is admitted through metadata alone.
func encodeShapeOutput(state *cudaShapeForward, binding *dsv4shape.Binding, hc *cudaShapeHCMix, collapse *cudaShapeHC, weightEntry driver.Function, w shapeOutputWeights, quant cudamem.Slice[byte], logits cudamem.Slice[float32], rmsEpsilon, hcEpsilon float32) error {
	if state == nil || state.closed || hc == nil || hc.closed || hc.module == 0 || hc.norm == 0 || hc.project == 0 || weightEntry == 0 || hc.r != state.r || hc.config != state.plan.config {
		return errors.New("cuda shape output: invalid state or module")
	}
	if err := state.r.checkOpen(); err != nil {
		return err
	}
	if err := collapse.checkState(state); err != nil {
		return err
	}
	for _, epsilon := range []float32{rmsEpsilon, hcEpsilon} {
		if epsilon <= 0 || math.IsNaN(float64(epsilon)) || math.IsInf(float64(epsilon), 0) {
			return errors.New("cuda shape output: invalid epsilon")
		}
	}
	c := state.plan.config
	if _, err := planShapeForward(c, binding, state.plan.context); err != nil {
		return err
	}
	specs, err := c.Weights()
	if err != nil {
		return err
	}
	byName := make(map[string]dsv4shape.TensorSpec)
	for _, spec := range specs {
		byName[spec.Name] = spec
	}
	views := make(map[string]dsv4shape.WeightView)
	for _, view := range binding.Weights {
		views[view.Name] = view
	}
	var ranges []mxfp4DeviceRange
	for _, entry := range []struct {
		name   string
		kind   uint32
		tensor cudaModelTensor
	}{
		{"output_hc_fn.weight", 1, w.fn}, {"output_hc_scale.weight", 0, w.scale}, {"output_hc_base.weight", 0, w.base}, {"output_norm.weight", 0, w.norm}, {"output.weight", 8, w.output},
	} {
		v, ok := views[entry.name]
		t := entry.tensor
		if !ok || v.Type != entry.kind || t.name != entry.name || t.typeID != v.Type || t.device == 0 || t.bytes != v.Bytes || t.fileOffset != v.Offset || !slices.Equal(t.dimensions, byName[entry.name].Dimensions) {
			return fmt.Errorf("cuda shape output: invalid uploaded weight %s", entry.name)
		}
		ranges = append(ranges, mxfp4DeviceRange{uint64(t.device), t.bytes})
	}
	for _, entry := range []struct {
		name  string
		count int
	}{
		{"cur", c.HC * c.Embed}, {"output_flat", c.HC * c.Embed}, {"output_pre", c.HC}, {"output_weights", c.HC}, {"output_embd", c.Embed}, {"output_norm", c.Embed},
	} {
		v := state.scratch[entry.name]
		if v.Len() != entry.count {
			return errors.New("cuda shape output: invalid State extent")
		}
		ranges = append(ranges, mxfp4DeviceRange{uint64(v.Ptr()), uint64(v.Len()) * 4})
	}
	if logits.Len() != c.Vocab || quant.Len() != c.Embed/32*36 {
		return errors.New("cuda shape output: invalid logits or quantized extent")
	}
	ranges = append(ranges, mxfp4DeviceRange{uint64(logits.Ptr()), uint64(logits.Len()) * 4}, mxfp4DeviceRange{uint64(quant.Ptr()), uint64(quant.Len())})
	if err := checkMXFP4DeviceRanges(ranges...); err != nil {
		return err
	}
	r := state.r
	if err := r.withContext(func() error {
		stream := r.stream()
		if err := cudalaunch.Kernel(hc.norm).Grid(1, 1, 1).Block(256, 1, 1).On(stream).Launch(state.scratch["output_flat"], state.scratch["cur"], uint32(c.HC*c.Embed), uint32(1), rmsEpsilon); err != nil {
			return err
		}
		if err := cudalaunch.Kernel(hc.project).Grid(uint32(c.HC), 1, 1).Block(32, 1, 1).On(stream).Launch(state.scratch["output_pre"], w.fn.device, state.scratch["output_flat"], uint64(c.HC*c.Embed), uint64(c.HC), uint64(1)); err != nil {
			return err
		}
		if err := cudalaunch.Kernel(weightEntry).Grid(1, 1, 1).Block(256, 1, 1).On(stream).Launch(state.scratch["output_weights"], state.scratch["output_pre"], w.scale.device, w.base.device, uint32(c.HC), uint32(1), hcEpsilon); err != nil {
			return err
		}
		return cudalaunch.Kernel(collapse.collapse).Grid(uint32((c.Embed+255)/256), 1, 1).Block(256, 1, 1).On(stream).Launch(state.scratch["output_embd"], state.scratch["cur"], state.scratch["output_weights"], uint32(c.Embed), uint32(c.HC), uint32(1), uint32(c.HC))
	}); err != nil {
		return err
	}
	if err := r.encodeRMS(state.scratch["output_embd"], w.norm, state.scratch["output_norm"], 1, true); err != nil {
		return err
	}
	if err := r.encodeQuantQ81(state.scratch["output_norm"], quant, c.Embed); err != nil {
		return err
	}
	return r.encodeQ80MatvecQ81(w.output, quant, logits)
}
