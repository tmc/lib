//go:build linux

package main

import (
	_ "embed"
	"errors"
	"fmt"
	"slices"

	"github.com/tmc/lib/examples/ds4-min/dsv4shape"
	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cudalaunch"
	"github.com/tmc/lib/nvidia/x/cudamem"
)

//go:embed cuda_shape_router.cu
var cudaShapeRouterSource string

// This initial executable FFN route admits only raw IQ2_XXS gate/up, Q2_K
// down and Q8_0 shared weights. Other valid GGUF layouts need other consumers.
type shapeFFNWeights struct {
	router, bias, hash               cudaModelTensor
	gate, up, down                   cudaModelTensor
	sharedGate, sharedUp, sharedDown cudaModelTensor
}

// All extra buffers are borrowed. They must remain live through the stream
// drain; this object does not own resources or broaden full backend Supports.
type shapeFFNBuffers struct {
	logits, probs, weights          cudamem.Slice[float32]
	ids                             cudamem.Slice[int32]
	gate, up, down                  cudamem.Slice[float32]
	inputQ81, expertsQ81, sharedQ80 cudamem.Slice[byte]
	invalid                         cudamem.Slice[uint32]
}

func validateShapeFFNWeights(c dsv4shape.Config, b *dsv4shape.Binding, layer int, w shapeFFNWeights) error {
	if _, err := planShapeForward(c, b, 1); err != nil {
		return err
	}
	return validateShapeFFNWeightsAdmitted(c, b, layer, w)
}

// The caller must have admitted the unchanged binding with planShapeForward.
func validateShapeFFNWeightsAdmitted(c dsv4shape.Config, b *dsv4shape.Binding, layer int, w shapeFFNWeights) error {
	l, err := c.Layer(layer)
	if err != nil {
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
	for _, view := range b.Weights {
		views[view.Name] = view
	}
	for _, entry := range []struct {
		name   string
		kind   uint32
		tensor cudaModelTensor
	}{
		{"ffn_gate_inp.weight", 1, w.router}, {"ffn_gate_exps.weight", 16, w.gate}, {"ffn_up_exps.weight", 16, w.up}, {"ffn_down_exps.weight", 10, w.down},
		{"ffn_gate_shexp.weight", 8, w.sharedGate}, {"ffn_up_shexp.weight", 8, w.sharedUp}, {"ffn_down_shexp.weight", 8, w.sharedDown},
	} {
		name := fmt.Sprintf("blk.%d.%s", layer, entry.name)
		v, ok := views[name]
		t := entry.tensor
		if !ok || v.Type != entry.kind || t.typeID != v.Type || t.name != name || t.device == 0 || t.bytes != v.Bytes || t.fileOffset != v.Offset || !slices.Equal(t.dimensions, byName[name].Dimensions) {
			return fmt.Errorf("cuda shape FFN: invalid uploaded tensor %s", name)
		}
	}
	optional := w.bias
	name := fmt.Sprintf("blk.%d.exp_probs_b.bias", layer)
	if l.HashRouter {
		optional = w.hash
		name = fmt.Sprintf("blk.%d.ffn_gate_tid2eid.weight", layer)
	}
	_, hasOptional := views[name]
	if optional.device != 0 || l.HashRouter || hasOptional {
		v, ok := views[name]
		if !ok || optional.name != name || optional.device == 0 || optional.typeID != v.Type || optional.bytes != v.Bytes || optional.fileOffset != v.Offset || !slices.Equal(optional.dimensions, byName[name].Dimensions) {
			return fmt.Errorf("cuda shape FFN: invalid uploaded router table %s", name)
		}
	}
	return nil
}

// encodeShapeFFN consumes normalized actual State input and fills ffn_out.
// Caller must compile the router entry, own the borrowed scratch, and refuse
// nonfinite output before HC expansion. No callback stands in for arithmetic.
func encodeShapeFFN(state *cudaShapeForward, binding *dsv4shape.Binding, layer int, token uint32, router driver.Function, w shapeFFNWeights, buf shapeFFNBuffers) error {
	if state == nil || state.closed || router == 0 {
		return errors.New("cuda shape FFN: closed state or missing router")
	}
	if err := state.r.checkOpen(); err != nil {
		return err
	}
	c := state.plan.config
	if token >= uint32(c.Vocab) {
		return errors.New("cuda shape FFN: token outside vocabulary")
	}
	if err := validateShapeFFNWeights(c, binding, layer, w); err != nil {
		return err
	}
	l, _ := c.Layer(layer)
	if buf.logits.Len() != c.Experts || buf.probs.Len() != c.Experts || buf.weights.Len() != c.ExpertsUsed || buf.ids.Len() != c.ExpertsUsed || buf.gate.Len() != c.ExpertsUsed*c.ExpertFF || buf.up.Len() != buf.gate.Len() || buf.down.Len() != c.ExpertsUsed*c.Embed || buf.inputQ81.Len() != c.Embed/32*36 || buf.expertsQ81.Len() != c.ExpertsUsed*c.ExpertFF/32*36 || buf.sharedQ80.Len() != c.ExpertFF/32*36 || buf.invalid.Len() != 1 {
		return errors.New("cuda shape FFN: invalid borrowed scratch extents")
	}
	for _, entry := range []struct {
		name string
		n    int
	}{{"ffn_norm", c.Embed}, {"shared_gate", c.ExpertFF}, {"shared_up", c.ExpertFF}, {"ffn_shared", c.Embed}, {"ffn_out", c.Embed}} {
		if state.scratch[entry.name].Len() != entry.n {
			return errors.New("cuda shape FFN: invalid State scratch extents")
		}
	}
	ranges := []mxfp4DeviceRange{}
	for _, v := range []cudamem.Slice[float32]{buf.logits, buf.probs, buf.weights, buf.gate, buf.up, buf.down, state.scratch["ffn_norm"], state.scratch["shared_gate"], state.scratch["shared_up"], state.scratch["ffn_shared"], state.scratch["ffn_out"]} {
		ranges = append(ranges, mxfp4DeviceRange{uint64(v.Ptr()), uint64(v.Len()) * 4})
	}
	for _, v := range []cudamem.Slice[byte]{buf.inputQ81, buf.expertsQ81, buf.sharedQ80} {
		ranges = append(ranges, mxfp4DeviceRange{uint64(v.Ptr()), uint64(v.Len())})
	}
	ranges = append(ranges, mxfp4DeviceRange{uint64(buf.ids.Ptr()), uint64(buf.ids.Len()) * 4}, mxfp4DeviceRange{uint64(buf.invalid.Ptr()), 4})
	for _, t := range []cudaModelTensor{w.router, w.gate, w.up, w.down, w.sharedGate, w.sharedUp, w.sharedDown} {
		ranges = append(ranges, mxfp4DeviceRange{uint64(t.device), t.bytes})
	}
	if l.HashRouter {
		ranges = append(ranges, mxfp4DeviceRange{uint64(w.hash.device), w.hash.bytes})
	} else if w.bias.device != 0 {
		ranges = append(ranges, mxfp4DeviceRange{uint64(w.bias.device), w.bias.bytes})
	}
	if err := checkMXFP4DeviceRanges(ranges...); err != nil {
		return err
	}
	r := state.r
	if err := r.withContext(func() error {
		return r.stream().WithHandle(func(h driver.Stream) error { return buf.invalid.Ptr().MemsetD32Async(0, 1, h) })
	}); err != nil {
		return err
	}
	if err := r.encodeF16Matvec(w.router, state.scratch["ffn_norm"], buf.logits); err != nil {
		return err
	}
	hashMode, hasBias := uint32(0), uint32(0)
	var bias, hash driver.DevicePtr
	if l.HashRouter {
		hashMode = 1
		hash = w.hash.device
	} else if w.bias.device != 0 {
		hasBias = 1
		bias = w.bias.device
	}
	if err := r.withContext(func() error {
		return cudalaunch.Kernel(router).Grid(1, 1, 1).Block(1, 1, 1).On(r.stream()).Launch(buf.ids, buf.weights, buf.probs, bias, hash, buf.logits, driver.DevicePtr(0), int32(token), uint32(c.Vocab), uint32(1), hasBias, hashMode, uint32(c.Experts), c.ExpertWeightScale, buf.invalid)
	}); err != nil {
		return err
	}
	// Invalid route data must not be fed to any expert, even if the expert
	// kernels have their own bounds checks. This bounded route drains here.
	if err := r.withContext(func() error { return syncStream(r.stream()) }); err != nil {
		return err
	}
	var invalid [1]uint32
	if err := buf.invalid.CopyToHost(invalid[:]); err != nil {
		return err
	}
	if invalid[0] != 0 {
		return errors.New("cuda shape FFN: invalid or nonfinite router")
	}
	if err := r.encodeQuantQ81(state.scratch["ffn_norm"], buf.inputQ81, c.Embed); err != nil {
		return err
	}
	if err := r.encodeQ80MatvecQ81(w.sharedGate, buf.inputQ81, state.scratch["shared_gate"]); err != nil {
		return err
	}
	if err := r.encodeQ80MatvecQ81(w.sharedUp, buf.inputQ81, state.scratch["shared_up"]); err != nil {
		return err
	}
	if err := r.encodeSwiGLUQuantQ80(state.scratch["shared_gate"], state.scratch["shared_up"], buf.sharedQ80, c.ExpertFF, 1); err != nil {
		return err
	}
	if err := r.encodeQ80MatvecQuantized(w.sharedDown, buf.sharedQ80, state.scratch["ffn_shared"]); err != nil {
		return err
	}
	if err := r.encodeMatvecIQ2SelectedQ81(w.gate, w.up, buf.ids, buf.inputQ81, buf.gate, buf.up); err != nil {
		return err
	}
	if err := r.encodeSwiGLUQuantQ81(buf.gate, buf.up, buf.weights, buf.expertsQ81, c.ExpertFF, c.ExpertsUsed); err != nil {
		return err
	}
	if err := r.encodeMatvecQ2KSelectedQ81(w.down, buf.ids, buf.expertsQ81, buf.down); err != nil {
		return err
	}
	return r.encodeSumRoutedShared(buf.down, state.scratch["ffn_shared"], state.scratch["ffn_out"], c.Embed)
}

// FFNPhase runs the concrete FFN between the HC phase consumers. Until a
// device finite-reduction gate is qualified, it reads the whole FFN output
// before expansion; this conservative gate is not a performance claim.
func (s *cudaShapeHCMix) FFNPhase(hc *cudaShapeHC, state *cudaShapeForward, binding *dsv4shape.Binding, layer int, token uint32, router driver.Function, w shapeFFNWeights, buf shapeFFNBuffers, rmsEpsilon, hcEpsilon float32) error {
	if err := s.BeforePhase(hc, state, layer, "ffn", rmsEpsilon, hcEpsilon); err != nil {
		return err
	}
	if err := encodeShapeFFN(state, binding, layer, token, router, w, buf); err != nil {
		return err
	}
	if err := readShapeFiniteOutput(state, "ffn_out"); err != nil {
		return err
	}
	return s.AfterPhase(hc, state, layer, "ffn")
}
