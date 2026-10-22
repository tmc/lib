package main

import (
	"fmt"
	"math"

	"github.com/tmc/lib/gguf"
)

// swigluClampExp is the symmetric bound applied to the gate and up halves of
// the SwiGLU activation: gate is clamped above at swigluClampExp, up to
// [-swigluClampExp, +swigluClampExp].
//
// This is not a free choice. In the C reference it is a per-shape field,
// ds4.c:956 DS4_SWIGLU_CLAMP_EXP = g_ds4_shape.swiglu_clamp_exp, whose default
// is 10.0f (ds4.c:408) but which the GLM 5.2 preset sets to 0.0f, i.e. clamping
// disabled (ds4.c:747). The value is also carried in the model file, and the
// reference validates the file against the shape on load
// (validate_swiglu_clamp_metadata, ds4.c:6474). ds4-min is compiled for one
// shape and one pinned checkpoint, so it keeps the value as a constant and
// checks the file against it in validateSwiGLUClamp.
const swigluClampExp = 10.0

// swigluClampKey is the GGUF metadata key holding the per-layer clamp.
const swigluClampKey = "deepseek4.swiglu_clamp_exp"

// validateSwiGLUClamp reports whether the model's per-layer SwiGLU clamp
// metadata agrees with swigluClampExp, mirroring the C reference's
// validate_swiglu_clamp_metadata (ds4.c:6474): the key must be present, be a
// float32 or float64 array, hold at least layers entries, and every one of the
// first layers entries must equal swigluClampExp to the reference's tolerance.
//
// On the only checkpoint ds4-min runs, this check passes vacuously: that file
// carries 43 entries, all exactly 10.0, which is what the kernels already
// assume. It fires only against a model whose metadata disagrees, and its worth
// is entirely prospective -- it converts a silent miscomputation into a load
// error. It is not evidence that anything currently in the tree is correct.
func validateSwiGLUClamp(g *gguf.File, layers int) error {
	v, ok := g.Metadata[swigluClampKey]
	if !ok || v.TypeID != gguf.TypeArray {
		return fmt.Errorf("model metadata %q is missing or not an array", swigluClampKey)
	}
	values, ok := v.Value.([]any)
	if !ok {
		return fmt.Errorf("model metadata %q is not an array", swigluClampKey)
	}
	if len(values) < layers {
		return fmt.Errorf("model metadata %q has %d entries, want at least %d", swigluClampKey, len(values), layers)
	}
	for i, value := range values[:layers] {
		var got float32
		switch n := value.(type) {
		case float32:
			got = n
		case float64:
			got = float32(n)
		default:
			return fmt.Errorf("model metadata %q layer %d is %T, want a float", swigluClampKey, i, value)
		}
		if !swigluClampEqual(got, swigluClampExp) {
			return fmt.Errorf("model metadata %q layer %d is %.9g, want %.9g", swigluClampKey, i, got, float32(swigluClampExp))
		}
	}
	return nil
}

// swigluClampEqual reports whether got matches want within the tolerance the C
// reference uses for float metadata, config_expect_f32 (ds4.c:6513): a relative
// 1e-6, with the scale floored at one.
func swigluClampEqual(got, want float32) bool {
	scale := float64(want)
	if scale < 0 {
		scale = -scale
	}
	if scale < 1 {
		scale = 1
	}
	return math.Abs(float64(got)-float64(want)) <= scale*1e-6
}
