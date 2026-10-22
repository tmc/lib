package cudamega

import (
	"fmt"
	"math"
)

// MoEInput supplies the real expert slab, row scales, hidden vector and
// routing coefficients. BindMoE copies retained slices. The slab preserves
// all three global layer slots; the current ready layers are zero and two.
type MoEInput struct {
	Config  MoEWeightConfig
	Layer   int
	Weights []byte
	Scales  []float32
	Hidden  []float32
	Routing [2]float32
}

// MoEBinding is an immutable host input snapshot. Its zero value is invalid.
// It contains no device allocation or lifetime claim.
type MoEBinding struct{ data *MoEInput }

// BindMoE validates and snapshots one finite typed invocation before workers
// or output buffers are created. Nonfinite values and conservatively bounded
// FP32 overflow are refused. This does not execute the expert activation.
func BindMoE(input MoEInput) (MoEBinding, error) {
	if err := input.Config.validate(); err != nil {
		return MoEBinding{}, err
	}
	c := input.Config
	if input.Layer != 0 && input.Layer != 2 {
		return MoEBinding{}, fmt.Errorf("MoE layer %d is not ready", input.Layer)
	}
	rows := c.Layers * c.Experts * 2 * c.Intermediate
	if len(input.Weights) != rows*c.Hidden || len(input.Scales) != rows || len(input.Hidden) != c.Hidden {
		return MoEBinding{}, fmt.Errorf("MoE input shape differs from its configuration")
	}
	for _, raw := range input.Weights {
		if raw&127 == 127 {
			return MoEBinding{}, fmt.Errorf("MoE weights contain E4M3FN NaN")
		}
	}
	for _, values := range [][]float32{input.Scales, input.Hidden, input.Routing[:]} {
		for _, x := range values {
			if !moeFinite(x) {
				return MoEBinding{}, fmt.Errorf("MoE input contains a nonfinite value")
			}
		}
	}
	// Nonnegative RN sums bound absolute signed RN sums by monotonicity.
	// SiLU's magnitude is at most the magnitude of its finite argument.
	// Checking before worker startup avoids a lane abandoning a barrier.
	for n := 0; n < c.Intermediate; n++ {
		var mixture float32
		for expert := 0; expert < c.Experts; expert++ {
			row := (input.Layer*c.Experts+expert)*2*c.Intermediate + (n/8)*16 + n%8
			var gate, up float32
			for h, x := range input.Hidden {
				gate = float32(gate + float32(moeAbs(moeDecode(input.Weights[row*c.Hidden+h]))*moeAbs(x)))
				up = float32(up + float32(moeAbs(moeDecode(input.Weights[(row+8)*c.Hidden+h]))*moeAbs(x)))
			}
			gate = float32(gate * moeAbs(input.Scales[row]))
			up = float32(up * moeAbs(input.Scales[row+8]))
			bound := float32(gate * up)
			mixture = float32(mixture + float32(bound*moeAbs(input.Routing[expert])))
			if !moeFinite(gate) || !moeFinite(up) || !moeFinite(bound) || !moeFinite(mixture) {
				return MoEBinding{}, fmt.Errorf("MoE FP32 magnitude bound overflows")
			}
		}
	}
	input.Weights = append([]byte(nil), input.Weights...)
	input.Scales = append([]float32(nil), input.Scales...)
	input.Hidden = append([]float32(nil), input.Hidden...)
	return MoEBinding{data: &input}, nil
}

// Snapshot returns a detached copy of the checked invocation. A performing
// backend calls it once before starting workers and retains that private copy.
func (b MoEBinding) Snapshot() (MoEInput, error) {
	if b.data == nil {
		return MoEInput{}, fmt.Errorf("unbound MoE inputs")
	}
	input := *b.data
	input.Weights = append([]byte(nil), input.Weights...)
	input.Scales = append([]float32(nil), input.Scales...)
	input.Hidden = append([]float32(nil), input.Hidden...)
	return input, nil
}

func moeFinite(x float32) bool { return !math.IsNaN(float64(x)) && !math.IsInf(float64(x), 0) }
func moeAbs(x float32) float32 { return float32(math.Abs(float64(x))) }

// Used only for the preflight magnitude bound, not by an execution backend.
func moeDecode(raw byte) float32 {
	magnitude := uint32(raw & 127)
	e, m := magnitude>>3, magnitude&7
	var v float32
	if e == 0 {
		v = float32(m) * 0.001953125
	} else {
		v = math.Float32frombits((e+120)<<23 | m<<20)
	}
	return math.Float32frombits(math.Float32bits(v) | uint32(raw&128)<<24)
}
