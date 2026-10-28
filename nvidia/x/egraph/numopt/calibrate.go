package numopt

import (
	"fmt"
	"math"
	"slices"
)

// Calibrate computes the sensitivity statistics of one K-tile from its
// weights and calibration data.
//
// acts holds the activations that multiply the tile, tokens rows of equal
// length, and grads the output gradients for the same tokens, also tokens
// rows. HessianTrace is the mean over tokens of ‖x_t‖²·‖g_t‖², the diagonal
// empirical Fisher of the tile summed over its inputs. grads may be nil, in
// which case ‖g_t‖² is taken as 1 and the statistic is activation energy
// alone, as in activation-aware weight quantization.
func Calibrate(id string, weights, acts, grads []float32, tokens int) (TileSensitivity, error) {
	if len(weights) == 0 {
		return TileSensitivity{}, fmt.Errorf("calibrate %q: no weights", id)
	}
	if tokens <= 0 || len(acts) == 0 || len(acts)%tokens != 0 {
		return TileSensitivity{}, fmt.Errorf("calibrate %q: %d activations do not form %d tokens", id, len(acts), tokens)
	}
	if grads != nil && (len(grads) == 0 || len(grads)%tokens != 0) {
		return TileSensitivity{}, fmt.Errorf("calibrate %q: %d gradients do not form %d tokens", id, len(grads), tokens)
	}
	for _, v := range slices.Concat(weights, acts, grads) {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			return TileSensitivity{}, fmt.Errorf("calibrate %q: non-finite input", id)
		}
	}
	t := TileSensitivity{TileID: id, Elements: len(weights), NonNegativeAct: true}

	ka := len(acts) / tokens
	var h float64
	for i := range tokens {
		x := sumSquares(acts[i*ka : (i+1)*ka])
		g := 1.0
		if grads != nil {
			kg := len(grads) / tokens
			g = sumSquares(grads[i*kg : (i+1)*kg])
		}
		h += x * g
	}
	t.HessianTrace = h / float64(tokens)

	var mean, m2, m4 float64
	for _, w := range weights {
		mean += float64(w)
	}
	mean /= float64(len(weights))
	for _, w := range weights {
		d := float64(w) - mean
		m2 += d * d
		m4 += d * d * d * d
	}
	if m2 > 0 {
		t.WeightKurtosis = m4 * float64(len(weights)) / (m2 * m2)
	}

	var mags []float64
	for _, x := range acts {
		if x < 0 {
			t.NonNegativeAct = false
		}
		if x != 0 {
			mags = append(mags, math.Abs(float64(x)))
		}
	}
	if len(mags) > 0 {
		slices.Sort(mags)
		if med := mags[len(mags)/2]; med > 0 {
			t.ActDynamicRange = 20 * math.Log10(mags[len(mags)-1]/med)
		}
	}
	return t, nil
}

func sumSquares(v []float32) float64 {
	var s float64
	for _, x := range v {
		s += float64(x) * float64(x)
	}
	return s
}
