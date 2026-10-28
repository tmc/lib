package numopt

import (
	"math"
	"math/rand/v2"
	"testing"
)

func TestCalibrate(t *testing.T) {
	r := rand.New(rand.NewPCG(1, 1))
	gauss := make([]float32, 1<<16)
	heavy := make([]float32, 1<<16)
	for i := range gauss {
		gauss[i] = float32(r.NormFloat64())
		// A Gaussian scale mixture: 5% of weights ten times wider.
		heavy[i] = gauss[i]
		if r.IntN(20) == 0 {
			heavy[i] *= 10
		}
	}
	relu := []float32{0, 1, 2, 3, 0, 4}
	signed := []float32{-1, 1, 2, 3, 0, 4}

	tg, err := Calibrate("g", gauss, relu, nil, 2)
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(tg.WeightKurtosis-3) > 0.1 {
		t.Errorf("Gaussian kurtosis = %v, want about 3", tg.WeightKurtosis)
	}
	if !tg.NonNegativeAct {
		t.Error("ReLU activations reported signed")
	}
	// Tokens [0 1 2] and [3 0 4]: ‖x‖² = 5 and 25, no gradients: mean 15.
	if tg.HessianTrace != 15 {
		t.Errorf("HessianTrace = %v, want 15", tg.HessianTrace)
	}
	// Non-zero magnitudes 1 2 3 4: max 4, median 3.
	if want := 20 * math.Log10(4.0/3); math.Abs(tg.ActDynamicRange-want) > 1e-12 {
		t.Errorf("ActDynamicRange = %v, want %v", tg.ActDynamicRange, want)
	}
	if tg.Elements != len(gauss) {
		t.Errorf("Elements = %d", tg.Elements)
	}

	th, err := Calibrate("h", heavy, signed, []float32{1, 1, 2, 0}, 2)
	if err != nil {
		t.Fatal(err)
	}
	if th.WeightKurtosis < 10 {
		t.Errorf("mixture kurtosis = %v, want well above 3", th.WeightKurtosis)
	}
	if th.NonNegativeAct {
		t.Error("signed activations reported non-negative")
	}
	// ‖x‖² = 6, 25; ‖g‖² = 2, 4: mean (12 + 100)/2.
	if th.HessianTrace != 56 {
		t.Errorf("HessianTrace = %v, want 56", th.HessianTrace)
	}

	for _, tt := range []struct {
		name    string
		w, x, g []float32
		tokens  int
	}{
		{"no weights", nil, relu, nil, 2},
		{"ragged acts", gauss, relu, nil, 4},
		{"ragged grads", gauss, relu, []float32{1, 2, 3}, 2},
		{"nan", []float32{float32(math.NaN())}, relu, nil, 2},
	} {
		if _, err := Calibrate("x", tt.w, tt.x, tt.g, tt.tokens); err == nil {
			t.Errorf("%s: no error", tt.name)
		}
	}
}
