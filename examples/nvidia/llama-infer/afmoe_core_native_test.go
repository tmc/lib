package main

import (
	"encoding/json"
	"math"
	"os"
	"testing"
)

// TestAFMoENativeCore loads the common module rather than relying on the
// attention/MoE gates to cover it. The shared device guard skips ordinary
// host tests before CUDA initialization. All expected values below are
// independent fixture values or exact binary-arithmetic witnesses.
func TestAFMoENativeCore(t *testing.T) {
	dev := newAFMoENativeDevice(t)
	m, err := compileAfmoeCoreModule(dev)
	if m != nil {
		t.Cleanup(func() {
			if err := m.Close(); err != nil {
				t.Error(err)
			}
		})
	}
	if err != nil {
		t.Fatal(err)
	}
	emit := afmoeNativeEmit(dev)
	weight := func(t *testing.T, values []float32) deviceWeight {
		t.Helper()
		return deviceWeight{buf: afmoeNativeUpload(t, dev, roundBF16(values)), dtype: weightBF16}
	}

	t.Run("mup_independent_fixture", func(t *testing.T) {
		var fixture struct {
			Embedding struct {
				Input    afmoeAttentionTensor `json:"embedding"`
				Hidden   int                  `json:"hidden_size"`
				Expected afmoeAttentionTensor
			} `json:"mup_embedding"`
		}
		data, err := os.ReadFile("testdata/afmoe-mini/reference.json")
		if err != nil {
			t.Fatal(err)
		}
		if err := json.Unmarshal(data, &fixture); err != nil {
			t.Fatal(err)
		}
		f := fixture.Embedding
		if f.Hidden != 6 || len(f.Input.Data) != f.Hidden || len(f.Expected.Data) != f.Hidden {
			t.Fatal("invalid independent embedding fixture")
		}
		// A distinct first row also checks selection of token1.
		table := make([]float32, 2*f.Hidden)
		for i := range table[:f.Hidden] {
			table[i] = 7
		}
		copy(table[f.Hidden:], f.Input.Data)
		w := weight(t, table)
		out := afmoeNativeUpload(t, dev, make([]float32, f.Hidden))
		if err := m.emitEmbedding(w, out, 1, 2, f.Hidden, float32(math.Sqrt(float64(f.Hidden))), emit); err != nil {
			t.Fatal(err)
		}
		afmoeNativeSync(t, dev)
		checkAfmoeAttentionValues(t, afmoeNativeDownload(t, out), f.Expected.Data)
	})

	t.Run("mup_hidden2048_scalar_discriminator", func(t *testing.T) {
		const hidden = 2048
		table, want := make([]float32, 2*hidden), make([]float32, hidden)
		table[hidden], table[hidden+1], table[len(table)-1] = 0.0654296875, -0.0654296875, 1
		// Target PyTorch2.13 uses an FP32 opmath scalar for BF16*sqrt(H).
		// Prematurely rounding sqrt2048 to BF16 gives2.953125 instead.
		// These values were independently checked on target CPU; its CUDA
		// MulFunctor/OpMathType/scalar-loading source has the same boundary.
		want[0], want[1], want[hidden-1] = 2.96875, -2.96875, 45.25
		out := afmoeNativeUpload(t, dev, make([]float32, hidden))
		if err := m.emitEmbedding(weight(t, table), out, 1, 2, hidden, float32(math.Sqrt(hidden)), emit); err != nil {
			t.Fatal(err)
		}
		afmoeNativeSync(t, dev)
		checkAfmoeAttentionValues(t, afmoeNativeDownload(t, out), want)
	})

	t.Run("linear_exact_BF16_weights", func(t *testing.T) {
		// Binary fractions make all four dot products exact, independent of
		// the reduction tree and any CPU model implementation.
		w := weight(t, []float32{1, 2, 0.5, -0.25, 0.25, -0.5, 1, 0.5, -1, 0.25, -0.5, 2, 0.03125, 0.0625, -0.125, 0.25})
		in := afmoeNativeUpload(t, dev, []float32{1, 0.5, -2, 4})
		out := afmoeNativeUpload(t, dev, make([]float32, 4))
		if err := m.emitLinear(w, in, out, 4, 4, true, emit); err != nil {
			t.Fatal(err)
		}
		afmoeNativeSync(t, dev)
		checkAfmoeAttentionValues(t, afmoeNativeDownload(t, out), []float32{0, 0, 8.125, 1.3125})
	})

	t.Run("lm_head_round_before_argmax", func(t *testing.T) {
		w := weight(t, []float32{1, 1.0 / 1024, 1, 3.0 / 1024})
		in := afmoeNativeUpload(t, dev, []float32{1, 1})
		out := afmoeNativeUpload(t, dev, make([]float32, 2))
		token := afmoeNativeUpload(t, dev, []int32{-9})
		if err := m.emitLinear(w, in, out, 2, 2, false, emit); err != nil {
			t.Fatal(err)
		}
		afmoeNativeSync(t, dev)
		checkAfmoeAttentionValues(t, afmoeNativeDownload(t, out), []float32{1.0009765625, 1.0029296875})
		// Both exact sums round to BF16 1. The reference BF16 lm_head must
		// therefore select ID0, whereas argmax on the FP32 sums selects ID1.
		if err := m.emitLinear(w, in, out, 2, 2, true, emit); err != nil {
			t.Fatal(err)
		}
		if err := m.emitArgmax(out, token, emit); err != nil {
			t.Fatal(err)
		}
		afmoeNativeSync(t, dev)
		checkAfmoeAttentionValues(t, afmoeNativeDownload(t, out), []float32{1, 1})
		if got := afmoeNativeDownload(t, token)[0]; got != 0 {
			t.Errorf("rounded head argmax %d, want0", got)
		}
	})

	t.Run("argmax_lowest_ID_across_threads", func(t *testing.T) {
		values := make([]float32, 520)
		for i := range values {
			values[i] = -1
		}
		values[17], values[257], values[513] = 9, 9, 9
		logits := afmoeNativeUpload(t, dev, values)
		token := afmoeNativeUpload(t, dev, []int32{-9})
		if err := m.emitArgmax(logits, token, emit); err != nil {
			t.Fatal(err)
		}
		afmoeNativeSync(t, dev)
		if got := afmoeNativeDownload(t, token)[0]; got != 17 {
			t.Errorf("tied argmax %d, want17", got)
		}
	})

	t.Run("argmax_rejects_nonfinite", func(t *testing.T) {
		for _, bad := range []float32{float32(math.NaN()), float32(math.Inf(1)), float32(math.Inf(-1))} {
			logits := afmoeNativeUpload(t, dev, []float32{1, bad, 2})
			token := afmoeNativeUpload(t, dev, []int32{-9})
			if err := m.emitArgmax(logits, token, emit); err != nil {
				t.Fatal(err)
			}
			afmoeNativeSync(t, dev)
			if got := afmoeNativeDownload(t, token)[0]; got != -1 {
				t.Errorf("nonfinite %g argmax %d, want-1", bad, got)
			}
		}
	})

	t.Run("residual_rounding", func(t *testing.T) {
		for _, alias := range []string{"separate", "alias_x", "alias_y"} {
			t.Run(alias, func(t *testing.T) {
				x := afmoeNativeUpload(t, dev, []float32{1, 1, -1, 256})
				y := afmoeNativeUpload(t, dev, []float32{1.0 / 256, 1.0 / 128, 1.0 / 256, 1})
				out := afmoeNativeUpload(t, dev, make([]float32, 4))
				if alias == "alias_x" {
					out = x
				} else if alias == "alias_y" {
					out = y
				}
				if err := m.emitResidual(out, x, y, emit); err != nil {
					t.Fatal(err)
				}
				afmoeNativeSync(t, dev)
				// Round-to-nearest-even at1+halfULP and256+halfULP;
				// the other two exact sums are representable BF16 values.
				checkAfmoeAttentionValues(t, afmoeNativeDownload(t, out), []float32{1, 1.0078125, -0.99609375, 256})
			})
		}
	})
}
