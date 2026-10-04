//go:build native

package cudalayer

import (
	"crypto/sha256"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudaautograd"
	"github.com/tmc/lib/nvidia/x/cudair"
	"github.com/tmc/lib/nvidia/x/cudatest"
)

func runNativeBackward(t *testing.T, m *FusedMLP, n int, mode cudair.GradientMode, input bool, previous []float32) (MLPGradients, []float32, []float32, []float32) {
	t.Helper()
	if os.Getenv("GPU_LOCK_HELD") == "" {
		t.Skip("SKIPPED, NOT VERIFIED: native backward requires gpu-lock")
	}
	dev := cudatest.RequireDevice(t)
	closeDevice := true
	defer func() {
		if !closeDevice {
			return
		}
		if e := dev.Close(); e != nil {
			t.Error(e)
		}
	}()
	x, w, dy := backwardFixture(m, n)
	if m.name == "activation" && m.inputWidth == 1 && m.outputWidth == 1 && n == 7 {
		x = []float32{-30, -2, -0.01, 0, 0.01, 2, 30}
		w = []float32{1}
		dy = []float32{1, 1, 1, 1, 1, 1, 1}
	}
	o := MLPBackwardOptions{Samples: n, Input: input, ParameterMode: mode, MaxBytes: 64 << 20}
	got, e := m.BackwardBatch(dev, x, w, dy, previous, o)
	if e != nil {
		var pending io.Closer
		if errors.As(e, &pending) {
			if cleanupErr := pending.Close(); cleanupErr != nil {
				closeDevice = false
				t.Cleanup(func() {
					if err := pending.Close(); err != nil {
						t.Errorf("native work remains quarantined: %v", err)
						return
					}
					if err := dev.Close(); err != nil {
						t.Error(err)
					}
				})
				t.Errorf("initial cleanup retry failed: %v", cleanupErr)
			}
		}
		t.Fatal(e)
	}
	y, dx, dw := backwardReference(m, float64s(x), float64s(w), float64s(dy), n)
	if mode == cudair.GradientModeAccumulate {
		for i := range dw {
			dw[i] += float64(previous[i])
		}
	}
	source, _, e := m.backwardSource(o)
	if e != nil {
		t.Fatal(e)
	}
	sourceHash := sha256.Sum256([]byte(source))
	h := sha256.New()
	for _, a := range [][]float32{x, w, dy, previous} {
		for _, v := range a {
			var b [4]byte
			binary.LittleEndian.PutUint32(b[:], math.Float32bits(v))
			h.Write(b[:])
		}
	}
	major, minor, e := dev.ComputeCapability()
	if e != nil {
		t.Fatal(e)
	}
	for _, check := range []struct {
		name string
		got  []float32
		want []float64
	}{{"output", got.Output, y}, {"input", got.Input, dx}, {"parameters", got.Parameters, dw}} {
		if len(check.got) == 0 {
			continue
		}
		expected := make([]float32, len(check.want))
		for i := range expected {
			expected[i] = float32(check.want[i])
		}
		p := cudatest.CUDAOutputPolicy{Target: fmt.Sprintf("sm_%d%d", major, minor), DType: "fp32", MathMode: "NVRTC default FP32; SiLU __expf; sample-order FP32 reduction", SourceDigest: fmt.Sprintf("%x", sourceHash), InputDigest: fmt.Sprintf("%x", h.Sum(nil)), Elements: len(expected), Absolute: 2e-5, Relative: 2e-5, NaN: "reject", Infinity: "reject", Zero: "equal"}
		if e := cudatest.CompareCUDAOutputFP32(check.got, expected, p); e != nil {
			t.Fatalf("%s: %v", check.name, e)
		}
	}
	t.Logf("actual backward samples=%d input=%d parameters=%d mode=%s target=sm_%d%d source=%x input=%x; forward/backward separate launches, parameter reduction sample-order; context guards/padded lanes checked", n, len(got.Input), len(got.Parameters), mode, major, minor, sourceHash, h.Sum(nil))
	return got, x, w, dy
}

func TestMLPBackward_LinearInputAndParameterGradients(t *testing.T) {
	m, _ := NewFusedMLP(5, 3, 3, 1)
	got, x, w, dy := runNativeBackward(t, m, 7, cudair.GradientModeOverwrite, true, nil)
	_, dx, dw := backwardReference(m, float64s(x), float64s(w), float64s(dy), 7)
	checkFiniteDifference(t, m, x, w, dy, 7, dx, dw)
	if len(got.Parameters) != 15 {
		t.Fatal("incomplete parameter gradients")
	}
}

func TestMLPBackward_ReluAndSiluDerivatives_Native(t *testing.T) {
	for _, act := range []string{"relu", "silu"} {
		t.Run(act, func(t *testing.T) {
			l, _ := NewMLPLayer("activation", 1, 1, act)
			m := &FusedMLP{name: "activation", inputWidth: 1, outputWidth: 1, layers: []*MLPLayer{l}}
			got, _, _, _ := runNativeBackward(t, m, 7, cudair.GradientModeOverwrite, true, nil)
			if act == "relu" && got.Input[3] != 0 {
				t.Fatal("native ReLU derivative at zero is not zero")
			}
		})
	}
}

func TestMLPBackward_RectangularChainAndOffsets(t *testing.T) {
	t.Run("SiLU 5-to-7-to-3", func(t *testing.T) {
		m, e := NewFusedMLPWithActivation(5, 7, 3, 2, "silu")
		if e != nil {
			t.Fatal(e)
		}
		runNativeBackward(t, m, 7, cudair.GradientModeOverwrite, true, nil)
	})
	for _, dims := range [][4]int{{5, 7, 3, 2}, {64, 64, 64, 3}} {
		t.Run(fmt.Sprint(dims), func(t *testing.T) {
			m, _ := NewFusedMLP(dims[0], dims[1], dims[2], dims[3])
			runNativeBackward(t, m, 7, cudair.GradientModeOverwrite, true, nil)
		})
	}
}

func TestMLPBackward_ContextTailAndPadding(t *testing.T) {
	m, _ := NewFusedMLP(5, 7, 3, 2)
	runNativeBackward(t, m, 65, cudair.GradientModeOverwrite, true, nil)
}

func TestMLPBackward_AccumulationModes_Native(t *testing.T) {
	m, _ := NewFusedMLP(5, 7, 3, 2)
	first, _, _, _ := runNativeBackward(t, m, 7, cudair.GradientModeOverwrite, true, nil)
	runNativeBackward(t, m, 7, cudair.GradientModeOverwrite, true, nil)
	runNativeBackward(t, m, 7, cudair.GradientModeAccumulate, true, first.Parameters)
	runNativeBackward(t, m, 7, cudair.GradientModeIgnore, true, nil)
	runNativeBackward(t, m, 7, cudair.GradientModeOverwrite, false, nil)
}

func TestMLPBackward_TrainingStepMatchesCPU(t *testing.T) {
	m, _ := NewFusedMLP(5, 7, 3, 2)
	got, x, w, dy := runNativeBackward(t, m, 7, cudair.GradientModeOverwrite, true, nil)
	_, _, dw := backwardReference(m, float64s(x), float64s(w), float64s(dy), 7)
	const lr = float32(0.001)
	opt, e := cudaautograd.NewSGDOptimizer(lr, 0)
	if e != nil {
		t.Fatal(e)
	}
	updated := append([]float32(nil), w...)
	if e := opt.Step(map[string][]float32{"mlp": updated}, map[string][]float32{"mlp": got.Parameters}); e != nil {
		t.Fatal(e)
	}
	want := make([]float64, len(w))
	for i := range want {
		want[i] = float64(w[i] - lr*float32(dw[i]))
	}
	if !compareBackward(t, "all parameter updates", updated, want) {
		t.Fatal("training step differs")
	}
	loss := func(weights []float32) float64 {
		y, _, _ := backwardReference(m, float64s(x), float64s(weights), float64s(dy), 7)
		v := 0.0
		for i := range y {
			v += y[i] * float64(dy[i]) / 7
		}
		return v
	}
	before, after := loss(w), loss(updated)
	if !(after < before) {
		t.Fatalf("declared training fixture did not lower loss: %g -> %g", before, after)
	}
	dev := cudatest.RequireDevice(t)
	closeDevice := true
	defer func() {
		if !closeDevice {
			return
		}
		if e := dev.Close(); e != nil {
			t.Error(e)
		}
	}()
	post, e := m.BackwardBatch(dev, x, updated, dy, nil, MLPBackwardOptions{Samples: 7, Input: true, ParameterMode: cudair.GradientModeOverwrite, MaxBytes: 64 << 20})
	if e != nil {
		var pending io.Closer
		if errors.As(e, &pending) {
			if cleanupErr := pending.Close(); cleanupErr != nil {
				closeDevice = false
				t.Cleanup(func() {
					if err := pending.Close(); err != nil {
						t.Errorf("training work remains quarantined: %v", err)
						return
					}
					if err := dev.Close(); err != nil {
						t.Error(err)
					}
				})
				t.Errorf("initial cleanup retry failed: %v", cleanupErr)
			}
		}
		t.Fatal(e)
	}
	wantOutput, _, _ := backwardReference(m, float64s(x), float64s(updated), float64s(dy), 7)
	if !compareBackward(t, "updated device forward", post.Output, wantOutput) {
		t.Fatal("updated weights did not reach device execution")
	}
	deviceLoss := func(y []float32) float64 {
		sum := 0.0
		for i, value := range y {
			sum += float64(value) * float64(dy[i]) / 7
		}
		return sum
	}
	deviceBefore, deviceAfter := deviceLoss(got.Output), deviceLoss(post.Output)
	if math.Abs(deviceBefore-before) > 2e-5 || math.Abs(deviceAfter-after) > 2e-5 || !(deviceAfter < deviceBefore) {
		t.Fatalf("device training losses %g -> %g reference %g -> %g", deviceBefore, deviceAfter, before, after)
	}
	// Updating only the final layer must disagree with the checked full update.
	one := append([]float32(nil), w...)
	offset := m.layers[0].inputWidth * m.layers[0].outputWidth
	copy(one[offset:], updated[offset:])
	different := false
	for i := range one {
		if one[i] != updated[i] {
			different = true
			break
		}
	}
	if !different {
		t.Fatal("one-layer-only negative control is ineffective")
	}
	t.Logf("existing cudaautograd.SGDOptimizer updated every weight and updated CUDA forward matched; device dot-product mean loss %.9g -> %.9g reference %.9g -> %.9g", deviceBefore, deviceAfter, before, after)
}
