package main

import (
	"encoding/json"
	"errors"
	"math"
	"os"
	"reflect"
	"testing"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudalaunch"
	"github.com/tmc/lib/nvidia/x/cudamem"
)

// newAFMoENativeDevice must be called before any native-test CUDA operation.
// An ordinary host test never initializes CUDA. Once explicitly enabled,
// unavailable or broken CUDA is a failure, not a skipped native gate.
func newAFMoENativeDevice(t *testing.T) *cuda.Device {
	t.Helper()
	if os.Getenv("LLAMA_AFMOE_NATIVE_TEST") != "1" {
		t.Skip("LLAMA_AFMOE_NATIVE_TEST is not 1: native AFMoE not executed")
	}
	if err := driver.Load(""); err != nil {
		t.Fatal(err)
	}
	if err := cuda.Init(); err != nil {
		t.Fatal(err)
	}
	dev, err := cuda.NewDevice(0)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := dev.Close(); err != nil {
			t.Error(err)
		}
	})
	return dev
}

func afmoeNativeUpload[T any](t *testing.T, dev *cuda.Device, values []T) cudamem.Slice[T] {
	t.Helper()
	s, err := cudamem.Alloc[T](dev, len(values))
	t.Cleanup(func() {
		if err := s.Free(); err != nil {
			t.Error(err)
		}
	})
	if err != nil {
		t.Fatal(err)
	}
	if err := s.CopyFromHost(values); err != nil {
		t.Fatal(err)
	}
	return s
}

func afmoeNativeDownload[T any](t *testing.T, s cudamem.Slice[T]) []T {
	t.Helper()
	values := make([]T, s.Len())
	if err := s.CopyToHost(values); err != nil {
		t.Fatal(err)
	}
	return values
}

func afmoeNativeEmit(dev *cuda.Device) afmoeKernelEmitter {
	return func(l *cudalaunch.Launcher, args ...any) error {
		return dev.WithContext(func() error { return l.Launch(args...) })
	}
}

func afmoeNativeSync(t *testing.T, dev *cuda.Device) {
	t.Helper()
	if err := dev.WithContext(driver.CtxSynchronize); err != nil {
		t.Fatal(err)
	}
}

func afmoeNativeMoEKernels(t *testing.T, dev *cuda.Device) *afmoeMoEKernels {
	t.Helper()
	m, err := compileAFMoEMoEKernels(dev)
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
	return m
}

func TestAFMoENativeComponents(t *testing.T) {
	dev := newAFMoENativeDevice(t)
	m := afmoeNativeMoEKernels(t, dev)
	t.Run("router128", func(t *testing.T) { afmoeNativeRouter(t, dev, m) })
	t.Run("nonzero independent router", func(t *testing.T) { afmoeNativeRouterFixture(t, dev, m) })
	t.Run("combine fixture", func(t *testing.T) { afmoeNativeCombine(t, dev, m) })
	t.Run("invalid IDs fault before pointer access", func(t *testing.T) { afmoeNativeInvalidIDs(t, dev, m) })
	t.Run("128 pointer tables", func(t *testing.T) { afmoeNativePointerTables(t, dev, m) })
	t.Run("dense independent fixture", func(t *testing.T) { afmoeNativeDenseMLP(t, dev, m) })
}

func afmoeNativeRouterFixture(t *testing.T, dev *cuda.Device, m *afmoeMoEKernels) {
	type tensor struct{ Data []float32 }
	var fixture struct {
		Router struct {
			Logits, Bias, Weights tensor
			IDs                   []int `json:"selected_ids"`
		} `json:"router"`
	}
	data, err := os.ReadFile("testdata/afmoe_moe.json")
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(data, &fixture); err != nil {
		t.Fatal(err)
	}
	f := fixture.Router
	if len(f.Logits.Data) != 10 || len(f.Bias.Data) != 10 || len(f.IDs) != 8 || len(f.Weights.Data) != 8 {
		t.Fatal("invalid independent router fixture")
	}
	logits, bias, matrix := make([]float32, 128), make([]float32, 128), make([]float32, 128*4)
	for e := range logits {
		// Padding cannot enter the top8: all10 original selection scores are
		// greater than-1, while these have selection score below-15.
		logits[e], bias[e] = -16, -16
		if e < 10 {
			logits[e], bias[e] = f.Logits.Data[e], f.Bias.Data[e]
		}
		matrix[e*4] = logits[e]
	}
	x := afmoeNativeUpload(t, dev, []float32{1, 0, 0, 0})
	w := afmoeNativeUpload(t, dev, roundBF16(matrix))
	l := afmoeNativeUpload(t, dev, make([]float32, 128))
	b := afmoeNativeUpload(t, dev, bias)
	ids := afmoeNativeUpload(t, dev, make([]int32, 8))
	weights := afmoeNativeUpload(t, dev, make([]float32, 8))
	fault := afmoeNativeUpload(t, dev, []int32{0})
	emit := afmoeNativeEmit(dev)
	if err := emit(cudalaunch.Kernel(m.routerLinearFn).Grid(128, 1, 1).Block(128, 1, 1), x, w, l, int32(4), int32(128)); err != nil {
		t.Fatal(err)
	}
	if err := emit(cudalaunch.Kernel(m.routeFn).Grid(1, 1, 1).Block(1, 1, 1), l, b, ids, weights, fault, int32(128), int32(8), int32(1), float32(1e-20), float32(2.826)); err != nil {
		t.Fatal(err)
	}
	afmoeNativeSync(t, dev)
	if got := afmoeNativeDownload(t, l); !reflect.DeepEqual(got, logits) {
		t.Errorf("router one-hot projection %v, want exact fixture logits %v", got, logits)
	}
	if got := afmoeNativeDownload(t, fault)[0]; got != 0 {
		t.Fatalf("router fixture fault %d", got)
	}
	for k, id := range afmoeNativeDownload(t, ids) {
		if id != int32(f.IDs[k]) {
			t.Errorf("fixture route%d ID%d, want%d", k, id, f.IDs[k])
		}
	}
	for k, weight := range afmoeNativeDownload(t, weights) {
		// Prospective absolute tolerance permits FP32 expf rounding relative
		// to the independent scalar fixture; it does not excuse biased weights.
		if !afmoeFinite(weight) || math.Abs(float64(weight-f.Weights.Data[k])) > 2e-7 {
			t.Errorf("fixture route%d weight%g, want%g (abs2e-7)", k, weight, f.Weights.Data[k])
		}
	}
}

func afmoeNativeRouter(t *testing.T, dev *cuda.Device, m *afmoeMoEKernels) {
	logits, bias := make([]float32, 128), make([]float32, 128)
	for e := range bias {
		bias[e] = float32(e) / 128
	}
	l := afmoeNativeUpload(t, dev, logits)
	b := afmoeNativeUpload(t, dev, bias)
	ids := afmoeNativeUpload(t, dev, make([]int32, 8))
	weights := afmoeNativeUpload(t, dev, make([]float32, 8))
	fault := afmoeNativeUpload(t, dev, []int32{0})
	if err := afmoeNativeEmit(dev)(cudalaunch.Kernel(m.routeFn).Grid(1, 1, 1).Block(1, 1, 1), l, b, ids, weights, fault, int32(128), int32(8), int32(1), float32(1e-20), float32(2.826)); err != nil {
		t.Fatal(err)
	}
	afmoeNativeSync(t, dev)
	if got := afmoeNativeDownload(t, fault)[0]; got != 0 {
		t.Fatalf("router fault %d", got)
	}
	for k, id := range afmoeNativeDownload(t, ids) {
		if id != int32(127-k) {
			t.Errorf("route %d expert %d, want %d", k, id, 127-k)
		}
	}
	for k, weight := range afmoeNativeDownload(t, weights) {
		// All logits are zero: every unbiased score is0.5 regardless of bias.
		if math.Abs(float64(weight-float32(2.826/8))) > 1e-7 {
			t.Errorf("route %d weight %g, want scale/8", k, weight)
		}
	}
}

func afmoeNativeCombine(t *testing.T, dev *cuda.Device, m *afmoeMoEKernels) {
	type tensor struct {
		Shape []int
		Data  []float32
	}
	var fixture struct {
		MoE struct {
			Shared      tensor
			Outputs     tensor `json:"expert_outputs"`
			SelectedIDs []int  `json:"selected_ids"`
			Weights     tensor
			Sequential  tensor `json:"sequential_bf16_expected"`
			Wide        tensor `json:"f32_accumulation_expected"`
		} `json:"moe_combine"`
	}
	data, err := os.ReadFile("testdata/afmoe_moe.json")
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(data, &fixture); err != nil {
		t.Fatal(err)
	}
	f := fixture.MoE
	h := len(f.Shared.Data)
	ids, outputs := make([]int32, len(f.SelectedIDs)), make([]float32, len(f.SelectedIDs)*h)
	for k, id := range f.SelectedIDs {
		ids[k] = int32(id)
		copy(outputs[k*h:], f.Outputs.Data[id*h:(id+1)*h])
	}
	shared := afmoeNativeUpload(t, dev, f.Shared.Data)
	experts := afmoeNativeUpload(t, dev, outputs)
	dIDs := afmoeNativeUpload(t, dev, ids)
	weights := afmoeNativeUpload(t, dev, f.Weights.Data)
	output := afmoeNativeUpload(t, dev, make([]float32, h))
	fault := afmoeNativeUpload(t, dev, []int32{0})
	for _, test := range []struct {
		name string
		wide int32
		want []float32
	}{
		{"sequential BF16", 0, f.Sequential.Data},
		{"wide float32", 1, f.Wide.Data},
	} {
		t.Run(test.name, func(t *testing.T) {
			if err := afmoeNativeEmit(dev)(cudalaunch.Kernel(m.combineFn).Grid(1, 1, 1).Block(128, 1, 1), shared, experts, dIDs, weights, output, int32(h), int32(len(ids)), test.wide, fault); err != nil {
				t.Fatal(err)
			}
			afmoeNativeSync(t, dev)
			if got := afmoeNativeDownload(t, fault)[0]; got != 0 {
				t.Fatalf("combine fault %d", got)
			}
			if got := afmoeNativeDownload(t, output); !reflect.DeepEqual(got, test.want) {
				t.Errorf("combine %v, want fixture %v", got, test.want)
			}
		})
	}
}

func afmoeNativeInvalidIDs(t *testing.T, dev *cuda.Device, m *afmoeMoEKernels) {
	input := afmoeNativeUpload(t, dev, []float32{1, 1, 1, 1})
	activation := afmoeNativeUpload(t, dev, make([]float32, 3))
	output := afmoeNativeUpload(t, dev, make([]float32, 4))
	ids := afmoeNativeUpload(t, dev, []int32{0})
	fault := afmoeNativeUpload(t, dev, []int32{0})
	for _, id := range []int32{-1, 128} {
		if err := ids.CopyFromHost([]int32{id}); err != nil {
			t.Fatal(err)
		}
		if err := fault.CopyFromHost([]int32{0}); err != nil {
			t.Fatal(err)
		}
		// Null tables deliberately prove the ID guard runs before any indirect
		// table dereference. This negative test bypasses the checked emitter.
		if err := afmoeNativeEmit(dev)(cudalaunch.Kernel(m.selectedGateFn).Grid(3, 1, 1).Block(128, 1, 1), input, driver.DevicePtr(0), driver.DevicePtr(0), ids, activation, int32(4), int32(3), int32(1), fault); err != nil {
			t.Fatal(err)
		}
		afmoeNativeSync(t, dev)
		if got := afmoeNativeDownload(t, fault)[0]; got != 2 {
			t.Errorf("invalid ID %d fault %d, want2", id, got)
		}
		if err := fault.CopyFromHost([]int32{0}); err != nil {
			t.Fatal(err)
		}
		if err := afmoeNativeEmit(dev)(cudalaunch.Kernel(m.selectedDownFn).Grid(4, 1, 1).Block(128, 1, 1), activation, driver.DevicePtr(0), ids, output, int32(3), int32(4), int32(1), fault); err != nil {
			t.Fatal(err)
		}
		afmoeNativeSync(t, dev)
		if got := afmoeNativeDownload(t, fault)[0]; got != 2 {
			t.Errorf("invalid Down ID %d fault %d, want2", id, got)
		}
	}
}

func afmoeNativePointerTables(t *testing.T, dev *cuda.Device, m *afmoeMoEKernels) {
	const h, inter = 4, 3
	gate, up, down := make([]deviceWeight, 128), make([]deviceWeight, 128), make([]deviceWeight, 128)
	weight := func(values []float32) deviceWeight {
		return deviceWeight{buf: afmoeNativeUpload(t, dev, roundBF16(values)), dtype: weightBF16}
	}
	for e := 0; e < 128; e++ {
		g, u, d := make([]float32, h*inter), make([]float32, h*inter), make([]float32, h*inter)
		for row := 0; row < inter; row++ {
			g[row*h] = float32(1+e%4) / 4
			u[row*h+1] = 1 + float32(e%8)/8
		}
		for row, factor := range []float32{1, -1, 0.5, 2} {
			d[row*inter] = float32(e+1) / 128 * factor
		}
		gate[e], up[e], down[e] = weight(g), weight(u), weight(d)
	}
	tables, err := newAFMoEExpertTables(dev, gate, up, down, h, inter)
	if tables != nil {
		t.Cleanup(func() {
			if err := tables.Close(); err != nil {
				t.Error(err)
			}
		})
	}
	if err != nil {
		t.Fatal(err)
	}
	if err := gate[127].release(); !errors.Is(err, cudamem.ErrInUse) {
		t.Fatalf("pointer table did not retain expert127 allocation: %v", err)
	}
	router := weight(make([]float32, 128*h))
	biasValues := make([]float32, 128)
	for e := range biasValues {
		biasValues[e] = float32(e) / 128
	}
	bias := afmoeNativeUpload(t, dev, biasValues)
	sharedGate, sharedUp, sharedDown := make([]float32, h*inter), make([]float32, h*inter), make([]float32, h*inter)
	for row := 0; row < inter; row++ {
		sharedGate[row*h], sharedUp[row*h+1] = 0.5, 1
	}
	for row := 0; row < h; row++ {
		sharedDown[row*inter] = 0.25
	}
	sg, su, sd := weight(sharedGate), weight(sharedUp), weight(sharedDown)
	for _, topK := range []int{8, 128} {
		b := &afmoeMoEBuffers{
			Input:  afmoeNativeUpload(t, dev, []float32{1, 1, 0, 0}),
			Logits: afmoeNativeUpload(t, dev, make([]float32, 128)), Weights: afmoeNativeUpload(t, dev, make([]float32, topK)),
			SharedAct: afmoeNativeUpload(t, dev, make([]float32, inter)), SharedOut: afmoeNativeUpload(t, dev, make([]float32, h)),
			RoutedAct: afmoeNativeUpload(t, dev, make([]float32, topK*inter)), RoutedOut: afmoeNativeUpload(t, dev, make([]float32, topK*h)),
			Output: afmoeNativeUpload(t, dev, make([]float32, h)), IDs: afmoeNativeUpload(t, dev, make([]int32, topK)), Fault: afmoeNativeUpload(t, dev, []int32{0}),
		}
		if err := m.emitMoE(tables, router, bias, sg, su, sd, b, topK, true, 1e-20, 2.826, false, afmoeNativeEmit(dev)); err != nil {
			t.Fatal(err)
		}
		afmoeNativeSync(t, dev)
		if err := b.checkFault(); err != nil {
			t.Fatal(err)
		}
		gotIDs, gotAct, gotOut := afmoeNativeDownload(t, b.IDs), afmoeNativeDownload(t, b.RoutedAct), afmoeNativeDownload(t, b.RoutedOut)
		expectedIDs, expectedWeights, expectedOutputs := make([]int, topK), make([]float32, topK), make([][]float32, topK)
		for k, id := range gotIDs {
			wantID := 127 - k
			expectedIDs[k], expectedWeights[k] = wantID, float32(2.826)/float32(topK)
			expectedOutputs[k] = make([]float32, h)
			if id != int32(wantID) {
				t.Fatalf("top%d route%d ID%d, want%d", topK, k, id, wantID)
			}
			g, u := float32(1+wantID%4)/4, 1+float32(wantID%8)/8
			silu := afmoeBF16(float32(float64(g) / (1 + math.Exp(-float64(g)))))
			act := afmoeBF16(silu * u)
			for row := 0; row < inter; row++ {
				if gotAct[k*inter+row] != act {
					t.Errorf("expert%d activation%d %g, want%g", id, row, gotAct[k*inter+row], act)
				}
			}
			for row, factor := range []float32{1, -1, 0.5, 2} {
				want := afmoeBF16(act * (float32(wantID+1) / 128 * factor))
				expectedOutputs[k][row] = want
				if gotOut[k*h+row] != want {
					t.Errorf("expert%d output%d %g, want%g", id, row, gotOut[k*h+row], want)
				}
			}
		}
		for k, weight := range afmoeNativeDownload(t, b.Weights) {
			if weight != expectedWeights[k] {
				t.Errorf("top%d route%d weight %g, want%g", topK, k, weight, expectedWeights[k])
			}
		}
		weighted, err := afmoeWeightedExperts(expectedOutputs, expectedWeights)
		if err != nil {
			t.Fatal(err)
		}
		shared := make([]float32, h)
		for j := range shared {
			shared[j] = afmoeBF16(afmoeBF16(float32(0.5/(1+math.Exp(-0.5)))) * 0.25)
		}
		want, err := afmoeCombineExperts(shared, weighted, expectedIDs, false)
		if err != nil {
			t.Fatal(err)
		}
		if got := afmoeNativeDownload(t, b.Output); !reflect.DeepEqual(got, want) {
			t.Errorf("top%d composite output %v, want%v", topK, got, want)
		}
	}
}

func afmoeNativeDenseMLP(t *testing.T, dev *cuda.Device, m *afmoeMoEKernels) {
	type tensor struct{ Data []float32 }
	var fixture struct {
		Layer struct {
			Input  tensor `json:"ffn_input"`
			Output tensor `json:"ffn_output"`
			MLP    struct{ Gate, Up, Down [][]float32 }
		} `json:"four_norm_layer"`
	}
	data, err := os.ReadFile("testdata/afmoe-mini/reference.json")
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(data, &fixture); err != nil {
		t.Fatal(err)
	}
	f := fixture.Layer
	h, inter := len(f.Input.Data), len(f.MLP.Gate)
	if h != 4 || inter != 3 || len(f.Output.Data) != h {
		t.Fatal("invalid independent dense fixture dimensions")
	}
	weight := func(rows [][]float32) deviceWeight {
		var values []float32
		for _, row := range rows {
			values = append(values, row...)
		}
		return deviceWeight{buf: afmoeNativeUpload(t, dev, roundBF16(values)), dtype: weightBF16}
	}
	x := afmoeNativeUpload(t, dev, f.Input.Data)
	act := afmoeNativeUpload(t, dev, make([]float32, inter))
	output := afmoeNativeUpload(t, dev, make([]float32, h))
	if err := m.emitDenseMLP(weight(f.MLP.Gate), weight(f.MLP.Up), weight(f.MLP.Down), x, act, output, h, inter, afmoeNativeEmit(dev)); err != nil {
		t.Fatal(err)
	}
	afmoeNativeSync(t, dev)
	if got := afmoeNativeDownload(t, output); !reflect.DeepEqual(got, f.Output.Data) {
		t.Errorf("dense MLP %v, want independent fixture %v", got, f.Output.Data)
	}
}
