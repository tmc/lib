package gomlx

import (
	"math"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/ml/layers"
	"github.com/gomlx/gomlx/ml/model"
	"github.com/gomlx/gomlx/ml/model/checkpoint"
	"github.com/gomlx/gomlx/ml/model/initializer"
)

func TestNativeRandomInitialization(t *testing.T) {
	path := os.Getenv("TMC_COMPUTE_PLUGIN")
	if path == "" {
		t.Skip("set TMC_COMPUTE_PLUGIN")
	}
	b, err := New(path)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := b.Close(); err != nil {
			t.Error(err)
		}
	})
	for _, name := range []string{"default", "normal", "xavier", "glorot"} {
		run := func(seed int64) []float32 {
			t.Helper()
			store := model.NewStore()
			t.Cleanup(store.Finalize)
			if err := store.SetRNGStateFromSeed(seed); err != nil {
				t.Fatal(err)
			}
			e, err := model.NewExec(b, store, func(scope *model.Scope, x *graph.Node) *graph.Node {
				switch name {
				case "normal":
					scope = scope.WithInitializer(initializer.RandomNormalFn(scope, 0.5))
				case "xavier":
					scope = scope.WithInitializer(initializer.XavierNormalFn(scope))
				case "glorot":
					scope = scope.WithInitializer(initializer.GlorotUniformFn(scope))
				}
				return layers.Dense(scope, x, false, 8)
			})
			if err != nil {
				t.Fatal(err)
			}
			t.Cleanup(e.Finalize)
			input := [][]float32{{1, 1, 1, 1, 1, 1, 1, 1}}
			out, err := e.Call(input)
			if err != nil {
				t.Fatalf("%+v", err)
			}
			defer out[0].FinalizeAll()
			got := out[0].Value().([][]float32)[0]
			weights := store.GetVariable("/dense/weights").MustValue().Value().([][]float32)
			var flat []float32
			for _, row := range weights {
				flat = append(flat, row...)
			}
			for j := range got {
				want := 0.0
				for i := range weights {
					want += float64(weights[i][j])
				}
				if math.IsNaN(float64(got[j])) || math.Abs(float64(got[j])-want) > 1e-5 {
					t.Fatalf("dense column %d = %v, host reference %v", j, got[j], want)
				}
			}
			before := slices.Clone(store.GetVariable("/" + model.RNGStateVariableName).MustValue().Value().([]uint64))
			again, err := e.Call(input)
			if err != nil {
				t.Fatal(err)
			}
			defer again[0].FinalizeAll()
			after := store.GetVariable("/" + model.RNGStateVariableName).MustValue().Value().([]uint64)
			if !slices.Equal(before, after) || !slices.Equal(got, again[0].Value().([][]float32)[0]) {
				t.Fatal("forward call reinitialized weights or advanced RNG")
			}
			return flat
		}
		first, replay, other := run(123), run(123), run(456)
		if !slices.Equal(first, replay) {
			t.Fatalf("initializer=%s: same seed did not reproduce weights", name)
		}
		if slices.Equal(first, other) {
			t.Fatalf("initializer=%s: different seed did not change weights", name)
		}
		t.Logf("initializer=%s: seeded initialization replays, changed seed diverges, repeat forward retains state", name)
	}
}

func TestNativeDropoutCheckpoint(t *testing.T) {
	path := os.Getenv("TMC_COMPUTE_PLUGIN")
	if path == "" {
		t.Skip("set TMC_COMPUTE_PLUGIN")
	}
	b, err := New(path)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := b.Close(); err != nil {
			t.Error(err)
		}
	})
	store := model.NewStore()
	t.Cleanup(store.Finalize)
	if err := store.SetRNGStateFromSeed(123); err != nil {
		t.Fatal(err)
	}
	makeExec := func(s *model.Store, training bool) *model.Exec {
		e, err := model.NewExec(b, s, func(scope *model.Scope, x *graph.Node) []*graph.Node {
			s.SetTraining(x.Graph(), training)
			y := layers.DropoutStatic(scope, x, 0.25)
			return []*graph.Node{y, graph.Gradient(graph.ReduceAllSum(y), x)[0]}
		})
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(e.Finalize)
		return e
	}
	state := func(s *model.Store) []uint64 {
		return slices.Clone(s.GetVariable("/" + model.RNGStateVariableName).MustValue().Value().([]uint64))
	}
	input := make([]float32, 256)
	for i := range input {
		input[i] = 1
	}
	call := func(e *model.Exec, training bool) []float32 {
		t.Helper()
		out, err := e.Call(input)
		if err != nil {
			t.Fatalf("%+v", err)
		}
		for _, x := range out {
			defer x.FinalizeAll()
		}
		values := slices.Clone(out[0].Value().([]float32))
		grad := out[1].Value().([]float32)
		zeros := 0
		for i, v := range values {
			if training {
				if v == 0 {
					zeros++
				} else if math.Abs(float64(v)-4.0/3) > 1e-6 {
					t.Fatalf("dropout output %v", v)
				}
			} else if v != 1 {
				t.Fatalf("inference output %v", v)
			}
			if grad[i] != v {
				t.Fatalf("gradient %v differs from dropout multiplier %v", grad[i], v)
			}
		}
		if training && (zeros == 0 || zeros == len(input)) {
			t.Fatal("dropout did not keep and drop values")
		}
		return values
	}
	train := makeExec(store, true)
	before := state(store)
	first := call(train, true)
	saved := state(store)
	if slices.Equal(before, saved) {
		t.Fatal("training did not advance RNG")
	}
	infer := makeExec(store, false)
	call(infer, false)
	call(infer, false)
	if !slices.Equal(saved, state(store)) {
		t.Fatal("inference advanced RNG")
	}
	home, err := os.UserHomeDir()
	if err != nil {
		t.Fatal(err)
	}
	dir, err := os.MkdirTemp(filepath.Join(home, "tmp"), "tmc-rng-checkpoint-")
	if err != nil {
		t.Fatal(err)
	}
	h, err := checkpoint.Build(store).Dir(dir).Done()
	if err != nil {
		t.Fatal(err)
	}
	if err := h.Save(); err != nil {
		t.Fatal(err)
	}
	second := call(train, true)
	next := state(store)
	if slices.Equal(first, second) {
		t.Fatal("successive training calls repeated mask")
	}
	load := func() *model.Store {
		s := model.NewStore()
		t.Cleanup(s.Finalize)
		if _, err := checkpoint.Load(s).Dir(dir).Immediate().Done(); err != nil {
			t.Fatal(err)
		}
		return s
	}
	restored := load()
	restoredExec := makeExec(restored, true)
	if !slices.Equal(state(restored), saved) {
		t.Fatal("checkpoint lost RNG state")
	}
	if !slices.Equal(call(restoredExec, true), second) || !slices.Equal(state(restored), next) {
		t.Fatal("restored dropout did not reproduce next mask/state")
	}
	third := call(train, true)
	thirdState := state(store)
	if !slices.Equal(call(restoredExec, true), third) || !slices.Equal(state(restored), thirdState) {
		t.Fatal("second resumed mask/state mismatch")
	}
	reset := load()
	if err := reset.SetRNGStateFromSeed(123); err != nil {
		t.Fatal(err)
	}
	wrong := call(makeExec(reset, true), true)
	if !slices.Equal(wrong, first) || slices.Equal(wrong, second) {
		t.Fatal("reset-seed control did not replay first mask instead of continuation")
	}
	t.Logf("dropout gradients, training/inference RNG, checkpoint continuation and reset-seed control pass; checkpoint %s", dir)
}
