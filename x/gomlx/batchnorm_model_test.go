package gomlx

import (
	"math"
	"os"
	"path/filepath"
	"reflect"
	"testing"

	"github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/core/tensors"
	"github.com/gomlx/gomlx/ml/layers/norm"
	"github.com/gomlx/gomlx/ml/model"
	"github.com/gomlx/gomlx/ml/model/checkpoint"
)

func TestNativeBatchNormModel(t *testing.T) {
	path := os.Getenv("TMC_COMPUTE_PLUGIN")
	if path == "" {
		t.Skip("set TMC_COMPUTE_PLUGIN")
	}
	b, err := New(path)
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := b.Close(); err != nil {
			t.Error(err)
		}
	}()
	store := model.NewStore()
	defer store.Finalize()
	build := func(training bool) func(*model.Scope, *graph.Node) *graph.Node {
		return func(scope *model.Scope, x *graph.Node) *graph.Node {
			scope.Store().SetTraining(x.Graph(), training)
			return norm.BatchNorm(scope, x, 1).UseBackendInference(true).Momentum(.5).Epsilon(.125).Done()
		}
	}
	train, err := model.NewExec(b, store, build(true))
	if err != nil {
		t.Fatal(err)
	}
	defer train.Finalize()
	input := [][]float64{{1, 2}, {3, 4}, {5, 6}}
	for _, shift := range []float64{0, 2} {
		batch := make([][]float64, len(input))
		for i, row := range input {
			batch[i] = []float64{row[0] + shift, row[1] + shift}
		}
		out, err := train.Call(batch)
		if err != nil {
			t.Fatal(err)
		}
		got := out[0].Value().([][]float64)
		for i, row := range got {
			for _, v := range row {
				want := float64(2*i-2) / math.Sqrt(8.0/3+.125)
				if math.Abs(v-want) > 1e-12 {
					t.Fatalf("training row%d got%g want%g", i, v, want)
				}
			}
		}
		for _, v := range out {
			v.FinalizeAll()
		}
	}
	for name, want := range map[string][]float64{"mean": {4, 5}, "variance": {8.0 / 3, 8.0 / 3}, "avg_weight": {2, 2}} {
		v := store.GetVariable("/batch_normalization/" + name)
		if v == nil {
			t.Fatalf("missing %s", name)
		}
		got := v.MustValue().Value().([]float64)
		for i, w := range want {
			if math.Abs(got[i]-w) > 1e-12 {
				t.Fatalf("%s got%v want%v", name, got, want)
			}
		}
	}
	infer, err := model.NewExec(b, store, build(false))
	if err != nil {
		t.Fatal(err)
	}
	defer infer.Finalize()
	expected, err := infer.Call(input)
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		for _, v := range expected {
			v.FinalizeAll()
		}
	}()
	for i, row := range expected[0].Value().([][]float64) {
		for j, v := range row {
			want := (input[i][j] - float64(4+j)) / math.Sqrt(8.0/3+.125)
			if math.Abs(v-want) > 1e-12 {
				t.Fatalf("inference[%d,%d] got%g want%g", i, j, v, want)
			}
		}
	}
	dir, err := os.MkdirTemp(filepath.Join(os.Getenv("HOME"), "tmp"), "tmc-batchnorm-")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(dir)
	saver, err := checkpoint.Build(store).Dir(dir).Done()
	if err != nil {
		t.Fatal(err)
	}
	if err := saver.Save(); err != nil {
		t.Fatal(err)
	}
	restored := model.NewStore()
	defer restored.Finalize()
	if _, err := checkpoint.Load(restored).Dir(dir).Immediate().Done(); err != nil {
		t.Fatal(err)
	}
	fresh, err := New(path)
	if err != nil {
		t.Fatal(err)
	}
	defer fresh.Close()
	replay, err := model.NewExec(fresh, restored, build(false))
	if err != nil {
		t.Fatal(err)
	}
	defer replay.Finalize()
	out, err := replay.Call(input)
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		for _, v := range out {
			v.FinalizeAll()
		}
	}()
	if !reflect.DeepEqual(out[0].Value(), expected[0].Value()) {
		t.Fatal("restored inference changed")
	}
	restored.GetVariable("/batch_normalization/mean").MustSetValue(tensors.MustFromAnyValue([]float64{0, 0}))
	changed, err := replay.Call(input)
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		for _, v := range changed {
			v.FinalizeAll()
		}
	}()
	if reflect.DeepEqual(changed[0].Value(), out[0].Value()) {
		t.Fatal("changed running mean did not change inference")
	}
}
