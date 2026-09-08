package gomlx

import (
	"os"
	"path/filepath"
	"reflect"
	"testing"

	"github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/ml/model"
	"github.com/gomlx/gomlx/ml/model/checkpoint"
)

func TestNativeModelUpdates(t *testing.T) {
	path := os.Getenv("TMC_COMPUTE_PLUGIN")
	if path == "" {
		t.Skip("set TMC_COMPUTE_PLUGIN")
	}
	b, err := New(path)
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		b.Finalize()
		if b.closeErr != nil {
			t.Error(b.closeErr)
		}
	}()
	store := model.NewStore()
	defer store.Finalize()
	weights := store.RootScope().VariableWithValue("weights", []float32{4, -2})
	build := func(scope *model.Scope, g *graph.Graph) *graph.Node {
		variable := scope.Store().GetVariable("/weights")
		x := variable.NodeValue(g)
		loss := graph.ReduceAllSum(graph.Mul(x, x))
		grad := graph.Gradient(loss, x)[0]
		variable.SetNodeValue(graph.Add(x, graph.MulScalar(grad, float32(-0.25))))
		return loss
	}
	e, err := model.NewExec(b, store, build)
	if err != nil {
		t.Fatal(err)
	}
	defer e.Finalize()
	for i, want := range []float32{20, 5, 1.25} {
		out, err := e.Call()
		if err != nil {
			t.Fatal(err)
		}
		got := out[0].Value()
		for _, v := range out {
			v.FinalizeAll()
		}
		if !reflect.DeepEqual(got, want) {
			t.Fatalf("step %d loss: got %v, want %v", i, got, want)
		}
		t.Logf("step %d: loss %v", i, got)
	}
	if got := weights.MustValue().Value(); !reflect.DeepEqual(got, []float32{0.5, -0.25}) {
		t.Fatalf("stored weights: %v", got)
	}
	t.Log("GoMLX store retained updated weights [0.5 -0.25]")
	home, err := os.UserHomeDir()
	if err != nil {
		t.Fatal(err)
	}
	dir, err := os.MkdirTemp(filepath.Join(home, "tmp"), "tmc-gomlx-checkpoint-")
	if err != nil {
		t.Fatal(err)
	}
	handler, err := checkpoint.Build(store).Dir(dir).Done()
	if err != nil {
		t.Fatal(err)
	}
	if err := handler.Save(); err != nil {
		t.Fatal(err)
	}
	restored := model.NewStore()
	defer restored.Finalize()
	if _, err := checkpoint.Load(restored).Dir(dir).Immediate().Done(); err != nil {
		t.Fatal(err)
	}
	loaded := restored.GetVariable("/weights")
	if loaded == nil {
		t.Fatal("checkpoint did not restore weights")
	}
	if got := loaded.MustValue().Value(); !reflect.DeepEqual(got, []float32{0.5, -0.25}) {
		t.Fatalf("restored weights: %v", got)
	}
	resumed, err := model.NewExec(b, restored, build)
	if err != nil {
		t.Fatal(err)
	}
	defer resumed.Finalize()
	out, err := resumed.Call()
	if err != nil {
		t.Fatal(err)
	}
	got := out[0].Value()
	for _, v := range out {
		v.FinalizeAll()
	}
	if !reflect.DeepEqual(got, float32(0.3125)) {
		t.Fatalf("resumed loss: %v", got)
	}
	if got := loaded.MustValue().Value(); !reflect.DeepEqual(got, []float32{0.25, -0.125}) {
		t.Fatalf("resumed weights: %v", got)
	}
	t.Log("checkpoint restored into fresh store; resumed loss 0.3125, weights [0.25 -0.125]")
	t.Logf("checkpoint retained at %s", dir)
}
