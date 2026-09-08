package gomlx

import (
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/core/tensors"
	"github.com/gomlx/gomlx/ml/model"
	"github.com/gomlx/gomlx/ml/model/checkpoint"
	"github.com/gomlx/gomlx/ml/train/optimizer"
)

func TestNativeSGDCheckpoint(t *testing.T) {
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
	opt := optimizer.StochasticGradientDescent().WithLearningRate(0.25)
	build := func(scope *model.Scope, g *graph.Graph) *graph.Node {
		x := scope.Store().GetVariable("/weights").NodeValue(g)
		loss := graph.ReduceAllSum(graph.Mul(x, x))
		opt.UpdateGraph(scope, g, loss)
		return loss
	}
	store := model.NewStore()
	defer store.Finalize()
	store.RootScope().VariableWithValue("weights", []float32{4, -2})
	e, err := model.NewExec(b, store, build)
	if err != nil {
		t.Fatal(err)
	}
	defer e.Finalize()
	want := []float32{4, -2}
	step := func(e *model.Exec, store *model.Store, n int) {
		t.Helper()
		wantLoss := want[0]*want[0] + want[1]*want[1]
		out, err := e.Call()
		if err != nil {
			t.Fatal(err)
		}
		gotLoss := out[0].Value().(float32)
		for _, v := range out {
			v.FinalizeAll()
		}
		if math.Abs(float64(gotLoss-wantLoss)) > 1e-5 {
			t.Fatalf("step %d loss %v, want %v", n, gotLoss, wantLoss)
		}
		lr := float32(0.25 / math.Sqrt(float64(n)))
		for i := range want {
			want[i] -= lr * 2 * want[i]
		}
		got := store.GetVariable("/weights").MustValue().Value().([]float32)
		for i := range want {
			if math.Abs(float64(got[i]-want[i])) > 1e-5 {
				t.Fatalf("step %d weights %v, want %v", n, got, want)
			}
		}
		if got := optimizer.GetGlobalStep(store); got != int64(n) {
			t.Fatalf("global step %d, want %d", got, n)
		}
		t.Logf("built-in SGD step %d: loss %v, weights %v", n, gotLoss, got)
	}
	for n := 1; n <= 3; n++ {
		step(e, store, n)
	}
	home, err := os.UserHomeDir()
	if err != nil {
		t.Fatal(err)
	}
	dir, err := os.MkdirTemp(filepath.Join(home, "tmp"), "tmc-sgd-checkpoint-")
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
	load := func() *model.Store {
		s := model.NewStore()
		t.Cleanup(s.Finalize)
		if _, err := checkpoint.Load(s).Dir(dir).Immediate().Done(); err != nil {
			t.Fatal(err)
		}
		return s
	}
	restored := load()
	re, err := model.NewExec(b, restored, build)
	if err != nil {
		t.Fatal(err)
	}
	defer re.Finalize()
	step(re, restored, 4)
	reset := load()
	optimizer.GetGlobalStepVar(reset).MustSetValue(tensors.MustFromAnyValue(int64(0)))
	bad, err := model.NewExec(b, reset, build)
	if err != nil {
		t.Fatal(err)
	}
	defer bad.Finalize()
	out, err := bad.Call()
	if err != nil {
		t.Fatal(err)
	}
	for _, v := range out {
		v.FinalizeAll()
	}
	got := reset.GetVariable("/weights").MustValue().Value().([]float32)
	if math.Abs(float64(got[0]-want[0])) < 1e-3 {
		t.Fatal("reset-step negative control failed to diverge")
	}
	t.Log("restored optimizer step preserves decay; resetting step changes the next update")
	t.Logf("checkpoint retained at %s", dir)
}

func TestNativeAdamCheckpoint(t *testing.T) {
	path := os.Getenv("TMC_COMPUTE_PLUGIN")
	if path == "" {
		t.Skip("set TMC_COMPUTE_PLUGIN")
	}
	b, err := New(path)
	if err != nil {
		t.Fatalf("%+v", err)
	}
	defer func() {
		if err := b.Close(); err != nil {
			t.Error(err)
		}
	}()
	opt := optimizer.Adam().LearningRate(0.25).Betas(0.5, 0.75).Epsilon(1e-7).Done()
	build := func(scope *model.Scope, g *graph.Graph) *graph.Node {
		x := scope.Store().GetVariable("/weights").NodeValue(g)
		loss := graph.ReduceAllSum(graph.Mul(x, x))
		opt.UpdateGraph(scope, g, loss)
		return loss
	}
	store := model.NewStore()
	defer store.Finalize()
	store.RootScope().VariableWithValue("weights", []float32{4, -2})
	e, err := model.NewExec(b, store, build)
	if err != nil {
		t.Fatalf("%+v", err)
	}
	defer e.Finalize()
	want := []float32{4, -2}
	m1, m2 := [2]float64{}, [2]float64{}
	step := func(e *model.Exec, store *model.Store, n int) {
		t.Helper()
		wantLoss := want[0]*want[0] + want[1]*want[1]
		out, err := e.Call()
		if err != nil {
			t.Fatalf("%+v", err)
		}
		gotLoss := out[0].Value().(float32)
		for _, v := range out {
			v.FinalizeAll()
		}
		if math.Abs(float64(gotLoss-wantLoss)) > 1e-5 {
			t.Fatalf("step %d loss %v, want %v", n, gotLoss, wantLoss)
		}
		for i := range want {
			grad := 2 * float64(want[i])
			m1[i] = 0.5*m1[i] + 0.5*grad
			m2[i] = 0.75*m2[i] + 0.25*grad*grad
			mean := m1[i] / (1 - math.Pow(0.5, float64(n)))
			variance := m2[i] / (1 - math.Pow(0.75, float64(n)))
			want[i] -= float32(0.25 * mean / (math.Sqrt(variance) + 1e-7))
		}
		got := store.GetVariable("/weights").MustValue().Value().([]float32)
		for i := range want {
			if math.Abs(float64(got[i]-want[i])) > 1e-5 {
				t.Fatalf("step %d weights %v, want %v", n, got, want)
			}
		}
		if got := optimizer.GetGlobalStep(store); got != int64(n) {
			t.Fatalf("global step %d, want %d", got, n)
		}
		t.Logf("built-in Adam step %d: loss %v, weights %v", n, gotLoss, got)
	}
	for n := 1; n <= 3; n++ {
		step(e, store, n)
	}
	home, err := os.UserHomeDir()
	if err != nil {
		t.Fatalf("%+v", err)
	}
	dir, err := os.MkdirTemp(filepath.Join(home, "tmp"), "tmc-adam-checkpoint-")
	if err != nil {
		t.Fatalf("%+v", err)
	}
	h, err := checkpoint.Build(store).Dir(dir).Done()
	if err != nil {
		t.Fatalf("%+v", err)
	}
	if err := h.Save(); err != nil {
		t.Fatalf("%+v", err)
	}
	load := func() *model.Store {
		s := model.NewStore()
		t.Cleanup(s.Finalize)
		if _, err := checkpoint.Load(s).Dir(dir).Immediate().Done(); err != nil {
			t.Fatalf("%+v", err)
		}
		return s
	}
	restored := load()
	re, err := model.NewExec(b, restored, build)
	if err != nil {
		t.Fatalf("%+v", err)
	}
	defer re.Finalize()
	step(re, restored, 4)
	reset := load()
	for _, name := range []string{"weights_1st_moment", "weights_2nd_moment"} {
		reset.GetVariable("/" + optimizer.AdamDefaultScope + "/" + name).MustSetValue(tensors.MustFromAnyValue([]float32{0, 0}))
	}
	bad, err := model.NewExec(b, reset, build)
	if err != nil {
		t.Fatalf("%+v", err)
	}
	defer bad.Finalize()
	out, err := bad.Call()
	if err != nil {
		t.Fatalf("%+v", err)
	}
	for _, v := range out {
		v.FinalizeAll()
	}
	got := reset.GetVariable("/weights").MustValue().Value().([]float32)
	if math.Abs(float64(got[0]-want[0])) < 1e-3 {
		t.Fatal("reset-moment negative control failed to diverge")
	}
	t.Log("restored Adam moments preserve updates; resetting only moments changes the next update")
	t.Logf("checkpoint retained at %s", dir)
}
