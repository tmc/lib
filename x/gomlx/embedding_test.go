package gomlx

import (
	"bytes"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"testing"

	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/core/tensors"
	"github.com/gomlx/gomlx/ml/layers"
	"github.com/gomlx/gomlx/ml/model"
	"github.com/gomlx/gomlx/ml/model/checkpoint"
	"github.com/gomlx/gomlx/ml/train/optimizer"
)

func TestNativeEmbeddingTraining(t *testing.T) {
	path := os.Getenv("TMC_COMPUTE_PLUGIN")
	if path == "" {
		t.Skip("set TMC_COMPUTE_PLUGIN")
	}
	newBackend := func() *Backend {
		b, err := New(path)
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() {
			if err := b.Close(); err != nil {
				t.Error(err)
			}
		})
		return b
	}
	b := newBackend()
	opt := optimizer.Adam().LearningRate(0.25).Betas(0.5, 0.75).Epsilon(1e-7).Done()
	build := func(scope *model.Scope, indices *graph.Node) []*graph.Node {
		g := indices.Graph()
		embedded := layers.Embedding(scope, indices, dtypes.Float32, 3, 2)
		loss := graph.ReduceAllSum(graph.Mul(embedded, embedded))
		grad := graph.Gradient(loss, scope.Store().GetVariable("/embeddings").NodeValue(g))[0]
		opt.UpdateGraph(scope, g, loss)
		return []*graph.Node{loss, grad}
	}
	newExec := func(b *Backend, s *model.Store) *model.Exec {
		e, err := model.NewExec(b, s, build)
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(e.Finalize)
		return e
	}
	store := model.NewStore()
	t.Cleanup(store.Finalize)
	store.RootScope().VariableWithValue("embeddings", [][]float32{{1, 2}, {3, 4}, {5, 6}})
	e := newExec(b, store)
	want := [3][2]float32{{1, 2}, {3, 4}, {5, 6}}
	var m1, m2 [3][2]float64
	step := func(e *model.Exec, s *model.Store, n int) {
		t.Helper()
		out, err := e.Call([]int32{2, 0, 2})
		if err != nil {
			t.Fatalf("step %d: %+v", n, err)
		}
		defer func() {
			for _, v := range out {
				v.FinalizeAll()
			}
		}()
		grad := out[1].Value().([][]float32)
		var loss float32
		for row, count := range []float32{1, 0, 2} {
			for col, x := range want[row] {
				loss += count * x * x
				g := 2 * count * x
				if math.Abs(float64(grad[row][col]-g)) > 1e-5 {
					t.Fatalf("step%d gradient[%d,%d]=%v want%v", n, row, col, grad[row][col], g)
				}
				m1[row][col] = .5*m1[row][col] + .5*float64(g)
				m2[row][col] = .75*m2[row][col] + .25*float64(g)*float64(g)
				mean := m1[row][col] / (1 - math.Pow(.5, float64(n)))
				variance := m2[row][col] / (1 - math.Pow(.75, float64(n)))
				want[row][col] -= float32(.25 * mean / (math.Sqrt(variance) + 1e-7))
			}
		}
		if math.Abs(float64(out[0].Value().(float32)-loss)) > 1e-4 {
			t.Fatalf("step%d loss %v want%v", n, out[0].Value(), loss)
		}
		got := s.GetVariable("/embeddings").MustValue().Value().([][]float32)
		for row := range want {
			for col := range want[row] {
				if math.Abs(float64(got[row][col]-want[row][col])) > 1e-5 {
					t.Fatalf("step%d weights %v want%v", n, got, want)
				}
			}
		}
		if optimizer.GetGlobalStep(s) != int64(n) {
			t.Fatal("incorrect optimizer step")
		}
	}
	for n := 1; n <= 3; n++ {
		step(e, store, n)
	}
	home, err := os.UserHomeDir()
	if err != nil {
		t.Fatal(err)
	}
	dir, err := os.MkdirTemp(filepath.Join(home, "tmp"), "tmc-embedding-checkpoint-")
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
	snapshot := func(s *model.Store) map[string][]byte {
		m := make(map[string][]byte)
		for v := range s.RootScope().IterVariables() {
			if err := v.MustValue().ConstBytes(func(data []byte) { m[v.Path()] = bytes.Clone(data) }); err != nil {
				t.Fatal(err)
			}
		}
		return m
	}
	load := func() *model.Store {
		s := model.NewStore()
		t.Cleanup(s.Finalize)
		if _, err := checkpoint.Load(s).Dir(dir).Immediate().Done(); err != nil {
			t.Fatal(err)
		}
		return s
	}
	savedWant, savedM1, savedM2 := want, m1, m2
	states := make([]map[string][]byte, 3)
	for n := 4; n <= 6; n++ {
		step(e, store, n)
		states[n-4] = snapshot(store)
	}
	restored := load()
	re := newExec(newBackend(), restored)
	want, m1, m2 = savedWant, savedM1, savedM2
	for n := 4; n <= 6; n++ {
		step(re, restored, n)
		if !reflect.DeepEqual(snapshot(restored), states[n-4]) {
			t.Fatalf("step%d restored state mismatch", n)
		}
	}
	reset := load()
	for _, name := range []string{"embeddings_1st_moment", "embeddings_2nd_moment"} {
		v := reset.GetVariable("/" + optimizer.AdamDefaultScope + "/" + name)
		if v == nil {
			t.Fatalf("missing moment %s", name)
		}
		v.MustSetValue(tensors.MustFromAnyValue([][]float32{{0, 0}, {0, 0}, {0, 0}}))
	}
	bad := newExec(b, reset)
	out, err := bad.Call([]int32{2, 0, 2})
	if err != nil {
		t.Fatal(err)
	}
	for _, v := range out {
		v.FinalizeAll()
	}
	if bytes.Equal(snapshot(reset)["/embeddings"], states[0]["/embeddings"]) {
		t.Fatal("reset moments did not change update")
	}
	t.Logf("repeated-index embedding gradients, host Adam updates, three restored steps and reset-moment control pass; checkpoint %s", dir)
}
