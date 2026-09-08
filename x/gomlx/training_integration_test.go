package gomlx

import (
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/shapes"
	"github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/core/tensors"
	"github.com/gomlx/gomlx/ml/layers"
	"github.com/gomlx/gomlx/ml/layers/attention"
	"github.com/gomlx/gomlx/ml/layers/norm"
	"github.com/gomlx/gomlx/ml/model"
	"github.com/gomlx/gomlx/ml/model/checkpoint"
	"github.com/gomlx/gomlx/ml/train/optimizer"
)

func TestNativeModelTrainingCheckpoint(t *testing.T) {
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
	input := [][][]float32{{{1, -0.5}, {0.25, 0.75}, {-0.5, 1}}}
	init := func(g *graph.Graph, s shapes.Shape) *graph.Node {
		if len(s.Dimensions) == 1 {
			return graph.Zeros(g, s)
		}
		flat := make([]float32, s.Size())
		for i := range flat {
			flat[i] = float32(i+1) / 16
		}
		return graph.Reshape(graph.Const(g, flat), s.Dimensions...)
	}
	build := func(scope *model.Scope, x *graph.Node) *graph.Node {
		scope = scope.WithInitializer(init)
		n := norm.RMSNorm(scope.In("norm"), x).Done()
		project := func(name string) *graph.Node { return layers.Dense(scope.In("%s", name), n, true, 1, 2) }
		a, _ := attention.Core(project("query"), project("key"), project("value"), attention.LayoutBSHD, attention.CoreOptions{UseCausalMask: true, WantCoefficients: true})
		y := layers.Dense(scope.In("output"), graph.Reshape(a, 1, 3, 2), true, 2)
		y = graph.Add(x, y)
		loss := graph.ReduceAllSum(graph.Square(y))
		optimizer.Adam().LearningRate(0.01).Done().UpdateGraph(scope, x.Graph(), loss)
		return loss
	}
	makeExec := func(s *model.Store) *model.Exec {
		e, err := model.NewExec(b, s, build)
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(e.Finalize)
		return e
	}
	step := func(e *model.Exec, s *model.Store, n int) float64 {
		out, err := e.Call(input)
		if err != nil {
			t.Fatalf("%+v", err)
		}
		defer out[0].FinalizeAll()
		loss := float64(out[0].Value().(float32))
		if math.IsNaN(loss) || math.IsInf(loss, 0) {
			t.Fatalf("step %d nonfinite loss %v", n, loss)
		}
		if got := optimizer.GetGlobalStep(s); got != int64(n) {
			t.Fatalf("step %d counter %d", n, got)
		}
		t.Logf("step %d loss %.9g", n, loss)
		return loss
	}
	snapshot := func(s *model.Store) map[string][]float64 {
		result := make(map[string][]float64)
		for v := range s.RootScope().IterVariables() {
			var values []float64
			var err error
			switch v.Shape().DType {
			case dtypes.Float32:
				err = tensors.ConstFlatData[float32](v.MustValue(), func(x []float32) {
					for _, n := range x {
						values = append(values, float64(n))
					}
				})
			case dtypes.Int64:
				err = tensors.ConstFlatData[int64](v.MustValue(), func(x []int64) {
					for _, n := range x {
						values = append(values, float64(n))
					}
				})
			default:
				t.Fatalf("unhandled state dtype %v", v.Shape().DType)
			}
			if err != nil {
				t.Fatal(err)
			}
			result[v.Path()] = values
		}
		return result
	}
	compare := func(got, want map[string][]float64) {
		t.Helper()
		if len(got) != len(want) {
			t.Fatalf("state count %d, want %d", len(got), len(want))
		}
		for path, w := range want {
			g := got[path]
			if len(g) != len(w) {
				t.Fatalf("state shape mismatch %s", path)
			}
			for i := range w {
				if math.IsNaN(g[i]) || math.Abs(g[i]-w[i]) > 1e-5+1e-4*math.Abs(w[i]) {
					t.Fatalf("state %s[%d] = %v, want %v", path, i, g[i], w[i])
				}
			}
		}
	}
	store := model.NewStore()
	t.Cleanup(store.Finalize)
	e := makeExec(store)
	previous := math.Inf(1)
	for n := 1; n <= 3; n++ {
		loss := step(e, store, n)
		if loss >= previous {
			t.Fatalf("loss did not decrease: %v >= %v", loss, previous)
		}
		if n == 1 {
			want := initialAttentionModelLoss(input[0])
			if math.Abs(loss-want) > 1e-5 {
				t.Fatalf("initial loss %v, independent reference %v", loss, want)
			}
		}
		previous = loss
	}
	home, err := os.UserHomeDir()
	if err != nil {
		t.Fatal(err)
	}
	dir, err := os.MkdirTemp(filepath.Join(home, "tmp"), "tmc-training-checkpoint-")
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
	losses := make(map[int]float64)
	var state4 map[string][]float64
	for n := 4; n <= 8; n++ {
		losses[n] = step(e, store, n)
		if losses[n] >= previous {
			t.Fatalf("step %d loss did not decrease", n)
		}
		previous = losses[n]
		if n == 4 {
			state4 = snapshot(store)
		}
	}
	restored := load()
	re := makeExec(restored)
	for n := 4; n <= 8; n++ {
		got := step(re, restored, n)
		if math.Abs(got-losses[n]) > 1e-5 {
			t.Fatalf("resumed step %d loss %v, want %v", n, got, losses[n])
		}
	}
	compare(snapshot(restored), snapshot(store))
	reset := load()
	moments := 0
	for v := range reset.RootScope().IterVariables() {
		if strings.HasSuffix(v.Path(), "_1st_moment") || strings.HasSuffix(v.Path(), "_2nd_moment") {
			v.MustSetValue(tensors.FromFlatDataAndDimensions(make([]float32, v.Shape().Size()), v.Shape().Dimensions...))
			moments++
		}
	}
	if moments != 18 {
		t.Fatalf("reset %d moments, want 18 for 9 trainable variables", moments)
	}
	step(makeExec(reset), reset, 4)
	changed := false
	resetState := snapshot(reset)
	for v := range reset.RootScope().IterVariables() {
		if !v.Trainable {
			continue
		}
		got := resetState[v.Path()]
		want := state4[v.Path()]
		for i := range got {
			if math.Abs(got[i]-want[i]) > 1e-5 {
				changed = true
			}
		}
	}
	if !changed {
		t.Fatal("reset-moments control failed to change next parameter update")
	}
	t.Logf("checkpoint restored all %d state tensors; reset %d moments changed updates; checkpoint %s", len(snapshot(store)), moments, dir)
}

// initialAttentionModelLoss evaluates the deterministic initial model in float64.
func initialAttentionModelLoss(input [][]float32) float64 {
	var projected [3][2]float64
	for s, x := range input {
		rms := math.Sqrt((float64(x[0])*float64(x[0])+float64(x[1])*float64(x[1]))/2 + 1e-6)
		for j := 0; j < 2; j++ {
			for d := 0; d < 2; d++ {
				projected[s][j] += float64(x[d]) / rms * float64(d*2+j+1) / 16
			}
		}
	}
	loss := 0.0
	for s := 0; s < 3; s++ {
		scores := [3]float64{}
		sum := 0.0
		for k := 0; k <= s; k++ {
			for d := 0; d < 2; d++ {
				scores[k] += projected[s][d] * projected[k][d]
			}
			scores[k] = math.Exp(scores[k] / math.Sqrt(2))
			sum += scores[k]
		}
		var a [2]float64
		for k := 0; k <= s; k++ {
			for d := 0; d < 2; d++ {
				a[d] += scores[k] / sum * projected[k][d]
			}
		}
		for j := 0; j < 2; j++ {
			y := float64(input[s][j])
			for d := 0; d < 2; d++ {
				y += a[d] * float64(d*2+j+1) / 16
			}
			loss += y * y
		}
	}
	return loss
}
