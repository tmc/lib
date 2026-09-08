package gomlx

import (
	"bytes"
	"os"
	"path/filepath"
	"reflect"
	"testing"

	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/shapes"
	"github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/core/tensors"
	"github.com/gomlx/gomlx/ml/layers"
	"github.com/gomlx/gomlx/ml/layers/attention"
	"github.com/gomlx/gomlx/ml/layers/attention/kvcache"
	"github.com/gomlx/gomlx/ml/layers/attention/pos"
	"github.com/gomlx/gomlx/ml/model"
	"github.com/gomlx/gomlx/ml/model/checkpoint"
)

func TestNativeCachedModelCheckpoint(t *testing.T) {
	path := os.Getenv("TMC_COMPUTE_PLUGIN")
	if path == "" {
		t.Skip("set TMC_COMPUTE_PLUGIN")
	}
	backend, err := New(path)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := backend.Close(); err != nil {
			t.Error(err)
		}
	})
	init := func(g *graph.Graph, s shapes.Shape) *graph.Node {
		values := make([]float32, s.Size())
		for i := range values {
			values[i] = float32(i%7+1) / 16
		}
		return graph.Reshape(graph.Const(g, values), s.Dimensions...)
	}
	build := func(scope *model.Scope, x *graph.Node) *graph.Node {
		scope = scope.WithInitializer(init)
		g := x.Graph()
		state := scope.In("state").WithInitializer(func(g *graph.Graph, s shapes.Shape) *graph.Node { return graph.Zeros(g, s) })
		position := state.VariableWithValue("position", int32(0)).SetTrainable(false)
		kvShape := shapes.Make(dtypes.Float32, 1, 2, 8, 2)
		keys := state.VariableWithShape("key", kvShape).SetTrainable(false)
		values := state.VariableWithShape("value", kvShape).SetTrainable(false)
		p := position.NodeValue(g)
		project := func(name string) *graph.Node { return layers.Dense(scope.In("%s", name), x, true, 2, 2) }
		q, k := pos.NewRoPE(10000).EncodeQK(project("query"), project("key"), graph.Reshape(p, 1), 1)
		v := project("value")
		cfg := kvcache.NewKVCache().WithSinkPositions(1).WithMaxSeqLenPerLayerType([]int{8, 8}).WithMinSeqLenPerLayerType([]int{8, 8})
		cache := kvcache.KVCacheNodes{scope.Scope() + kvcache.KeySuffix: keys.NodeValue(g), scope.Scope() + kvcache.ValueSuffix: values.NodeValue(g)}
		cfg.Update(scope, cache, graph.TransposeAllAxes(k, 0, 2, 1, 3), graph.TransposeAllAxes(v, 0, 2, 1, 3), p)
		nextK, nextV := cfg.Get(cache, scope.Scope())
		keys.SetNodeValue(nextK)
		values.SetNodeValue(nextV)
		mask := cfg.BuildAttentionMask(scope, cache, q, p, true, 4)
		a, _ := attention.Core(q, graph.TransposeAllAxes(nextK, 0, 2, 1, 3), graph.TransposeAllAxes(nextV, 0, 2, 1, 3), attention.LayoutBSHD, attention.CoreOptions{AttentionMask: graph.InsertAxes(mask, 2)})
		position.SetNodeValue(graph.AddScalar(p, int32(1)))
		return graph.Add(x, layers.Dense(scope.In("output"), graph.Reshape(a, 1, 1, 4), true, 4))
	}
	makeExec := func(b *Backend, s *model.Store) *model.Exec {
		e, err := model.NewExec(b, s, build)
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(e.Finalize)
		return e
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
	step := func(e *model.Exec, s *model.Store, p int) []byte {
		input := [][][]float32{{{float32(p%5) / 4, -.5, .75, float32(p+1) / 16}}}
		out, err := e.Call(input)
		if err != nil {
			t.Fatalf("p%d: %+v", p, err)
		}
		defer out[0].FinalizeAll()
		var result []byte
		if err := out[0].ConstBytes(func(data []byte) { result = bytes.Clone(data) }); err != nil {
			t.Fatal(err)
		}
		variable := s.GetVariable("/state/position")
		if variable == nil {
			t.Fatal("missing position")
		}
		if got := variable.MustValue().Value().(int32); got != int32(p+1) {
			t.Fatalf("position %d want%d", got, p+1)
		}
		return result
	}
	store := model.NewStore()
	t.Cleanup(store.Finalize)
	e := makeExec(backend, store)
	for p := 0; p < 4; p++ {
		step(e, store, p)
	}
	saved := snapshot(store)
	if len(saved) != 11 {
		t.Fatalf("state tensors %d, want11", len(saved))
	}
	dir, err := os.MkdirTemp(filepath.Join(os.Getenv("HOME"), "tmp"), "tmc-cached-model-")
	if err != nil {
		t.Fatal(err)
	}
	saver, err := checkpoint.Build(store).Dir(dir).Done()
	if err != nil {
		t.Fatal(err)
	}
	if err := saver.Save(); err != nil {
		t.Fatal(err)
	}
	saved = snapshot(store)
	if len(saved) != 12 {
		t.Fatalf("checkpoint state tensors %d,want12", len(saved))
	}
	outputs := make(map[int][]byte)
	states := make(map[int]map[string][]byte)
	for p := 4; p < 13; p++ {
		outputs[p] = step(e, store, p)
		states[p] = snapshot(store)
	}
	load := func() *model.Store {
		s := model.NewStore()
		t.Cleanup(s.Finalize)
		if _, err := checkpoint.Load(s).Dir(dir).Immediate().Done(); err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(snapshot(s), saved) {
			got := snapshot(s)
			for name, want := range saved {
				if !bytes.Equal(got[name], want) {
					t.Errorf("restored %s: %x want %x", name, got[name], want)
				}
			}
			t.Fatalf("restored state entries %d, want %d", len(got), len(saved))
		}
		return s
	}
	restored := load()
	fresh, err := New(path)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := fresh.Close(); err != nil {
			t.Error(err)
		}
	})
	resumed := makeExec(fresh, restored)
	for p := 4; p < 13; p++ {
		if got := step(resumed, restored, p); !bytes.Equal(got, outputs[p]) {
			t.Fatalf("p%d output differs after restore", p)
		}
		if !reflect.DeepEqual(snapshot(restored), states[p]) {
			t.Fatalf("p%d state differs after restore", p)
		}
	}
	mutant := load()
	v := mutant.GetVariable("/state/value")
	if err := v.SetValue(tensors.FromShape(v.Shape())); err != nil {
		t.Fatal(err)
	}
	changed := snapshot(mutant)
	for name, want := range saved {
		if name != "/state/value" && !bytes.Equal(changed[name], want) {
			t.Fatalf("mutant changed %s", name)
		}
	}
	if got := step(makeExec(fresh, mutant), mutant, 4); bytes.Equal(got, outputs[4]) {
		t.Fatal("erased cache values did not change continuation")
	}
	t.Logf("restored %d tensors (8 weights/biases,2 caches,position,checkpoint global step),9 exact continuation steps; cache-only reset changes output; checkpoint %s", len(saved), dir)
}
