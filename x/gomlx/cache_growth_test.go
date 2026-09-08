package gomlx

import (
	"fmt"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/core/tensors"
	"github.com/gomlx/gomlx/ml/layers/attention"
	"github.com/gomlx/gomlx/ml/layers/attention/kvcache"
	"github.com/gomlx/gomlx/ml/model"
)

func TestNativeGrowingCachedAttention(t *testing.T) {
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
	scope := model.NewStore().Scope("/layer")
	cfg := kvcache.NewKVCache().WithOrderedScopes([]string{scope.Scope()}).WithSinkPositions(2).WithMinSeqLenPerLayerType([]int{2, 2}).WithMaxSeqLenPerLayerType([]int{8, 8})
	graphFn := func(k, v, nk, nv, q, p *graph.Node) []*graph.Node {
		cache := cfg.DeserializeNodes([]*graph.Node{k, v})
		cfg.Update(scope, cache, nk, nv, p)
		key, value := cfg.Get(cache, scope.Scope())
		mask := cfg.BuildAttentionMask(scope, cache, q, p, true, 3)
		result, _ := attention.Core(q, graph.TransposeAllAxes(key, 0, 2, 1, 3), graph.TransposeAllAxes(value, 0, 2, 1, 3), attention.LayoutBSHD, attention.CoreOptions{AttentionMask: graph.InsertAxes(mask, 2)})
		nodes, err := cfg.SerializeNodes(cache)
		if err != nil {
			panic(err)
		}
		return append(nodes, result)
	}
	e, err := graph.NewExec(b, graphFn)
	if err != nil {
		t.Fatal(err)
	}
	defer e.Finalize()
	cache := cfg.InitializeTensors(1, 2, 2, dtypes.Float32, 1)
	defer func() {
		for _, v := range cache {
			v.FinalizeAll()
		}
	}()
	positions := []int{-1, -1}
	for p := 0; p < 18; p++ {
		grown, err := cfg.PadTensors(cache, p+1)
		if err != nil {
			t.Fatal(err)
		}
		for name, old := range cache {
			if grown[name] != old {
				old.FinalizeAll()
			}
		}
		cache = grown
		flat, err := cfg.SerializeTensors(cache)
		if err != nil {
			t.Fatal(err)
		}
		capacity := flat[0].Shape().Dimensions[2]
		if capacity != len(positions) {
			for _, tensor := range flat {
				values := tensor.Value().([][][][]float32)
				for h := 0; h < 2; h++ {
					for j := len(positions); j < capacity; j++ {
						for d := 0; d < 2; d++ {
							if values[0][h][j][d] != 0 {
								t.Fatal("nonzero padding")
							}
						}
					}
				}
			}
			for len(positions) < capacity {
				positions = append(positions, -1)
			}
			t.Logf("position %d grew cache to %d", p, capacity)
		}
		keyData, valueData := make([]float32, 4), make([]float32, 4)
		for h := 0; h < 2; h++ {
			for d := 0; d < 2; d++ {
				keyData[h*2+d] = float32((p%7)-3+h+d) / 8
				valueData[h*2+d] = float32(p*4+h*2+d+1) / 16
			}
		}
		nk := tensors.FromFlatDataAndDimensions(keyData, 1, 2, 1, 2)
		nv := tensors.FromFlatDataAndDimensions(valueData, 1, 2, 1, 2)
		q := tensors.FromFlatDataAndDimensions([]float32{.25, -.5, .75, .125}, 1, 1, 2, 2)
		out, err := e.Call(flat[0], flat[1], nk, nv, q, int32(p))
		nk.FinalizeAll()
		nv.FinalizeAll()
		q.FinalizeAll()
		if err != nil {
			t.Fatalf("position %d: %+v", p, err)
		}
		for _, old := range cache {
			old.FinalizeAll()
		}
		cache = cfg.DeserializeTensors(out[:2])
		slot := p
		if p >= 8 {
			slot = 2 + (p-2)%6
		}
		positions[slot] = p
		got := out[2].Value().([][][][]float32)
		out[2].FinalizeAll()
		for h := 0; h < 2; h++ {
			queries := [][]float64{{.25, -.5}, {.75, .125}}
			total := 0.0
			want := [2]float64{}
			for _, pos := range positions {
				if pos < 0 || (pos >= 2 && p-pos >= 3) {
					continue
				}
				score := 0.0
				for d := 0; d < 2; d++ {
					score += queries[h][d] * float64((pos%7)-3+h+d) / 8
				}
				weight := math.Exp(score / math.Sqrt(2))
				total += weight
				for d := 0; d < 2; d++ {
					want[d] += weight * float64(pos*4+h*2+d+1) / 16
				}
			}
			for d := 0; d < 2; d++ {
				want[d] /= total
				if math.Abs(float64(got[0][0][h][d])-want[d]) > 1e-5 {
					t.Fatalf("p%d h%d d%d got%g want%g", p, h, d, got[0][0][h][d], want[d])
				}
			}
		}
		if p == 4 {
			dir, err := os.MkdirTemp(filepath.Join(os.Getenv("HOME"), "tmp"), "tmc-cache-checkpoint-")
			if err != nil {
				t.Fatal(err)
			}
			flat, err := cfg.SerializeTensors(cache)
			if err != nil {
				t.Fatal(err)
			}
			for i, tensor := range flat {
				if err := tensor.Save(filepath.Join(dir, fmt.Sprintf("%d.tensor", i))); err != nil {
					t.Fatal(err)
				}
			}
			restored := make([]*tensors.Tensor, len(flat))
			for i := range restored {
				restored[i], err = tensors.Load(filepath.Join(dir, fmt.Sprintf("%d.tensor", i)))
				if err != nil {
					t.Fatal(err)
				}
			}
			for _, tensor := range cache {
				tensor.FinalizeAll()
			}
			cache = cfg.DeserializeTensors(restored)
			e.Finalize()
			restoredBackend, err := New(path)
			if err != nil {
				t.Fatal(err)
			}
			defer func() {
				if err := restoredBackend.Close(); err != nil {
					t.Error(err)
				}
			}()
			scope = model.NewStore().Scope("/layer")
			e, err = graph.NewExec(restoredBackend, graphFn)
			if err != nil {
				t.Fatal(err)
			}
			defer e.Finalize()
			t.Logf("restored cache at position 5 into fresh backend from %s", dir)
		}

	}
}
