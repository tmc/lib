package gomlx

import (
	"fmt"
	"os"
	"testing"

	"github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/core/tensors"
	"github.com/gomlx/gomlx/ml/layers/attention/kvcache"
	"github.com/gomlx/gomlx/ml/model"
)

func TestNativeKVCache(t *testing.T) {
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
	for _, sinks := range []int{0, 2} {
		for _, window := range []int{0, 3} {
			for _, chunk := range []int{1, 3} {
				t.Run(fmt.Sprintf("sinks%d/window%d/chunk%d", sinks, window, chunk), func(t *testing.T) {
					const capacity = 6
					store := model.NewStore()
					scope := store.Scope("/layer")
					cfg := kvcache.NewKVCache().WithSinkPositions(sinks).WithMinSeqLenPerLayerType([]int{capacity, capacity}).WithMaxSeqLenPerLayerType([]int{capacity, capacity})
					e, err := graph.NewExec(b, func(k, v, nk, nv, q, p *graph.Node) []*graph.Node {
						cache := kvcache.KVCacheNodes{scope.Scope() + kvcache.KeySuffix: k, scope.Scope() + kvcache.ValueSuffix: v}
						cfg.Update(scope, cache, nk, nv, p)
						key, value := cfg.Get(cache, scope.Scope())
						mask := cfg.BuildAttentionMask(scope, cache, q, p, true, window)
						return []*graph.Node{key, value, mask}
					})
					if err != nil {
						t.Fatal(err)
					}
					defer e.Finalize()
					kt := tensors.FromFlatDataAndDimensions(make([]float32, capacity*4), 1, 2, capacity, 2)
					vt := tensors.FromFlatDataAndDimensions(make([]float32, capacity*4), 1, 2, capacity, 2)
					defer func() { kt.FinalizeAll(); vt.FinalizeAll() }()
					positions := make([]int, capacity)
					for i := range positions {
						positions[i] = -1
					}
					for p := 0; p < 14; p += chunk {
						data := make([]float32, chunk*4)
						for h := 0; h < 2; h++ {
							for c := 0; c < chunk; c++ {
								for d := 0; d < 2; d++ {
									data[(h*chunk+c)*2+d] = float32((p+c)*10 + h*2 + d + 1)
								}
							}
						}
						nk := tensors.FromFlatDataAndDimensions(data, 1, 2, chunk, 2)
						vals := make([]float32, chunk*4)
						for i := range vals {
							vals[i] = -data[i]
						}
						nv := tensors.FromFlatDataAndDimensions(vals, 1, 2, chunk, 2)
						q := tensors.FromFlatDataAndDimensions(make([]float32, chunk*4), 1, chunk, 2, 2)
						out, err := e.Call(kt, vt, nk, nv, q, int32(p))
						nk.FinalizeAll()
						nv.FinalizeAll()
						q.FinalizeAll()
						if err != nil {
							t.Fatalf("position %d: %+v", p, err)
						}
						kt.FinalizeAll()
						vt.FinalizeAll()
						kt, vt = out[0], out[1]
						for c := 0; c < chunk; c++ {
							pos := p + c
							slot := pos % capacity
							if sinks > 0 && pos >= capacity {
								slot = sinks + (pos-sinks)%(capacity-sinks)
							}
							positions[slot] = pos
						}
						keys := kt.Value().([][][][]float32)
						values := vt.Value().([][][][]float32)
						mask := out[2].Value().([][][]bool)
						out[2].FinalizeAll()
						for j, position := range positions {
							for h := 0; h < 2; h++ {
								for d := 0; d < 2; d++ {
									want := float32(0)
									if position >= 0 {
										want = float32(position*10 + h*2 + d + 1)
									}
									if keys[0][h][j][d] != want || values[0][h][j][d] != -want {
										t.Fatalf("p%d slot%d head%d dim%d: key%g value%g want%g", p, j, h, d, keys[0][h][j][d], values[0][h][j][d], want)
									}
								}
							}
							for c := 0; c < chunk; c++ {
								want := position >= 0 && position <= p+c && (window == 0 || position < sinks || p+c-position < window)
								if mask[0][c][j] != want {
									t.Fatalf("p%d slot%d position%d: mask%v want%v", p, j, position, mask[0][c][j], want)
								}
							}
						}
					}
				})
			}
		}
	}
}
