package gomlx

import (
	"fmt"
	"math"
	"os"
	"testing"

	"github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/core/tensors"
	"github.com/gomlx/gomlx/ml/layers/attention/pos"
)

func TestNativeRoPE(t *testing.T) {
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
	for _, seqAxis := range []int{1, 2} {
		for _, interleaved := range []bool{false, true} {
			for _, partial := range []bool{false, true} {
				t.Run(fmt.Sprintf("axis%d/interleaved%v/partial%v", seqAxis, interleaved, partial), func(t *testing.T) {
					const heads, seq, width = 2, 3, 8
					dims := []int{1, seq, heads, width}
					if seqAxis == 2 {
						dims = []int{1, heads, seq, width}
					}
					index := func(h, s, d int) int {
						if seqAxis == 1 {
							return (s*heads+h)*width + d
						}
						return (h*seq+s)*width + d
					}
					q, k := make([]float32, heads*seq*width), make([]float32, heads*seq*width)
					for i := range q {
						q[i] = float32(i%13-6) / 4
						k[i] = float32(i%11-5) / 8
					}
					positions := []float32{0, 7, 23}
					start, end, scale := 0, width, 1.0
					if partial {
						start, end, scale = 2, 6, 2
					}
					rope := pos.NewRoPEWithDimRange(10000, start, end).WithInterleaved(interleaved).WithLinearScaling(scale)
					e, err := graph.NewExec(b, func(q, k, p *graph.Node) []*graph.Node {
						a, c := rope.EncodeQK(q, k, p, seqAxis)
						return []*graph.Node{a, c}
					})
					if err != nil {
						t.Fatal(err)
					}
					t.Cleanup(e.Finalize)
					qt := tensors.FromFlatDataAndDimensions(q, dims...)
					kt := tensors.FromFlatDataAndDimensions(k, dims...)
					defer qt.FinalizeAll()
					defer kt.FinalizeAll()
					out, err := e.Call(qt, kt, positions)
					if err != nil {
						t.Fatalf("%+v", err)
					}
					for _, x := range out {
						defer x.FinalizeAll()
					}
					changed := false
					for n, data := range [][]float32{q, k} {
						got := out[n].Value().([][][][]float32)
						for h := 0; h < heads; h++ {
							for s := 0; s < seq; s++ {
								want := make([]float64, width)
								for d := range want {
									want[d] = float64(data[index(h, s, d)])
								}
								half := (end - start) / 2
								for j := 0; j < half; j++ {
									a, c := start+j, start+j+half
									if interleaved {
										a, c = start+2*j, start+2*j+1
									}
									angle := float64(positions[s]) / scale / math.Pow(10000, float64(2*j)/float64(end-start))
									x, y := want[a], want[c]
									want[a] = x*math.Cos(angle) - y*math.Sin(angle)
									want[c] = x*math.Sin(angle) + y*math.Cos(angle)
								}
								for d, w := range want {
									var g float64
									if seqAxis == 1 {
										g = float64(got[0][s][h][d])
									} else {
										g = float64(got[0][h][s][d])
									}
									if math.IsNaN(g) || math.Abs(g-w) > 1e-5 {
										t.Fatalf("output%d [%d,%d,%d] = %v, want %v", n, h, s, d, g, w)
									}
									if math.Abs(g-float64(data[index(h, s, d)])) > 1e-3 {
										changed = true
									}
								}
							}
						}
					}
					if !changed {
						t.Fatal("zero-position control failed to differ from rotated output")
					}
				})
			}
		}
	}
}

func TestNativeFloat64Preservation(t *testing.T) {
	path := os.Getenv("TMC_COMPUTE_PLUGIN")
	if path == "" {
		t.Skip("set TMC_COMPUTE_PLUGIN")
	}
	b, err := New(path)
	if err != nil {
		t.Fatal(err)
	}
	defer b.Close()
	values := []float64{math.Nextafter(1, 2), 9007199254740991, math.SmallestNonzeroFloat64, math.Copysign(0, -1)}
	e, err := graph.NewExec(b, func(x *graph.Node) []*graph.Node {
		return []*graph.Node{x, graph.Const(x.Graph(), values)}
	})
	if err != nil {
		t.Fatal(err)
	}
	defer e.Finalize()
	out, err := e.Call(values)
	if err != nil {
		t.Fatal(err)
	}
	for _, v := range out {
		defer v.FinalizeAll()
		for i, x := range v.Value().([]float64) {
			if math.Float64bits(x) != math.Float64bits(values[i]) {
				t.Fatalf("element %d: bits %x, want %x", i, math.Float64bits(x), math.Float64bits(values[i]))
			}
		}
	}
}
