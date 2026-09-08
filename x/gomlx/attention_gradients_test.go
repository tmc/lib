package gomlx

import (
	"fmt"
	"math"
	"os"
	"testing"

	"github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/core/tensors"
	"github.com/gomlx/gomlx/ml/layers/attention"
)

func TestNativeAttentionGradients(t *testing.T) {
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
	const seq, dim, vd, kvh = 3, 2, 3, 2
	for _, layout := range []attention.AxesLayout{attention.LayoutBHSD, attention.LayoutBSHD} {
		for _, heads := range []int{2, 4} {
			for _, mode := range []string{"plain", "masked", "causal"} {
				t.Run(fmt.Sprintf("layout%d/heads%d/%s", layout, heads, mode), func(t *testing.T) {
					index := func(h, s, d, nh, width int) int {
						if layout == attention.LayoutBHSD {
							return (h*seq+s)*width + d
						}
						return (s*nh+h)*width + d
					}
					shape := func(nh, width int) []int {
						if layout == attention.LayoutBHSD {
							return []int{1, nh, seq, width}
						}
						return []int{1, seq, nh, width}
					}
					data := [][]float64{make([]float64, heads*seq*dim), make([]float64, kvh*seq*dim), make([]float64, kvh*seq*vd)}
					for n, x := range data {
						for i := range x {
							x[i] = float64((i+3*n)%9-4) / 8
						}
					}
					mask := make([]bool, heads*seq*seq)
					for h := 0; h < heads; h++ {
						for s := 0; s < seq; s++ {
							for j := 0; j < seq; j++ {
								mask[index(h, s, j, heads, seq)] = j != seq-1
							}
						}
					}
					loss := func() float64 {
						total := 0.0
						for h := 0; h < heads; h++ {
							for s := 0; s < seq; s++ {
								kh := h / (heads / kvh)
								scores := [seq]float64{}
								max := math.Inf(-1)
								for j := 0; j < seq; j++ {
									if mode == "masked" && j == seq-1 || mode == "causal" && j > s {
										scores[j] = math.Inf(-1)
										continue
									}
									for d := 0; d < dim; d++ {
										scores[j] += data[0][index(h, s, d, heads, dim)] * data[1][index(kh, j, d, kvh, dim)]
									}
									scores[j] /= math.Sqrt(dim)
									max = math.Max(max, scores[j])
								}
								sum := 0.0
								for j := range scores {
									scores[j] = math.Exp(scores[j] - max)
									sum += scores[j]
								}
								for j := range scores {
									for d := 0; d < vd; d++ {
										total += scores[j] / sum * data[2][index(kh, j, d, kvh, vd)]
									}
								}
							}
						}
						return total
					}
					e, err := graph.NewExec(b, func(q, k, v, mask *graph.Node) []*graph.Node {
						opts := attention.CoreOptions{WantCoefficients: true}
						if mode == "masked" {
							opts.AttentionMask = mask
						}
						if mode == "causal" {
							opts.UseCausalMask = true
						}
						out, _ := attention.Core(q, k, v, layout, opts)
						return graph.Gradient(graph.ReduceAllSum(out), q, k, v)
					})
					if err != nil {
						t.Fatal(err)
					}
					defer e.Finalize()
					var inputs []*tensors.Tensor
					for n, x := range data {
						flat := make([]float32, len(x))
						for i := range flat {
							flat[i] = float32(x[i])
						}
						nh, width := kvh, dim
						if n == 0 {
							nh = heads
						}
						if n == 2 {
							width = vd
						}
						inputs = append(inputs, tensors.FromFlatDataAndDimensions(flat, shape(nh, width)...))
					}
					inputs = append(inputs, tensors.FromFlatDataAndDimensions(mask, shape(heads, seq)...))
					for _, x := range inputs {
						defer x.FinalizeAll()
					}
					out, err := e.Call(inputs[0], inputs[1], inputs[2], inputs[3])
					if err != nil {
						t.Fatalf("%+v", err)
					}
					for _, x := range out {
						defer x.FinalizeAll()
					}
					maxAbs, maxRel, wrongDelta := 0.0, 0.0, 0.0
					for n, x := range data {
						got := out[n].Value().([][][][]float32)
						nh, width := kvh, dim
						if n == 0 {
							nh = heads
						}
						if n == 2 {
							width = vd
						}
						for h := 0; h < nh; h++ {
							for s := 0; s < seq; s++ {
								for d := 0; d < width; d++ {
									i := index(h, s, d, nh, width)
									saved := x[i]
									const epsilon = 1e-4
									x[i] = saved + epsilon
									plus := loss()
									x[i] = saved - epsilon
									minus := loss()
									x[i] = saved
									want := (plus - minus) / (2 * epsilon)
									var g float64
									if layout == attention.LayoutBHSD {
										g = float64(got[0][h][s][d])
									} else {
										g = float64(got[0][s][h][d])
									}
									diff := math.Abs(g - want)
									maxAbs = math.Max(maxAbs, diff)
									if math.Abs(want) > 1e-5 {
										maxRel = math.Max(maxRel, diff/math.Abs(want))
									}
									if math.IsNaN(g) || diff > 1e-5+1e-4*math.Abs(want) {
										t.Fatalf("gradient %d [%d,%d,%d] = %v, finite difference %v", n, h, s, d, g, want)
									}
									if n == 0 {
										wrongDelta = math.Max(wrongDelta, math.Abs(g+want))
									}
									if mode == "masked" && n > 0 && s == seq-1 && g != 0 {
										t.Fatalf("masked key/value gradient %v, want zero", g)
									}
								}
							}
						}
					}
					if wrongDelta < 1e-3 {
						t.Fatal("wrong-sign query-gradient control did not diverge")
					}
					t.Logf("dQ/dK/dV finite differences: max abs %.3g, max relative %.3g; wrong-sign delta %.3g", maxAbs, maxRel, wrongDelta)
				})
			}
		}
	}
}
