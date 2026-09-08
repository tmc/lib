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

func TestNativeAttentionCore(t *testing.T) {
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
	const seq, dim, valueDim, kvHeads = 3, 2, 3, 2
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
					q, k, v := make([]float32, heads*seq*dim), make([]float32, kvHeads*seq*dim), make([]float32, kvHeads*seq*valueDim)
					for i := range q {
						q[i] = float32(i%7-3) / 4
					}
					for i := range k {
						k[i] = float32(i%5-2) / 3
					}
					for i := range v {
						v[i] = float32(i%11-5) / 2
					}
					mask := make([]bool, heads*seq*seq)
					for h := 0; h < heads; h++ {
						for s := 0; s < seq; s++ {
							for j := 0; j < seq; j++ {
								mask[index(h, s, j, heads, seq)] = j != seq-1
							}
						}
					}
					e, err := graph.NewExec(b, func(q, k, v, mask *graph.Node) []*graph.Node {
						opts := attention.CoreOptions{WantCoefficients: true}
						if mode == "masked" {
							opts.AttentionMask = mask
						}
						if mode == "causal" {
							opts.UseCausalMask = true
						}
						out, coeff := attention.Core(q, k, v, layout, opts)
						return []*graph.Node{out, coeff}
					})
					if err != nil {
						t.Fatal(err)
					}
					defer e.Finalize()
					run := func() []float32 {
						t.Helper()
						inputs := []*tensors.Tensor{
							tensors.FromFlatDataAndDimensions(q, shape(heads, dim)...),
							tensors.FromFlatDataAndDimensions(k, shape(kvHeads, dim)...),
							tensors.FromFlatDataAndDimensions(v, shape(kvHeads, valueDim)...),
							tensors.FromFlatDataAndDimensions(mask, shape(heads, seq)...),
						}
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
						got := out[0].Value().([][][][]float32)
						coeff := out[1].Value().([][][][]float32)
						at := func(x [][][][]float32, h, s, d int) float64 {
							if layout == attention.LayoutBHSD {
								return float64(x[0][h][s][d])
							}
							return float64(x[0][s][h][d])
						}
						var result []float32
						for h := 0; h < heads; h++ {
							for s := 0; s < seq; s++ {
								kh := h / (heads / kvHeads)
								scores := make([]float64, seq)
								max := math.Inf(-1)
								for j := 0; j < seq; j++ {
									if (mode == "masked" && j == seq-1) || (mode == "causal" && j > s) {
										scores[j] = math.Inf(-1)
										continue
									}
									for d := 0; d < dim; d++ {
										scores[j] += float64(q[index(h, s, d, heads, dim)]) * float64(k[index(kh, j, d, kvHeads, dim)])
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
									scores[j] /= sum
									if got := at(coeff, h, s, j); math.IsNaN(got) || math.Abs(got-scores[j]) > 2e-5 {
										t.Fatalf("coefficient [%d,%d,%d] = %v, want %v", h, s, j, got, scores[j])
									}
								}
								for d := 0; d < valueDim; d++ {
									want := 0.0
									for j := range scores {
										want += scores[j] * float64(v[index(kh, j, d, kvHeads, valueDim)])
									}
									g := at(got, h, s, d)
									if math.IsNaN(g) || math.Abs(g-want) > 2e-5 {
										t.Fatalf("output [%d,%d,%d] = %v, want %v", h, s, d, g, want)
									}
									result = append(result, float32(g))
								}
							}
						}
						return result
					}
					before := run()
					if mode == "masked" {
						for h := 0; h < kvHeads; h++ {
							for d := 0; d < dim; d++ {
								k[index(h, seq-1, d, kvHeads, dim)] = 10000
							}
							for d := 0; d < valueDim; d++ {
								v[index(h, seq-1, d, kvHeads, valueDim)] = -10000
							}
						}
						after := run()
						for i := range before {
							if before[i] != after[i] {
								t.Fatal("masked key/value perturbation changed output")
							}
						}
					}
				})
			}
		}
	}
}
