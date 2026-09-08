package gomlx

import (
	"math"
	"os"
	"reflect"
	"testing"

	"github.com/gomlx/gomlx/core/graph"
)

func TestNativeSoftmax(t *testing.T) {
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
	for _, axis := range []int{0, 1} {
		t.Run(map[int]string{0: "columns", 1: "rows"}[axis], func(t *testing.T) {
			e, err := graph.NewExec(b, func(x *graph.Node) []*graph.Node {
				p := graph.Softmax(x, axis)
				lp := graph.LogSoftmax(x, axis)
				weights := graph.Const(x.Graph(), [][]float32{{-1, 0, 0}, {0, -1, -1}})
				loss := graph.ReduceAllSum(graph.Mul(lp, weights))
				return []*graph.Node{p, lp, graph.Gradient(loss, x)[0]}
			})
			if err != nil {
				t.Fatal(err)
			}
			defer e.Finalize()
			for _, shift := range []float32{0, 10000, -10000} {
				input := [][]float32{{shift + 1, shift + 2, shift + 3}, {shift + 4, shift + 1, shift - 2}}
				out, err := e.Call(input)
				if err != nil {
					t.Fatalf("%+v", err)
				}
				got := [3][][]float32{}
				for i, tensor := range out {
					got[i] = tensor.Value().([][]float32)
					defer tensor.FinalizeAll()
				}
				weights := [2][3]float64{{-1, 0, 0}, {0, -1, -1}}
				for row := 0; row < 2; row++ {
					for col := 0; col < 3; col++ {
						max := math.Inf(-1)
						count := 3
						if axis == 0 {
							count = 2
						}
						value := func(k int) float64 {
							if axis == 0 {
								return float64(input[k][col])
							}
							return float64(input[row][k])
						}
						weight := func(k int) float64 {
							if axis == 0 {
								return weights[k][col]
							}
							return weights[row][k]
						}
						for k := 0; k < count; k++ {
							max = math.Max(max, value(k))
						}
						sum, weightSum := 0.0, 0.0
						for k := 0; k < count; k++ {
							sum += math.Exp(value(k) - max)
							weightSum += weight(k)
						}
						logp := float64(input[row][col]) - max - math.Log(sum)
						p := math.Exp(logp)
						want := [3]float64{p, logp, weights[row][col] - p*weightSum}
						for i, w := range want {
							g := float64(got[i][row][col])
							if math.IsNaN(g) || math.Abs(g-w) > 2e-5 {
								t.Fatalf("axis %d shift %v output %d [%d,%d] = %v, want %v", axis, shift, i, row, col, g, w)
							}
						}
					}
				}
				t.Logf("axis %d shift %v: probabilities, log probabilities and cross-entropy gradients match", axis, shift)
			}
		})
	}
}

func TestNativeReduceMax(t *testing.T) {
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
	for _, tc := range []struct {
		name        string
		input, want any
	}{
		{"negative", []float32{-3, -2, -9}, float32(-2)},
		{"infinity", []float32{float32(math.Inf(-1))}, float32(math.Inf(-1))},
		{"int32", []int32{math.MinInt32, math.MinInt32 + 1}, int32(math.MinInt32 + 1)},
		{"int64", []int64{math.MinInt64, math.MinInt64 + 1}, int64(math.MinInt64 + 1)},
	} {
		t.Run(tc.name, func(t *testing.T) {
			e, err := graph.NewExec(b, func(x *graph.Node) *graph.Node { return graph.ReduceMax(x) })
			if err != nil {
				t.Fatal(err)
			}
			defer e.Finalize()
			out, err := e.Call(tc.input)
			if err != nil {
				t.Fatal(err)
			}
			defer out[0].FinalizeAll()
			if got := out[0].Value(); !reflect.DeepEqual(got, tc.want) {
				t.Fatalf("got %v, want %v", got, tc.want)
			}
		})
	}
}
