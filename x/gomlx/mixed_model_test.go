package gomlx

import (
	"math"
	"os"
	"path/filepath"
	"reflect"
	"testing"

	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/dtypes/bfloat16"
	"github.com/gomlx/compute/dtypes/float16"
	"github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/core/tensors"
	"github.com/gomlx/gomlx/ml/layers"
	"github.com/gomlx/gomlx/ml/layers/norm"
	"github.com/gomlx/gomlx/ml/model"
	"github.com/gomlx/gomlx/ml/model/checkpoint"
)

func TestNativeMixedModel(t *testing.T) {
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
	weights := [][]float32{{.125, -.5, .75}, {-.25, .375, .5}}
	inputs := [][]float32{{1, -.5}, {.25, .75}}
	for _, dtype := range []dtypes.DType{dtypes.Float16, dtypes.BFloat16} {
		t.Run(dtype.String(), func(t *testing.T) {
			quant := func(x float32) float32 {
				if dtype == dtypes.Float16 {
					return float16.FromFloat32(x).Float32()
				}
				return bfloat16.FromFloat32(x).Float32()
			}
			half := func(x [][]float32) any {
				if dtype == dtypes.Float16 {
					v := make([][]float16.Float16, len(x))
					for i, row := range x {
						v[i] = make([]float16.Float16, len(row))
						for j, n := range row {
							v[i][j] = float16.FromFloat32(n)
						}
					}
					return v
				}
				v := make([][]bfloat16.BFloat16, len(x))
				for i, row := range x {
					v[i] = make([]bfloat16.BFloat16, len(row))
					for j, n := range row {
						v[i][j] = bfloat16.FromFloat32(n)
					}
				}
				return v
			}
			store := model.NewStore()
			defer store.Finalize()
			variable := store.RootScope().In("projection").In("dense").VariableWithValue("weights", half(weights))
			build := func(scope *model.Scope, x *graph.Node) []*graph.Node {
				full := graph.ConvertDType(x, dtypes.Float32)
				normalized := norm.RMSNorm(scope.In("norm"), full).WithScale(false).Done()
				projected := layers.Dense(scope.In("projection"), graph.ConvertDType(normalized, dtype), false, 3)
				logits := graph.ConvertDType(projected, dtypes.Float32)
				return []*graph.Node{logits, graph.Softmax(logits, -1)}
			}
			e, err := model.NewExec(b, store, build)
			if err != nil {
				t.Fatal(err)
			}
			defer e.Finalize()
			out, err := e.Call(half(inputs))
			if err != nil {
				t.Fatal(err)
			}
			defer func() {
				for _, v := range out {
					v.FinalizeAll()
				}
			}()
			gotLogits := out[0].Value().([][]float32)
			gotProb := out[1].Value().([][]float32)
			strict, excess := true, true
			for row, x := range inputs {
				a, z := quant(x[0]), quant(x[1])
				denom := float32(math.Sqrt(float64((a*a+z*z)/2 + float32(1e-6))))
				a, z = quant(a/denom), quant(z/denom)
				logits := make([]float32, 3)
				max := float32(math.Inf(-1))
				for col := range logits {
					// XLA may retain excess precision in an intermediate dot
					// followed by conversion to F32. Accept either permitted
					// result, without relaxing the arithmetic comparison.
					full := a*quant(weights[0][col]) + z*quant(weights[1][col])
					got := gotLogits[row][col]
					strict = strict && got == quant(full)
					excess = excess && got == full
					if got != quant(full) && got != full {
						t.Fatalf("logit[%d,%d]=%v want rounded %v or excess %v", row, col, got, quant(full), full)
					}
					logits[col] = got
					if logits[col] > max {
						max = logits[col]
					}
				}
				var sum float64
				for _, v := range logits {
					sum += math.Exp(float64(v - max))
				}
				for col := range logits {
					want := math.Exp(float64(logits[col]-max)) / sum
					if math.IsNaN(float64(gotProb[row][col])) || math.Abs(float64(gotProb[row][col])-want) > 2e-6 {
						t.Fatalf("probability[%d,%d]=%v want%v", row, col, gotProb[row][col], want)
					}
				}
			}
			if !strict && !excess {
				t.Fatal("logits mix strict and excess precision results")
			}
			if variable.MustValue().DType() != dtype {
				t.Fatal("weights widened in model store")
			}
			dir, err := os.MkdirTemp(filepath.Join(os.Getenv("HOME"), "tmp"), "tmc-mixed-model-")
			if err != nil {
				t.Fatal(err)
			}
			defer os.RemoveAll(dir)
			saver, err := checkpoint.Build(store).Dir(dir).Done()
			if err != nil {
				t.Fatal(err)
			}
			if err := saver.Save(); err != nil {
				t.Fatal(err)
			}
			restored := model.NewStore()
			defer restored.Finalize()
			if _, err := checkpoint.Load(restored).Dir(dir).Immediate().Done(); err != nil {
				t.Fatal(err)
			}
			rv := restored.GetVariable("/projection/dense/weights")
			if rv == nil || rv.MustValue().DType() != dtype || !reflect.DeepEqual(rv.MustValue().Value(), variable.MustValue().Value()) {
				t.Fatal("checkpoint changed stored weights or dtype")
			}
			fresh, err := New(path)
			if err != nil {
				t.Fatal(err)
			}
			defer func() {
				if err := fresh.Close(); err != nil {
					t.Error(err)
				}
			}()
			re, err := model.NewExec(fresh, restored, build)
			if err != nil {
				t.Fatal(err)
			}
			defer re.Finalize()
			replayed, err := re.Call(half(inputs))
			if err != nil {
				t.Fatal(err)
			}
			defer func() {
				for _, v := range replayed {
					v.FinalizeAll()
				}
			}()
			for i, v := range replayed {
				if !reflect.DeepEqual(v.Value(), out[i].Value()) {
					t.Fatalf("restored output %d changed", i)
				}
			}
			// Changing one stored weight must change the actual model's next output.
			changed := [][]float32{{2, -.5, .75}, {-.25, .375, .5}}
			variable.MustSetValue(tensors.MustFromAnyValue(half(changed)))
			bad, err := e.Call(half(inputs))
			if err != nil {
				t.Fatal(err)
			}
			defer func() {
				for _, v := range bad {
					v.FinalizeAll()
				}
			}()
			if bad[0].Value().([][]float32)[0][0] == gotLogits[0][0] {
				t.Fatal("changed-weight control did not change logits")
			}
		})
	}
}
