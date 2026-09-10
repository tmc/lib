package gomlx

import (
	"math"
	"os"
	"testing"

	"github.com/gomlx/compute"
	. "github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/core/tensors"
	"github.com/gomlx/gomlx/ml/model"
	"github.com/gomlx/gomlx/ml/nn"
	"github.com/gomlx/gomlx/ml/train"
	"github.com/gomlx/gomlx/ml/train/loss"
	"github.com/gomlx/gomlx/ml/train/optimizer"
)

// TestNativeTrainerDenseAdapter exercises the public Trainer lifecycle with a
// real nn.Dense model, loss metric, and one optimizer update. The model uses
// the backend fused path; the measured loss and updated parameter are checked
// against the scalar linear-model oracle below.
func TestNativeTrainerDenseAdapter(t *testing.T) {
	path := os.Getenv("TMC_COMPUTE_PLUGIN")
	if path == "" {
		t.Skip("set TMC_COMPUTE_PLUGIN")
	}
	b, err := New(path)
	if err != nil {
		t.Fatal(err)
	}
	store := model.NewStore()
	t.Cleanup(func() {
		store.Finalize()
		if err := b.Close(); err != nil {
			t.Errorf("close backend: %v", err)
		}
	})
	modelFn := func(scope *model.Scope, _ any, inputs []*Node) []*Node {
		x := inputs[0]
		g := x.Graph()
		modelScope := scope.In("model")
		w := modelScope.VariableWithValue("weights", [][]float32{{1}}).NodeValue(g)
		bias := modelScope.VariableWithValue("bias", []float32{0}).NodeValue(g)
		dense := nn.Dense(x, w, bias, compute.DenseLayoutInputOutputs)
		if dense.Type() != NodeTypeFusedDense {
			panic("trainer model did not select fused dense")
		}
		return []*Node{dense}
	}
	trainer := train.NewTrainer(
		b, store, modelFn, loss.MeanSquaredError,
		optimizer.StochasticGradientDescent().WithDecay(false).WithLearningRate(0.1).Done(),
		nil, nil,
	)
	input := tensors.MustFromAnyValue([][]float32{{2}, {4}})
	labels := tensors.MustFromAnyValue([][]float32{{4}, {8}})
	metrics, err := trainer.TrainStep(train.Batch{Inputs: []*tensors.Tensor{input}, Labels: []*tensors.Tensor{labels}})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		for _, metric := range metrics {
			if err := metric.FinalizeAll(); err != nil {
				t.Errorf("finalize metric: %v", err)
			}
		}
	})
	if len(metrics) == 0 || trainer.GlobalStep() != 1 {
		t.Fatalf("metrics=%d global step=%d", len(metrics), trainer.GlobalStep())
	}
	if got := metrics[0].Value().(float32); math.IsNaN(float64(got)) || math.IsInf(float64(got), 0) || math.Abs(float64(got-10)) > 1e-4 {
		t.Fatalf("loss=%g want 10", got)
	}
	weights := store.GetVariable("/model/weights")
	if weights == nil {
		t.Fatal("trainer did not retain model weights")
	}
	updated := weights.MustValue().Value().([][]float32)[0][0]
	if math.IsNaN(float64(updated)) || math.IsInf(float64(updated), 0) || math.Abs(float64(updated-3)) > 1e-4 {
		t.Fatalf("updated weight=%g want 3", updated)
	}
	bias := store.GetVariable("/model/bias")
	if bias == nil {
		t.Fatal("trainer did not retain model bias")
	}
	updatedBias := bias.MustValue().Value().([]float32)[0]
	if math.IsNaN(float64(updatedBias)) || math.IsInf(float64(updatedBias), 0) || math.Abs(float64(updatedBias-0.6)) > 1e-4 {
		t.Fatalf("updated bias=%g want 0.6", updatedBias)
	}
}

// TestTrainerDenseAdapterRejectsLabelShapeMismatch is a negative consumer
// witness: Trainer must reject a batch whose labels cannot match the model's
// [batch, 1] prediction before claiming a training step.
func TestTrainerDenseAdapterRejectsLabelShapeMismatch(t *testing.T) {
	path := os.Getenv("TMC_COMPUTE_PLUGIN")
	if path == "" {
		t.Skip("set TMC_COMPUTE_PLUGIN")
	}
	b, err := New(path)
	if err != nil {
		t.Fatal(err)
	}
	store := model.NewStore()
	t.Cleanup(func() {
		store.Finalize()
		if err := b.Close(); err != nil {
			t.Errorf("close backend: %v", err)
		}
	})
	modelFn := func(scope *model.Scope, _ any, inputs []*Node) []*Node {
		x := inputs[0]
		g := x.Graph()
		w := scope.VariableWithValue("weights", [][]float32{{1}}).NodeValue(g)
		return []*Node{nn.Dense(x, w, nil, compute.DenseLayoutInputOutputs)}
	}
	trainer := train.NewTrainer(b, store, modelFn, loss.MeanSquaredError,
		optimizer.StochasticGradientDescent().WithDecay(false).WithLearningRate(0.1).Done(), nil, nil)
	beforeStep := trainer.GlobalStep()
	err = func() error {
		_, e := trainer.TrainStep(train.Batch{
			Inputs: []*tensors.Tensor{tensors.MustFromAnyValue([][]float32{{2}, {4}})},
			Labels: []*tensors.Tensor{tensors.MustFromAnyValue([]float32{4, 8})},
		})
		return e
	}()
	if err == nil {
		t.Fatal("accepted labels with incompatible rank")
	}
	if trainer.GlobalStep() != beforeStep {
		t.Fatalf("failed step advanced global step from %d to %d", beforeStep, trainer.GlobalStep())
	}
}
