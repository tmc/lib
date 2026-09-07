package llama_test

import (
	"context"
	"fmt"
	"os"

	"github.com/tmc/lib/x/xla"
	"github.com/tmc/lib/x/xla/llama"
)

// This example runs when a CPU PJRT plugin is supplied. Its small synthetic
// weights demonstrate ownership and forwarding, not trained-model quality.
func Example() {
	path := os.Getenv("XLA_PJRT_CPU_PLUGIN")
	if path == "" {
		return
	}
	engine, err := xla.NewEngine(path)
	if err != nil {
		fmt.Println(err)
		return
	}
	defer engine.Close()
	c := llama.Config{BatchSize: 1, Capacity: 4, VocabSize: 2, HiddenSize: 2, IntermediateSize: 2, QueryHeads: 1, KVHeads: 1, HeadSize: 2, NormEpsilon: 1e-5, RopeTheta: 10000}
	zero := make([]float32, 4)
	w := llama.Weights{Embedding: []float32{1, 0, 0, 1}, Norm: []float32{1, 1}, Layers: []llama.LayerWeights{{AttentionNorm: []float32{1, 1}, MLPNorm: []float32{1, 1}, Query: zero, Key: zero, Value: zero, AttentionOutput: zero, Gate: zero, Up: zero, Down: zero}}}
	model, err := llama.New(context.Background(), engine, c, w)
	if err != nil {
		fmt.Println(err)
		return
	}
	defer model.Close()
	logits, err := model.Forward(context.Background(), []int32{0}, nil)
	if err != nil {
		fmt.Println(err)
		return
	}
	defer logits.Close()
	if model.Lengths()[0] != 1 {
		fmt.Println("unexpected length")
	}
	// Output:
}

func ExampleConfig() {
	c := llama.Config{QueryHeads: 8, KVHeads: 2}
	fmt.Println(c.QueryHeads/c.KVHeads, "query heads per KV head")
	// Output: 4 query heads per KV head
}
func ExampleLayerWeights() {
	var weights llama.LayerWeights
	fmt.Println(weights.QueryBias == nil)
	// Output: true
}
func ExampleWeights() {
	weights := llama.Weights{Embedding: []float32{1, 0, 0, 1}}
	fmt.Println(weights.LMHead == nil)
	// Output: true
}
func ExampleModel() {
	var model llama.Model
	fmt.Println(model.Lengths())
	// Output: []
}
func ExampleNew() {
	_, err := llama.New(context.Background(), nil, llama.Config{}, llama.Weights{})
	fmt.Println(err)
	// Output: llama: dimensions must be positive
}
func ExampleModel_Lengths() {
	var model llama.Model
	fmt.Println(model.Lengths())
	// Output: []
}
func ExampleModel_Forward() {
	var model llama.Model
	_, err := model.Forward(context.Background(), []int32{1}, nil)
	fmt.Println(err)
	// Output: llama: model is closed
}
func ExampleModel_Device() {
	var model llama.Model
	_, err := model.Device()
	fmt.Println(err)
	// Output: llama: model is closed
}
func ExampleModel_Reset() {
	var model llama.Model
	fmt.Println(model.Reset())
	// Output: llama: model is closed
}
func ExampleModel_Close() {
	var model llama.Model
	fmt.Println(model.Close())
	// Output: <nil>
}

func ExampleEncodedTensor() {
	tensor := llama.EncodedTensor{ElementType: xla.BF16, Data: []byte{0x80, 0x3f}}
	fmt.Println(tensor.ElementType == xla.BF16, len(tensor.Data))
	// Output: true 2
}
func ExampleEncodedLayerWeights() {
	var layer llama.EncodedLayerWeights
	fmt.Println(layer.QueryBias.Data == nil)
	// Output: true
}
func ExampleEncodedWeights() {
	var weights llama.EncodedWeights
	fmt.Println(weights.LMHead.Data == nil)
	// Output: true
}
func ExampleNewEncoded() {
	_, err := llama.NewEncoded(context.Background(), nil, llama.Config{}, llama.EncodedWeights{})
	fmt.Println(err)
	// Output: llama: dimensions must be positive
}
