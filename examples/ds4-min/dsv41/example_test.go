package dsv41

import (
	"bytes"
	"fmt"

	"github.com/tmc/lib/examples/ds4-min/dsv41engram"
	"github.com/tmc/lib/examples/ds4-min/dsv4cpu"
)

func Example() {
	layer, _ := Schedule(14)
	fmt.Println(layer.Ratio, layer.Owner, layer.PublishKV, layer.Engram)
	// Output: 2 2 true 1
}

func ExampleSchedule() {
	layer, _ := Schedule(24)
	fmt.Println(layer.Ratio, layer.Owner, layer.RefreshIndex)
	// Output: 1 3 true
}

func ExampleBF16() {
	fmt.Printf("%.6f\n", BF16(1.001))
	// Output: 1.000000
}

func ExampleQuantize() {
	x := make([]float32, 32)
	x[0] = 1
	_ = Quantize(x, 32, FP8E8M0)
	fmt.Println(x[0], x[1])
	// Output: 1 0
}

func ExamplePool_Append() {
	var pool Pool
	out := make([]float32, 1)
	_ = pool.Append(nil, []float32{2}, []float32{0}, 1, 0)
	_ = pool.Append(out, []float32{4}, []float32{0}, 1, 1)
	fmt.Println(out)
	// Output: [3]
}

func ExamplePool_Reset() {
	var pool Pool
	_ = pool.Append(nil, []float32{2}, []float32{0}, 1, 0)
	pool.Reset()
	fmt.Println(pool.Append(nil, []float32{4}, []float32{0}, 1, 0))
	// Output: <nil>
}

func ExampleCandidates_Pick() {
	var candidates Candidates
	ids, _ := candidates.Pick(20, 2, []float32{1, 3, 2})
	fmt.Println(ids)
	// Output: [1 2 0]
}

func ExampleCandidates_Reset() {
	var candidates Candidates
	_, _ = candidates.Pick(20, 0, []float32{1})
	candidates.Reset()
	_, err := candidates.Pick(21, 0, nil)
	fmt.Println(err)
	// Output: missing index selection
}

func ExampleNewKVState() {
	state, err := NewKVState(2048)
	fmt.Println(state != nil, err)
	// Output: true <nil>
}

func ExampleKVState_Commit() {
	state, _ := NewKVState(2)
	fmt.Println(state.Commit(0, 0, make([]float32, HeadDim), nil, nil))
	// Output: <nil>
}

func ExampleKVState_Rows() {
	state, _ := NewKVState(2)
	_ = state.Commit(0, 0, make([]float32, HeadDim), nil, nil)
	raw, kv, index, err := state.Rows(0, 0)
	fmt.Println(len(raw), len(kv), len(index), err)
	// Output: 512 0 0 <nil>
}

func ExampleKVState_Reset() {
	state, _ := NewKVState(2)
	_ = state.Commit(0, 0, make([]float32, HeadDim), nil, nil)
	state.Reset()
	fmt.Println(state.Commit(0, 0, make([]float32, HeadDim), nil, nil))
	// Output: <nil>
}

func ExampleAddEngram() {
	residual := []float32{1, 1, 1, 1}
	_ = AddEngram(residual, []float32{0, 0, 0, 0, 2}, make([]float32, 4), make([]float32, 4), 1, 1e-20)
	fmt.Println(residual)
	// Output: [2 2 2 2]
}

func exampleEngram() *EngramState {
	spec := dsv41engram.LayoutSpec{TokenMap: []uint32{0}, CompressedVocabSize: 1, Rows: [2]uint32{48, 48}}
	for i := range spec.Multipliers {
		for j := range spec.Multipliers[i] {
			spec.Multipliers[i][j] = 1
		}
		for j := range spec.Primes[i] {
			spec.Primes[i][j] = 2
		}
	}
	layout, err := dsv41engram.NewLayout(spec)
	if err != nil {
		panic(err)
	}
	raw := make([]byte, 48*dsv41engram.RowBytes)
	table, err := dsv41engram.NewTable(bytes.NewReader(raw), int64(len(raw)), 0, 48)
	if err != nil {
		panic(err)
	}
	weights := make([]byte, 5*24*256*2)
	matrix, err := dsv4cpu.NewMatrix(bytes.NewReader(weights), 0, int64(len(weights)), 5, 24*256, dsv4cpu.F16)
	if err != nil {
		panic(err)
	}
	w := EngramWeights{table, matrix, make([]float32, 4), make([]float32, 4)}
	state, err := NewEngramState(layout, [2]EngramWeights{w, w})
	if err != nil {
		panic(err)
	}
	return state
}

func ExampleNewEngramState() {
	state := exampleEngram() // Bounded synthetic table and immutable F16 weights.
	fmt.Println(state.Prepare(0, true))
	// Output: <nil>
}

func ExampleEngramState_Prepare() {
	state := exampleEngram()
	fmt.Println(state.Prepare(0, false)) // Image position breaks n-gram history.
	// Output: <nil>
}

func ExampleEngramState_Apply() {
	state := exampleEngram()
	_ = state.Prepare(0, true)
	residual := []float32{1, 1, 1, 1}
	fmt.Println(state.Apply(1, residual), residual)
	// Output: <nil> [1 1 1 1]
}

func ExampleEngramState_Reset() {
	state := exampleEngram()
	_ = state.Prepare(0, true)
	state.Reset()
	fmt.Println(state.Apply(1, make([]float32, 4)))
	// Output: engram token not prepared
}
