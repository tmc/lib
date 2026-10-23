package cudaplan_test

import (
	"errors"
	"fmt"
	"math"
	"slices"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudaattn"
	"github.com/tmc/lib/nvidia/x/cudakernel"
	"github.com/tmc/lib/nvidia/x/cudalayer"
)

const (
	hybridTokens  = 3
	hybridDim     = 4
	hybridFFN     = 6
	hybridVocab   = 3
	hybridEpsilon = float32(0.125)
	hybridScale   = float32(0.5)
)

var errHybridNumericalMismatch = errors.New("hybrid: numerical mismatch")

// This fixed fixture is not a graph compiler or a new execution backend. The
// five CUDA entries below are unchanged production producers. The final logits
// consume DownResidual through cublasltx.MatMulRowMajor in the native fixture.
type hybridFixture struct {
	data        map[string][]float32
	rows        []int32
	generations []uint64
	masks       []uint64
}

func newHybridFixture() hybridFixture {
	f := hybridFixture{data: make(map[string][]float32), rows: []int32{0, 1, 2}, generations: []uint64{11, 12, 13}, masks: []uint64{1, 3, 7}}
	for name, size := range map[string]int{
		"input": hybridTokens * hybridDim, "norm_weight": hybridDim, "post_weight": hybridDim,
		"q_weight": hybridDim * hybridDim, "k_weight": hybridDim * hybridDim, "v_weight": hybridDim * hybridDim,
		"o_weight": hybridDim * hybridDim, "gate_weight": hybridFFN * hybridDim, "up_weight": hybridFFN * hybridDim,
		"down_weight": hybridDim * hybridFFN, "head_weight": hybridDim * hybridVocab,
	} {
		v := make([]float32, size)
		for i := range v {
			v[i] = float32((i*5+len(name))%17-7) / 16
		}
		f.data[name] = v
	}
	f.data["norm_weight"] = []float32{0.5, 1, 1.5, 0.75}
	f.data["post_weight"] = []float32{1.25, 0.5, 1, 1.5}
	return f
}

// hybridReference is a separate token-major scalar reference. It calls no
// production shader, tree-attention reference or native result producer.
// Matrix weights are output-major except the final row-major logits head.
func hybridReference(f hybridFixture) (map[string][]float32, error) {
	if len(f.masks) != hybridTokens || len(f.rows) != hybridTokens || len(f.generations) != hybridTokens {
		return nil, errors.New("hybrid: invalid metadata extent")
	}
	for i := range f.rows {
		if f.rows[i] != int32(i) || f.generations[i] == 0 || f.masks[i] == 0 || f.masks[i]>>hybridTokens != 0 || f.masks[i]>>uint(i+1) != 0 {
			return nil, errors.New("hybrid: unsupported row, generation or causal mask")
		}
	}
	for name, size := range map[string]int{"input": 12, "norm_weight": 4, "post_weight": 4, "q_weight": 16, "k_weight": 16, "v_weight": 16, "o_weight": 16, "gate_weight": 24, "up_weight": 24, "down_weight": 24, "head_weight": 12} {
		if len(f.data[name]) != size {
			return nil, fmt.Errorf("hybrid: invalid %s extent", name)
		}
		for _, v := range f.data[name] {
			if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
				return nil, errors.New("hybrid: nonfinite input")
			}
		}
	}
	r := make(map[string][]float32)
	normalize := func(x, weight []float32) []float32 {
		y := make([]float32, len(x))
		for token := 0; token < hybridTokens; token++ {
			var squares float64
			for d := 0; d < hybridDim; d++ {
				v := float64(x[token*hybridDim+d])
				squares += v * v
			}
			scale := 1 / math.Sqrt(squares/hybridDim+float64(hybridEpsilon))
			for d := 0; d < hybridDim; d++ {
				y[token*hybridDim+d] = float32(float64(x[token*hybridDim+d]) * scale * float64(weight[d]))
			}
		}
		return y
	}
	project := func(x, w []float32, in, out int) []float32 {
		y := make([]float32, hybridTokens*out)
		for token := 0; token < hybridTokens; token++ {
			for j := 0; j < out; j++ {
				var sum float64
				for k := 0; k < in; k++ {
					sum += float64(x[token*in+k]) * float64(w[j*in+k])
				}
				y[token*out+j] = float32(sum)
			}
		}
		return y
	}
	r["normalized"] = normalize(f.data["input"], f.data["norm_weight"])
	for _, name := range []string{"q", "k", "v"} {
		r[name] = project(r["normalized"], f.data[name+"_weight"], hybridDim, hybridDim)
	}
	r["attention"] = make([]float32, hybridTokens*hybridDim)
	for token := 0; token < hybridTokens; token++ {
		scores := make([]float64, hybridTokens)
		maximum := math.Inf(-1)
		for key := 0; key < hybridTokens; key++ {
			if f.masks[token]&(1<<uint(key)) == 0 {
				scores[key] = math.Inf(-1)
				continue
			}
			var dot float64
			for d := 0; d < hybridDim; d++ {
				dot += float64(r["q"][token*hybridDim+d]) * float64(r["k"][key*hybridDim+d])
			}
			scores[key] = dot * float64(hybridScale)
			maximum = math.Max(maximum, scores[key])
		}
		var denominator float64
		for key := range scores {
			scores[key] = math.Exp(scores[key] - maximum)
			denominator += scores[key]
		}
		for d := 0; d < hybridDim; d++ {
			var sum float64
			for key := range scores {
				sum += scores[key] * float64(r["v"][key*hybridDim+d])
			}
			r["attention"][token*hybridDim+d] = float32(sum / denominator)
		}
	}
	r["residual"] = project(r["attention"], f.data["o_weight"], hybridDim, hybridDim)
	for i := range r["residual"] {
		r["residual"][i] += f.data["input"][i]
	}
	r["post_norm"] = normalize(r["residual"], f.data["post_weight"])
	r["gate"] = project(r["post_norm"], f.data["gate_weight"], hybridDim, hybridFFN)
	r["up"] = project(r["post_norm"], f.data["up_weight"], hybridDim, hybridFFN)
	r["activation"] = make([]float32, hybridTokens*hybridFFN)
	for i, g := range r["gate"] {
		r["activation"][i] = float32(float64(g) / (1 + math.Exp(-float64(g))) * float64(r["up"][i]))
	}
	r["hidden"] = project(r["activation"], f.data["down_weight"], hybridFFN, hybridDim)
	for i := range r["hidden"] {
		r["hidden"][i] += r["residual"][i]
	}
	r["logits"] = make([]float32, hybridTokens*hybridVocab)
	for token := 0; token < hybridTokens; token++ {
		for j := 0; j < hybridVocab; j++ {
			var sum float64
			for d := 0; d < hybridDim; d++ {
				sum += float64(r["hidden"][token*hybridDim+d]) * float64(f.data["head_weight"][d*hybridVocab+j])
			}
			r["logits"][token*hybridVocab+j] = float32(sum)
		}
	}
	return r, nil
}

type hybridKernel struct {
	entry, source string
	schema        cudakernel.Schema
	outputs       []string
	shared        uint32
}

func hybridKernels() ([]hybridKernel, error) {
	var out []hybridKernel
	config := cudalayer.ProjectionNormConfig{Dim: hybridDim, Threads: 32, Epsilon: hybridEpsilon}
	reductionBytes, err := config.SharedMemoryBytes()
	if err != nil {
		return nil, err
	}
	add := func(entry, source string, ptrs []string, ints []string, scalar string, outputs []string, shared uint32) error {
		var params []cudakernel.Param
		for _, name := range ptrs {
			params = append(params, cudakernel.Param{Name: name, Type: "ptr", IsPointer: true})
		}
		for _, name := range ints {
			params = append(params, cudakernel.Param{Name: name, Type: "s32"})
		}
		if scalar != "" {
			params = append(params, cudakernel.Param{Name: scalar, Type: "f32"})
		}
		schema, err := cudakernel.NewSchema(params...)
		if err != nil {
			return err
		}
		out = append(out, hybridKernel{entry, source, schema, outputs, shared})
		return nil
	}
	layer := cudalayer.ProjectionResidualRMSNormTemplate
	if err := add("RMSNormQKV", layer, []string{"input", "norm_weight", "q_weight", "k_weight", "v_weight", "token_rows", "row_generations", "token_generations", "row_status", "normalized", "q", "k", "v"}, []string{"dim", "q_dim", "kv_dim", "num_tokens", "metadata_rows"}, "epsilon", []string{"row_status", "normalized", "q", "k", "v"}, uint32(reductionBytes)); err != nil {
		return nil, err
	}
	// Tree attention's output precedes the mask pointer in the actual ABI.
	if err := add("TreeFlashAttentionCUDA", cudaattn.TreeAttentionCUDA, []string{"q", "k", "v", "attention", "masks"}, []string{"num_nodes", "num_heads", "head_dim"}, "scale", []string{"attention"}, uint32(cudaattn.TreeAttentionSharedBytes(hybridTokens, 32))); err != nil {
		return nil, err
	}
	if err := add("ProjectionResidualRMSNorm", layer, []string{"attention", "input", "o_weight", "post_weight", "token_rows", "row_status", "residual", "post_norm"}, []string{"dim", "num_tokens", "metadata_rows"}, "epsilon", []string{"residual", "post_norm"}, uint32(reductionBytes)); err != nil {
		return nil, err
	}
	if err := add("GateUpSiLU", layer, []string{"post_norm", "gate_weight", "up_weight", "token_rows", "row_status", "gate", "up", "activation"}, []string{"dim", "intermediate_dim", "num_tokens", "metadata_rows"}, "", []string{"gate", "up", "activation"}, 0); err != nil {
		return nil, err
	}
	if err := add("DownResidual", layer, []string{"activation", "residual", "down_weight", "token_rows", "row_status", "hidden"}, []string{"dim", "intermediate_dim", "num_tokens", "metadata_rows"}, "", []string{"hidden"}, 0); err != nil {
		return nil, err
	}
	return out, nil
}

func checkHybridValues(got, want []float32) error {
	if len(got) != len(want) {
		return fmt.Errorf("%w: output extent", errHybridNumericalMismatch)
	}
	for i, v := range got {
		tolerance := 1e-4 + 1e-4*math.Abs(float64(want[i]))
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) || math.Abs(float64(v)-float64(want[i])) > tolerance {
			return fmt.Errorf("%w: output %d differs: got %g want %g", errHybridNumericalMismatch, i, v, want[i])
		}
	}
	return nil
}

func TestHybridTransformerFixtureOracle(t *testing.T) {
	f := newHybridFixture()
	want, err := hybridReference(f)
	if err != nil {
		t.Fatal(err)
	}
	// Compare only the attention stage against its independent existing CPU
	// implementation. The full connected result remains our scalar reference.
	attention, err := cudaattn.TreeAttentionReference(want["q"], want["k"], want["v"], f.masks, hybridTokens, 1, hybridDim, hybridScale)
	if err != nil || checkHybridValues(attention, want["attention"]) != nil {
		t.Fatal("attention reference disagreement", err)
	}
	if len(want["logits"]) != 9 || slices.Equal(want["hidden"], f.data["input"]) {
		t.Fatal("fixture omitted block or logits")
	}
	kernels, err := hybridKernels()
	if err != nil || len(kernels) != 5 {
		t.Fatal("production kernels", err)
	}
	for _, k := range kernels {
		if len(k.source) == 0 || len(k.schema.Params) == 0 {
			t.Fatal("missing actual producer", k.entry)
		}
	}
	// Uniform attention and zero FFN provide a separate analytic reference.
	for _, name := range []string{"q_weight", "k_weight", "gate_weight", "up_weight", "down_weight"} {
		clear(f.data[name])
	}
	r, err := hybridReference(f)
	if err != nil {
		t.Fatal(err)
	}
	for token := 0; token < hybridTokens; token++ {
		for d := 0; d < hybridDim; d++ {
			var sum float64
			for key := 0; key <= token; key++ {
				sum += float64(r["v"][key*hybridDim+d])
			}
			if math.Abs(float64(r["attention"][token*hybridDim+d])-sum/float64(token+1)) > 1e-6 {
				t.Fatal("nonuniform zero-score attention")
			}
		}
	}
	if !slices.Equal(r["hidden"], r["residual"]) {
		t.Fatal("zero FFN changed residual")
	}
}

func TestHybridTransformerFixtureControls(t *testing.T) {
	f := newHybridFixture()
	want, err := hybridReference(f)
	if err != nil {
		t.Fatal(err)
	}
	changed := newHybridFixture()
	changed.data["q_weight"][0] += 1
	other, err := hybridReference(changed)
	if err != nil || checkHybridValues(other["logits"], want["logits"]) == nil {
		t.Fatal("projection omission control is insensitive", err)
	}
	changed = newHybridFixture()
	changed.masks[2] = 4
	other, err = hybridReference(changed)
	if err != nil || checkHybridValues(other["logits"], want["logits"]) == nil {
		t.Fatal("mask control is insensitive", err)
	}
	changed = newHybridFixture()
	changed.masks[0] = 2
	if _, err := hybridReference(changed); err == nil {
		t.Fatal("future-token mask admitted")
	}
	changed = newHybridFixture()
	delete(changed.data, "o_weight")
	if _, err := hybridReference(changed); err == nil {
		t.Fatal("missing producer weight admitted")
	}
	missing := slices.Clone(want["logits"])
	missing[0] = float32(math.NaN())
	if checkHybridValues(missing, want["logits"]) == nil {
		t.Fatal("unwritten output admitted")
	}
}
