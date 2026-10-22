package dsv41

import (
	"fmt"
	"io"
	"testing"

	"github.com/tmc/lib/gguf"
)

func metadataFixture() *gguf.File {
	g := &gguf.File{Metadata: map[string]gguf.Value{"general.architecture": {TypeID: gguf.TypeString, Value: "deepseek41"}}}
	for key, value := range flashIntegers {
		g.Metadata["deepseek41."+key] = gguf.Value{TypeID: gguf.TypeUint32, Value: value}
	}
	for key, value := range flashFloats {
		g.Metadata["deepseek41."+key] = gguf.Value{TypeID: gguf.TypeFloat32, Value: value}
	}
	for key, value := range map[string]string{"scoring_func": "sqrtsoftplus", "hidden_act": "silu", "topk_method": "noaux_tc", "engram.encoding": "e4m3_e8m0_32_row264"} {
		g.Metadata["deepseek41."+key] = gguf.Value{TypeID: gguf.TypeString, Value: value}
	}
	g.Metadata["deepseek41.norm_topk_prob"] = gguf.Value{TypeID: gguf.TypeBool, Value: true}
	arrays := map[string][]uint32{"kv_source_layer_ids": {2, 8, 14, 20}, "index_source_layer_ids": {2, 8, 14, 20, 24, 28, 32, 36}, "engram.layer_ids": {1, 14}, "engram.rows": {384006168, 384016682}, "engram.pad_id": nil}
	delete(arrays, "engram.pad_id")
	ratios := make([]uint32, 40)
	for i := range ratios {
		l, _ := Schedule(i)
		ratios[i] = uint32(l.Ratio)
	}
	arrays["compress_ratios"] = ratios
	arrays["engram.token_map"] = make([]uint32, 129280)
	primes := make([]uint32, 48)
	for i := range primes {
		primes[i] = 2
	}
	primes[23] = 384006168 - 46
	primes[47] = 384016682 - 46
	arrays["engram.primes"] = primes
	for key, values := range arrays {
		a := make([]any, len(values))
		for i, v := range values {
			a[i] = v
		}
		g.Metadata["deepseek41."+key] = gguf.Value{TypeID: gguf.TypeArray, Value: a}
	}
	g.Metadata["deepseek41.engram.pad_id"] = gguf.Value{TypeID: gguf.TypeUint32, Value: uint32(0)}
	multipliers := make([]any, 8)
	for i := range multipliers {
		multipliers[i] = uint64(1)
	}
	g.Metadata["deepseek41.engram.multipliers"] = gguf.Value{TypeID: gguf.TypeArray, Value: multipliers}
	return g
}

type countReader struct{ reads int }

func (r *countReader) ReadAt(dst []byte, off int64) (int, error) {
	r.reads++
	clear(dst)
	return len(dst), nil
}

func modelFixture(t *testing.T) *gguf.File {
	t.Helper()
	g := metadataFixture()
	g.DataOffset = 4096
	var offset uint64
	for _, need := range tensorRequirements() {
		var kind uint32
		for need.types&(uint64(1)<<kind) == 0 {
			kind++
		}
		if need.types&(1<<8) != 0 {
			kind = 8
		}
		if len(need.dims) == 3 {
			kind = 16
			if need.dims[0] == 2304 {
				kind = 10
			}
		}
		tensor := gguf.Tensor{Name: need.name, TypeID: kind, Offset: offset}
		for _, dim := range need.dims {
			tensor.Dimensions = append(tensor.Dimensions, uint64(dim))
		}
		n, err := tensor.Bytes()
		if err != nil {
			t.Fatal(err)
		}
		offset += n
		g.Tensors = append(g.Tensors, tensor)
	}
	g.Size = g.DataOffset + offset
	return g
}

func TestBindReaderLifetimeAndBounds(t *testing.T) {
	g := modelFixture(t)
	r := new(countReader)
	m, err := Bind(r, int64(g.Size), g)
	if err != nil {
		t.Fatal(err)
	}
	if r.reads != 0 {
		t.Fatal("Bind read payloads")
	}
	e, err := m.NewEngram()
	if err != nil {
		t.Fatal(err)
	}
	if r.reads != 16 {
		t.Fatal("expected only eight normalization rows per site", r.reads)
	}
	if err := e.Prepare(0, true); err != nil {
		t.Fatal(err)
	}
	if r.reads <= 16 {
		t.Fatal("Engram did not use bound tables")
	}
	for _, test := range []struct {
		name   string
		mutate func(*gguf.File)
	}{
		{"owner", func(g *gguf.File) { g.Metadata["deepseek41.kv_source_layer_ids"].Value.([]any)[1] = uint32(7) }},
		{"ratio", func(g *gguf.File) { g.Metadata["deepseek41.compress_ratios"].Value.([]any)[20] = uint32(2) }},
		{"encoding", func(g *gguf.File) { g.Tensors[0].TypeID = 0 }},
		{"bounds", func(g *gguf.File) { g.Tensors[0].Offset = g.Size }},
		{"shape", func(g *gguf.File) { g.Tensors[0].Dimensions[0]-- }},
		{"duplicate", func(g *gguf.File) { g.Tensors = append(g.Tensors, g.Tensors[0]) }},
		{"missing", func(g *gguf.File) { g.Tensors = g.Tensors[1:] }},
	} {
		t.Run(test.name, func(t *testing.T) {
			g := modelFixture(t)
			test.mutate(g)
			if _, err := Bind(new(countReader), int64(g.Size), g); err == nil {
				t.Fatal("mutated directory admitted")
			}
		})
	}
}

func TestMetadataFloatCompatibility(t *testing.T) {
	g := metadataFixture()
	g.Metadata["deepseek41.rope_scaling.original_max_position_embeddings"] = gguf.Value{TypeID: gguf.TypeUint32, Value: uint32(65536)}
	if _, err := ValidateMetadata(g); err != nil {
		t.Fatal(err)
	}
	g.Metadata["deepseek41.rms_norm_eps"] = gguf.Value{TypeID: gguf.TypeFloat64, Value: float64(1e-20)}
	if _, err := ValidateMetadata(g); err != nil {
		t.Fatal(err)
	}
}

func ExampleValidateMetadata() {
	_, err := ValidateMetadata(&gguf.File{})
	fmt.Println(err)
	// Output: expected deepseek41 architecture
}

func ExampleBind() {
	_, err := Bind(io.NewSectionReader(nil, 0, 0), 0, nil)
	fmt.Println(err)
	// Output: invalid V4.1 file source
}

func ExampleModel_NewEngram() {
	var model Model
	_, err := model.NewEngram()
	fmt.Println(err)
	// Output: invalid V4.1 model
}
