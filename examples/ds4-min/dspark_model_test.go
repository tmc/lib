package main

import (
	"fmt"
	"testing"

	"github.com/tmc/lib/gguf"
)

func supportFixture() *gguf.File {
	g := &gguf.File{Size: 1 << 40, Metadata: map[string]gguf.Value{}}
	for key, v := range map[string]any{"block_size": uint32(5), "markov_rank": uint32(256), "noise_token_id": uint32(0), "target_layer_ids": []any{uint32(1), uint32(20)}} {
		g.Metadata["deepseek4.dspark."+key] = gguf.Value{Value: v}
	}
	m := &dsparkModel{stages: 3, rank: 256, targets: []int{1, 20}}
	for stage := 0; stage < 3; stage++ {
		for _, r := range dsparkRules(stage, m) {
			typ := uint32(0)
			if r.kind == 3 {
				typ = 8
			}
			g.Tensors = append(g.Tensors, gguf.Tensor{Name: fmt.Sprintf("mtp.%d.%s", stage, r.name), TypeID: typ, Dimensions: append([]uint64(nil), r.dims...)})
		}
	}
	return g
}

func TestDSparkAdmission(t *testing.T) {
	g := supportFixture()
	m, err := admitDSparkModel(g)
	if err != nil {
		t.Fatal(err)
	}
	if m.stages != 3 || m.block != 5 || m.rank != 256 || len(m.targets) != 2 {
		t.Fatalf("descriptor: %+v", m)
	}
	name := g.Tensors[0].Name
	g.Tensors[0].Dimensions[0]++
	if m.tensors[name].Dimensions[0] == g.Tensors[0].Dimensions[0] {
		t.Fatal("descriptor aliases source")
	}
	copy, ok := m.tensor(name)
	if !ok {
		t.Fatal("missing copied tensor")
	}
	copy.Dimensions[0]++
	if m.tensors[name].Dimensions[0] == copy.Dimensions[0] {
		t.Fatal("descriptor aliases accessor")
	}
	for _, tc := range []struct {
		name   string
		mutate func(*gguf.File)
	}{
		{"missing", func(g *gguf.File) { g.Tensors = g.Tensors[1:] }},
		{"duplicate", func(g *gguf.File) { g.Tensors = append(g.Tensors, g.Tensors[0]) }},
		{"dimension", func(g *gguf.File) { g.Tensors[0].Dimensions[0]++ }},
		{"type", func(g *gguf.File) { g.Tensors[0].TypeID = 10 }},
		{"short file", func(g *gguf.File) { g.Size = 1 }},
		{"block", func(g *gguf.File) { g.Metadata["deepseek4.dspark.block_size"] = gguf.Value{Value: uint32(17)} }},
		{"targets", func(g *gguf.File) {
			g.Metadata["deepseek4.dspark.target_layer_ids"] = gguf.Value{Value: []any{uint32(20), uint32(1)}}
		}},
		{"noise", func(g *gguf.File) {
			g.Metadata["deepseek4.dspark.noise_token_id"] = gguf.Value{Value: uint32(Flash.Vocab)}
		}},
		{"stage", func(g *gguf.File) { g.Tensors[0].Name = "mtp.8.hc_attn_fn.weight" }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			g := supportFixture()
			tc.mutate(g)
			if _, err := admitDSparkModel(g); err == nil {
				t.Fatal("mutation admitted")
			}
		})
	}
}

func TestDSparkAliases(t *testing.T) {
	for _, prefix := range []string{"deepseek4.dspark.", "deepseek4.dspark_", "dspark."} {
		g := supportFixture()
		old := g.Metadata
		g.Metadata = map[string]gguf.Value{}
		for key, v := range old {
			g.Metadata[prefix+key[len("deepseek4.dspark."):]] = v
		}
		if _, err := admitDSparkModel(g); err != nil {
			t.Fatal(err)
		}
	}
}

func Example_readDSparkModel() {
	_, err := readDSparkModel(nil, 0, "")
	fmt.Println(err)
	// Output: dspark: full support sha256 required
}
