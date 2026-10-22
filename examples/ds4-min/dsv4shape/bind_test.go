package dsv4shape

import (
	"fmt"
	"testing"

	"github.com/tmc/lib/gguf"
)

func syntheticWeights(t *testing.T, c Config) *gguf.File {
	t.Helper()
	s, err := c.Weights()
	if err != nil {
		t.Fatal(err)
	}
	g := &gguf.File{DataOffset: 4096, Size: 4096}
	for _, v := range s {
		if v.Optional {
			continue
		}
		typ := v.Types[0]
		if len(v.Dimensions) == 3 {
			typ = 16
			if v.Dimensions[0] == uint64(c.ExpertFF) {
				typ = 10
			}
		}
		x := gguf.Tensor{Name: v.Name, TypeID: typ, Dimensions: append([]uint64(nil), v.Dimensions...), Offset: g.Size - g.DataOffset}
		n, err := x.Bytes()
		if err != nil {
			t.Fatal(err)
		}
		g.Tensors = append(g.Tensors, x)
		g.Size += n
	}
	return g
}

func TestBinding(t *testing.T) {
	for _, v := range []Variant{Flash, Pro} {
		c, _ := For(v)
		g := syntheticWeights(t, c)
		b, err := c.Bind(g)
		if err != nil {
			t.Fatal(err)
		}
		if b.ResidentBytes+b.RoutedBytes != g.Size-g.DataOffset {
			t.Fatal("bytes not reconciled")
		}
		for _, x := range b.Weights {
			if x.ExpertBytes != 0 && x.Bytes/x.ExpertBytes != uint64(c.Experts) {
				t.Fatal("wrong expert stride")
			}
		}
		if err := b.AdmitResident(g.Size, 0, 0); err != nil {
			t.Fatal(err)
		}
		if err := b.AdmitResident(1, 0, 0); err == nil {
			t.Fatal("insufficient memory admitted")
		}
		if err := b.AdmitResident(^uint64(0), ^uint64(0), 0); err == nil {
			t.Fatal("overflow admitted")
		}
		for _, tc := range []struct {
			name   string
			change func(*gguf.File)
		}{
			{"truncated", func(g *gguf.File) { g.Size-- }},
			{"missing", func(g *gguf.File) { g.Tensors = g.Tensors[1:] }},
			{"wrong type", func(g *gguf.File) { g.Tensors[0].TypeID = 26 }},
			{"wrong shape", func(g *gguf.File) { g.Tensors[0].Dimensions[0]++ }},
			{"offset overflow", func(g *gguf.File) { g.Tensors[0].Offset = ^uint64(0) }},
			{"duplicate", func(g *gguf.File) { g.Tensors = append(g.Tensors, g.Tensors[0]) }},
		} {
			t.Run(fmt.Sprintf("%d/%s", v, tc.name), func(t *testing.T) {
				g := syntheticWeights(t, c)
				tc.change(g)
				if _, err := c.Bind(g); err == nil {
					t.Fatal("invalid model admitted")
				}
			})
		}
	}
}

func TestProInitialCompressors(t *testing.T) {
	c, _ := For(Pro)
	g := syntheticWeights(t, c)
	for _, layer := range []int{0, 1} {
		name := fmt.Sprintf("blk.%d.attn_compressor_kv.weight", layer)
		for i, x := range g.Tensors {
			if x.Name == name {
				if x.Dimensions[0] != 7168 || x.Dimensions[1] != 512 {
					t.Fatal("wrong Pro compressor shape")
				}
				bad := *g
				bad.Tensors = append([]gguf.Tensor(nil), g.Tensors[:i]...)
				bad.Tensors = append(bad.Tensors, g.Tensors[i+1:]...)
				if _, err := c.Bind(&bad); err == nil {
					t.Fatal("missing initial compressor admitted")
				}
			}
		}
	}
}

func TestCacheState(t *testing.T) {
	c, _ := For(Pro)
	p, err := c.CacheState(2048)
	if err != nil {
		t.Fatal(err)
	}
	if len(p.Layers) != 61 || p.Layers[0].CompressedKV != 18*512 || p.Layers[1].Ratio != 128 || p.Layers[2].IndexerKV != 514*128 {
		t.Fatalf("wrong schedule %+v", p.Layers[:3])
	}
	if p.Layers[0].CompressorKV != 128*512 || p.Layers[2].CompressorKV != 8*1024 {
		t.Fatal("wrong accumulator layout")
	}
	if _, err := c.CacheState(0); err == nil {
		t.Fatal("zero context admitted")
	}
	if _, err := c.CacheState(^uint64(0)); err == nil {
		t.Fatal("huge context admitted")
	}
}

func ExampleConfig_Weights() {
	c, _ := For(Pro)
	s, _ := c.Weights()
	fmt.Println(s[0].Name, s[0].Dimensions)
	// Output: token_embd.weight [7168 129280]
}

func ExampleConfig_Bind() {
	c, _ := For(Pro)
	_, err := c.Bind(nil)
	fmt.Println(err)
	// Output: shape: invalid GGUF file extent
}

func ExampleBinding_AdmitResident() {
	b := Binding{ResidentBytes: 1024, RoutedBytes: 2048}
	fmt.Println(b.AdmitResident(4096, 512, 512))
	// Output: <nil>
}

func ExampleConfig_CacheState() {
	c, _ := For(Pro)
	p, _ := c.CacheState(2048)
	fmt.Println(len(p.Layers), p.Layers[0].Ratio)
	// Output: 61 128
}
