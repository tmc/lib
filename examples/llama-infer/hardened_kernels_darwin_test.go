//go:build darwin

package main

import (
	"math"
	"reflect"
	"testing"

	"github.com/tmc/lib/apple/x/metal"
	"github.com/tmc/lib/apple/x/metal/mempool"
)

func TestTiledServingArgmaxEdgeSemantics(t *testing.T) {
	eng, err := GetOrCompileEngine(DefaultTinyLlamaConfig())
	if err != nil {
		t.Fatal(err)
	}
	const vocab = 513
	cases := []struct {
		name string
		set  func([]float32)
		want int32
	}{
		{name: "tie across tiles", set: func(x []float32) { x[1], x[300] = 7, 7 }, want: 1},
		{name: "mixed NaN", set: func(x []float32) { x[0], x[257] = float32(math.NaN()), 4 }, want: 257},
		{name: "all NaN", set: func(x []float32) {
			for i := range x {
				x[i] = float32(math.NaN())
			}
		}, want: 0},
		{name: "all negative infinity", set: func(x []float32) {
			for i := range x {
				x[i] = float32(math.Inf(-1))
			}
		}, want: 0},
	}
	for _, half := range []bool{false, true} {
		for _, test := range cases {
			t.Run(test.name+map[bool]string{false: "/f32", true: "/f16"}[half], func(t *testing.T) {
				head := make([]float32, vocab)
				test.set(head)
				if got := runTiledServingArgmax(t, eng, head, half); got != test.want {
					t.Fatalf("token=%d, want %d", got, test.want)
				}
			})
		}
	}
}

func runTiledServingArgmax(t *testing.T, eng *ModelEngine, head []float32, half bool) int32 {
	t.Helper()
	const dim = 1
	tiles := (len(head) + 255) / 256
	x, err := mempool.AllocFromSlice(eng.dev, []float32{1})
	if err != nil {
		t.Fatal(err)
	}
	defer x.Free()
	scale, err := mempool.AllocFromSlice(eng.dev, []float32{1})
	if err != nil {
		t.Fatal(err)
	}
	defer scale.Free()
	candidates, err := mempool.Alloc[float32](eng.dev, tiles*2)
	if err != nil {
		t.Fatal(err)
	}
	defer candidates.Free()
	token, err := mempool.Alloc[int32](eng.dev, 1)
	if err != nil {
		t.Fatal(err)
	}
	defer token.Free()
	cb := eng.dev.Queue().CommandBuffer()
	ce := cb.ComputeCommandEncoder()
	if half {
		norm, err := mempool.AllocFromSlice(eng.dev, []uint16{0x3c00})
		if err != nil {
			t.Fatal(err)
		}
		defer norm.Free()
		weights := make([]uint16, len(head))
		for i, value := range head {
			weights[i] = testFloat16(value)
		}
		gpuHead, err := mempool.AllocFromSlice(eng.dev, weights)
		if err != nil {
			t.Fatal(err)
		}
		defer gpuHead.Free()
		ce.SetComputePipelineState(eng.pso.LMHeadTileArgmaxH)
		ce.SetBufferWithOffsetAtIndex(x.Buffer(), 0, 0)
		ce.SetBufferWithOffsetAtIndex(norm.Buffer(), 0, 1)
		ce.SetBufferWithOffsetAtIndex(gpuHead.Buffer(), 0, 2)
	} else {
		norm, err := mempool.AllocFromSlice(eng.dev, []float32{1})
		if err != nil {
			t.Fatal(err)
		}
		defer norm.Free()
		gpuHead, err := mempool.AllocFromSlice(eng.dev, head)
		if err != nil {
			t.Fatal(err)
		}
		defer gpuHead.Free()
		ce.SetComputePipelineState(eng.pso.LMHeadTileArgmax)
		ce.SetBufferWithOffsetAtIndex(x.Buffer(), 0, 0)
		ce.SetBufferWithOffsetAtIndex(norm.Buffer(), 0, 1)
		ce.SetBufferWithOffsetAtIndex(gpuHead.Buffer(), 0, 2)
	}
	ce.SetBufferWithOffsetAtIndex(scale.Buffer(), 0, 3)
	ce.SetBufferWithOffsetAtIndex(candidates.Buffer(), 0, 4)
	metal.SetScalar(ce, uint32(dim), 5)
	metal.SetScalar(ce, uint32(len(head)), 6)
	metal.SetScalar(ce, uint32(tiles), 7)
	ce.DispatchThreadgroupsThreadsPerThreadgroup(metal.Threads(tiles), metal.Threads(256))
	ce.SetComputePipelineState(eng.pso.ArgmaxTiles)
	ce.SetBufferWithOffsetAtIndex(candidates.Buffer(), 0, 0)
	ce.SetBufferWithOffsetAtIndex(token.Buffer(), 0, 1)
	metal.SetScalar(ce, uint32(1), 2)
	metal.SetScalar(ce, uint32(tiles), 3)
	ce.DispatchThreadgroupsThreadsPerThreadgroup(metal.Threads(1), metal.Threads(128))
	ce.EndEncoding()
	cb.Commit()
	cb.WaitUntilCompleted()
	return token.HostSlice()[0]
}

func testFloat16(value float32) uint16 {
	switch {
	case math.IsNaN(float64(value)):
		return 0x7e00
	case math.IsInf(float64(value), -1):
		return 0xfc00
	case value == 0:
		return 0
	case value == 4:
		return 0x4400
	case value == 7:
		return 0x4700
	default:
		panic("unsupported test float16 value")
	}
}

func TestBatchedArgmaxEdgeSemantics(t *testing.T) {
	eng, err := GetOrCompileEngine(DefaultTinyLlamaConfig())
	if err != nil {
		t.Fatal(err)
	}
	rows := [][]float32{
		{1, 7, 7, 2},
		{float32(math.NaN()), -2, -1, float32(math.NaN())},
		{float32(math.Inf(-1)), float32(math.Inf(-1)), float32(math.Inf(-1)), float32(math.Inf(-1))},
		{float32(math.NaN()), float32(math.NaN()), float32(math.NaN()), float32(math.NaN())},
	}
	for _, b := range []int{1, 2, 4} {
		logits := make([]float32, 0, b*4)
		for _, row := range rows[:b] {
			logits = append(logits, row...)
		}
		got := runBatchedArgmax(t, eng, logits, b, 4)
		want := []int32{1, 2, 0, 0}
		if !reflect.DeepEqual(got, want[:b]) {
			t.Fatalf("B=%d tokens=%v, want %v", b, got, want[:b])
		}
	}
}

func TestRowIndexedAttentionRejectsInvalidShapes(t *testing.T) {
	cfg := DefaultTinyLlamaConfig()
	eng, err := GetOrCompileEngine(cfg)
	if err != nil {
		t.Fatal(err)
	}
	cache, err := eng.NewBatchedAttentionCache(cfg, 2)
	if err != nil {
		t.Fatal(err)
	}
	defer cache.Free()
	q := make([]float32, cfg.NumHeads*cfg.HeadDim)
	k := make([]float32, cfg.NumKVHeads*cfg.HeadDim)
	v := make([]float32, len(k))
	tests := []struct {
		name            string
		edit            func(*Config)
		positions, rows []int
	}{
		{name: "odd head", edit: func(c *Config) { c.HeadDim = 15 }, positions: []int{1}, rows: []int{0}},
		{name: "zero kv heads", edit: func(c *Config) { c.NumKVHeads = 0 }, positions: []int{1}, rows: []int{0}},
		{name: "position bound", edit: func(c *Config) {}, positions: []int{cfg.MaxSeqLen}, rows: []int{0}},
		{name: "row bound", edit: func(c *Config) {}, positions: []int{1}, rows: []int{2}},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			c := cfg
			test.edit(&c)
			if _, err := RunBatchedAttention(eng, c, 0, cache, q, k, v, test.positions, test.rows); err == nil {
				t.Fatal("invalid shape succeeded")
			}
		})
	}
}

func runBatchedArgmax(t *testing.T, eng *ModelEngine, logits []float32, batch, vocab int) []int32 {
	t.Helper()
	in, err := mempool.AllocFromSlice(eng.dev, logits)
	if err != nil {
		t.Fatal(err)
	}
	defer in.Free()
	out, err := mempool.Alloc[int32](eng.dev, batch)
	if err != nil {
		t.Fatal(err)
	}
	defer out.Free()
	cb := eng.dev.Queue().CommandBuffer()
	ce := cb.ComputeCommandEncoder()
	ce.SetComputePipelineState(eng.pso.BatchedSample)
	ce.SetBufferWithOffsetAtIndex(in.Buffer(), 0, 0)
	ce.SetBufferWithOffsetAtIndex(out.Buffer(), 0, 1)
	metal.SetScalar(ce, uint32(batch), 2)
	metal.SetScalar(ce, uint32(vocab), 3)
	ce.DispatchThreadgroupsThreadsPerThreadgroup(metal.Threads(batch), metal.Threads(128))
	ce.EndEncoding()
	cb.Commit()
	cb.WaitUntilCompleted()
	return append([]int32(nil), out.HostSlice()...)
}
