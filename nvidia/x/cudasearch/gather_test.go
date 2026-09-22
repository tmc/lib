package cudasearch

import (
	"context"
	"errors"
	"math"
	"slices"
	"testing"

	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudatrial"
	"github.com/tmc/lib/nvidia/x/egraph"
)

func TestGatherCompiler(t *testing.T) {
	data := []float32{0, math.Float32frombits(0x80000000), math.Float32frombits(0x7fc01234), 1, 2, 3, -1, -2, -3}
	routes := []int32{2, 0, 2}
	c, err := NewGatherCompiler(&cuda.Device{}, "sm_121a", 3, 3, data, routes, func() bool { return false })
	if err != nil {
		t.Fatal(err)
	}
	values := c.References()["gather_bytes"]
	want := []uint32{0xbf800000, 0xc0000000, 0xc0400000, 0, 0x80000000, 0x7fc01234, 0xbf800000, 0xc0000000, 0xc0400000}
	for i, w := range want {
		b := values[i*4 : i*4+4]
		got := uint32(b[0]) | uint32(b[1])<<8 | uint32(b[2])<<16 | uint32(b[3])<<24
		if got != w {
			t.Fatalf("word %d = %08x want %08x", i, got, w)
		}
	}
	key := c.Bindings().Key()
	data[0] = 42
	routes[0] = 1
	values[0] = 42
	if c.Bindings().Key() != key || c.References()["gather_bytes"][0] != 0 {
		t.Fatal("caller data aliases compiler")
	}
	b := c.Bindings()
	b.InputNames[0] = "changed"
	b.InputIdentities["routes"] = "changed"
	b.WeightIdentities["experts"] = "changed"
	if c.Bindings().Key() != key {
		t.Fatal("bindings alias compiler")
	}
	for _, name := range []string{"gather_scalar", "gather16", "gather32"} {
		expr := &egraph.Expr{Op: name}
		if _, err := c.admit(expr, c.Bindings()); err != nil {
			t.Fatal(err)
		}
		if x, err := c.Compile(context.Background(), expr, c.Bindings()); x != nil || !errors.Is(err, cudatrial.ErrHardwareLockRequired) {
			t.Fatalf("compile without lease: %v %v", x, err)
		}
	}
	for _, expr := range []*egraph.Expr{nil, {Op: "gather64"}, {Op: "gather16", Data: 1}, {Op: "gather32", Children: []*egraph.Expr{{Op: "x"}}}} {
		if _, err := c.admit(expr, c.Bindings()); err == nil {
			t.Fatal("accepted malformed expression")
		}
	}
	if _, err := c.admit(&egraph.Expr{Op: "gather16"}, b); err == nil {
		t.Fatal("accepted changed workload")
	}
}

func TestGatherCompilerAdmission(t *testing.T) {
	keys := make(map[string]bool)
	for _, tc := range []struct {
		data   []float32
		routes []int32
	}{{[]float32{1, 2}, []int32{0, 1}}, {[]float32{1, 2}, []int32{1, 0}}, {[]float32{1, 3}, []int32{0, 1}}} {
		c, err := NewGatherCompiler(&cuda.Device{}, "sm_121a", 1, 2, tc.data, tc.routes, func() bool { return false })
		if err != nil {
			t.Fatal(err)
		}
		key := c.Bindings().Key()
		if keys[key] {
			t.Fatal("route/weight identity collision")
		}
		keys[key] = true
	}
	for _, routes := range [][]int32{nil, {-1}, {2}} {
		if _, err := NewGatherCompiler(&cuda.Device{}, "sm_121a", 1, 2, []float32{1, 2}, routes, func() bool { return false }); err == nil {
			t.Fatalf("accepted routes %v", routes)
		}
	}
	for _, target := range []string{"", "compute_121a", "sm_bad"} {
		if _, err := NewGatherCompiler(&cuda.Device{}, target, 1, 1, []float32{1}, []int32{0}, func() bool { return false }); err == nil {
			t.Fatalf("accepted target %q", target)
		}
	}
	if _, err := NewGatherCompiler(&cuda.Device{}, "sm_121a", 2, 2, []float32{1}, []int32{0}, func() bool { return false }); err == nil {
		t.Fatal("accepted short experts")
	}
}

func TestSequenceOutputDecode(t *testing.T) {
	bytes, err := (sequenceOutput{bytes: true}).decode([]byte{0, 127, 255})
	if err != nil || !slices.Equal(bytes, []float32{0, 127, 255}) {
		t.Fatalf("bytes=%v err=%v", bytes, err)
	}
	v, err := (sequenceOutput{}).decode([]byte{0, 0, 0, 128, 0, 0, 0, 63})
	if err != nil || len(v) != 2 || math.Float32bits(v[0]) != 0x80000000 || v[1] != 0.5 {
		t.Fatalf("floats=%v err=%v", v, err)
	}
	if _, err := (sequenceOutput{}).decode([]byte{1, 2, 3}); err == nil {
		t.Fatal("truncated float accepted")
	}
	x := &sequenceExecutable{}
	if err := x.load(nil); err == nil {
		t.Fatal("empty sequence accepted")
	}
}
