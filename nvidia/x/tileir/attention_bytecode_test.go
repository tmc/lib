package tileir

import (
	"bytes"
	"context"
	"errors"
	"math"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudair"
	"github.com/tmc/lib/nvidia/x/cudakernel"
	"github.com/tmc/lib/nvidia/x/tileiras"
)

type attentionFixture struct{ snapshot *cudair.Snapshot }

func (f attentionFixture) Snapshot() *cudair.Snapshot { return f.snapshot }
func (f attentionFixture) Identity() string           { return strings.Repeat("1", 64) }
func (f attentionFixture) Output() []float32          { return []float32{1} }
func (f attentionFixture) Reference() []float32       { return []float32{1} }
func (f attentionFixture) Inputs() map[string][]float32 {
	return map[string][]float32{"scores": {0, 0}, "values": {3, -1}}
}

func attentionFixtureLowering(t *testing.T) *AttentionLowering {
	t.Helper()
	g := cudair.NewGraph()
	s := g.AddInput("scores", []int{2}, cudair.Float32)
	v := g.AddInput("values", []int{2, 1}, cudair.Float32)
	e := g.AddUnary("exp", g.AddBinary("sub", s, g.AddReduce("max", s, 0)))
	p := g.AddBinary("div", e, g.AddReduce("sum", e, 0))
	out := g.AddMatMul(g.AddReshape(p, []int{1, 2}), v)
	if err := g.SetOutputs(out); err != nil {
		t.Fatal(err)
	}
	snap, err := g.Snapshot()
	if err != nil {
		t.Fatal(err)
	}
	a, err := LowerOnlineAttention(attentionFixture{snap}, AttentionOptions{TileSize: 1})
	if err != nil {
		t.Fatal(err)
	}
	return a
}

func TestExpBytecodeContract(t *testing.T) {
	module := func(typ Type, attrs string) *Module {
		return &Module{Functions: []*Function{{Name: "exp", Entry: true, Operations: []*Operation{
			{OpCode: "cuda_tile.constant", TypeVal: typ, Results: []string{"%a"}, Attributes: "{value = 0}"},
			{OpCode: "cuda_tile.exp", TypeVal: typ, Results: []string{"%b"}, Operands: []string{"%a"}, Attributes: attrs},
			{OpCode: "cuda_tile.return"},
		}}}}
	}
	for _, typ := range []Type{NewTile(nil, Float32), NewTile([]int64{4}, Float64)} {
		data, err := module(typ, "{rounding_mode = full}").EncodeBytecode()
		if err != nil {
			t.Fatal(err)
		}
		decoded, err := DecodeBytecodeBytes(data)
		if err != nil {
			t.Fatal(err)
		}
		op := decoded.Functions[0].Operations[1]
		if op.OpCode != "cuda_tile.exp" || op.Attributes != "{rounding_mode = full}" || !typeEqual(op.TypeVal, typ) {
			t.Fatal("exp opcode, attributes or type lost")
		}
		for _, attrs := range []string{"", "{rounding_mode = approx}", "{rounding_mode = nearest_even}", "{rounding_mode = full, flush_to_zero}"} {
			if data, err := module(typ, attrs).EncodeBytecode(); data != nil || !errors.Is(err, ErrUnsupportedASTEncoding) {
				t.Fatalf("exp attributes %q admitted: %v", attrs, err)
			}
		}
	}
	if data, err := module(NewTile(nil, Int32), "{rounding_mode = full}").EncodeBytecode(); data != nil || !errors.Is(err, ErrUnsupportedASTEncoding) {
		t.Fatalf("integer exp admitted: %v", err)
	}
	bad := module(NewTile(nil, Float32), "{rounding_mode = full}")
	bad.Functions[0].Operations[1].Operands = []string{"%undefined"}
	if data, err := bad.EncodeBytecode(); data != nil || err == nil {
		t.Fatal("exp with undefined input admitted")
	}
}

func TestAttentionAssemblyComposition(t *testing.T) {
	a := attentionFixtureLowering(t)
	r := a.Result()
	want, err := r.Module.EncodeBytecode()
	if err != nil {
		t.Fatal(err)
	}
	called := 0
	got, err := assembleSemanticMemory(context.Background(), r, "sm_90", func(_ context.Context, data []byte, target string) (*tileiras.Result, error) {
		called++
		if !bytes.Equal(data, want) || target != "sm_90" {
			t.Fatal("assembler received another module or target")
		}
		return &tileiras.Result{CUBIN: syntheticMemoryELF(t, "tile_online_attention", 90), Log: "metadata-only fake; no compiler"}, nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if called != 1 || got.Artifact.Kind != cudakernel.KindCUBIN || got.Artifact.EntryPoint != "tile_online_attention" || got.BytecodeDigest != r.Memory.BytecodeDigest() || got.Artifact.Schema.TotalParamBytes != 24 {
		t.Fatal("typed attention assembly lost ABI or source identity")
	}
	for _, edit := range []func(){
		func() { r.Schema.Params[0].Name = "other" },
		func() { r.Module.Functions[0].Operations[0].Attributes = "{unknown = full}" },
	} {
		r = attentionFixtureLowering(t).Result()
		edit()
		called = 0
		_, err := assembleSemanticMemory(context.Background(), r, "sm_90", func(context.Context, []byte, string) (*tileiras.Result, error) {
			called++
			return nil, nil
		})
		if err == nil || called != 0 {
			t.Fatal("changed binding reached assembler")
		}
	}
}

func TestAttentionFiniteTolerance(t *testing.T) {
	opts := AttentionOptions{Absolute: math.MaxFloat64, Relative: math.MaxFloat64}
	for _, got := range []float32{2, 3} {
		if attentionMatches([]float32{got}, []float32{2}, opts) {
			t.Fatal("overflowing tolerance admitted, including equality fast path")
		}
	}
	for _, v := range []float32{float32(math.Inf(1)), float32(math.NaN())} {
		if attentionMatches([]float32{v}, []float32{v}, AttentionOptions{}) {
			t.Fatal("nonfinite dense output admitted by equality")
		}
	}
	opts.Excluded = []bool{true}
	if !attentionMatches([]float32{math.Float32frombits(0x7fc00000)}, []float32{math.Float32frombits(0x7fc00000)}, opts) {
		t.Fatal("canonical fully masked policy changed")
	}
}
