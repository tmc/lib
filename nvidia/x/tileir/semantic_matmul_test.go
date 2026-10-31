package tileir_test

import (
	"bytes"
	"encoding/binary"
	"encoding/hex"
	"errors"
	"fmt"
	"math"
	"reflect"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudair"
	"github.com/tmc/lib/nvidia/x/cudaref"
	"github.com/tmc/lib/nvidia/x/cudashape"
	"github.com/tmc/lib/nvidia/x/tileir"
)

func matmulGraph(t *testing.T, m, n, k int, reverse bool) (*cudair.Graph, *cudair.Snapshot) {
	t.Helper()
	g := cudair.NewGraph()
	var a, b *cudair.Node
	if reverse {
		b = g.AddInput("right", []int{k, n}, cudair.Float32)
		a = g.AddInput("left", []int{m, k}, cudair.Float32)
	} else {
		a = g.AddInput("left", []int{m, k}, cudair.Float32)
		b = g.AddInput("right", []int{k, n}, cudair.Float32)
	}
	a.SourceLoc = cudair.SourceLocation{File: "left.go", Line: 10}
	b.SourceLoc = cudair.SourceLocation{File: "right.go", Line: 20}
	out := g.AddMatMul(a, b)
	out.Name = "product"
	out.SourceLoc = cudair.SourceLocation{File: "product.go", Line: 30, Callers: []cudair.Frame{{File: "caller.go", Line: 40}}}
	if err := g.SetOutputs(out); err != nil {
		t.Fatal(err)
	}
	s, err := g.Snapshot()
	if err != nil {
		t.Fatal(err)
	}
	return g, s
}

type matmulPointer struct{ slot, index int }
type matmulValue struct {
	kind tileir.TypeKind
	f    float64
	i    int64
	ptr  matmulPointer
}

// Interpret only decoded public SSA. Host slices are guarded, distinct ABI
// slots, not device pointers; no source graph or C++ is evaluated here.
func executeMatmul(data []byte, a, b []float32, outputs int) ([]float32, error) {
	bm, err := tileir.DecodeBytecodeBytes(data)
	if err != nil {
		return nil, err
	}
	if len(bm.Functions) != 1 || bm.Functions[0].BodyDecodeError != nil {
		return nil, fmt.Errorf("oracle: incomplete decoded function")
	}
	fn := bm.Functions[0]
	sig, ok := fn.Signature.(*tileir.FunctionType)
	if !ok || len(sig.Inputs) != 3 || len(sig.Results) != 0 || !fn.IsEntry {
		return nil, fmt.Errorf("oracle: wrong entry ABI")
	}
	guard := math.Float32frombits(0x7f123456)
	buffers := make([][]float32, 3)
	for slot, input := range [][]float32{a, b, make([]float32, outputs)} {
		buffers[slot] = make([]float32, len(input)+2)
		copy(buffers[slot][1:], input)
		buffers[slot][0], buffers[slot][len(input)+1] = guard, guard
	}
	for i := 1; i <= outputs; i++ {
		buffers[2][i] = 777
	}
	beforeA, beforeB := append([]float32(nil), buffers[0]...), append([]float32(nil), buffers[1]...)
	writes := make([]int, outputs)
	values := map[string]matmulValue{}
	for i := 0; i < 3; i++ {
		values[fmt.Sprintf("%%arg%d", i)] = matmulValue{kind: tileir.KindPointer, ptr: matmulPointer{slot: i}}
	}
	tokens := make(map[string]int)
	lastStore := ""
	get := func(name string) (matmulValue, error) {
		v, ok := values[name]
		if !ok {
			return v, fmt.Errorf("oracle: undefined SSA %s", name)
		}
		return v, nil
	}
	consume := func(name string) error {
		uses, ok := tokens[name]
		if !ok || uses != 0 {
			return fmt.Errorf("oracle: missing or reused token %s", name)
		}
		tokens[name]++
		return nil
	}
	checkPointer := func(v matmulValue) error {
		if v.kind != tileir.KindPointer || v.ptr.slot < 0 || v.ptr.slot >= 3 || v.ptr.index < 0 || v.ptr.index >= len(buffers[v.ptr.slot])-2 {
			return fmt.Errorf("oracle: pointer outside ABI range %+v", v.ptr)
		}
		return nil
	}
	for _, op := range fn.Operations {
		args := make([]matmulValue, len(op.Operands))
		for i, name := range op.Operands {
			args[i], err = get(name)
			if err != nil {
				return nil, err
			}
		}
		var result matmulValue
		switch op.OpCode {
		case "cuda_tile.constant":
			typ, ok := op.TypeVal.(*tileir.TileType)
			if !ok || len(typ.Shape) != 0 {
				return nil, fmt.Errorf("oracle: non-scalar constant")
			}
			text := strings.TrimSuffix(strings.TrimPrefix(op.Attributes, "{value = dense<\"0x"), "\">}")
			raw, e := hex.DecodeString(text)
			if e != nil {
				return nil, e
			}
			result.kind = typ.ElementType.Kind()
			switch result.kind {
			case tileir.KindInt1:
				if len(raw) != 1 || raw[0] > 1 {
					return nil, fmt.Errorf("oracle: invalid mask")
				}
				result.i = int64(raw[0])
			case tileir.KindInt64:
				if len(raw) != 8 {
					return nil, fmt.Errorf("oracle: invalid offset")
				}
				result.i = int64(binary.LittleEndian.Uint64(raw))
			case tileir.KindFloat32:
				if len(raw) != 4 {
					return nil, fmt.Errorf("oracle: invalid f32 constant")
				}
				result.f = float64(math.Float32frombits(binary.LittleEndian.Uint32(raw)))
			case tileir.KindFloat64:
				if len(raw) != 8 {
					return nil, fmt.Errorf("oracle: invalid f64 constant")
				}
				result.f = math.Float64frombits(binary.LittleEndian.Uint64(raw))
			default:
				return nil, fmt.Errorf("oracle: unsupported constant type")
			}
		case "cuda_tile.make_token":
			result.kind = tileir.KindToken
		case "cuda_tile.offset":
			if len(args) != 2 || args[0].kind != tileir.KindPointer || args[1].kind != tileir.KindInt64 || args[1].i < 0 || args[1].i > 16 {
				return nil, fmt.Errorf("oracle: invalid scalar offset")
			}
			result = args[0]
			result.ptr.index += int(args[1].i)
			if e := checkPointer(result); e != nil {
				return nil, e
			}
		case "cuda_tile.load_ptr_tko":
			if op.Attributes != "{memory_ordering_semantics = weak}" || len(args) != 4 || len(op.Results) != 2 || args[1].kind != tileir.KindInt1 || args[1].i != 1 || args[2].kind != tileir.KindFloat32 {
				return nil, fmt.Errorf("oracle: invalid load")
			}
			if e := checkPointer(args[0]); e != nil {
				return nil, e
			}
			if args[0].ptr.slot == 2 {
				return nil, fmt.Errorf("oracle: output used as input")
			}
			if e := consume(op.Operands[3]); e != nil {
				return nil, e
			}
			result = matmulValue{kind: tileir.KindFloat32, f: float64(buffers[args[0].ptr.slot][args[0].ptr.index+1])}
			values[op.Results[1]] = matmulValue{kind: tileir.KindToken}
			tokens[op.Results[1]] = 0
		case "cuda_tile.ftof":
			to, ok := op.TypeVal.(*tileir.TileType)
			if !ok || len(args) != 1 || op.Attributes != "{rounding_mode = nearest_even}" {
				return nil, fmt.Errorf("oracle: invalid conversion")
			}
			result = args[0]
			result.kind = to.ElementType.Kind()
			if result.kind == tileir.KindFloat32 && args[0].kind == tileir.KindFloat64 {
				result.f = float64(float32(result.f))
			} else if result.kind != tileir.KindFloat64 || args[0].kind != tileir.KindFloat32 {
				return nil, fmt.Errorf("oracle: conversion types")
			}
		case "cuda_tile.mulf", "cuda_tile.addf":
			typ, ok := op.TypeVal.(*tileir.TileType)
			if !ok || typ.ElementType.Kind() != tileir.KindFloat64 || len(args) != 2 || args[0].kind != tileir.KindFloat64 || args[1].kind != tileir.KindFloat64 || op.Attributes != "{rounding_mode = nearest_even}" {
				return nil, fmt.Errorf("oracle: arithmetic is not ordered f64")
			}
			result.kind = tileir.KindFloat64
			if op.OpCode == "cuda_tile.mulf" {
				result.f = args[0].f * args[1].f
			} else {
				result.f = args[0].f + args[1].f
			}
		case "cuda_tile.store_ptr_tko":
			if op.Attributes != "{memory_ordering_semantics = weak}" || len(args) != 4 || args[1].kind != tileir.KindFloat32 || args[2].kind != tileir.KindInt1 || args[2].i != 1 {
				return nil, fmt.Errorf("oracle: invalid store")
			}
			if e := checkPointer(args[0]); e != nil {
				return nil, e
			}
			if args[0].ptr.slot != 2 {
				return nil, fmt.Errorf("oracle: input overwritten")
			}
			if e := consume(op.Operands[3]); e != nil {
				return nil, e
			}
			index := args[0].ptr.index
			writes[index]++
			buffers[2][index+1] = float32(args[1].f)
			result.kind = tileir.KindToken
			lastStore = op.Results[0]
		case "cuda_tile.return":
			if len(op.Operands) != 0 || len(op.Results) != 0 {
				return nil, fmt.Errorf("oracle: non-void return")
			}
			continue
		default:
			return nil, fmt.Errorf("oracle: unsupported decoded operation %s", op.OpCode)
		}
		if len(op.Results) == 0 {
			return nil, fmt.Errorf("oracle: missing SSA result")
		}
		if _, exists := values[op.Results[0]]; exists {
			return nil, fmt.Errorf("oracle: duplicate SSA")
		}
		values[op.Results[0]] = result
		if result.kind == tileir.KindToken {
			tokens[op.Results[0]] = 0
		}
	}
	for name, uses := range tokens {
		if uses != 1 && !(uses == 0 && name == lastStore) {
			return nil, fmt.Errorf("oracle: orphan token %s", name)
		}
	}
	for i, count := range writes {
		if count != 1 {
			return nil, fmt.Errorf("oracle: output %d has %d writes", i, count)
		}
	}
	if !reflect.DeepEqual(beforeA, buffers[0]) || !reflect.DeepEqual(beforeB, buffers[1]) || buffers[2][0] != guard || buffers[2][outputs+1] != guard {
		return nil, fmt.Errorf("oracle: input or guard changed")
	}
	return append([]float32(nil), buffers[2][1:outputs+1]...), nil
}

func TestSemanticMatMulDecoded(t *testing.T) {
	for _, tc := range []struct {
		name    string
		m, n, k int
		reverse bool
	}{
		{"singleton", 1, 1, 1, false},
		{"rectangular", 2, 4, 3, false},
		{"input_order", 2, 4, 3, true},
		{"largest", 4, 4, 4, false},
		{"cancellation", 1, 1, 3, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			_, snap := matmulGraph(t, tc.m, tc.n, tc.k, tc.reverse)
			r, err := tileir.LowerSemanticRegion(snap, tileir.LowerOptions{})
			if err != nil || r.Memory == nil {
				t.Fatalf("typed matmul dispatch: %v", err)
			}
			direct, err := tileir.LowerSemanticMatMul(snap, tileir.LowerOptions{})
			if err != nil {
				t.Fatal(err)
			}
			data, err := r.Module.EncodeBytecode()
			if err != nil {
				t.Fatal(err)
			}
			other, err := direct.Module.EncodeBytecode()
			if err != nil || !bytes.Equal(data, other) {
				t.Fatal("dispatch and public adapter disagree")
			}
			bm, err := tileir.DecodeBytecodeBytes(data)
			if err != nil {
				t.Fatal(err)
			}
			decodedModule := bm.ToModule()
			if decodedModule == nil {
				t.Fatal("typed body did not decode to an AST")
			}
			roundtrip, err := decodedModule.EncodeBytecode()
			if err != nil || !bytes.Equal(data, roundtrip) {
				t.Fatalf("bytecode round trip: %v", err)
			}
			a, b := make([]float32, tc.m*tc.k), make([]float32, tc.k*tc.n)
			for i := range a {
				a[i] = float32(i%5-2) * 1.25
			}
			for i := range b {
				b[i] = float32(i%7-3) * 0.5
			}
			if tc.name == "cancellation" {
				a, b = []float32{1e8, 1, -1e8}, []float32{1, 1, 1}
			}
			want, err := cudaref.EvaluateSnapshotAll(snap, map[string][]float32{"left": a, "right": b})
			if err != nil {
				t.Fatal(err)
			}
			got, err := executeMatmul(data, a, b, tc.m*tc.n)
			if err != nil || !reflect.DeepEqual(got, want[snap.Outputs()[0].ID]) {
				t.Fatalf("decoded matmul = %v, %v; want %v", got, err, want[snap.Outputs()[0].ID])
			}
			if tc.name == "cancellation" && got[0] != 1 {
				t.Fatal("widened accumulation lost small term")
			}
			if r.Source != "" || r.Grid != [3]uint32{1, 1, 1} || r.Schema.TotalParamBytes != 24 || len(r.Schema.Params) != 3 {
				t.Fatal("invented source or wrong launch/ABI")
			}
			for i, name := range []string{"left", "right", "product"} {
				p := r.Schema.Params[i]
				if p.Name != name || p.Offset != 8*i || p.Alignment != 8 || p.ByteWidth != 8 || !p.IsPointer {
					t.Fatalf("ABI slot %d: %+v", i, p)
				}
			}
			if err := r.Memory.Verify(r); err != nil {
				t.Fatal(err)
			}
			refs := r.Memory.References()
			out := snap.Outputs()[0]
			for i, node := range []*cudair.Node{out.Inputs[0], out.Inputs[1], out, out} {
				if refs[i].NodeID != node.ID || refs[i].Location.File != node.SourceLoc.File || len(refs[i].Operations) == 0 || len(refs[i].Values) == 0 {
					t.Fatal("source origin missing")
				}
				for _, op := range refs[i].Operations {
					if op < 0 || op >= len(r.Module.Functions[0].Operations) {
						t.Fatal("source operation outside module")
					}
				}
			}
			refs[2].Location.Callers[0].Line++
			refs[0].Operations[0]++
			if reflect.DeepEqual(refs, r.Memory.References()) || r.Memory.Verify(r) != nil {
				t.Fatal("returned source map aliases binding")
			}
		})
	}
}

func TestSemanticMatMulRefusals(t *testing.T) {
	for _, tc := range []struct {
		name string
		edit func(*cudair.Graph)
	}{
		{"dtype", func(g *cudair.Graph) {
			for _, n := range g.Nodes() {
				n.Type = cudair.Float16
			}
		}},
		{"layout", func(g *cudair.Graph) { g.Nodes()[0].Layout = cudashape.LayoutColumnMajor }},
		{"strides", func(g *cudair.Graph) { g.Nodes()[0].Strides = []int{1, 2} }},
		{"names", func(g *cudair.Graph) { g.Nodes()[1].Name = g.Nodes()[0].Name }},
		{"empty_name", func(g *cudair.Graph) { g.Nodes()[0].Name = "" }},
		{"effect", func(g *cudair.Graph) { g.Nodes()[2].Effect = cudair.EffectMutating }},
		{"alias", func(g *cudair.Graph) { g.Nodes()[2].StorageAlias = g.Nodes()[0] }},
		{"extra_node", func(g *cudair.Graph) { g.AddInput("unused", []int{2, 2}, cudair.Float32) }},
		{"multiple_outputs", func(g *cudair.Graph) { _ = g.SetOutputs(g.Nodes()[0], g.Nodes()[2]) }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			g, _ := matmulGraph(t, 2, 2, 2, false)
			tc.edit(g)
			snap, err := g.Snapshot()
			if err != nil {
				return // Authoritative semantic admission already refuses it.
			}
			if r, err := tileir.LowerSemanticMatMul(snap, tileir.LowerOptions{}); r != nil || err == nil {
				t.Fatal("unsupported graph admitted")
			}
		})
	}
	for _, dims := range [][3]int{{0, 2, 2}, {2, 0, 2}, {2, 2, 0}, {5, 2, 2}, {2, 5, 2}, {2, 2, 5}} {
		_, snap := matmulGraph(t, dims[0], dims[1], dims[2], false)
		if r, err := tileir.LowerSemanticMatMul(snap, tileir.LowerOptions{}); r != nil || !errors.Is(err, tileir.ErrUnsupportedTileOp) {
			t.Fatalf("extent %v admitted: %v", dims, err)
		}
	}
	_, snap := matmulGraph(t, 2, 2, 2, false)
	for _, opts := range []tileir.LowerOptions{{TileM: 16}, {TileN: 2}, {TileK: -1}, {KernelName: "bad-name"}} {
		if r, err := tileir.LowerSemanticMatMul(snap, opts); r != nil || err == nil {
			t.Fatal("unsupported options admitted")
		}
	}
	legacy, err := tileir.LowerSemanticRegion(snap, tileir.LowerOptions{TileM: 16, TileN: 16, TileK: 16})
	if err != nil || legacy.Memory != nil {
		t.Fatalf("legacy analysis changed: %v", err)
	}
	if data, err := legacy.Module.EncodeBytecode(); data != nil || !errors.Is(err, tileir.ErrUnsupportedASTEncoding) {
		t.Fatal("legacy debug projection silently published bytecode")
	}
	if r, err := tileir.LowerSemanticMatMul(nil, tileir.LowerOptions{}); r != nil || err == nil {
		t.Fatal("nil snapshot admitted")
	}
}

func TestSemanticMatMulBindingAndTokenControls(t *testing.T) {
	for _, tc := range []struct {
		name string
		edit func(*tileir.LoweringResult)
	}{
		{"grid", func(r *tileir.LoweringResult) { r.Grid[0] = 2 }},
		{"ABI", func(r *tileir.LoweringResult) { r.Schema.Params[0].Name = "changed" }},
		{"type", func(r *tileir.LoweringResult) {
			for _, op := range r.Module.Functions[0].Operations {
				if op.OpCode == "cuda_tile.mulf" {
					op.TypeVal = tileir.NewTile(nil, tileir.Float32)
					return
				}
			}
		}},
		{"token", func(r *tileir.LoweringResult) {
			for _, op := range r.Module.Functions[0].Operations {
				if op.OpCode == "cuda_tile.store_ptr_tko" {
					op.Operands[3] = "%missing"
					return
				}
			}
		}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			_, snap := matmulGraph(t, 2, 2, 2, false)
			r, err := tileir.LowerSemanticMatMul(snap, tileir.LowerOptions{})
			if err != nil {
				t.Fatal(err)
			}
			tc.edit(r)
			if r.Memory.Verify(r) == nil {
				t.Fatal("changed contract admitted")
			}
			if tc.name == "type" || tc.name == "token" {
				if b, err := r.Module.EncodeBytecode(); b != nil || err == nil {
					t.Fatal("broken SSA/type/token encoded")
				}
			}
		})
	}
}
