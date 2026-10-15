package tileir

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"strings"
	"testing"
	"unsafe"

	"github.com/tmc/lib/nvidia/x/cudair"
	"github.com/tmc/lib/nvidia/x/cudakernel"
	"github.com/tmc/lib/nvidia/x/tileiras"
)

func reductionLowering(t *testing.T, op string, shape []int, axis int, width int64) (*cudair.Snapshot, *LoweringResult) {
	t.Helper()
	g := cudair.NewGraph()
	a := g.AddInput("a", shape, cudair.Float32)
	out := g.AddReduce(op, a, axis)
	out.SourceLoc = cudair.SourceLocation{File: "reduce.go", Line: 12, Function: "Reduce"}
	snap, err := g.Snapshot()
	if err != nil {
		t.Fatal(err)
	}
	r, err := LowerSemanticRegion(snap, LowerOptions{TileM: width})
	if err != nil {
		t.Fatal(err)
	}
	return snap, r
}

func TestReductionBytecodeRoutes(t *testing.T) {
	prologue := []string{"make_token", "get_tile_block_id", "exti", "constant", "muli", "reshape", "broadcast", "iota", "addi", "constant"}
	for _, op := range []string{"sum", "max"} {
		for _, tc := range []struct {
			shape []int
			axis  int
		}{{[]int{5}, 0}, {[]int{5, 3}, 1}, {[]int{3, 5}, 0}, {[]int{64}, 0}} {
			for _, width := range []int64{1, 4, 1024} {
				t.Run(fmt.Sprint(op, tc.shape, tc.axis, width), func(t *testing.T) {
					snap, r := reductionLowering(t, op, tc.shape, tc.axis, width)
					fn := r.Module.Functions[0]
					bm, err := r.Module.ToBytecode()
					if err != nil {
						t.Fatal(err)
					}
					if len(bm.Functions) != 1 || len(bm.Functions[0].Operations) != len(fn.Operations) || !bm.Functions[0].IsEntry {
						t.Fatal("decoded body differs from the emitted operation list")
					}
					data, err := r.Module.EncodeBytecode()
					if err != nil || len(data) == 0 {
						t.Fatalf("encode: %v", err)
					}
					if err := r.Memory.Verify(r); err != nil {
						t.Fatal(err)
					}
					if errs := fn.verifyReductionSSA(); len(errs) != 0 {
						t.Fatalf("policy: %v", errs)
					}
					if err := ValidateTileCPPSource(r.Source); err != nil {
						t.Fatalf("C++ reference source: %v", err)
					}
					if len(r.Module.Locations) != 0 || fn.OptHints != "" || len(fn.Params) != 2 {
						t.Fatal("module carries debug sections, hints or a different ABI")
					}
					for i, code := range prologue {
						if fn.Operations[i].OpCode != "cuda_tile."+code {
							t.Fatalf("operation %d = %s, want %s", i, fn.Operations[i].OpCode, code)
						}
					}
					if n := len(fn.Operations); fn.Operations[n-2].OpCode != "cuda_tile.store_ptr_tko" || fn.Operations[n-1].OpCode != "cuda_tile.return" || len(fn.Operations[n-1].Operands) != 0 {
						t.Fatal("kernel must end in a store and a void return")
					}
					conversions, loads := 0, 0
					for _, o := range fn.Operations {
						if strings.Contains(o.Attributes, "semantic_id") || o.LocRef != "" || o.Loc != (Location{}) {
							t.Fatalf("semantic identity in bytecode operation %s", o.OpCode)
						}
						switch o.OpCode {
						case "cuda_tile.ftof":
							conversions++
						case "cuda_tile.load_ptr_tko":
							loads++
						}
					}
					in, out := snap.Inputs()[0], snap.Outputs()[0]
					terms := in.Shape[tc.axis]
					if want := map[string]int{"sum": terms + 1, "max": 0}[op]; conversions != want || loads != terms {
						t.Fatalf("%d conversions and %d loads, want %d and %d", conversions, loads, want, terms)
					}
					roles := make(map[string]MemorySource)
					for _, ref := range r.Memory.References() {
						roles[ref.Role] = ref
					}
					if len(roles) != 3 || roles["input"].NodeID != in.ID || roles[op].NodeID != out.ID || roles["output"].ABIIndex != 1 ||
						roles[op].Location.File != "reduce.go" || roles["output"].Location.Line != 12 || len(roles["input"].Operations) != 2+2*terms {
						t.Fatalf("semantic records: %+v", roles)
					}
					for _, i := range roles["output"].Operations {
						if code := fn.Operations[i].OpCode; code != "cuda_tile.reshape" && code != "cuda_tile.broadcast" && code != "cuda_tile.offset" && code != "cuda_tile.store_ptr_tko" {
							t.Fatalf("output record holds %s", code)
						}
					}
				})
			}
		}
	}
}

func TestReductionBytecodeFloatConversion(t *testing.T) {
	f32, f64 := NewTile([]int64{4}, Float32), NewTile([]int64{4}, Float64)
	module := func(from, to Type, attrs string) *Module {
		return &Module{Functions: []*Function{{Name: "k", Entry: true, Operations: []*Operation{
			{OpCode: "cuda_tile.constant", Results: []string{"%x"}, TypeVal: from, Attributes: "{value = 1.0}"},
			{OpCode: "cuda_tile.ftof", Results: []string{"%y"}, TypeVal: to, Operands: []string{"%x"}, Attributes: attrs},
			{OpCode: "cuda_tile.return"},
		}}}}
	}
	for _, tc := range []struct{ from, to Type }{{f32, f64}, {f64, f32}} {
		bm, err := module(tc.from, tc.to, "{rounding_mode = nearest_even}").ToBytecode()
		if err != nil {
			t.Fatal(err)
		}
		index := func(want Type) byte {
			for i, typ := range bm.Types {
				if typeEqual(typ, want) {
					return byte(i)
				}
			}
			t.Fatalf("type %s not registered", want)
			return 0
		}
		// Ops.td:2198-2240 and BytecodeGen.cpp:660-666: opcode 0x2A, the
		// result type, the required rounding_mode (AttrDefs.td nearest_even
		// is 0) and the single operand. No flags or operand count.
		body := bm.Functions[0].BodyBytes
		want := []byte{0x10, index(tc.from), 0, 0x2a, index(tc.to), 0, 0, 0x5c, 0, 0}
		if !bytes.Equal(body, want) {
			t.Fatalf("body=%x want=%x", body, want)
		}
		bad := bytes.Clone(body)
		bad[5] = 1 // rounding_mode = zero
		if ops, err := decodeASTBody(bad, bm.Types, bm.Constants, bm.Functions[0].Signature); ops != nil || err == nil {
			t.Fatal("decoded ftof with a directed rounding mode")
		}
	}
	for _, tc := range []struct {
		name     string
		from, to Type
		attrs    string
	}{
		{"zero", f32, f64, "{rounding_mode = zero}"},
		{"nearest away", f64, f32, "{rounding_mode = nearest_away}"},
		{"missing mode", f32, f64, ""},
		{"extra attribute", f32, f64, "{rounding_mode = nearest_even, flush_to_zero}"},
		{"identity", f32, f32, "{rounding_mode = nearest_even}"},
		{"half", f32, NewTile([]int64{4}, Float16), "{rounding_mode = nearest_even}"},
		{"integer", f32, NewTile([]int64{4}, Int32), "{rounding_mode = nearest_even}"},
		{"shape", f32, NewTile([]int64{2}, Float64), "{rounding_mode = nearest_even}"},
	} {
		if data, err := module(tc.from, tc.to, tc.attrs).EncodeBytecode(); data != nil || !errors.Is(err, ErrUnsupportedASTEncoding) {
			t.Errorf("%s: ftof accepted: %x %v", tc.name, data, err)
		}
	}
}

// Well-typed bytecode can still violate the reduction policy. The policy
// check, not the encoder, refuses these.
func TestReductionBytecodePolicy(t *testing.T) {
	for _, tc := range []struct {
		name   string
		mutate func(*Operation) bool
	}{
		{"unmasked load", func(op *Operation) bool {
			if op.OpCode != "cuda_tile.load_ptr_tko" {
				return false
			}
			op.Operands = []string{op.Operands[0], op.Operands[2], op.Operands[3]}
			return true
		}},
		{"flushed accumulation", func(op *Operation) bool {
			if op.OpCode != "cuda_tile.addf" {
				return false
			}
			op.Attributes = "{rounding_mode = nearest_even, flush_to_zero}"
			return true
		}},
		{"directed accumulation", func(op *Operation) bool {
			if op.OpCode != "cuda_tile.addf" {
				return false
			}
			op.Attributes = "{rounding_mode = zero}"
			return true
		}},
		{"signed bound", func(op *Operation) bool {
			if op.OpCode != "cuda_tile.cmpi" {
				return false
			}
			op.Attributes = "{comparison_predicate = less_than, signedness = signed}"
			return true
		}},
		{"wrapping index", func(op *Operation) bool {
			if op.OpCode != "cuda_tile.addi" {
				return false
			}
			op.Attributes = "{overflow = none}"
			return true
		}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			_, r := reductionLowering(t, "sum", []int{5, 3}, 1, 4)
			fn := r.Module.Functions[0]
			mutated := false
			for _, op := range fn.Operations {
				if mutated = tc.mutate(op); mutated {
					break
				}
			}
			if !mutated {
				t.Fatal("mutation did not apply")
			}
			if _, err := r.Module.EncodeBytecode(); err != nil {
				t.Fatalf("control is not well typed: %v", err)
			}
			if len(fn.verifyReductionSSA()) == 0 {
				t.Fatal("policy violation accepted")
			}
		})
	}
	_, r := reductionLowering(t, "max", []int{5, 3}, 1, 4)
	for _, op := range r.Module.Functions[0].Operations {
		if op.Attributes == reductionGreater {
			op.Attributes = "{comparison_predicate = greater_than_or_equal, comparison_ordering = ordered}"
			break
		}
	}
	if _, err := r.Module.EncodeBytecode(); err != nil {
		t.Fatal(err)
	}
	if len(r.Module.Functions[0].verifyReductionSSA()) == 0 {
		t.Fatal("max tie policy violation accepted")
	}
}

func TestReductionBytecodeRefusals(t *testing.T) {
	for _, op := range []string{"mean"} {
		g := cudair.NewGraph()
		g.AddReduce(op, g.AddInput("a", []int{4}, cudair.Float32), 0)
		snap, err := g.Snapshot()
		if err != nil {
			t.Fatal(err)
		}
		if r, err := LowerSemanticRegion(snap, LowerOptions{TileM: 4}); r != nil || !errors.Is(err, ErrUnsupportedTileOp) {
			t.Errorf("%s reduction lowered: %v", op, err)
		}
	}
	// Attention's row max and normalizer reduce computed scores, not direct
	// inputs; cudair has no broadcasting, so the normalizing division is
	// omitted. The ordered route must refuse the composition, not part of it.
	g := cudair.NewGraph()
	q := g.AddInput("q", []int{4, 8}, cudair.Float32)
	k := g.AddInput("k", []int{8, 4}, cudair.Float32)
	v := g.AddInput("v", []int{4, 8}, cudair.Float32)
	scores := g.AddMatMul(q, k)
	weights := g.AddUnary("exp", scores)
	g.AddReduce("max", scores, 1)
	g.AddReduce("sum", weights, 1)
	g.AddMatMul(weights, v)
	snap, err := g.Snapshot()
	if err != nil {
		t.Fatal(err)
	}
	if r, err := LowerSemanticRegion(snap, LowerOptions{TileM: 4}); r != nil || !errors.Is(err, ErrUnsupportedTileOp) {
		t.Fatalf("attention lowered: %v", err)
	}
	g = cudair.NewGraph()
	g.AddReduce("sum", g.AddInput("a", []int{4}, cudair.Float32), 0)
	snap, err = g.Snapshot()
	if err != nil {
		t.Fatal(err)
	}
	if r, err := LowerSemanticRegion(snap, LowerOptions{TileM: 2048}); r != nil || !errors.Is(err, ErrUnsupportedTileOp) {
		t.Fatalf("pointer tile wider than 1024 lowered: %v", err)
	}
}

func TestReductionAssemblyEntryABI(t *testing.T) {
	assemble := func(r *LoweringResult) (called bool, err error) {
		_, err = assembleSemanticMemory(context.Background(), r, "sm_90", func(context.Context, []byte, string) (*tileiras.Result, error) {
			called = true
			return nil, tileiras.ErrNotFound
		})
		return called, err
	}

	// Every output is empty: the entry has no parameters, the schema two.
	_, empty := reductionLowering(t, "sum", []int{0, 3}, 1, 4)
	if empty.Grid != [3]uint32{0, 1, 1} || len(empty.Module.Functions[0].Params) != 0 || len(empty.Schema.Params) != 2 {
		t.Fatalf("empty reduction: grid %v, %d entry parameters, schema %v", empty.Grid, len(empty.Module.Functions[0].Params), empty.Schema)
	}
	if err := empty.Memory.Verify(empty); err != nil {
		t.Fatalf("binding: %v", err)
	}
	if called, err := assemble(empty); called || err == nil || !strings.Contains(err.Error(), "entry has 0 parameters but schema has 2") {
		t.Fatalf("empty reduction: assembler called %v, err %v", called, err)
	}

	// The same graph with work reaches the assembler.
	_, work := reductionLowering(t, "sum", []int{5, 3}, 1, 4)
	if called, err := assemble(work); !called || !errors.Is(err, tileiras.ErrNotFound) {
		t.Fatalf("reduction with work: assembler called %v, err %v", called, err)
	}

	// A rebound schema whose output is not a pointer keeps the count.
	work.Schema.Params = append([]cudakernel.Param(nil), work.Schema.Params...)
	work.Schema.Params[1].IsPointer = false
	work.Memory.contractDigest = memoryContract(work, work.Memory.references)
	if err := work.Memory.Verify(work); err != nil {
		t.Fatalf("rebound binding: %v", err)
	}
	if called, err := assemble(work); called || err == nil || !strings.Contains(err.Error(), "parameter 1 pointer kind") {
		t.Fatalf("pointer kind: assembler called %v, err %v", called, err)
	}
}

// Reduction parameters are bare pointers and use the 8-byte slots of the
// singleton memory and vector routes, so Pack leaves no gap between them.
func TestReductionSchemaPointerSlots(t *testing.T) {
	want := []byte{
		0x88, 0x77, 0x66, 0x55, 0x44, 0x33, 0x22, 0x11,
		0x00, 0xff, 0xee, 0xdd, 0xcc, 0xbb, 0xaa, 0x99,
	}
	for _, op := range []string{"sum", "max"} {
		for _, tc := range []struct {
			shape []int
			axis  int
		}{{[]int{5}, 0}, {[]int{5, 3}, 1}} {
			_, r := reductionLowering(t, op, tc.shape, tc.axis, 4)
			s := r.Schema
			if len(s.Params) != 2 || s.TotalParamBytes != 16 {
				t.Fatalf("%s %v: %d parameters, %d bytes, want 2 and 16", op, tc.shape, len(s.Params), s.TotalParamBytes)
			}
			for i, p := range s.Params {
				if !p.IsPointer || p.ByteWidth != 8 || p.Alignment != 8 || p.Offset != 8*i {
					t.Fatalf("%s %v: parameter %d = %+v, want an 8-byte pointer at offset %d", op, tc.shape, i, p, 8*i)
				}
			}
			args := []any{uint64(0x1122334455667788), uint64(0x99aabbccddeeff00)}
			packed, err := s.Pack(args...)
			if err != nil {
				t.Fatal(err)
			}
			if !bytes.Equal(packed, want) {
				t.Fatalf("%s %v: Pack = %x, want %x", op, tc.shape, packed, want)
			}
			ptrs, err := s.PackPointers(args...)
			if err != nil {
				t.Fatal(err)
			}
			for i, p := range ptrs {
				if got := unsafe.Slice((*byte)(p), 8); !bytes.Equal(got, want[8*i:8*i+8]) {
					t.Fatalf("%s %v: PackPointers[%d] = %x, want %x", op, tc.shape, i, got, want[8*i:8*i+8])
				}
			}
		}
	}
}

// The entry check reads the pointer kind from the module parameter's
// element type, not only from the tile wrapper.
func TestReductionEntryPointerKind(t *testing.T) {
	schema, err := cudakernel.NewSchema(cudakernel.Param{Name: "a", Type: "f32", ByteWidth: 8, Alignment: 8, IsPointer: true})
	if err != nil {
		t.Fatal(err)
	}
	param := &Param{Name: "%arg0", TypeVal: NewTile(nil, Float32)}
	r := &LoweringResult{Module: &Module{Functions: []*Function{{Entry: true, Params: []*Param{param}}}}, Schema: schema}
	if checkEntryABI(r) == nil {
		t.Fatal("numeric entry parameter admitted as a pointer")
	}
	param.TypeVal = NewTile(nil, NewPointer(Float32))
	if err := checkEntryABI(r); err != nil {
		t.Fatal(err)
	}
	r.Schema.Params[0].IsPointer = false
	if checkEntryABI(r) == nil {
		t.Fatal("numeric schema parameter admitted as a pointer")
	}
}

func TestReductionReferenceQuietNaN(t *testing.T) {
	for _, width := range []int64{1, 4} {
		_, r := reductionLowering(t, "max", []int{3, 2}, 0, width)
		want := fmt.Sprintf("ct::full<ct::tile<float, ct::shape<%d>>>(std::numeric_limits<float>::quiet_NaN())", width)
		if n := strings.Count(r.Source, want); n != 2 {
			t.Fatalf("width %d: %d quiet NaN tiles, want 2:\n%s", width, n, r.Source)
		}
		if n := strings.Count(r.Source, "quiet_NaN()"); n != 2 {
			t.Fatalf("width %d: %d quiet NaN operands, want 2", width, n)
		}
	}
}

// The C++ reference source recomputes the row base for every term, as it
// did before the route was rebuilt on typed bytecode.
func TestReductionReferenceRowBase(t *testing.T) {
	_, r := reductionLowering(t, "sum", []int{5, 3}, 1, 4)
	// One block start multiply and one row base multiply per term.
	if n := strings.Count(r.Source, " * "); n != 1+3 {
		t.Fatalf("%d multiplies, want 4:\n%s", n, r.Source)
	}
}
