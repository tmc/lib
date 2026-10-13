package tileir

import (
	"bytes"
	"errors"
	"math"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudair"
	"github.com/tmc/lib/nvidia/x/cudashape"
)

func memoryGraph(t *testing.T) (*cudair.Graph, *cudair.Snapshot) {
	t.Helper()
	g := cudair.NewGraph()
	x := g.AddInput("x", []int{1}, cudair.Float32)
	y := g.AddInput("y", []int{1}, cudair.Float32)
	x.SourceLoc = cudair.SourceLocation{File: "input.go", Line: 11, Function: "build", Callers: []cudair.Frame{{File: "caller.go", Line: 22}}}
	add := g.AddBinary("add", x, y)
	add.SourceLoc = cudair.SourceLocation{File: "add.go", Line: 33, Function: "build"}
	if err := g.SetOutputs(add); err != nil {
		t.Fatal(err)
	}
	s, err := g.Snapshot()
	if err != nil {
		t.Fatal(err)
	}
	return g, s
}

func memoryLowering(t *testing.T) *LoweringResult {
	t.Helper()
	_, snap := memoryGraph(t)
	r, err := LowerSemanticMemory(snap, LowerOptions{KernelName: "singleton"})
	if err != nil {
		t.Fatal(err)
	}
	return r
}

func TestSemanticMemoryLowering(t *testing.T) {
	r := memoryLowering(t)
	if r.Source != "" || r.Grid != [3]uint32{1, 1, 1} || len(r.Schema.Params) != 3 || r.Schema.TotalParamBytes != 24 {
		t.Fatal("incorrect standalone contract")
	}
	for i, p := range r.Schema.Params {
		if !p.IsPointer || p.ByteWidth != 8 || p.Alignment != 8 || p.Offset != i*8 {
			t.Fatalf("pointer ABI %d: %+v", i, p)
		}
		if err := checkMemoryPointer(r.Module.Functions[0].Params[i].TypeVal.(*TileType)); err != nil {
			t.Fatal(err)
		}
	}
	fn := r.Module.Functions[0]
	if !fn.Entry || len(fn.Returns) != 0 || len(fn.Operations[3].Results) != 2 || len(fn.Operations[7].Operands) != 0 {
		t.Fatal("load/entry results are wrong")
	}
	if len(r.Module.Locations) != 0 || fn.Operations[3].Attributes != "{memory_ordering_semantics = weak}" {
		t.Fatal("invented wire metadata")
	}
	if err := r.Memory.Verify(r); err != nil {
		t.Fatal(err)
	}
	refs := r.Memory.References()
	if len(refs) != 4 || refs[0].Location.File != "input.go" || refs[0].Location.Callers[0].Line != 22 || refs[2].Location.Line != 33 || refs[3].ABIIndex != 2 {
		t.Fatal("source attribution lost")
	}
	refs[0].Values[0] = "mutated"
	refs[0].Location.Callers[0].Line = 0
	if r.Memory.References()[0].Values[0] != "%arg0" || r.Memory.References()[0].Location.Callers[0].Line != 22 {
		t.Fatal("sidecar accessor aliases storage")
	}
	_, snap := memoryGraph(t)
	dispatched, err := LowerSemanticRegion(snap, LowerOptions{})
	if err != nil || dispatched.Memory == nil {
		t.Fatalf("singleton dispatch: %v", err)
	}
	legacy, err := LowerSemanticRegion(snap, LowerOptions{TileM: 16})
	if err != nil || legacy.Memory != nil {
		t.Fatalf("legacy explicit tile route changed: %v", err)
	}
	// The independently selected C++ reference route remains available.
	if source, _, err := GenerateSemanticTileCPP(snap, LowerOptions{}); err != nil || source == "" {
		t.Fatalf("legacy C++ route changed: %v", err)
	}
}

func TestSemanticMemoryBindingRefusals(t *testing.T) {
	for _, mutate := range []func(*LoweringResult){
		func(r *LoweringResult) { r.Module.Functions[0].Operations[5].Operands[1] = "%x" },
		func(r *LoweringResult) { r.Grid[0] = 2 },
		func(r *LoweringResult) { r.Schema.Params[0].Offset = 8 },
		func(r *LoweringResult) { r.Memory.references[0].NodeID++ },
		func(r *LoweringResult) { r.Memory.byteDigest = "wrong" },
		func(r *LoweringResult) {
			// SSA spelling is not a wire identity, but must still match
			// the source map bound to the original typed module.
			r.Module.Functions[0].Operations[3].Results[0] = "%left"
			r.Module.Functions[0].Operations[5].Operands[0] = "%left"
		},
	} {
		r := memoryLowering(t)
		mutate(r)
		if err := r.Memory.Verify(r); err == nil {
			t.Fatal("mutated binding accepted")
		}
	}
}

func TestSemanticMemoryGraphRefusals(t *testing.T) {
	for _, tt := range []struct {
		name   string
		mutate func(*cudair.Graph)
	}{
		{"shape", func(g *cudair.Graph) {
			for _, n := range g.Nodes() {
				n.Shape = []int{2}
			}
		}},
		{"dtype", func(g *cudair.Graph) {
			for _, n := range g.Nodes() {
				n.Type = cudair.Float16
			}
		}},
		{"operation", func(g *cudair.Graph) { g.Nodes()[2].Op = "mul" }},
		{"strides", func(g *cudair.Graph) { g.Nodes()[0].Strides = []int{2} }},
		{"layout", func(g *cudair.Graph) { g.Nodes()[0].Layout = cudashape.LayoutColumnMajor }},
		{"duplicate_roles", func(g *cudair.Graph) { g.Nodes()[1].Name = "x" }},
		{"reused_input", func(g *cudair.Graph) { g.Nodes()[2].Inputs[1] = g.Nodes()[0] }},
		{"multiple_outputs", func(g *cudair.Graph) { _ = g.SetOutputs(g.Nodes()[0], g.Nodes()[2]) }},
	} {
		t.Run(tt.name, func(t *testing.T) {
			g, _ := memoryGraph(t)
			tt.mutate(g)
			snap, err := g.Snapshot()
			if err != nil {
				return
			} // Authoritative snapshot already refuses it.
			if r, err := LowerSemanticMemory(snap, LowerOptions{}); err == nil || r != nil {
				t.Fatal("unsupported graph lowered")
			}
		})
	}
	_, snap := memoryGraph(t)
	if r, err := LowerSemanticMemory(snap, LowerOptions{TileM: 16}); r != nil || !errors.Is(err, ErrUnsupportedTileOp) {
		t.Fatal("oversize admitted")
	}
	if r, err := LowerSemanticMemory(nil, LowerOptions{}); r != nil || err == nil {
		t.Fatal("nil admitted")
	}
}

// Evaluate just the decoded singleton topology. Pointer slots refer to three
// separate host scalars; this is an independent CPU oracle, never CUDA code.
func evaluateMemory(t *testing.T, data []byte, x, y float32) float32 {
	t.Helper()
	bm, err := DecodeBytecodeBytes(data)
	if err != nil {
		t.Fatal(err)
	}
	values := make(map[string]float32)
	buffers := []float32{x, y, 12345}
	pointers := map[string]int{"%arg0": 0, "%arg1": 1, "%arg2": 2}
	for _, op := range bm.Functions[0].Operations {
		switch op.OpCode {
		case "cuda_tile.load_ptr_tko":
			values[op.Results[0]] = buffers[pointers[op.Operands[0]]]
		case "cuda_tile.addf":
			values[op.Results[0]] = values[op.Operands[0]] + values[op.Operands[1]]
		case "cuda_tile.store_ptr_tko":
			buffers[pointers[op.Operands[0]]] = values[op.Operands[1]]
		case "cuda_tile.constant", "cuda_tile.make_token", "cuda_tile.return":
		default:
			t.Fatalf("unexpected oracle opcode %s", op.OpCode)
		}
	}
	return buffers[2]
}

func TestSemanticMemoryDecodedOracle(t *testing.T) {
	r := memoryLowering(t)
	data, err := r.Module.EncodeBytecode()
	if err != nil {
		t.Fatal(err)
	}
	for _, tt := range []struct{ x, y, want float32 }{{1.25, -10.5, -9.25}, {100, -99, 1}, {math.SmallestNonzeroFloat32, math.SmallestNonzeroFloat32, 2 * math.SmallestNonzeroFloat32}} {
		if got := evaluateMemory(t, data, tt.x, tt.y); got != tt.want {
			t.Fatalf("oracle = %g, want %g", got, tt.want)
		}
	}
	for _, mutate := range []func(*LoweringResult){
		func(r *LoweringResult) { r.Module.Functions[0].Operations[3].Operands[0] = "%arg1" },
		func(r *LoweringResult) { r.Module.Functions[0].Operations[5].Operands[1] = "%x" },
		func(r *LoweringResult) { r.Module.Functions[0].Operations[6].Operands[0] = "%arg0" },
	} {
		r := memoryLowering(t)
		mutate(r)
		changed, err := r.Module.EncodeBytecode()
		if err != nil {
			t.Fatal(err)
		}
		if bytes.Equal(changed, data) || evaluateMemory(t, changed, 1.25, -10.5) == -9.25 {
			t.Fatal("omitted pointer/value topology escaped oracle")
		}
		if r.Memory.Verify(r) == nil {
			t.Fatal("changed topology escaped binding")
		}
	}
}
