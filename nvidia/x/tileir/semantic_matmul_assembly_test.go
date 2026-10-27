package tileir

import (
	"bytes"
	"context"
	"errors"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudair"
	"github.com/tmc/lib/nvidia/x/cudakernel"
	"github.com/tmc/lib/nvidia/x/tileiras"
)

func matmulAssemblyLowering(t *testing.T) *LoweringResult {
	t.Helper()
	g := cudair.NewGraph()
	a := g.AddInput("a", []int{2, 3}, cudair.Float32)
	b := g.AddInput("b", []int{3, 4}, cudair.Float32)
	product := g.AddMatMul(a, b)
	product.Name = "product"
	if err := g.SetOutputs(product); err != nil {
		t.Fatal(err)
	}
	snap, err := g.Snapshot()
	if err != nil {
		t.Fatal(err)
	}
	r, err := LowerSemanticRegion(snap, LowerOptions{})
	if err != nil {
		t.Fatal(err)
	}
	return r
}

func TestSemanticMatMulAssemblyComposition(t *testing.T) {
	r := matmulAssemblyLowering(t)
	data, err := r.Module.EncodeBytecode()
	if err != nil {
		t.Fatal(err)
	}
	calls := 0
	got, err := assembleSemanticMemory(context.Background(), r, "sm_90", func(_ context.Context, input []byte, target string) (*tileiras.Result, error) {
		calls++
		if !bytes.Equal(data, input) || target != "sm_90" {
			t.Fatal("another bytecode/target reached assembler")
		}
		return &tileiras.Result{CUBIN: syntheticMemoryELF(t, "tile_matmul", 90), Log: "synthetic metadata-only fixture; no compiler executed", Remarks: []tileiras.Remark{{Kind: "Analysis", Pass: "HostFixture", Name: "Synthetic", Function: "tile_matmul"}}}, nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if calls != 1 || got.Artifact.Kind() != cudakernel.KindCUBIN || got.Artifact.Target() != "sm_90" || got.Artifact.EntryPoint() != "tile_matmul" || got.Artifact.Schema().TotalParamBytes != 24 || got.BytecodeDigest != r.Memory.BytecodeDigest() || got.Artifact.Toolchain() != "tileiras (version unobserved)" || len(got.Remarks) != 1 {
		t.Fatal("assembly composition lost identity/ABI/diagnostics")
	}
	if got, err := AssembleSemanticMemory(context.Background(), r, "sm_90a"); got != nil || !errors.Is(err, tileiras.ErrArchSuffix) {
		t.Fatal("public feature-suffix preflight changed")
	}
	for _, tc := range []struct {
		name      string
		edit      func(*LoweringResult)
		result    func() (*tileiras.Result, error)
		wantCalls int
	}{
		{"binding", func(r *LoweringResult) { r.Grid[0] = 2 }, nil, 0},
		{"source", func(r *LoweringResult) { r.Memory.references[0].NodeID++ }, nil, 0},
		{"source_location", func(r *LoweringResult) { r.Memory.references[0].Location.Line++ }, nil, 0},
		{"missing_tool", nil, func() (*tileiras.Result, error) { return nil, tileiras.ErrNotFound }, 1},
		{"diagnostic", nil, func() (*tileiras.Result, error) {
			return &tileiras.Result{CUBIN: syntheticMemoryELF(t, "tile_matmul", 90), Remarks: []tileiras.Remark{{Kind: "invented", Pass: "HostFixture", Name: "Bad"}}}, nil
		}, 1},
		{"wrong_entry", nil, func() (*tileiras.Result, error) {
			return &tileiras.Result{CUBIN: syntheticMemoryELF(t, "other", 90)}, nil
		}, 1},
	} {
		t.Run(tc.name, func(t *testing.T) {
			r := matmulAssemblyLowering(t)
			if tc.edit != nil {
				tc.edit(r)
			}
			calls := 0
			got, err := assembleSemanticMemory(context.Background(), r, "sm_90", func(context.Context, []byte, string) (*tileiras.Result, error) {
				calls++
				if tc.result != nil {
					return tc.result()
				}
				return nil, nil
			})
			if got != nil || err == nil || calls != tc.wantCalls {
				t.Fatalf("refusal=%v calls=%d", err, calls)
			}
			if tc.name == "missing_tool" && !errors.Is(err, tileiras.ErrNotFound) {
				t.Fatal("tool absence identity lost")
			}
		})
	}
}
