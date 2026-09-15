package egraph_test

import (
	"context"
	"errors"
	"testing"

	"github.com/tmc/lib/nvidia/x/egraph"
	"github.com/tmc/lib/nvidia/x/tileiras"
)

func TestAssembleTileIRRejectsNil(t *testing.T) {
	if _, err := egraph.AssembleTileIR(context.Background(), nil, egraph.CodegenOptions{}); err == nil {
		t.Fatal("AssembleTileIR accepted nil expression")
	}
}

// TestAssembleTileIRReportsMissingAssembler verifies that an absent tileiras is
// reported as an error rather than masquerading as a successful assembly by
// returning the unassembled bytecode.
func TestAssembleTileIRReportsMissingAssembler(t *testing.T) {
	expr := &egraph.Expr{Op: "tile_swiglu"}
	got, err := egraph.AssembleTileIR(context.Background(), expr, egraph.CodegenOptions{
		KernelName: "swiglu_kernel",
		Precision:  "f16",
	}, tileiras.WithGPUName("sm_121"))
	if err != nil {
		if !errors.Is(err, tileiras.ErrNotFound) {
			t.Fatalf("AssembleTileIR: %v", err)
		}
		if got != nil {
			t.Errorf("AssembleTileIR returned %d bytes alongside ErrNotFound, want nil", len(got))
		}
		t.Skipf("tileiras absent: %v", err)
	}
	if len(got) == 0 {
		t.Fatal("AssembleTileIR returned no bytes and no error")
	}
	if tileiras.IsBytecode(got) {
		t.Fatal("AssembleTileIR returned Tile IR bytecode, want an assembled CUBIN")
	}
}
