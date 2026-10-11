package egraph_test

import (
	"context"
	"errors"
	"testing"

	"github.com/tmc/lib/nvidia/x/egraph"
	"github.com/tmc/lib/nvidia/x/tileir"
)

func TestAssembleTileIRRejectsNil(t *testing.T) {
	if _, err := egraph.AssembleTileIR(context.Background(), nil, egraph.CodegenOptions{}); err == nil {
		t.Fatal("AssembleTileIR accepted nil expression")
	}
}

// TestAssembleTileIRRefusesIncompleteEncoding verifies that unsupported
// encoding is reported before any external assembler is invoked.
func TestAssembleTileIRRefusesIncompleteEncoding(t *testing.T) {
	expr := &egraph.Expr{Op: "tile_swiglu"}
	got, err := egraph.AssembleTileIR(context.Background(), expr, egraph.CodegenOptions{
		KernelName: "swiglu_kernel",
		Precision:  "f16",
	})
	if got != nil || !errors.Is(err, tileir.ErrUnsupportedASTEncoding) {
		t.Fatalf("AssembleTileIR = %x, %v; want unsupported AST encoding", got, err)
	}
}
