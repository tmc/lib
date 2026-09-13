package egraph_test

import (
	"testing"

	"github.com/tmc/lib/nvidia/x/egraph"
)

func TestAssembleTileIRRejectsNil(t *testing.T) {
	if _, err := egraph.AssembleTileIR(nil, egraph.CodegenOptions{}); err == nil {
		t.Fatal("AssembleTileIR accepted nil expression")
	}
}
