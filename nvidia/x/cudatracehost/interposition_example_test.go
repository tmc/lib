package cudatracehost_test

import (
	"fmt"
	"github.com/tmc/lib/nvidia/x/cudaprobe"
	"github.com/tmc/lib/nvidia/x/cudatracehost"
)

func ExampleInterpositionDeclaration() {
	cfg := cudatracehost.Config{Symbols: []string{"cuLaunchKernel"}, Interposition: []cudatracehost.InterpositionDeclaration{{Symbol: "cuLaunchKernel", Interposition: cudaprobe.Interposition{Status: "declared_active", Evidence: "operator run manifest"}}}}
	fmt.Println(cfg.Interposition[0].Symbol, cfg.Interposition[0].Interposition.Status)
	// Output: cuLaunchKernel declared_active
}
