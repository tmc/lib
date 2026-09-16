package cudaoccupancy

import (
	"fmt"

	"github.com/tmc/lib/nvidia/x/nvrtcx"
)

// InspectArtifact reports kernel resources in a compiled CUBIN. PTX and LTOIR
// have no final register allocation and must be linked or loaded first.
func InspectArtifact(cm *nvrtcx.CompiledModule) ([]KernelResource, error) {
	if cm == nil {
		return nil, fmt.Errorf("cudaoccupancy: nil artifact")
	}
	cubin := cm.CUBIN()
	if len(cubin) == 0 {
		return nil, fmt.Errorf("cudaoccupancy: artifact has no CUBIN")
	}
	return InspectCubin(cubin)
}
