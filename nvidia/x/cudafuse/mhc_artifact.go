package cudafuse

import (
	"fmt"
	"github.com/tmc/lib/nvidia/x/cudakernel"
	"github.com/tmc/lib/nvidia/x/nvrtcx"
)

func mhcSchema(names []string) (cudakernel.Schema, error) {
	var params []cudakernel.Param
	for _, name := range names {
		params = append(params, cudakernel.Param{Name: name, Type: "ptr", ByteWidth: 8, Alignment: 8, IsPointer: true})
	}
	params = append(params, cudakernel.Param{Name: "count", Type: "s32", ByteWidth: 4, Alignment: 4})
	return cudakernel.NewSchema(params...)
}

// CompileMHC compiles the specialized pair to immutable PTX artifacts with
// explicit parameter schemas. It performs no device load or launch. Resources
// not reported by the compiler remain unknown (zero), rather than estimates.
// ExecuteMHC compiles the identical source and flags for its stream's device.
func CompileMHC(c MHCConfig, major, minor int) ([2]*cudakernel.Artifact, error) {
	var out [2]*cudakernel.Artifact
	sources, err := MHCSource(c)
	if err != nil {
		return out, err
	}
	for i, s := range sources {
		cm, err := nvrtcx.Compile(s.Source, nvrtcx.WithArch(major, minor), nvrtcx.WithFlag("--fmad=false"))
		if err != nil {
			return out, fmt.Errorf("cudafuse: compile mhc %s: %w", s.Name, err)
		}
		names := append(append([]string(nil), s.Inputs...), s.Outputs...)
		schema, err := mhcSchema(names)
		if err != nil {
			return out, err
		}
		out[i], err = cm.ToArtifact(s.Name, schema, fmt.Sprintf("compute_%d%d", major, minor))
		if err != nil {
			return out, err
		}
	}
	return out, nil
}
