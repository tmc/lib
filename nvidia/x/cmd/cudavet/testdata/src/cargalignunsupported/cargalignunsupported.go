package cargalignunsupported

import "github.com/tmc/lib/nvidia/x/cudakernel"

func build(names []string) (cudakernel.Schema, error) {
	params := make([]cudakernel.Param, 0, len(names))
	for _, n := range names {
		params = append(params, cudakernel.Param{Name: n, Type: "u32"})
	}
	return cudakernel.NewSchema(params...) // want `kernel parameter layout not checked: parameters are not written as literals`
}
