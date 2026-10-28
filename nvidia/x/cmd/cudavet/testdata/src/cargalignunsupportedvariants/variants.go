package cargalignunsupportedvariants

import "github.com/tmc/lib/nvidia/x/cudakernel"

func layouts(params []cudakernel.Param, total int) {
	_ = cudakernel.Schema{Params: []cudakernel.Param{}, TotalParamBytes: total}             // want `kernel parameter layout not checked: total parameter bytes are not constant`
	_ = cudakernel.Schema{[]cudakernel.Param{}, 0}                                          // want `kernel parameter layout not checked: schema fields are not keyed literals`
	_ = cudakernel.Schema{Params: params}                                                   // want `kernel parameter layout not checked: parameters are not written as constant literals`
	_, _ = cudakernel.NewSchema(params...)                                                  // want `kernel parameter layout not checked: parameters are not written as literals`
	_, _ = cudakernel.NewSchema(cudakernel.Param{Name: "n", Type: "u32", ByteWidth: total}) // want `kernel parameter layout not checked: parameters are not written as constant literals`
}
