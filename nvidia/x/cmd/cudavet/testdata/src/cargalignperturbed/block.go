// Package block states one parameter layout two ways: beside a NewSchema
// call, and as a Go struct. Its copy under cargalignperturbed is the same
// source, checked against a perturbed authority; only the expected offsets
// differ, which is what the perturbation control asserts.
package block

import "github.com/tmc/lib/nvidia/x/cudakernel"

var schema, _ = cudakernel.NewSchema(
	cudakernel.Param{Name: "n", Type: "u32", Offset: 0},
	cudakernel.Param{Name: "p", Type: "ptr", IsPointer: true, Offset: 4}, // want `parameter p declares offset 4; cudakernel.Schema places it at offset 16`
)

//cuda:kernelparams
type Tail struct { // want `kernel parameter block Tail is 12 bytes in Go; cudakernel.Schema computes 48 total parameter bytes`
	a uint32
	b uint32 // want `kernel parameter b is at Go offset 4; cudakernel.Schema places it at offset 16`
	c uint32 // want `kernel parameter c is at Go offset 8; cudakernel.Schema places it at offset 32`
}
