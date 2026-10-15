// Package cargalignclean states every layout the way the authority computes
// it, and must produce no diagnostics.
package cargalignclean

import "github.com/tmc/lib/nvidia/x/cudakernel"

var schema = cudakernel.Schema{
	Params: []cudakernel.Param{
		{Name: "n", Type: "u32", Offset: 0},
		{Name: "p", Type: "ptr", IsPointer: true, Offset: 8},
	},
	TotalParamBytes: 16,
}

var built, _ = cudakernel.NewSchema(
	cudakernel.Param{Name: "a", Type: "u32"},
	cudakernel.Param{Name: "b", Type: "u64"},
	cudakernel.Param{Name: "out", Type: "ptr", IsPointer: true},
)

// The first raw parameter can omit zero; later parameters need their offsets.
var pointers = cudakernel.Schema{
	Params: []cudakernel.Param{
		{Name: "first", Type: "ptr", IsPointer: true},
		{Name: "second", Type: "ptr", IsPointer: true, Offset: 8},
		{Name: "third", Type: "ptr", IsPointer: true, Offset: 16},
	},
	TotalParamBytes: 24,
}

var builtPointers, _ = cudakernel.NewSchema([]cudakernel.Param{
	{Name: "first", Type: "ptr", IsPointer: true},
	{Name: "second", Type: "ptr", IsPointer: true},
	{Name: "third", Type: "ptr", IsPointer: true},
}...)

// Names alone do not make a local type a cudakernel.Schema declaration.
type Param struct {
	Name string
	Type string
}

type Schema struct {
	Params          []Param
	TotalParamBytes int
}

var local = Schema{Params: []Param{{Name: "n", Type: "u32"}, {Name: "p", Type: "ptr"}}, TotalParamBytes: 16}

// Args is laid out the same way by the Go compiler and by the authority.
//
//cuda:kernelparams
type Args struct {
	n     uint32
	scale float32
	out   uintptr
}

// runtimeBuilt assembles parameters from values the analyzer cannot see. It
// is outside the check, not clean, and says so only under -unsupported.
func runtimeBuilt(names []string) (cudakernel.Schema, error) {
	params := make([]cudakernel.Param, 0, len(names))
	for _, n := range names {
		params = append(params, cudakernel.Param{Name: n, Type: "u32"})
	}
	return cudakernel.NewSchema(params...)
}
