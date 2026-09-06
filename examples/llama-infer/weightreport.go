package main

import (
	"fmt"
	"strings"
)

// WeightReport describes the weight storage this engine is actually running on.
//
// Every field is read back from the objects in use rather than from the
// configuration that requested them: DType comes from the bound module, Kernels
// are the entry point names that module resolved, and Bytes is the measured
// size of the weight allocations. A run can therefore be shown to have executed
// in the mode it claims, and an f32 execution can never be recorded as an f16
// success.
type WeightReport struct {
	DType   string   // dtype the kernels were bound for
	Storage string   // dtype the weights are stored in on the device
	Bytes   int      // device bytes occupied by weights alone
	PerElem int      // bytes per weight
	Kernels []string // entry point names actually bound
}

// String renders the report as one line.
func (r WeightReport) String() string {
	return fmt.Sprintf("weights=%s storage=%s bytes=%d per-weight=%d kernels=[%s]",
		r.DType, r.Storage, r.Bytes, r.PerElem, strings.Join(r.Kernels, " "))
}

