package cargalign

import "github.com/tmc/lib/nvidia/x/cudakernel"

// packed is the hazard this check exists for: a parameter block written as
// twelve bytes, a u32 followed by a device pointer at offset 4. The device
// reads the pointer at offset 8, so four bytes of it are whatever the
// scalar left behind.
var packed = cudakernel.Schema{
	Params: []cudakernel.Param{
		{Name: "n", Type: "u32", Offset: 0},
		{Name: "p", Type: "ptr", IsPointer: true, Offset: 4}, // want `parameter p declares offset 4; cudakernel.Schema places it at offset 8`
	},
	TotalParamBytes: 12, // want `layout declares 12 total parameter bytes; cudakernel.Schema computes 16`
}

// Omitting Offset in a raw Schema leaves zero, even when the total is right.
var implicit = cudakernel.Schema{
	Params: []cudakernel.Param{
		{Name: "n", Type: "u32", ByteWidth: 4, Alignment: 4},
		{Name: "implicit", Type: "ptr", ByteWidth: 8, Alignment: 8, IsPointer: true}, // want `parameter implicit declares offset 0; cudakernel.Schema places it at offset 8`
	},
	TotalParamBytes: 16,
}

// Explicit and implicit zero offsets must both be checked in a mixed list.
var mixed = cudakernel.Schema{
	Params: []cudakernel.Param{
		{Name: "first", Type: "ptr", IsPointer: true},
		{Name: "explicit", Type: "ptr", IsPointer: true, Offset: 0}, // want `parameter explicit declares offset 0; cudakernel.Schema places it at offset 8`
		{Name: "third", Type: "ptr", IsPointer: true},               // want `parameter third declares offset 0; cudakernel.Schema places it at offset 16`
	},
	TotalParamBytes: 24,
}

// declared writes offsets beside a NewSchema call that recomputes them, so
// the written value is a second and wrong statement of the same fact.
var declared, _ = cudakernel.NewSchema(
	cudakernel.Param{Name: "a", Type: "u32", Offset: 0},
	cudakernel.Param{Name: "b", Type: "u64", Offset: 4}, // want `parameter b declares offset 4; cudakernel.Schema places it at offset 8`
)

// rejected asks for an alignment the ABI has no bank for.
var rejected, _ = cudakernel.NewSchema( // want `cudakernel.NewSchema rejects these parameters: `
	cudakernel.Param{Name: "n", Type: "u32", Alignment: 3},
)

// Tail is the twelve-byte struct in its Go form. Three u32 parameters occupy
// sixteen bytes on the device; the Go struct is twelve, so the tail is short.
//
//cuda:kernelparams
type Tail struct { // want `kernel parameter block Tail is 12 bytes in Go; cudakernel.Schema computes 16 total parameter bytes`
	a uint32
	b uint32
	c uint32
}

// Counted holds a parameter whose width the Go language does not fix, which
// the fixed-width device ABI has no type for.
//
//cuda:kernelparams
type Counted struct {
	n int // want `kernel parameter n has Go type int, which has no device parameter type`
}

// NotAStruct carries the marker on a type that has no fields to lay out.
//
//cuda:kernelparams
type NotAStruct uint64 // want `NotAStruct is marked cuda:kernelparams but is not a struct type`
