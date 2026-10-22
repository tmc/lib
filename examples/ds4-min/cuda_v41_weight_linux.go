//go:build linux

package main

import (
	"github.com/tmc/lib/examples/ds4-min/dsv41"
	"github.com/tmc/lib/nvidia/x/cudamem"
)

// These borrowed tensor records carry no operator or request state. The names
// remain unchanged so separately reviewed composition packets need no rewrite.
type cudaV41Layer0Q8Weight struct {
	Descriptor dsv41.TensorDescriptor
	Data       cudamem.Slice[byte]
}

type cudaV41Layer0FloatWeight struct {
	Descriptor dsv41.TensorDescriptor
	Data       cudamem.Slice[float32]
}
