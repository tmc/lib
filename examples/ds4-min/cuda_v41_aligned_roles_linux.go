//go:build linux

package main

import (
	"errors"
	"fmt"
	"slices"

	"github.com/tmc/lib/examples/ds4-min/dsv41"
)

// These are exact Model.tensorRequirements names, not interchangeable shapes.
// QB, grouped output A and shared down retain their separately qualified route.
func cudaV41AlignedQ8NamedShape(name string) (k, m uint64, ok bool) {
	if name == "output.weight" {
		return 5120, 129280, true
	}
	for layer := 0; layer < 40; layer++ {
		prefix := fmt.Sprintf("blk.%d.", layer)
		switch name {
		case prefix + "attn_q_a.weight":
			return 5120, 1280, true
		case prefix + "attn_kv.weight":
			return 5120, 512, true
		case prefix + "attn_output_b.weight":
			return 8192, 5120, true
		case prefix + "ffn_gate_shexp.weight", prefix + "ffn_up_shexp.weight":
			return 5120, 2304, true
		}
	}
	return 0, 0, false
}

func cudaV41AlignedQ8RoleMatches(d, bound dsv41.TensorDescriptor, layer int, role string) bool {
	name := "output.weight"
	if layer == -1 {
		if role != name {
			return false
		}
	} else {
		if layer < 0 || layer >= 40 {
			return false
		}
		switch role {
		case "attn_q_a.weight", "attn_kv.weight", "attn_output_b.weight", "ffn_gate_shexp.weight", "ffn_up_shexp.weight":
		default:
			return false
		}
		name = fmt.Sprintf("blk.%d.%s", layer, role)
	}
	return d.Name == name && bound.Name == name &&
		cudaV41AlignedQ8Eligible(d) && cudaV41AlignedQ8Eligible(bound) &&
		d.Encoding == bound.Encoding && d.Offset == bound.Offset &&
		d.Bytes == bound.Bytes && slices.Equal(d.Dimensions, bound.Dimensions)
}

// newCUDAV41AlignedQ8Role binds exactly one validated Model role. No descriptor
// is renamed, and no other layer's same-shaped weights may satisfy the binding.
// A nonnil owner returned with an error still requires Close. Success releases
// the child's source borrow; a containing owner's raw-weight contract may be
// longer. Shared gate/up callers retain distinct children through Reset/Close.
// The head uses layer=-1 and role="output.weight". No head is created per layer.
func newCUDAV41AlignedQ8Role(r *cudaFoundation, model *dsv41.Model, layer int, role string, w cudaV41Layer0Q8Weight) (*cudaV41AlignedQ8, error) {
	bound, ok := model.Tensor(w.Descriptor.Name)
	if !ok || !cudaV41AlignedQ8RoleMatches(w.Descriptor, bound, layer, role) {
		return nil, errors.New("V4.1 aligned Q8: descriptor does not bind requested model role")
	}
	return newCUDAV41AlignedQ8(r, w)
}
