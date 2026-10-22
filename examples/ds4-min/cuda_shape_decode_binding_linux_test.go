//go:build linux

package main

import (
	"testing"

	"github.com/tmc/lib/examples/ds4-min/dsv4shape"
	"github.com/tmc/lib/nvidia/driver"
)

func TestShapeDecodeBinding(t *testing.T) {
	c, _ := dsv4shape.For(dsv4shape.Pro)
	b := forwardTestBinding(t, c)
	specs, _ := c.Weights()
	for _, name := range []string{"token_embd.weight", "blk.2.indexer.attn_q_b.weight", "output.weight"} {
		var tensor cudaModelTensor
		for _, spec := range specs {
			if spec.Name == name {
				tensor.dimensions = append([]uint64(nil), spec.Dimensions...)
			}
		}
		for _, view := range b.Weights {
			if view.Name == name {
				tensor.name, tensor.typeID, tensor.bytes, tensor.fileOffset, tensor.device = name, view.Type, view.Bytes, view.Offset, driver.DevicePtr(4096)
			}
		}
		if err := validateShapeDecodeTensor(c, b, tensor); err != nil {
			t.Fatal(err)
		}
		for _, fault := range []string{"name", "type", "bytes", "offset", "dimensions", "device"} {
			t.Run(name+"/"+fault, func(t *testing.T) {
				bad := tensor
				bad.dimensions = append([]uint64(nil), tensor.dimensions...)
				switch fault {
				case "name":
					bad.name = "missing.weight"
				case "type":
					bad.typeID++
				case "bytes":
					bad.bytes--
				case "offset":
					bad.fileOffset++
				case "dimensions":
					bad.dimensions[0]++
				case "device":
					bad.device = 0
				}
				if err := validateShapeDecodeTensor(c, b, bad); err == nil {
					t.Fatal("mismatched uploaded weight accepted")
				}
			})
		}
	}
}
