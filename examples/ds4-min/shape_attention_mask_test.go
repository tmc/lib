package main

import (
	"testing"

	"github.com/tmc/lib/examples/ds4-min/dsv4shape"
)

func TestShapeAttentionStepZeroMaskAdmission(t *testing.T) {
	for _, variant := range []dsv4shape.Variant{dsv4shape.Flash, dsv4shape.Pro} {
		c, err := dsv4shape.For(variant)
		if err != nil {
			t.Fatal(err)
		}
		l, err := c.Layer(2)
		if err != nil || !l.Indexer {
			t.Fatal("fixture has no indexer layer")
		}
		// These numeric ranges exercise the real production range admission;
		// they are never converted to host pointers or sent to a driver.
		base := []mxfp4DeviceRange{{0x1000, 512 * 4}, {0x10000, uint64(c.Heads * c.HeadDim * 4)}}
		useMask := shapeAttentionUseMask(l.Indexer, 0)
		ranges, err := appendShapeAttentionMaskRange(base, useMask, 0, mxfp4DeviceRange{})
		if err != nil || useMask || len(ranges) != len(base) {
			t.Fatalf("STEP0_MASK_ADMISSION variant=%d use=%v ranges=%d err=%v", variant, useMask, len(ranges), err)
		}
		if err := checkMXFP4DeviceRanges(ranges...); err != nil {
			t.Fatalf("STEP0_MASK_RANGE variant=%d: %v", variant, err)
		}
		for _, rows := range []uint32{1, uint32(c.IndexerTopK)} {
			useMask = shapeAttentionUseMask(l.Indexer, rows)
			ranges, err = appendShapeAttentionMaskRange(base, useMask, rows, mxfp4DeviceRange{0x10000000, uint64(rows) * 4})
			if err != nil || !useMask || len(ranges) != len(base)+1 {
				t.Fatalf("ACTIVE_MASK_ADMISSION rows=%d use=%v err=%v", rows, useMask, err)
			}
			if err := checkMXFP4DeviceRanges(ranges...); err != nil {
				t.Fatal(err)
			}
		}
	}
	if _, err := appendShapeAttentionMaskRange(nil, true, 0, mxfp4DeviceRange{}); err == nil {
		t.Fatal("accepted a requested empty mask")
	}
	if _, err := appendShapeAttentionMaskRange(nil, true, 1, mxfp4DeviceRange{0x1000, 8}); err == nil {
		t.Fatal("accepted a capacity mask instead of one active row")
	}
}
