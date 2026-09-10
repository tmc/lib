//go:build darwin && arm64

package main

import (
	"encoding/binary"
	"testing"

	libmetal "github.com/tmc/lib/apple/x/metal"
	"github.com/tmc/lib/apple/x/metal/metalmem"
)

// The output head submits one command buffer per token and used to create it
// itself, so neither the submission counter nor the region log saw it. That is
// the whole of the discrepancy the centralization gates were written for:
// cbufs/tok read 86.00 (43 layers x 2 buffers) against a gputrace count of
// 87.0, and a metric with zero variance gives no hint that it is short a term.
//
// TestCommandBufferCreationIsCentralized is the source half of that gate. This
// is the runtime half: it drives the real head kernel and asserts the buffer
// now reaches both counters. A source test alone cannot tell a routed call
// from a routed call that is never made.
func TestMetalHeadProjectionBufferIsCounted(t *testing.T) {
	dev, err := libmetal.NewDevice()
	if err != nil {
		t.Fatal(err)
	}
	defer dev.Close()
	ops, err := newMetalPinnedOps(dev)
	if err != nil {
		t.Fatal(err)
	}
	defer ops.Close()

	arena := metalmem.NewArena(dev)
	defer arena.Free()

	// Four rows of 16384 F16 ones: the one shape MatvecF16 routes to the head.
	const cols, rows = 16384, 4
	data := metalmem.AllocIn[byte](arena, "head-weight", 2*cols*rows)
	for i := 0; i < cols*rows; i++ {
		binary.LittleEndian.PutUint16(data.HostSlice()[2*i:], 0x3c00)
	}
	weight := boundMetalTensor{name: "output.weight", typeID: 1, dimensions: []uint64{cols, rows}, data: data}
	x := make([]float32, cols)
	for i := range x {
		x[i] = float32(i%7) - 3
	}

	log := &metalRegionLog{}
	ops.regions = log
	defer func() { ops.regions = nil }()

	before := ops.submissions
	out, err := ops.MatvecF16(weight, x)
	if err != nil {
		t.Fatal(err)
	}
	if len(out) != rows {
		t.Fatalf("got %d outputs, want %d", len(out), rows)
	}
	if got := ops.submissions.sub(before).Buffers; got != 1 {
		t.Errorf("head projection consumed %d command buffers, want 1; it is creating them outside (*metalPinnedOps).commandBuffer", got)
	}
	spans := log.all()
	if len(spans) != 1 {
		t.Fatalf("got %d recorded spans, want 1; the head's buffer is missing from cbufs/tok", len(spans))
	}
	if spans[0].Region != "output-head" {
		t.Errorf("got region %q, want %q", spans[0].Region, "output-head")
	}
	if spans[0].Layer != -1 {
		t.Errorf("got layer %d, want -1; the head belongs to no layer", spans[0].Layer)
	}
}
