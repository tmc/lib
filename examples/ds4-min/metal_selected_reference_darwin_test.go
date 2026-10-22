//go:build darwin && arm64

package main

import (
	"encoding/binary"
	"math"
	"testing"

	libmetal "github.com/tmc/lib/apple/x/metal"
	"github.com/tmc/lib/apple/x/metal/metalmem"
)

// TestMatvecSelectedReferences exercises the per-token references for the two
// routed expert paths.
//
// They exist so a batched selected kernel has something to be bit-identical
// against, and a reference nothing runs is worth nothing: it can be wired to
// the wrong buffer index, ignore the expert IDs entirely, or read every slot
// from slot zero, and still compile. So this checks the properties that would
// catch those -- the IDs reach the kernel, slots are independent, and each
// call costs exactly one command buffer -- rather than pinning values, which
// is the batched kernel's test to write.
func TestMatvecSelectedReferences(t *testing.T) {
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

	const cols = 512
	const rows = 8
	const experts = 16
	const slots = 6

	gate := selectedExpertTensor(arena, "ffn_gate_exps.weight", 16, cols, rows, experts, 1)
	up := selectedExpertTensor(arena, "ffn_up_exps.weight", 16, cols, rows, experts, 2)
	down := selectedExpertTensor(arena, "ffn_down_exps.weight", 10, cols, rows, experts, 3)

	x := make([]float32, cols)
	for i := range x {
		x[i] = float32(i%17)*0.125 - 1
	}
	slotted := make([]float32, slots*cols)
	for slot := range slots {
		for i := range cols {
			slotted[slot*cols+i] = x[i] * float32(slot+1)
		}
	}

	ids := []int32{0, 1, 2, 3, 4, 5}
	other := []int32{9, 1, 2, 3, 4, 5}

	before := ops.submissions
	gateOut, upOut, err := ops.MatvecIQ2Selected(gate, up, ids, x)
	if err != nil {
		t.Fatal(err)
	}
	if got := ops.submissions.sub(before).Buffers; got != 1 {
		t.Errorf("MatvecIQ2Selected used %d command buffers, want 1", got)
	}
	requireFiniteLen(t, "IQ2 gate", gateOut, slots*rows)
	requireFiniteLen(t, "IQ2 up", upOut, slots*rows)

	// The gate and up tensors hold different bytes, so an implementation that
	// bound one buffer twice would return the same numbers for both.
	if equalFloats(gateOut, upOut) {
		t.Error("IQ2 gate and up outputs are identical; the two weight tensors are not both reaching the kernel")
	}

	otherGate, _, err := ops.MatvecIQ2Selected(gate, up, other, x)
	if err != nil {
		t.Fatal(err)
	}
	// Only slot 0's expert changed, so slot 0 must move and the rest must not.
	if equalFloats(gateOut[:rows], otherGate[:rows]) {
		t.Error("changing slot 0's expert ID left slot 0's output unchanged; the IDs are not reaching the kernel")
	}
	if !equalFloats(gateOut[rows:], otherGate[rows:]) {
		t.Error("changing slot 0's expert ID moved another slot's output; the slots are not independent")
	}

	before = ops.submissions
	downOut, err := ops.MatvecQ2KSelected(down, ids, slotted)
	if err != nil {
		t.Fatal(err)
	}
	if got := ops.submissions.sub(before).Buffers; got != 1 {
		t.Errorf("MatvecQ2KSelected used %d command buffers, want 1", got)
	}
	requireFiniteLen(t, "Q2_K down", downOut, slots*rows)

	// Slot inputs are scaled multiples of one vector, so a kernel reading
	// every slot from slot zero would return identical slot outputs.
	if equalFloats(downOut[:rows], downOut[rows:2*rows]) {
		t.Error("Q2_K slots 0 and 1 agree exactly; the kernel is not reading each slot's own input")
	}
}

// selectedExpertTensor builds a 3-D expert weight tensor of raw, finite blocks
// without going through a quantizer, the way cudaExpertFixture does on the CUDA
// side. typeID 16 is IQ2_XXS at 66 bytes per 256 elements, 10 is Q2_K at 84.
func selectedExpertTensor(arena *metalmem.Arena, name string, typeID uint32, cols, rows, experts int, seed uint32) boundMetalTensor {
	blockBytes := 66
	if typeID == 10 {
		blockBytes = 84
	}
	data := metalmem.AllocIn[byte](arena, name, experts*rows*(cols/256)*blockBytes)
	raw := data.HostSlice()
	rng := seed*2654435761 + 1
	next := func() uint32 {
		rng = rng*1664525 + 1013904223
		return rng
	}
	for offset := 0; offset < len(raw); offset += blockBytes {
		p := raw[offset : offset+blockBytes]
		for i := range p {
			p[i] = byte(next())
		}
		if typeID == 16 {
			binary.LittleEndian.PutUint16(p, 0x1c00+uint16(next()%1024))
		} else {
			binary.LittleEndian.PutUint16(p[80:], 0x1c00+uint16(next()%1024))
			binary.LittleEndian.PutUint16(p[82:], 0x1800+uint16(next()%1024))
		}
	}
	return boundMetalTensor{
		name:       name,
		typeID:     typeID,
		dimensions: []uint64{uint64(cols), uint64(rows), uint64(experts)},
		data:       data,
	}
}

func requireFiniteLen(t *testing.T, what string, got []float32, want int) {
	t.Helper()
	if len(got) != want {
		t.Fatalf("%s: got %d elements, want %d", what, len(got), want)
	}
	for i, v := range got {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Fatalf("%s: element %d is %g", what, i, v)
		}
	}
}

func equalFloats(a, b []float32) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if math.Float32bits(a[i]) != math.Float32bits(b[i]) {
			return false
		}
	}
	return true
}
