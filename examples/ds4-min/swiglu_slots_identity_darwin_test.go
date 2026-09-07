//go:build darwin && arm64

package main

import (
	"math"
	"testing"

	libmetal "github.com/tmc/lib/apple/x/metal"
	"github.com/tmc/lib/apple/x/metal/metalmem"
)

// TestMetalSwiGLUSlotsBitIdentity pins the relationship between the two routed
// SwiGLU encoders. evalLayer reads the six routing weights back to the host so
// each slot dispatches with an encode-time scalar; the slot kernel keeps them
// GPU-resident and would let the layer close without that readback. The two
// kernels compute the same expression, so the question is whether they agree
// bit-for-bit. The existing routed-expert coverage compares to a tolerance,
// which cannot answer that.
func TestMetalSwiGLUSlotsBitIdentity(t *testing.T) {
	dev, err := libmetal.NewDevice()
	if err != nil {
		t.Skip("no Metal device")
	}
	defer dev.Close()
	ops, err := newMetalPinnedOps(dev)
	if err != nil {
		t.Fatal(err)
	}
	defer ops.Close()

	const slots, width = 6, 2048
	gate, err := metalmem.Alloc[float32](dev, slots*width)
	if err != nil {
		t.Fatal(err)
	}
	defer gate.Free()
	up, err := metalmem.Alloc[float32](dev, slots*width)
	if err != nil {
		t.Fatal(err)
	}
	defer up.Free()
	weights, err := metalmem.Alloc[float32](dev, slots)
	if err != nil {
		t.Fatal(err)
	}
	defer weights.Free()
	scalarOut, err := metalmem.Alloc[float32](dev, slots*width)
	if err != nil {
		t.Fatal(err)
	}
	defer scalarOut.Free()
	slotsOut, err := metalmem.Alloc[float32](dev, slots*width)
	if err != nil {
		t.Fatal(err)
	}
	defer slotsOut.Free()

	g, u := gate.HostSlice(), up.HostSlice()
	for i := range g {
		g[i] = float32(math.Sin(float64(i)*0.37)) * 7.5
		u[i] = float32(math.Cos(float64(i)*0.11)) * 6.25
	}
	w := weights.HostSlice()
	for s := range w {
		w[s] = float32(0.1234567 + 0.0891*float64(s))
	}

	batch, err := newMetalBatch(ops)
	if err != nil {
		t.Fatal(err)
	}
	for s := range slots {
		gs, err := gate.Slice(s*width, (s+1)*width)
		if err != nil {
			t.Fatal(err)
		}
		us, err := up.Slice(s*width, (s+1)*width)
		if err != nil {
			t.Fatal(err)
		}
		os_, err := scalarOut.Slice(s*width, (s+1)*width)
		if err != nil {
			t.Fatal(err)
		}
		if err := batch.EncodeSwiGLUWeighted(gs, us, os_, w[s]); err != nil {
			t.Fatal(err)
		}
	}
	// The scalar kernel declares weight as `constant float &`. Metal lets such
	// a parameter be bound from a buffer as well as from SetBytes, and either
	// way the kernel sees one uniform value. Binding expertW directly would
	// keep the weight GPU-resident without changing the kernel the production
	// path already uses.
	for s := range slots {
		gs, err := gate.Slice(s*width, (s+1)*width)
		if err != nil {
			t.Fatal(err)
		}
		us, err := up.Slice(s*width, (s+1)*width)
		if err != nil {
			t.Fatal(err)
		}
		os_, err := slotsOut.Slice(s*width, (s+1)*width)
		if err != nil {
			t.Fatal(err)
		}
		ws, err := weights.Slice(s, s+1)
		if err != nil {
			t.Fatal(err)
		}
		if err := batch.EncodeSwiGLUWeightedGPU(gs, us, os_, ws); err != nil {
			t.Fatal(err)
		}
	}
	if err := batch.Wait(t.Context()); err != nil {
		t.Fatal(err)
	}

	got, want := slotsOut.HostSlice(), scalarOut.HostSlice()
	nonzero := 0
	for _, v := range want {
		if v != 0 {
			nonzero++
		}
	}
	if nonzero < len(want)/2 {
		t.Fatalf("scalar reference is %d/%d nonzero; inputs do not exercise the kernel", nonzero, len(want))
	}
	diff, first, worst := 0, -1, 0.0
	for i := range want {
		if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
			if first < 0 {
				first = i
			}
			diff++
			if d := math.Abs(float64(got[i]) - float64(want[i])); d > worst {
				worst = d
			}
		}
	}
	if diff != 0 {
		t.Errorf("slot kernel differs from scalar kernel in %d/%d values; first at %d (slot %d): got %v want %v; max |diff| %g",
			diff, len(want), first, first/width, got[first], want[first], worst)
	}
}
