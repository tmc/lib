//go:build darwin

package main

import (
	"context"
	"encoding/binary"
	"math"
	"testing"

	libmetal "github.com/tmc/lib/apple/x/metal"
	"github.com/tmc/lib/apple/x/metal/metalmem"
)

// TestMetalMatvecQ80BlockDiagMatchesLoop pins the attention output projection's
// fusion to bit identity. The runtime encodes that projection as eight Q8_0
// matvecs of 1024 rows over disjoint 4096-element slices of the attention heads
// (metal_runtime_darwin.go, the "for group := range 8" loop), and the CUDA port
// does the same at cuda_session_linux.go:664-672. Eight dispatches of 512
// threadgroups each measure 0.89 waves per SM on GB10, so all eight are latency
// bound; 094936CE measured the phase at 20.50 ms of a 111.25 ms decode step and
// 344 dispatches per token.
//
// EncodeMatvecQ80BlockDiag replaces the loop with one dispatch carrying the group
// on group.y. The claim this test defends is not that the two agree to a
// tolerance but that they agree exactly: matvec_q8_0_blockdiag is matvec_q8_0
// with three base pointers moved, so the accumulation order per output element is
// unchanged and every bit of the result must match. A tolerance-based assertion
// here would pass on a kernel that had quietly reassociated the sum, which is the
// mutation that would move the registered logits digest.
//
// Model free: the weights are synthetic Q8_0 built by testBatchQ80Tensor, so this
// runs without a model slot and gates the runtime swap before the anchor run.
func TestMetalMatvecQ80BlockDiagMatchesLoop(t *testing.T) {
	const (
		diag = 8
		cols = 4096
		rows = 1024
	)
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

	// Deliberately not testBatchQ80Tensor: that helper gives every block the
	// scale 0.125 and small integer weights, so with an equally tidy input every
	// partial sum is exactly representable and float addition is associative on
	// the data. A reassociating mutation then cannot be detected -- verified,
	// this test passed one until the values below replaced it. Vary the scales
	// and give the input a full mantissa so the sums actually round.
	weights := blockDiagQ80Tensor(arena, "attn_output_a.weight", cols, diag*rows)
	in := metalmem.AllocIn[float32](arena, "heads", diag*cols)
	for i := range in.HostSlice() {
		in.HostSlice()[i] = float32(math.Sin(float64(i) * 0.37))
	}

	run := func(t *testing.T, fused bool) []float32 {
		t.Helper()
		out := metalmem.AllocIn[float32](arena, "out", diag*rows)
		for i := range out.HostSlice() {
			out.HostSlice()[i] = math.Float32frombits(0x7fc00000) // Poison: every row must be written.
		}
		batch, err := newMetalBatch(ops)
		if err != nil {
			t.Fatal(err)
		}
		if fused {
			if err := batch.EncodeMatvecQ80BlockDiag(weights, in, out, diag); err != nil {
				t.Fatal(err)
			}
		} else {
			for group := range diag {
				w, err := weights.Rows(group*rows, rows)
				if err != nil {
					t.Fatal(err)
				}
				x, err := in.Slice(group*cols, (group+1)*cols)
				if err != nil {
					t.Fatal(err)
				}
				y, err := out.Slice(group*rows, (group+1)*rows)
				if err != nil {
					t.Fatal(err)
				}
				if err := batch.EncodeMatvecQ80(w, x, y); err != nil {
					t.Fatal(err)
				}
			}
		}
		// Commit alone is asynchronous; without the Wait the host reads the
		// poison back and the comparison passes vacuously on two NaN buffers.
		if err := batch.Wait(context.Background()); err != nil {
			t.Fatal(err)
		}
		return append([]float32(nil), out.HostSlice()...)
	}

	loop := run(t, false)
	fused := run(t, true)

	nonzero := 0
	for i := range loop {
		if math.IsNaN(float64(loop[i])) {
			t.Fatalf("row %d unwritten by the eight-dispatch loop", i)
		}
		if loop[i] != 0 {
			nonzero++
		}
		if math.Float32bits(loop[i]) != math.Float32bits(fused[i]) {
			t.Fatalf("row %d: loop %v (%#08x) != fused %v (%#08x)",
				i, loop[i], math.Float32bits(loop[i]), fused[i], math.Float32bits(fused[i]))
		}
	}
	// Guard against the vacuous pass where both arms write zeros: a kernel that
	// returned early for every row would otherwise agree with itself bit for bit.
	if nonzero < len(loop)/2 {
		t.Fatalf("only %d of %d outputs nonzero; the comparison is not exercising the kernel", nonzero, len(loop))
	}
}

// blockDiagQ80Tensor builds a Q8_0 tensor whose per-block F16 scales differ from
// one another, so that the products entering each dot product land on unrelated
// exponents and the accumulation rounds. See the comment at its call site.
func blockDiagQ80Tensor(arena *metalmem.Arena, name string, cols, rows int) boundMetalTensor {
	blocks := cols / 32
	data := metalmem.AllocIn[byte](arena, name, rows*blocks*34)
	for row := range rows {
		for block := range blocks {
			p := data.HostSlice()[(row*blocks+block)*34:]
			// Raw F16 bits, as cuda_experts_linux_test.go:199 does: exponents
			// around 2^-4 with the mantissa swept, which is all this needs.
			binary.LittleEndian.PutUint16(p, uint16(0x2c00+(row*31+block*17)%2048))
			for i := range 32 {
				p[2+i] = byte(int8((row*13+block*7+i*5)%251%127 - 63))
			}
		}
	}
	return boundMetalTensor{name: name, typeID: 8, dimensions: []uint64{uint64(cols), uint64(rows)}, data: data}
}
