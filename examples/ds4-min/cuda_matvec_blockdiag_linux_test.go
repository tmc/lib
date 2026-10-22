//go:build linux

package main

import (
	"context"
	"encoding/binary"
	"math"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudamem"
)

// TestCUDAMatvecQ80BlockDiagMatchesLoop pins the attention output projection's
// fusion to bit identity on CUDA. The runtime encoded that projection as eight
// Q8_0 matvecs of 1024 rows over disjoint 4096-element slices of the attention heads
// (cuda_session_linux.go:664-672). Eight dispatches of 1024 blocks each measure 0.89
// waves per SM on GB10, so all eight are latency bound; 094936CE measured the phase
// at 20.50 ms of a 111.25 ms decode step and 344 dispatches per token.
//
// encodeQ80MatvecQuantizedBlockDiag replaces the loop with one dispatch carrying
// the group on blockIdx.y. The claim this test defends is that the two agree exactly:
// ds4_q8_0_matvec_quantized_blockdiag is ds4_q8_0_matvec_quantized with base pointers
// moved and the exact same multiplication order (d * xb[b].d * float(dot)), so every
// bit of the result must match Float32bits.
//
// Model free: the weights are synthetic Q8_0 with swept F16 scales and full-mantissa
// inputs, ensuring float addition is non-associative on the test data and catches any
// summation permutation or reassociation mutations.
func TestCUDAMatvecQ80BlockDiagMatchesLoop(t *testing.T) {
	const (
		diag = 8
		cols = 4096
		rows = 1024
	)
	r := cudaSynthFoundation(t)

	blocks := cols / 32
	rawWeights := make([]byte, diag*rows*blocks*34)
	for totalRow := range diag * rows {
		for block := range blocks {
			p := rawWeights[(totalRow*blocks+block)*34:]
			binary.LittleEndian.PutUint16(p, uint16(0x2c00+(totalRow*31+block*17)%2048))
			for i := range 32 {
				p[2+i] = byte(int8((totalRow*13+block*7+i*5)%251%127 - 63))
			}
		}
	}
	dims := []uint64{uint64(cols), uint64(diag * rows)}
	weights := cudaSynthTensor(t, r, "attn_output_a.weight", 8, dims, rawWeights)

	inHost := make([]float32, diag*cols)
	for i := range inHost {
		inHost[i] = float32(math.Sin(float64(i) * 0.37))
	}
	inDev, err := cudamem.Alloc[float32](r.device, len(inHost))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = inDev.Free() })
	if err := inDev.CopyFromHost(inHost); err != nil {
		t.Fatal(err)
	}

	run := func(t *testing.T, fused bool) []float32 {
		t.Helper()
		outDev, err := cudamem.Alloc[float32](r.device, diag*rows)
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() { _ = outDev.Free() })

		poison := make([]float32, diag*rows)
		for i := range poison {
			poison[i] = math.Float32frombits(0x7fc00000) // NaN poison
		}
		if err := outDev.CopyFromHost(poison); err != nil {
			t.Fatal(err)
		}

		if fused {
			qAll, err := cudamem.Alloc[byte](r.device, diag*cols/32*36)
			if err != nil {
				t.Fatal(err)
			}
			t.Cleanup(func() { _ = qAll.Free() })
			if err := r.encodeQuantQ80(inDev, qAll, inDev.Len()); err != nil {
				t.Fatal(err)
			}
			if err := r.encodeQ80MatvecQuantizedBlockDiag(weights, qAll, outDev, diag); err != nil {
				t.Fatal(err)
			}
		} else {
			qGroup, err := cudamem.Alloc[byte](r.device, cols/32*36)
			if err != nil {
				t.Fatal(err)
			}
			t.Cleanup(func() { _ = qGroup.Free() })
			for group := range diag {
				w, err := cudaTensorRows(weights, group*rows, rows)
				if err != nil {
					t.Fatal(err)
				}
				x := inDev.Slice(group*cols, (group+1)*cols)
				y := outDev.Slice(group*rows, (group+1)*rows)
				if err := r.encodeQuantQ80(x, qGroup, cols); err != nil {
					t.Fatal(err)
				}
				if err := r.encodeQ80MatvecQuantized(w, qGroup, y); err != nil {
					t.Fatal(err)
				}
			}
		}

		if err := r.device.DefaultStream().Synchronize(context.Background()); err != nil {
			t.Fatal(err)
		}
		res := make([]float32, diag*rows)
		if err := outDev.CopyToHost(res); err != nil {
			t.Fatal(err)
		}
		return res
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
	if nonzero < len(loop)/2 {
		t.Fatalf("only %d of %d outputs nonzero; comparison not exercising kernel", nonzero, len(loop))
	}
}
