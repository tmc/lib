//go:build linux

package main

import (
	"encoding/binary"
	"math"
	"os"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudamem"
)

func TestCUDAFoundation(t *testing.T) {
	path := os.Getenv("DS4_MODEL")
	if path == "" {
		t.Skip("DS4_MODEL is not set")
	}
	r, err := newCUDAFoundation(path)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := r.Close(); err != nil {
			t.Error(err)
		}
	})
	if len(r.model) != pinnedModelBytes {
		t.Fatalf("mapped bytes = %d, want %d", len(r.model), pinnedModelBytes)
	}
	if r.name == "" || r.major == 0 {
		t.Fatalf("invalid CUDA identity %q sm_%d%d", r.name, r.major, r.minor)
	}
	if r.mapped.host == 0 || r.mapped.device == 0 || r.mapped.bytes == 0 {
		t.Fatalf("model range was not registered: %+v", r.mapped)
	}
	if r.mapped.bytes > cudaMaxRegisteredModelRange+uint(os.Getpagesize()) {
		t.Fatalf("registered bytes = %d, want a bounded range", r.mapped.bytes)
	}
	if r.probeTensor == "" || r.probeDevice != r.probeHost {
		t.Fatalf("CUDA read of %q = %#04x, host = %#04x", r.probeTensor, r.probeDevice, r.probeHost)
	}
	t.Logf("CUDA read of %q = host %#04x, device %#04x; registered %d bytes", r.probeTensor, r.probeHost, r.probeDevice, r.mapped.bytes)
}

func TestCUDAQ80Row(t *testing.T) {
	path := os.Getenv("DS4_MODEL")
	if path == "" {
		t.Skip("DS4_MODEL is not set")
	}
	r, err := newCUDAFoundation(path)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := r.Close(); err != nil {
			t.Error(err)
		}
	})

	x := make([]float32, r.probeCols)
	for i := range x {
		x[i] = float32(i%17-8) / 64
	}
	dx, err := cudamem.Alloc[float32](r.device, len(x))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = dx.Free() })
	dout, err := cudamem.Alloc[float32](r.device, 1)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = dout.Free() })
	if err := dx.CopyFromHost(x); err != nil {
		t.Fatal(err)
	}
	if err := r.q80Row(dx, dout); err != nil {
		t.Fatal(err)
	}
	var got [1]float32
	if err := dout.CopyToHost(got[:]); err != nil {
		t.Fatal(err)
	}

	p := r.model[r.probeOffset : r.probeOffset+r.probeBytes]
	var want float32
	for block := 0; block < len(x)/32; block++ {
		scale := cudaTestFloat16(binary.LittleEndian.Uint16(p[block*34:]))
		for i := range 32 {
			want += scale * float32(int8(p[block*34+2+i])) * x[block*32+i]
		}
	}
	delta := math.Abs(float64(got[0] - want))
	limit := 1e-4 + 2e-5*math.Abs(float64(want))
	if delta > limit {
		t.Fatalf("Q8_0 row = %g, want %g (delta %g > %g)", got[0], want, delta, limit)
	}
	t.Logf("Q8_0 tensor %q row 0: cols=%d bytes=%d CUDA=%g CPU=%g delta=%g limit=%g",
		r.probeTensor, r.probeCols, r.probeBytes, got[0], want, delta, limit)
}

func cudaTestFloat16(h uint16) float32 {
	sign := uint32(h&0x8000) << 16
	exp := uint32(h>>10) & 0x1f
	frac := uint32(h & 0x03ff)
	switch exp {
	case 0:
		if frac == 0 {
			return math.Float32frombits(sign)
		}
		e := int32(-14)
		for frac&0x0400 == 0 {
			frac <<= 1
			e--
		}
		frac &= 0x03ff
		return math.Float32frombits(sign | uint32(e+127)<<23 | frac<<13)
	case 0x1f:
		return math.Float32frombits(sign | 0x7f800000 | frac<<13)
	default:
		return math.Float32frombits(sign | uint32(int32(exp)-15+127)<<23 | frac<<13)
	}
}
