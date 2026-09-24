package egraph_test

import (
	"context"
	"math"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/x/cudalaunch"
	"github.com/tmc/lib/nvidia/x/cudatest"
	"github.com/tmc/lib/nvidia/x/egraph"
	"github.com/tmc/lib/nvidia/x/nvrtcx"
)

// halfBits converts a float32 to its float16 bit pattern, round-to-nearest-even.
// Only finite values within the half range are used by these tests.
func halfBits(f float32) uint16 {
	b := math.Float32bits(f)
	sign := uint16(b >> 16 & 0x8000)
	exp := int32(b>>23&0xff) - 127 + 15
	mant := b & 0x7fffff
	switch {
	case exp >= 0x1f:
		return sign | 0x7c00
	case exp <= 0:
		return sign
	}
	h := sign | uint16(exp)<<10 | uint16(mant>>13)
	// round to nearest even on the dropped bits
	if rest := mant & 0x1fff; rest > 0x1000 || (rest == 0x1000 && h&1 == 1) {
		h++
	}
	return h
}

// halfFloat converts a float16 bit pattern back to float32.
func halfFloat(h uint16) float32 {
	sign := uint32(h&0x8000) << 16
	exp := uint32(h >> 10 & 0x1f)
	mant := uint32(h & 0x3ff)
	switch {
	case exp == 0 && mant == 0:
		return math.Float32frombits(sign)
	case exp == 0x1f:
		return math.Float32frombits(sign | 0x7f800000 | mant<<13)
	}
	return math.Float32frombits(sign | (exp-15+127)<<23 | mant<<13)
}

// TestGeneratedHalfAddMatchesReference executes the generated half-precision
// elementwise add on the device and compares it against a float64 reference.
//
// Half operands are carried as uint16_t bit patterns, so the kernel must
// convert through half_to_float/float_to_half. A C cast instead of a
// conversion reads the bit pattern as an integer: 1.0 arrives as 15360, and
// this test turns red.
func TestGeneratedHalfAddMatchesReference(t *testing.T) {
	dev := cudatest.Require(t)
	defer dev.Close()

	src, err := egraph.GenerateCUDA(
		&egraph.Expr{Op: "add", Children: []*egraph.Expr{{Op: "a"}, {Op: "b"}}},
		egraph.CodegenOptions{KernelName: "HalfAdd", Precision: "half"},
	)
	if err != nil {
		t.Fatalf("GenerateCUDA: %v", err)
	}
	compiled, err := nvrtcx.Compile(src,
		nvrtcx.WithName("halfadd.cu"),
		nvrtcx.WithDeviceArch(dev),
		nvrtcx.WithNoCache(),
	)
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	mod, err := compiled.Load(dev)
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	defer mod.Unload()
	fn, err := mod.Function("HalfAdd")
	if err != nil {
		t.Fatal(err)
	}
	stream, err := dev.NewStream()
	if err != nil {
		t.Fatal(err)
	}
	defer stream.Destroy()

	// Exactly representable half values, so the expected sums are exact and the
	// comparison can be on bit patterns.
	const n = 64
	a := make([]uint16, n)
	b := make([]uint16, n)
	af := make([]float32, n)
	bf := make([]float32, n)
	for i := range a {
		af[i] = float32(i%9-4) + 0.25
		bf[i] = float32(i%5-2) * 0.5
		a[i] = halfBits(af[i])
		b[i] = halfBits(bf[i])
	}

	da := deviceSlice(t, dev, a)
	db := deviceSlice(t, dev, b)
	dout := deviceSlice(t, dev, make([]uint16, n))
	for _, free := range []func() error{da.Free, db.Free, dout.Free} {
		defer free()
	}

	if err := cudalaunch.Kernel(fn).Grid(1, 1).Block(n).On(stream).Launch(
		da, db, dout, int32(n),
	); err != nil {
		t.Fatalf("launch: %v", err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	if err := stream.Synchronize(ctx); err != nil {
		t.Fatalf("synchronize: %v", err)
	}
	got := make([]uint16, n)
	if err := dev.WithContext(func() error { return dout.CopyToHost(got) }); err != nil {
		t.Fatal(err)
	}

	for i := range got {
		want := halfBits(float32(float64(af[i]) + float64(bf[i])))
		if got[i] != want {
			t.Errorf("out[%d] = %#04x (%v), want %#04x (%v)",
				i, got[i], halfFloat(got[i]), want, halfFloat(want))
		}
	}
}
