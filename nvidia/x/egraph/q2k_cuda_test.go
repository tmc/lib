package egraph_test

import (
	"context"
	"fmt"
	"math"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/x/cudalaunch"
	"github.com/tmc/lib/nvidia/x/cudatest"
	"github.com/tmc/lib/nvidia/x/egraph"
	"github.com/tmc/lib/nvidia/x/nvrtcx"
)

// Q2_K super-block geometry, matching the block_q2_k structure GenerateCUDA
// emits: 16 packed scale/min nibble pairs, 256 two-bit quants, a half scale for
// the quants and a half scale for the minimums.
const (
	q2kSuperBlock = 256
	q2kBlockBytes = 84
)

// halfToFloat32 decodes an IEEE 754 binary16 value. The generated kernel
// decodes the same bits with PTX cvt.f32.f16, which is exact, so the reference
// and the device agree on the block scales by construction.
func halfToFloat32(h uint16) float32 {
	sign := uint32(h>>15) << 31
	exp := int32(h >> 10 & 0x1f)
	frac := uint32(h & 0x3ff)
	switch {
	case exp == 0 && frac == 0:
		return math.Float32frombits(sign)
	case exp == 0:
		// Subnormal: normalise into a float32 normal.
		e := int32(-1)
		for frac&0x400 == 0 {
			frac <<= 1
			e--
		}
		frac &= 0x3ff
		return math.Float32frombits(sign | uint32(127+e-14)<<23 | frac<<13)
	case exp == 0x1f:
		return math.Float32frombits(sign | 0xff<<23 | frac<<13)
	default:
		return math.Float32frombits(sign | uint32(127-15+exp)<<23 | frac<<13)
	}
}

// dequantizeQ2K unpacks one Q2_K super-block into 256 float32 weights.
//
// The element-to-quant mapping is group-interleaved, not sequential: element i
// belongs to group i/16, whose two-bit field lives in byte
// 32*(group/8) + 16*(group&1) + i%16 at shift ((group/2)&3)*2. Each scale byte
// packs two nibbles, the low one scaling the quant and the high one scaling the
// minimum that is subtracted. This mirrors q2_k_value_f32 in the ds4 oracle
// (ds4.c:3608) and quant.DequantizeQ2_K (lm/quant/kquant.go:49); a kernel that
// reads the quants sequentially, or that multiplies by the whole scale byte,
// produces finite plausible numbers that are wrong for every element.
func dequantizeQ2K(block []byte) []float32 {
	scales := block[:16]
	quants := block[16:80]
	d := float64(halfToFloat32(uint16(block[80]) | uint16(block[81])<<8))
	dmin := float64(halfToFloat32(uint16(block[82]) | uint16(block[83])<<8))

	out := make([]float32, q2kSuperBlock)
	for i := range out {
		group := i / 16
		l := i - group*16
		base := 32*(group/8) + 16*(group&1)
		shift := (group / 2 & 3) * 2
		q := quants[base+l] >> shift & 0x03
		sc := scales[group]
		out[i] = float32(d*float64(sc&0x0f)*float64(q) - dmin*float64(sc>>4))
	}
	return out
}

// q2kMatvecReference computes y = W*x in float64 from the packed blocks, which
// is what a "q2_k_matvec" kernel promises its callers.
func q2kMatvecReference(blocks []byte, x []float32, rows, cols int) []float32 {
	blocksPerRow := cols / q2kSuperBlock
	y := make([]float32, rows)
	for row := 0; row < rows; row++ {
		var acc float64
		for b := 0; b < blocksPerRow; b++ {
			off := (row*blocksPerRow + b) * q2kBlockBytes
			w := dequantizeQ2K(blocks[off : off+q2kBlockBytes])
			for i, wv := range w {
				acc += float64(wv) * float64(x[b*q2kSuperBlock+i])
			}
		}
		y[row] = float32(acc)
	}
	return y
}

// q2kTestBlocks builds deterministic Q2_K blocks. Both nibbles of every scale
// byte are non-zero so that dropping either one changes every row, and the
// quant bytes differ across the interleaved byte groups so that a sequential
// element mapping reads different values rather than a permutation of the same
// ones.
func q2kTestBlocks(nblocks int) []byte {
	// Exactly representable half values, used as the (d, dmin) pair per block.
	scaleBits := []uint16{0x3800, 0x3400, 0x3c00, 0x3000} // 0.5, 0.25, 1, 0.125
	minBits := []uint16{0x3000, 0x2c00, 0x3400, 0x2800}   // 0.125, 0.0625, 0.25, 0.03125

	blocks := make([]byte, nblocks*q2kBlockBytes)
	for b := 0; b < nblocks; b++ {
		blk := blocks[b*q2kBlockBytes : (b+1)*q2kBlockBytes]
		for g := 0; g < 16; g++ {
			lo := byte(1 + (g*3+b)%15)
			hi := byte(1 + (g*5+b)%15)
			blk[g] = hi<<4 | lo
		}
		for i := 0; i < 64; i++ {
			blk[16+i] = byte((i*37 + b*11) & 0xff)
		}
		d := scaleBits[b%len(scaleBits)]
		m := minBits[b%len(minBits)]
		blk[80], blk[81] = byte(d), byte(d>>8)
		blk[82], blk[83] = byte(m), byte(m>>8)
	}
	return blocks
}

// TestGeneratedQ2KMatvecMatchesReference runs the generated q2_k_matvec kernel
// on the device and compares every row against q2kMatvecReference.
//
// The compile gate cannot distinguish a correct unpacking from a wrong one, and
// a gate that recomputes the reference formula twice in Go cannot either: both
// stay green when the kernel changes. Only running the generated kernel does.
func TestGeneratedQ2KMatvecMatchesReference(t *testing.T) {
	for _, threads := range []uint32{32, 64} {
		t.Run(fmt.Sprintf("block%d", threads), func(t *testing.T) {
			testQ2KMatvec(t, threads)
		})
	}
}

func testQ2KMatvec(t *testing.T, threads uint32) {
	dev := cudatest.Require(t)
	defer dev.Close()
	src, err := egraph.GenerateCUDA(
		&egraph.Expr{Op: "q2_k_matvec", Children: []*egraph.Expr{{Op: "w"}, {Op: "x"}}},
		egraph.CodegenOptions{KernelName: "Q2KMatvec", Precision: "q2_k"},
	)
	if err != nil {
		t.Fatalf("GenerateCUDA: %v", err)
	}
	compiled, err := nvrtcx.Compile(src,
		nvrtcx.WithName("q2k.cu"),
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
	fn, err := mod.GetFunction("Q2KMatvec")
	if err != nil {
		t.Fatal(err)
	}
	stream, err := dev.NewStream()
	if err != nil {
		t.Fatal(err)
	}
	defer stream.Destroy()

	const (
		rows = 4
		cols = 2 * q2kSuperBlock
	)
	blocks := q2kTestBlocks(rows * cols / q2kSuperBlock)
	x := make([]float32, cols)
	for i := range x {
		x[i] = float32((i*13)%17-8) / 4
	}

	dw := deviceSlice(t, dev, blocks)
	dx := deviceSlice(t, dev, x)
	dy := deviceSlice(t, dev, make([]float32, rows))
	for _, free := range []func() error{dw.Free, dx.Free, dy.Free} {
		defer free()
	}

	// One warp per row: the kernel derives its row from the global warp index.
	grid := (rows*32 + threads - 1) / threads
	if err := cudalaunch.Kernel(fn).Grid(grid, 1).Block(threads).On(stream).Launch(
		dw, dx, dy, int32(rows), int32(cols),
	); err != nil {
		t.Fatalf("launch: %v", err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	if err := stream.Synchronize(ctx); err != nil {
		t.Fatalf("synchronize: %v", err)
	}
	got := make([]float32, rows)
	if err := dev.WithContext(func() error { return dy.CopyToHost(got) }); err != nil {
		t.Fatal(err)
	}

	want := q2kMatvecReference(blocks, x, rows, cols)
	for i := range want {
		tol := 1e-4 * math.Max(1, math.Abs(float64(want[i])))
		if diff := math.Abs(float64(got[i] - want[i])); diff > tol {
			t.Errorf("y[%d] = %v, want %v (diff %v)", i, got[i], want[i], diff)
		}
	}
}

func TestGeneratedQ2KMatvecRaggedColsReturnsNaN(t *testing.T) {
	dev := cudatest.Require(t)
	defer dev.Close()
	src, err := egraph.GenerateCUDA(
		&egraph.Expr{Op: "q2_k_matvec", Children: []*egraph.Expr{{Op: "w"}, {Op: "x"}}},
		egraph.CodegenOptions{KernelName: "Q2KMatvec", Precision: "q2_k"},
	)
	if err != nil {
		t.Fatalf("GenerateCUDA: %v", err)
	}
	compiled, err := nvrtcx.Compile(src,
		nvrtcx.WithName("q2k.cu"),
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
	fn, err := mod.GetFunction("Q2KMatvec")
	if err != nil {
		t.Fatal(err)
	}
	stream, err := dev.NewStream()
	if err != nil {
		t.Fatal(err)
	}
	defer stream.Destroy()

	const (
		rows = 2
		cols = 250 // ragged column count (not a multiple of 256)
	)
	blocks := q2kTestBlocks(2)
	x := make([]float32, cols)
	yInit := []float32{123.456, 789.012} // pre-populate with known values

	dw := deviceSlice(t, dev, blocks)
	dx := deviceSlice(t, dev, x)
	dy := deviceSlice(t, dev, yInit)
	for _, free := range []func() error{dw.Free, dx.Free, dy.Free} {
		defer free()
	}

	if err := cudalaunch.Kernel(fn).Grid(1, 1).Block(64).On(stream).Launch(
		dw, dx, dy, int32(rows), int32(cols),
	); err != nil {
		t.Fatalf("launch: %v", err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	if err := stream.Synchronize(ctx); err != nil {
		t.Fatalf("synchronize: %v", err)
	}
	got := make([]float32, rows)
	if err := dev.WithContext(func() error { return dy.CopyToHost(got) }); err != nil {
		t.Fatal(err)
	}

	for i, v := range got {
		if !math.IsNaN(float64(v)) {
			t.Errorf("got y[%d] = %v, want NaN for ragged cols=%d", i, v, cols)
		}
	}
}

