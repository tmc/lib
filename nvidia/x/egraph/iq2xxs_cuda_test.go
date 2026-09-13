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

const (
	iq2xxsSuperBlock = 256
	iq2xxsBlockBytes = 66
)

func iq2xxsTestBlocks(nblocks int) []byte {
	// Exactly representable half values for d
	scaleBits := []uint16{0x3800, 0x3400, 0x3c00, 0x3000} // 0.5, 0.25, 1, 0.125
	blocks := make([]byte, nblocks*iq2xxsBlockBytes)

	for b := 0; b < nblocks; b++ {
		blk := blocks[b*iq2xxsBlockBytes : (b+1)*iq2xxsBlockBytes]
		d := scaleBits[b%len(scaleBits)]
		blk[0] = byte(d)
		blk[1] = byte(d >> 8)

		qs := blk[2:]
		for ib32 := 0; ib32 < 8; ib32++ {
			// aux0: 4 grid indices (0..255)
			for l := 0; l < 4; l++ {
				qs[8*ib32+l] = byte((b*31 + ib32*7 + l*13 + 3) & 0xff)
			}
			// aux1: 4 7-bit signs (0..127) and 4-bit scale (0..15)
			s0 := uint32((b*17 + ib32*3 + 1) & 0x7f)
			s1 := uint32((b*19 + ib32*5 + 2) & 0x7f)
			s2 := uint32((b*23 + ib32*7 + 3) & 0x7f)
			s3 := uint32((b*29 + ib32*11 + 4) & 0x7f)
			sc := uint32((b*5 + ib32) & 0x0f)
			aux1 := s0 | (s1 << 7) | (s2 << 14) | (s3 << 21) | (sc << 28)
			qs[8*ib32+4] = byte(aux1)
			qs[8*ib32+5] = byte(aux1 >> 8)
			qs[8*ib32+6] = byte(aux1 >> 16)
			qs[8*ib32+7] = byte(aux1 >> 24)
		}
	}
	return blocks
}

func iq2xxsMatvecReference(blocks []byte, x []float32, rows, cols int) []float32 {
	blocksPerRow := cols / iq2xxsSuperBlock
	y := make([]float32, rows)
	wBuf := make([]float32, cols)

	for row := 0; row < rows; row++ {
		rowBlocks := blocks[row*blocksPerRow*iq2xxsBlockBytes : (row+1)*blocksPerRow*iq2xxsBlockBytes]
		if err := egraph.DequantizeIQ2XXS(wBuf, rowBlocks); err != nil {
			panic(err)
		}
		var acc float64
		for i := 0; i < cols; i++ {
			acc += float64(wBuf[i]) * float64(x[i])
		}
		y[row] = float32(acc)
	}
	return y
}

func TestIQ2XXSDequantizeTable(t *testing.T) {
	tests := []struct {
		name    string
		blockFn func() []byte
		check   func(t *testing.T, out []float32)
	}{
		{
			name: "AllZeroQuants",
			blockFn: func() []byte {
				blk := make([]byte, iq2xxsBlockBytes)
				// d = 1.0 (0x3c00 in float16)
				blk[0] = 0x00
				blk[1] = 0x3c
				// qs is all zeros:
				// aux0 = 0 -> gridIdx = 0 for all l
				// kgrid2bit256[0] = 0 -> all 8 l_val = 0 -> all grid_val = 1
				// aux1 = 0 -> signIdx = 0 -> signs = ksignsIQ2XS[0] = 0 -> all positive
				// aux1 >> 28 = 0 -> db = 1.0 * (0.5 + 0) * 0.25 = 0.125
				// Expected each weight = 0.125 * 1 * 1 = 0.125
				return blk
			},
			check: func(t *testing.T, out []float32) {
				const want = float32(0.125)
				for i, v := range out {
					if math.Abs(float64(v-want)) > 1e-6 {
						t.Fatalf("out[%d] = %v, want %v", i, v, want)
					}
				}
			},
		},
		{
			name: "SubBlockScaleAndSigns",
			blockFn: func() []byte {
				blk := make([]byte, iq2xxsBlockBytes)
				// d = 1.0 (0x3c00)
				blk[0] = 0x00
				blk[1] = 0x3c
				// In sub-block 0, set scale = 15 (0xf << 28)
				// and sign for l=0 to 1 (sign bit 0 set and parity bit 7 set in ksignsIQ2XS[1] = 129)
				// db = 1.0 * (0.5 + 15) * 0.25 = 15.5 * 0.25 = 3.875
				aux1 := uint32(1) | (uint32(15) << 28)
				blk[2+4] = byte(aux1)
				blk[2+5] = byte(aux1 >> 8)
				blk[2+6] = byte(aux1 >> 16)
				blk[2+7] = byte(aux1 >> 24)
				return blk
			},
			check: func(t *testing.T, out []float32) {
				const db = float32(3.875)
				// Element 0: sign bit 0 is set -> -3.875
				if math.Abs(float64(out[0]-(-db))) > 1e-6 {
					t.Errorf("out[0] = %v, want %v", out[0], -db)
				}
				// Elements 1..6: signs bits 1..6 not set -> +3.875
				for j := 1; j <= 6; j++ {
					if math.Abs(float64(out[j]-db)) > 1e-6 {
						t.Errorf("out[%d] = %v, want %v", j, out[j], db)
					}
				}
				// Element 7: sign bit 7 (parity bit) is set -> -3.875
				if math.Abs(float64(out[7]-(-db))) > 1e-6 {
					t.Errorf("out[7] = %v, want %v", out[7], -db)
				}
			},
		},
		{
			name: "GridCodebookValues",
			blockFn: func() []byte {
				blk := make([]byte, iq2xxsBlockBytes)
				// d = 2.0 (0x4000 in float16)
				blk[0] = 0x00
				blk[1] = 0x40
				// Grid index 1: kgrid2bit256[1] = 2.
				// In binary: 2 = 0b0000000000000010.
				// j=0: (2 >> 0) & 3 = 2 -> 2*2 + 1 = 5
				// j=1..7: (2 >> (2*j)) & 3 = 0 -> 2*0 + 1 = 1
				// scale = 0 -> db = 2.0 * 0.5 * 0.25 = 0.25
				// signs = 0
				// out[0] = 0.25 * 5 = 1.25
				// out[1..7] = 0.25 * 1 = 0.25
				blk[2] = 1 // aux0[0] = 1
				return blk
			},
			check: func(t *testing.T, out []float32) {
				if math.Abs(float64(out[0]-1.25)) > 1e-6 {
					t.Errorf("out[0] = %v, want 1.25", out[0])
				}
				for j := 1; j < 8; j++ {
					if math.Abs(float64(out[j]-0.25)) > 1e-6 {
						t.Errorf("out[%d] = %v, want 0.25", j, out[j])
					}
				}
			},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			blk := tc.blockFn()
			out := make([]float32, iq2xxsSuperBlock)
			if err := egraph.DequantizeIQ2XXS(out, blk); err != nil {
				t.Fatalf("DequantizeIQ2XXS failed: %v", err)
			}
			tc.check(t, out)
		})
	}
}

func TestIQ2XXSZeroAlloc(t *testing.T) {
	blk := make([]byte, iq2xxsBlockBytes)
	out := make([]float32, iq2xxsSuperBlock)
	allocs := testing.AllocsPerRun(100, func() {
		_ = egraph.DequantizeIQ2XXS(out, blk)
	})
	if allocs != 0 {
		t.Errorf("DequantizeIQ2XXS allocated %v times per run, want 0", allocs)
	}
}

func TestIQ2XXSInvalidBuffers(t *testing.T) {
	out := make([]float32, iq2xxsSuperBlock)
	if err := egraph.DequantizeIQ2XXS(out, make([]byte, 50)); err == nil {
		t.Error("expected error for short block buffer, got nil")
	}
	if err := egraph.DequantizeIQ2XXS(make([]float32, 100), make([]byte, iq2xxsBlockBytes)); err == nil {
		t.Error("expected error for short dst buffer, got nil")
	}
}

func TestGeneratedIQ2XXSMatvecMatchesReference(t *testing.T) {
	for _, threads := range []uint32{32, 64} {
		t.Run(fmt.Sprintf("block%d", threads), func(t *testing.T) {
			testIQ2XXSMatvec(t, threads)
		})
	}
}

func testIQ2XXSMatvec(t *testing.T, threads uint32) {
	dev := cudatest.Require(t)
	defer dev.Close()
	src, err := egraph.GenerateCUDA(
		&egraph.Expr{Op: "iq2_xxs_matvec", Children: []*egraph.Expr{{Op: "w"}, {Op: "x"}}},
		egraph.CodegenOptions{KernelName: "IQ2XXSMatvec", Precision: "iq2_xxs"},
	)
	if err != nil {
		t.Fatalf("GenerateCUDA: %v", err)
	}
	compiled, err := nvrtcx.Compile(src,
		nvrtcx.WithName("iq2xxs.cu"),
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
	fn, err := mod.GetFunction("IQ2XXSMatvec")
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
		cols = 2 * iq2xxsSuperBlock
	)
	blocks := iq2xxsTestBlocks(rows * cols / iq2xxsSuperBlock)
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

	want := iq2xxsMatvecReference(blocks, x, rows, cols)
	for i := range want {
		tol := 1e-4 * math.Max(1, math.Abs(float64(want[i])))
		if diff := math.Abs(float64(got[i] - want[i])); diff > tol {
			t.Errorf("y[%d] = %v, want %v (diff %v)", i, got[i], want[i], diff)
		}
	}
}

func TestGeneratedIQ2XXSMatvecRaggedColsReturnsNaN(t *testing.T) {
	dev := cudatest.Require(t)
	defer dev.Close()
	src, err := egraph.GenerateCUDA(
		&egraph.Expr{Op: "iq2_xxs_matvec", Children: []*egraph.Expr{{Op: "w"}, {Op: "x"}}},
		egraph.CodegenOptions{KernelName: "IQ2XXSMatvec", Precision: "iq2_xxs"},
	)
	if err != nil {
		t.Fatalf("GenerateCUDA: %v", err)
	}
	compiled, err := nvrtcx.Compile(src,
		nvrtcx.WithName("iq2xxs.cu"),
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
	fn, err := mod.GetFunction("IQ2XXSMatvec")
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
	blocks := iq2xxsTestBlocks(2)
	x := make([]float32, cols)
	yInit := []float32{123.456, 789.012}

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

func BenchmarkDequantizeIQ2XXS(b *testing.B) {
	const nblocks = 128
	blocks := iq2xxsTestBlocks(nblocks)
	dst := make([]float32, nblocks*iq2xxsSuperBlock)

	b.SetBytes(int64(len(blocks)))
	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		if err := egraph.DequantizeIQ2XXS(dst, blocks); err != nil {
			b.Fatal(err)
		}
	}
}
