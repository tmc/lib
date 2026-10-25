package hopper

import (
	"encoding/binary"
	"fmt"
	"math"
	"math/rand/v2"
	"slices"
	"strings"
	"testing"
)

func u32Bytes(v []uint32) []byte {
	b := make([]byte, 4*len(v))
	for i, x := range v {
		binary.LittleEndian.PutUint32(b[4*i:], x)
	}
	return b
}

// TestScaleE4M3ToBF16Exhaustive verifies that ScaleE4M3ToBF16 and Dequant9
// (plus MulBF16x2) match E2M1 * E4M3 * 2^-8 rounded to bfloat16 across all 4
// registers, all 256 pairs of E2M1 nibbles (all 16 codes in all 8 word slots),
// and all 256 E4M3 scale bytes (including ±0, subnormals, negatives, and
// 0x7f/0xff NaN), and that ScaleE4M3ToBF16Fast agrees on every positive normal
// scale byte.
func TestScaleE4M3ToBF16Exhaustive(t *testing.T) {
	// Check that dequant9Src is a bijection of the 32 word bits.
	var seen uint32
	for r := range 4 {
		for j := range 8 {
			seen |= 1 << dequant9Src[r][j]
		}
	}
	if seen != 0xffffffff {
		t.Fatalf("dequant9Src is not a 32-bit bijection: %#08x", seen)
	}

	// Check AllPositiveNormalE4M3 and NaN preservation on 0x7f and 0xff.
	for _, nanByte := range []uint8{0x7f, 0xff} {
		if IsPositiveNormalE4M3(nanByte) {
			t.Fatalf("IsPositiveNormalE4M3(%#02x) = true, want false", nanByte)
		}
		if got := ScaleE4M3ToBF16(nanByte); got != 0x7fc0 {
			t.Fatalf("ScaleE4M3ToBF16(%#02x) = %#04x, want 0x7fc0", nanByte, got)
		}
	}
	if AllPositiveNormalE4M3([]uint8{0x38, 0x7f}) {
		t.Fatal("AllPositiveNormalE4M3 accepted 0x7f")
	}
	if !AllPositiveNormalE4M3([]uint8{0x08, 0x38, 0x7e}) {
		t.Fatal("AllPositiveNormalE4M3 rejected positive normals")
	}

	// Exhaustively check all 4 registers x 256 code-pair bytes x 256 E4M3 scales.
	for r := range 4 {
		for cb := range 256 {
			codes := uint32(cb) << (8 * r)
			q := PackDequant9Word(codes)
			if got := UnpackDequant9Word(q); got != codes {
				t.Fatalf("UnpackDequant9Word(PackDequant9Word(%#08x)) = %#08x", codes, got)
			}
			a := Dequant9(q)
			w := DecodeWGMMAWord((uint32(cb)&0xf)<<(4*r)|((uint32(cb)>>4)<<(16+4*r)), r)
			if a[r] != w {
				t.Fatalf("reg %d cb %#02x: Dequant9 = %#08x, DecodeWGMMAWord = %#08x", r, cb, a[r], w)
			}
			for sb := range 256 {
				scaleByte := uint8(sb)
				sSlow := ScaleE4M3ToBF16x2(scaleByte, false)
				prodSlow := MulBF16x2(a[r], sSlow)
				for h := range 2 {
					code := uint8((cb >> (4 * h)) & 15)
					got := uint16(prodSlow >> (16 * h))
					// Reference: DequantNVFP4 at global = 2^-8 rounded to BF16.
					var ref [16]float32
					var pk [8]uint8
					pk[0] = code
					if err := DequantNVFP4(ref[:], pk[:], []uint8{scaleByte}, 1.0/256); err != nil {
						t.Fatal(err)
					}
					want := BF16Bits(ref[0])
					if want&0x7fff > 0x7f80 {
						if got&0x7fff <= 0x7f80 {
							t.Fatalf("slow reg %d code %#x scale %#02x: got %#04x, want NaN (%#04x)", r, code, sb, got, want)
						}
					} else if got != want {
						t.Fatalf("slow reg %d code %#x scale %#02x: got %#04x, want %#04x", r, code, sb, got, want)
					}
				}
				if IsPositiveNormalE4M3(scaleByte) {
					sFast := ScaleE4M3ToBF16x2(scaleByte, true)
					if sFast != sSlow {
						t.Fatalf("scale %#02x: fast %#08x != slow %#08x", sb, sFast, sSlow)
					}
					prodFast := MulBF16x2(a[r], sFast)
					if prodFast != prodSlow {
						t.Fatalf("reg %d cb %#02x scale %#02x: fast prod %#08x != slow %#08x", r, cb, sb, prodFast, prodSlow)
					}
				}
			}
		}
	}
}

// TestDequant9DeviceEmulated runs eg_dequant9 and eg_scale_e4m3_bf16x2 inside
// the C++ host emulator and checks bit-exact agreement with Go's Dequant9 and
// ScaleE4M3ToBF16x2 across all 256 E4M3 scale bytes and all 8 word nibble slots.
func TestDequant9DeviceEmulated(t *testing.T) {
	src := nvfp4WGMMAHelpers + `
__global__ void test_dq9(const uint32_t* words, uint32_t* out_a, uint32_t* out_s, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        uint32_t a0, a1, a2, a3;
        eg_dequant9(words[i], a0, a1, a2, a3);
        out_a[4 * i + 0] = a0;
        out_a[4 * i + 1] = a1;
        out_a[4 * i + 2] = a2;
        out_a[4 * i + 3] = a3;
    }
    if (i < 256) {
        out_s[i] = eg_scale_e4m3_bf16x2((uint32_t)i);
    }
}
`
	var words, wantCodes []uint32
	// All 16 codes in each of the 8 nibble slots, plus random words.
	for slot := range 8 {
		for c := range uint32(16) {
			codes := c << (4 * slot)
			wantCodes = append(wantCodes, codes)
			words = append(words, PackDequant9Word(codes))
		}
	}
	r := rand.New(rand.NewPCG(9, 10))
	for range 128 {
		codes := r.Uint32()
		wantCodes = append(wantCodes, codes)
		words = append(words, PackDequant9Word(codes))
	}
	n := len(words)
	res := runEmu(t, src, fmt.Sprintf("eg_emu_launch(dim3((%d+127)/128), dim3(128), 0, test_dq9, words, out_a, out_s, %d)", max(n, 256), n),
		emuBuf{"words", "uint32_t", u32Bytes(words)},
		emuBuf{"out_a", "uint32_t", poison(4 * 4 * n)},
		emuBuf{"out_s", "uint32_t", poison(4 * 256)},
	)
	gotA := i32s(res["out_a"])
	for i, w := range words {
		want := Dequant9(w)
		for j := range 4 {
			if uint32(gotA[4*i+j]) != want[j] {
				t.Fatalf("word %d (codes %#08x) reg %d: got %#08x, want %#08x", i, wantCodes[i], j, uint32(gotA[4*i+j]), want[j])
			}
		}
	}
	gotS := i32s(res["out_s"])
	for sb := range 256 {
		want := ScaleE4M3ToBF16x2(uint8(sb), false)
		got := uint32(gotS[sb])
		if want&0x7fff == 0x7fc0 {
			if got&0x7fff <= 0x7f80 || (got>>16)&0x7fff <= 0x7f80 {
				t.Fatalf("scale %#02x: got %#08x, want NaN", sb, got)
			}
		} else if got != want {
			t.Fatalf("scale %#02x: got %#08x, want %#08x", sb, got, want)
		}
	}
}

// TestRepackNVFP4Dequant9RoundTrip checks that RepackNVFP4Dequant9 and
// UnpackNVFP4Dequant9 round-trip losslessly (including 0x7f/0xff NaN scales and
// subnormals) and that decoding the repacked stream with Dequant9 +
// ScaleE4M3ToBF16x2 matches DequantNVFP4 bit-for-bit.
func TestRepackNVFP4Dequant9RoundTrip(t *testing.T) {
	r := rand.New(rand.NewPCG(77, 88))
	const n, k = 64, 512
	for _, glu := range []bool{false, true} {
		packed := make([]uint8, n*k/2)
		scales := make([]uint8, n*k/16)
		for i := range packed {
			packed[i] = uint8(r.IntN(256))
		}
		for i := range scales {
			scales[i] = uint8(i % 256) // covers all 256 E4M3 encodings incl. 0x00, subnormals, 0x7f, 0xff
		}
		stream, err := RepackNVFP4Dequant9(packed, scales, n, k, glu)
		if err != nil {
			t.Fatal(err)
		}
		gotPacked, gotScales, err := UnpackNVFP4Dequant9(stream, n, k, glu)
		if err != nil {
			t.Fatal(err)
		}
		if !slices.Equal(gotPacked, packed) {
			t.Fatalf("glu=%v: unpacked codes differ", glu)
		}
		if !slices.Equal(gotScales, scales) {
			t.Fatalf("glu=%v: unpacked scales differ", glu)
		}

		// Emulate the consumer warp decode on the stream and compare every element
		// against DequantNVFP4(packed, scales, 1.0/256) rounded to BF16.
		units := n / 32
		stages := k / 256
		tileRow := func(u, tile, row int) int {
			if glu {
				return tile*(n/2) + u*16 + row
			}
			return (2*u+tile)*16 + row
		}
		gotBF16 := make([]uint16, n*k)
		for u := range units {
			for hs := range stages {
				base := (u*stages + hs) * 4608
				for cc := range 4 {
					for tile := range 2 {
						for lane := range 32 {
							g, tq := lane>>2, lane&3
							sbase := base + (tile*32+lane)*8
							s0 := ScaleE4M3ToBF16x2(stream[sbase+2*cc], false)
							s1 := ScaleE4M3ToBF16x2(stream[sbase+2*cc+1], false)
							wbase := base + 512 + ((cc*2+tile)*32+lane)*16
							for s := range 4 {
								q := binary.LittleEndian.Uint32(stream[wbase+4*s:])
								a := Dequant9(q)
								a[0] = MulBF16x2(a[0], s0)
								a[1] = MulBF16x2(a[1], s1)
								a[2] = MulBF16x2(a[2], s0)
								a[3] = MulBF16x2(a[3], s1)
								for reg := range 4 {
									h, dp := reg&1, reg>>1
									row := tileRow(u, tile, g+8*h)
									col := hs*256 + cc*64 + tq*16 + s*4 + 2*dp
									gotBF16[row*k+col] = uint16(a[reg])
									gotBF16[row*k+col+1] = uint16(a[reg] >> 16)
								}
							}
						}
					}
				}
			}
		}
		rowVals := make([]float32, k)
		for row := range n {
			if err := DequantNVFP4(rowVals, packed[row*(k/2):(row+1)*(k/2)], scales[row*(k/16):(row+1)*(k/16)], 1.0/256); err != nil {
				t.Fatal(err)
			}
			for col, v := range rowVals {
				want := BF16Bits(v)
				got := gotBF16[row*k+col]
				if want&0x7fff > 0x7f80 {
					if got&0x7fff <= 0x7f80 {
						t.Fatalf("glu=%v [%d,%d]: got %#04x, want NaN", glu, row, col, got)
					}
				} else if got != want {
					t.Fatalf("glu=%v [%d,%d]: got %#04x, want %#04x", glu, row, col, got, want)
				}
			}
		}
	}
}

// TestRepackNVFP4WGMMARoundTrip checks that RepackNVFP4WGMMA and
// UnpackNVFP4WGMMA round-trip losslessly and that decoding each thread's
// fragment words matches DequantNVFP4 bit-for-bit across all 256 E4M3 scales.
func TestRepackNVFP4WGMMARoundTrip(t *testing.T) {
	r := rand.New(rand.NewPCG(101, 202))
	const n, k = 256, 128
	for _, glu := range []bool{false, true} {
		packed := make([]uint8, n*k/2)
		scales := make([]uint8, n*k/16)
		for i := range packed {
			packed[i] = uint8(r.IntN(256))
		}
		for i := range scales {
			scales[i] = uint8(i % 256)
		}
		wq, ws, err := RepackNVFP4WGMMA(packed, scales, n, k, glu)
		if err != nil {
			t.Fatal(err)
		}
		gotPacked, gotScales, err := UnpackNVFP4WGMMA(wq, ws, n, k, glu)
		if err != nil {
			t.Fatal(err)
		}
		if !slices.Equal(gotPacked, packed) {
			t.Fatalf("glu=%v: unpacked codes differ", glu)
		}
		if !slices.Equal(gotScales, scales) {
			t.Fatalf("glu=%v: unpacked scales differ", glu)
		}

		// Emulate thread fragment decode and compare against DequantNVFP4(..., 1.0/256).
		gotBF16 := make([]uint16, n*k)
		koff := [4]int{0, 0, 8, 8}
		for nt := range n / 128 {
			for kb := range k / 64 {
				for th := range 256 {
					cw, warp, lane := th/128, (th%128)/32, th%32
					g, tq := lane/4, lane%4
					rg := 64*cw + 16*warp + g
					r0 := wgmmaTileRow(n, nt, rg, glu)
					r1 := wgmmaTileRow(n, nt, rg+8, glu)
					trows := [4]int{r0, r1, r0, r1}
					wIdx := ((nt*(k/64)+kb)*256 + th) * 4
					sIdx := ((nt*(k/64)+kb)*256 + th) * 8
					for ks := range 4 {
						q := wq[wIdx+ks]
						s0 := ScaleE4M3ToBF16x2(ws[sIdx+2*ks], false)
						s1 := ScaleE4M3ToBF16x2(ws[sIdx+2*ks+1], false)
						sv := [4]uint32{s0, s1, s0, s1}
						for j := range 4 {
							prod := MulBF16x2(DecodeWGMMAWord(q, j), sv[j])
							row := trows[j]
							col := kb*64 + ks*16 + koff[j] + 2*tq
							gotBF16[row*k+col] = uint16(prod)
							gotBF16[row*k+col+1] = uint16(prod >> 16)
						}
					}
				}
			}
		}
		rowVals := make([]float32, k)
		for row := range n {
			if err := DequantNVFP4(rowVals, packed[row*(k/2):(row+1)*(k/2)], scales[row*(k/16):(row+1)*(k/16)], 1.0/256); err != nil {
				t.Fatal(err)
			}
			for col, v := range rowVals {
				want := BF16Bits(v)
				got := gotBF16[row*k+col]
				if want&0x7fff > 0x7f80 {
					if got&0x7fff <= 0x7f80 {
						t.Fatalf("glu=%v [%d,%d]: got %#04x, want NaN", glu, row, col, got)
					}
				} else if got != want {
					t.Fatalf("glu=%v [%d,%d]: got %#04x, want %#04x", glu, row, col, got, want)
				}
			}
		}
	}
}

// TestTileGEMMWGMMANVFP4Emulated tests TileGEMM with WGMMA: true and
// Weight: WeightNVFP4 across standard and repacked layouts, including Grouped,
// Gather, GLU, SumSq, Epilogue, and async WGMMA hazard checks.
func TestTileGEMMWGMMANVFP4Emulated(t *testing.T) {
	tests := []struct {
		name    string
		g       TileGEMM
		m       int
		tokens  int
		offsets []int32
	}{
		{
			name: "wgmma_nvfp4_128",
			g:    TileGEMM{N: 128, K: 128, Weight: WeightNVFP4, Out: F32, WGMMA: true, Stages: 2},
			m:    135,
		},
		{
			name: "wgmma_nvfp4_grouped_gather_glu_sumsq",
			g: TileGEMM{
				N: 128, K: 128, Weight: WeightNVFP4, Out: BF16, BN: 64, WGMMA: true,
				Grouped: true, Gather: true, GLU: true, SumSq: true,
				Epilogue: MustLambda("(add (mul (silu $0) $1) $2)"),
				Extras:   []Operand{{"bias", BF16, PerCol}},
			},
			m:       90,
			tokens:  30,
			offsets: []int32{0, 19, 19, 90},
		},
		{
			name: "wgmma_nvfp4_repacked_128",
			g: TileGEMM{
				N: 256, K: 128, Weight: WeightNVFP4, Out: F32, BN: 128, WGMMA: true,
				Repacked: true, SumSq: true,
			},
			m: 70,
		},
		{
			name: "wgmma_nvfp4_repacked_grouped_gather_glu_sumsq",
			g: TileGEMM{
				N: 256, K: 128, Weight: WeightNVFP4, Out: BF16, BN: 128, WGMMA: true,
				Repacked: true, Grouped: true, Gather: true, GLU: true, SumSq: true,
				Epilogue: MustLambda("(mul (silu $0) $1)"),
			},
			m:       140,
			tokens:  35,
			offsets: []int32{0, 65, 65, 140},
		},
	}
	r := rand.New(rand.NewPCG(41, 42))
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			g := tt.g
			g.Name = "tile_" + tt.name
			src, err := g.CUDA()
			if err != nil {
				t.Fatal(err)
			}
			// Verify async WGMMA fence/commit/wait are emitted.
			if !strings.Contains(src, "eg_wgmma_fence();") || !strings.Contains(src, "eg_wgmma_commit();") || !strings.Contains(src, "eg_wgmma_wait<0>();") {
				t.Fatalf("missing wgmma synchronization in emitted CUDA:\n%s", src)
			}
			n, k, m := g.N, g.K, tt.m
			xrows := m
			var rows []int32
			if g.Gather {
				xrows = tt.tokens
				for range m {
					rows = append(rows, int32(r.IntN(tt.tokens)))
				}
			}
			x := randBF16(r, xrows*k, 1)
			nout := n
			if g.GLU {
				nout = n / 2
			}
			experts := 1
			if g.Grouped {
				experts = len(tt.offsets) - 1
			}
			var ws []GEMMWeights
			var allPacked, allScales []uint8
			var allWQ []uint32
			var gs []float32
			for e := range experts {
				w := randNVFP4(r, n, k)
				// Include subnormal E4M3 scale in expert 0 to exercise both fast and slow scale decode paths.
				if e == 0 && len(w.Scales) > 0 {
					w.Scales[0] = 0x03 // subnormal E4M3
				}
				w.GlobalScal = float32(1.0 / 64.0) // power-of-two global scale so BF16 (e2m1 * sf) * gs has no FP32 rounding in Reference
				if g.Repacked {
					wq, wsf, err := RepackNVFP4WGMMA(w.Packed, w.Scales, n, k, g.GLU)
					if err != nil {
						t.Fatal(err)
					}
					allWQ = append(allWQ, wq...)
					allScales = append(allScales, wsf...)
				} else {
					allPacked = append(allPacked, w.Packed...)
					allScales = append(allScales, w.Scales...)
				}
				gs = append(gs, w.GlobalScal)
				ws = append(ws, w)
			}
			bufs := []emuBuf{{"x", "uint16_t", u16Bytes(x)}}
			args := []string{fmt.Sprint(m), "x"}
			if g.Gather {
				bufs = append(bufs, emuBuf{"rows", "int32_t", i32Bytes(rows)})
				args = append(args, "rows")
			}
			if g.Repacked {
				bufs = append(bufs, emuBuf{"w", "uint32_t", u32Bytes(allWQ)}, emuBuf{"w_sf", "uint8_t", allScales},
					emuBuf{"w_gs", "float", f32Bytes(gs)})
			} else {
				bufs = append(bufs, emuBuf{"w", "uint8_t", allPacked}, emuBuf{"w_sf", "uint8_t", allScales},
					emuBuf{"w_gs", "float", f32Bytes(gs)})
			}
			args = append(args, "w", "w_sf", "w_gs")
			maxRows := m
			if g.Grouped {
				bufs = append(bufs, emuBuf{"offsets", "int32_t", i32Bytes(tt.offsets)})
				args = append(args, "offsets")
				maxRows = 0
				for e := range experts {
					maxRows = max(maxRows, int(tt.offsets[e+1]-tt.offsets[e]))
				}
			}
			var extras [][]float32
			for _, e := range g.Extras {
				size := map[Broadcast]int{PerCol: nout, PerRow: m, Full: m * nout}[e.Broadcast]
				v := make([]float32, size)
				for i := range v {
					v[i] = e.DType.Round(float32(r.NormFloat64()))
				}
				extras = append(extras, v)
				bufs = append(bufs, emuBuf{e.Name, e.DType.cType(), typed{e.DType, f64s(v)}.bytes()})
				args = append(args, e.Name)
			}
			bufs = append(bufs, emuBuf{"out", g.Out.cType(), poison(m * nout * g.Out.Bits() / 8)})
			args = append(args, "out")
			if g.SumSq {
				bufs = append(bufs, emuBuf{"ss", "float", poison(4 * m * nout / 32)})
				args = append(args, "ss")
			}
			grid, block := g.Grid(maxRows, experts)
			launch := fmt.Sprintf("eg_emu_launch(dim3(%d, %d, %d), dim3(%d), %d, %s, %s)",
				grid[0], grid[1], grid[2], block[0], g.SharedBytes(), g.Name, strings.Join(args, ", "))
			res := runEmu(t, src, launch, bufs...)
			var offsets []int32
			if g.Grouped {
				offsets = tt.offsets
			}
			want, err := g.Reference(m, x, rows, offsets, ws, extras)
			if err != nil {
				t.Fatal(err)
			}
			got := make([]float32, len(want))
			for i, v := range decode(g.Out, res["out"]) {
				got[i] = float32(v)
			}
			rel := 1e-5
			if g.Out == BF16 {
				rel = 1.0 / 128
			}
			if i, ok := closeTo(got, want, rel, 1e-5); !ok {
				t.Fatalf("out[%d] = %g, want %g (relL2 %g)", i, got[i], want[i], relL2(got, want))
			}
			if g.SumSq {
				g32 := g
				g32.Out = F32
				y, err := g32.Reference(m, x, rows, offsets, ws, extras)
				if err != nil {
					t.Fatal(err)
				}
				wantSS := make([]float64, m*nout/32)
				for i, v := range y {
					wantSS[i/32] += float64(v) * float64(v)
				}
				gotSS := decode(F32, res["ss"])
				for i := range wantSS {
					if d := math.Abs(gotSS[i] - wantSS[i]); d > 1e-4*wantSS[i]+1e-6 {
						t.Fatalf("ss[%d] = %g, want %g", i, gotSS[i], wantSS[i])
					}
				}
			}
			// Verify that replacing eg_wgmma_wait<0>() with eg_wgmma_wait<1>() triggers an async WGMMA hazard (NaN),
			// because the final wgmma group is still pending when the epilogue reads the accumulator d[].
			mutSrc := strings.Replace(src, "eg_wgmma_wait<0>();", "eg_wgmma_wait<1>();", 1)
			mutSrc = strings.TrimSuffix(mutSrc, "        }\n    }\n}\n") + "        }\n    }\n    eg_wgmma_wait<0>();\n}\n"
			if mutSrc != src {
				mutRes := runEmu(t, mutSrc, launch, bufs...)
				hasNaN := false
				for _, v := range decode(g.Out, mutRes["out"]) {
					if math.IsNaN(v) {
						hasNaN = true
						break
					}
				}
				if !hasNaN {
					t.Fatalf("async WGMMA hazard mutant (wait<1> before epilogue) did not produce NaN")
				}
			}
		})
	}
}
