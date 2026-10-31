package hopper

import (
	"bytes"
	"fmt"
	"math"
	"math/rand/v2"
	"slices"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/egraph"
)

// TestUMMAGEMMEmulated runs UMMAGEMM through cudahostemu's descriptor-level
// tcgen05 emulation and checks it against a float64 reference. For NVFP4 the
// NVFP4Quant kernel quantizes x first and the reference uses its Go twin, so
// both sides multiply the same E2M1 values.
func TestUMMAGEMMEmulated(t *testing.T) {
	bias := Operand{Name: "bias", DType: F32, Broadcast: PerCol}
	res := Operand{Name: "res", DType: BF16, Broadcast: Full}
	tests := []struct {
		name string
		g    UMMAGEMM
		m    int
	}{
		{"decode bf16 bias", UMMAGEMM{N: 256, K: 192, Weight: WeightBF16, Out: F32, MaxM: 3, Epilogue: MustLambda("(add $0 $1)"), Extras: []Operand{bias}}, 3},
		{"decode bf16 wrap residual", UMMAGEMM{N: 128, K: 512, Weight: WeightBF16, Out: BF16, MaxM: 20, Stages: 2, Epilogue: MustLambda("(add $0 $1)"), Extras: []Operand{res}}, 17},
		{"prefill bf16", UMMAGEMM{N: 256, K: 128, Weight: WeightBF16, Out: F32, Stages: 2}, 130},
		{"prefill bf16 bn64 silu", UMMAGEMM{N: 128, K: 192, Weight: WeightBF16, Out: F32, BN: 64, Epilogue: MustLambda("(silu $0)")}, 40},
		{"decode nvfp4", UMMAGEMM{N: 128, K: 512, Weight: WeightNVFP4, Out: F32, MaxM: 2}, 2},
		{"prefill nvfp4 wrap", UMMAGEMM{N: 128, K: 768, Weight: WeightNVFP4, Out: F32, Stages: 2, Epilogue: MustLambda("(add $0 $1)"), Extras: []Operand{bias}}, 129},
	}
	for i, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			g := tt.g
			g.Name = fmt.Sprintf("umma_%d", i)
			m, n, k := tt.m, g.N, g.K
			r := rand.New(rand.NewPCG(uint64(i), 3))
			x := randBF16(r, m*k, 1)
			extras := make([][]float32, len(g.Extras))
			var bufs []emuBuf
			for j, e := range g.Extras {
				cnt := n
				if e.Broadcast == Full {
					cnt = m * n
				}
				v := make([]float32, cnt)
				for l := range v {
					v[l] = float32(r.NormFloat64())
					if e.DType == BF16 {
						v[l] = BF16.Round(v[l])
					}
				}
				extras[j] = v
				if e.DType == BF16 {
					u := make([]uint16, cnt)
					for l := range u {
						u[l] = BF16Bits(v[l])
					}
					bufs = append(bufs, emuBuf{e.Name, "uint16_t", u16Bytes(u)})
				} else {
					bufs = append(bufs, emuBuf{e.Name, "float", f32Bytes(v)})
				}
			}

			gemm, err := g.CUDA()
			if err != nil {
				t.Fatal(err)
			}
			gl, err := g.Launcher()
			if err != nil {
				t.Fatal(err)
			}
			src := gemm + gl
			xv := make([]float64, m*k) // x as the MMA sees it
			wv := make([]float64, n*k)
			var args []string
			if g.Weight == WeightNVFP4 {
				q := NVFP4Quant{Name: g.Name + "_quant", K: k}
				qs, err := q.CUDA()
				if err != nil {
					t.Fatal(err)
				}
				ql, _ := q.Launcher()
				src = qs + ql + src
				xq, xsf, err := q.Reference(m, x)
				if err != nil {
					t.Fatal(err)
				}
				row := make([]float32, k)
				for l := range m {
					if err := DequantNVFP4(row, xq[l*k/2:(l+1)*k/2], xsf[l*k/16:(l+1)*k/16], 1); err != nil {
						t.Fatal(err)
					}
					for c, f := range row {
						xv[l*k+c] = float64(f)
					}
				}
				w := randNVFP4(r, n, k)
				for l := range n {
					if err := DequantNVFP4(row, w.Packed[l*k/2:(l+1)*k/2], w.Scales[l*k/16:(l+1)*k/16], w.GlobalScal); err != nil {
						t.Fatal(err)
					}
					for c, f := range row {
						wv[l*k+c] = float64(f)
					}
				}
				bufs = append(bufs,
					emuBuf{"x", "uint16_t", u16Bytes(x)},
					emuBuf{"xq", "uint8_t", make([]byte, m*k/2)},
					emuBuf{"xsf", "uint8_t", make([]byte, m*k/16)},
					emuBuf{"w", "uint8_t", w.Packed},
					emuBuf{"w_sf", "uint8_t", w.Scales},
					emuBuf{"w_gs", "float", f32Bytes([]float32{w.GlobalScal})})
				args = []string{"xq", "xsf", "w", "w_sf", "w_gs"}
			} else {
				w := randBF16(r, n*k, 1)
				for l, b := range x {
					xv[l] = float64(FromBF16Bits(b))
				}
				for l, b := range w {
					wv[l] = float64(FromBF16Bits(b))
				}
				bufs = append(bufs, emuBuf{"x", "uint16_t", u16Bytes(x)}, emuBuf{"w", "uint16_t", u16Bytes(w)})
				args = []string{"x", "w"}
			}
			for _, e := range g.Extras {
				args = append(args, e.Name)
			}
			outBytes := 4 * m * n
			ctype := "float"
			if g.Out == BF16 {
				outBytes, ctype = 2*m*n, "uint16_t"
			}
			bufs = append(bufs, emuBuf{"out", ctype, make([]byte, outBytes)})
			launch := fmt.Sprintf("%s_launch(%d, %s, out, 0)", g.Name, m, strings.Join(args, ", "))
			if g.Weight == WeightNVFP4 {
				launch = fmt.Sprintf("%s_quant_launch(%d, x, xq, xsf, 0); %s", g.Name, m, launch)
			}
			got := runEmu(t, src, launch, bufs...)

			var outs []float32
			if g.Out == BF16 {
				for l := 0; l < m*n; l++ {
					outs = append(outs, FromBF16Bits(uint16(got["out"][2*l])|uint16(got["out"][2*l+1])<<8))
				}
			} else {
				outs = bytesF32(got["out"])
			}
			var worst float64
			for mi := range m {
				for ni := range n {
					var acc float64
					for c := range k {
						acc += xv[mi*k+c] * wv[ni*k+c]
					}
					ops := []float32{float32(acc)}
					for j, e := range g.Extras {
						if e.Broadcast == Full {
							ops = append(ops, extras[j][mi*n+ni])
						} else {
							ops = append(ops, extras[j][ni])
						}
					}
					epi := g.Epilogue
					if epi.root == nil {
						epi = Identity
					}
					want := float64(g.Out.Round(epi.Eval(ops...)))
					gotv := float64(outs[mi*n+ni])
					d := math.Abs(gotv-want) / (1 + math.Abs(want))
					if math.IsNaN(gotv) || d > worst {
						worst = d
						if math.IsNaN(gotv) {
							worst = math.Inf(1)
						}
					}
					tol := 1e-4
					if g.Out == BF16 {
						tol = 1e-2
					}
					if !(d <= tol) {
						t.Fatalf("out[%d, %d] = %v, want %v", mi, ni, gotv, want)
					}
				}
			}
			t.Logf("EMULATED: m=%d n=%d k=%d worst relative error %.2g", m, n, k, worst)
		})
	}
}

// TestUMMAGEMMEmulatedMutations checks that the emulation is not vacuous:
// each mutation of the kernel's synchronization or descriptors must make
// the emulated run fail or produce a wrong result.
func TestUMMAGEMMEmulatedMutations(t *testing.T) {
	g := UMMAGEMM{Name: "umma_mut", N: 128, K: 512, Weight: WeightBF16, Out: F32, MaxM: 4, Stages: 2}
	src, err := g.CUDA()
	if err != nil {
		t.Fatal(err)
	}
	l, _ := g.Launcher()
	const m = 4
	r := rand.New(rand.NewPCG(9, 9))
	x, w := randBF16(r, m*g.K, 1), randBF16(r, g.N*g.K, 1)
	want := make([]float64, m*g.N)
	for i := range m {
		for j := range g.N {
			for c := range g.K {
				want[i*g.N+j] += float64(FromBF16Bits(x[i*g.K+c])) * float64(FromBF16Bits(w[j*g.K+c]))
			}
		}
	}
	tests := []struct{ name, old, new string }{
		{"no accumulator wait", "    eg_umma_bar_wait(accum, 0);\n", ""},
		{"no stage-full wait", "            eg_umma_bar_wait(&full[s], (kt / S) & 1);\n", ""},
		{"no stage-empty wait", "            if (kt >= S) eg_umma_bar_wait(&empty[s], (kt / S - 1) & 1);\n", ""},
		{"unswizzled descriptor", "(1ull << 46) | (2ull << 61)", "(1ull << 46)"},
		{"K step not advanced", "eg_umma_desc_sw128(st + kk * 32)", "eg_umma_desc_sw128(st)"},
		{"accumulate from first step", "(kt | kk) != 0", "1"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			mut := strings.Replace(src, tt.old, tt.new, 1)
			if mut == src {
				t.Fatalf("mutation site %q not found", tt.old)
			}
			res, err := runEmuErr(t, mut+l, "umma_mut_launch(4, x, w, out, 0)",
				emuBuf{"x", "uint16_t", u16Bytes(x)},
				emuBuf{"w", "uint16_t", u16Bytes(w)},
				emuBuf{"out", "float", make([]byte, 4*m*g.N)})
			if err != nil {
				t.Logf("emulation failed: %.200s", err)
				return
			}
			got := bytesF32(res["out"])
			for i := range want {
				if d := math.Abs(float64(got[i]) - want[i]); !(d <= 1e-3*(1+math.Abs(want[i]))) {
					t.Logf("out[%d] = %v, want %v", i, got[i], want[i])
					return
				}
			}
			t.Errorf("mutation %q went undetected", tt.name)
		})
	}
}

// TestUMMAGEMMTensorMapsEmulated launches each kernel twice, once through
// its launcher and once directly with the maps TensorMaps describes, as a
// caller that encodes them once per buffer would, and checks that the two
// outputs agree bit for bit. A mutated map must not.
func TestUMMAGEMMTensorMapsEmulated(t *testing.T) {
	tests := []struct {
		name string
		g    UMMAGEMM
		m    int
	}{
		{"decode bf16", UMMAGEMM{N: 256, K: 192, Weight: WeightBF16, Out: F32, MaxM: 3}, 3},
		{"prefill bf16", UMMAGEMM{N: 128, K: 128, Weight: WeightBF16, Out: BF16, BN: 64, Stages: 2}, 130},
		{"decode nvfp4", UMMAGEMM{N: 128, K: 512, Weight: WeightNVFP4, Out: F32, MaxM: 2}, 2},
		{"prefill nvfp4", UMMAGEMM{N: 128, K: 256, Weight: WeightNVFP4, Out: F32, Stages: 2}, 129},
	}
	for i, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			g := tt.g
			g.Name = fmt.Sprintf("umma_tm_%d", i)
			g, err := g.norm()
			if err != nil {
				t.Fatal(err)
			}
			m, n, k := tt.m, g.N, g.K
			r := rand.New(rand.NewPCG(uint64(i), 11))
			src, err := g.Source()
			if err != nil {
				t.Fatal(err)
			}
			src = strings.TrimPrefix(src, Prelude)
			maps, err := g.TensorMaps(m)
			if err != nil {
				t.Fatal(err)
			}
			var bufs []emuBuf
			var operands []string
			if g.Weight == WeightNVFP4 {
				w := randNVFP4(r, n, k)
				bufs = []emuBuf{
					{"xq", "uint8_t", randBytes(r, m*k/2)},
					{"xsf", "uint8_t", e4m3Bytes(r, m*k/16)},
					{"w", "uint8_t", w.Packed},
					{"w_sf", "uint8_t", w.Scales},
					{"w_gs", "float", f32Bytes([]float32{w.GlobalScal})},
				}
				operands = []string{"xq", "xsf", "w", "w_sf", "w_gs"}
			} else {
				bufs = []emuBuf{{"x", "uint16_t", u16Bytes(randBF16(r, m*k, 1))}, {"w", "uint16_t", u16Bytes(randBF16(r, n*k, 1))}}
				operands = []string{"x", "w"}
			}
			outBytes := 4 * m * n
			if g.Out == BF16 {
				outBytes = 2 * m * n
			}
			ctype := g.Out.cType()
			bufs = append(bufs, emuBuf{"out", ctype, make([]byte, outBytes)}, emuBuf{"direct", ctype, make([]byte, outBytes)})
			direct := func(maps []TensorMap2D) string {
				var b strings.Builder
				args := []string{fmt.Sprint(m)}
				last := 0
				for j, tm := range maps {
					fmt.Fprintf(&b, "const eg_tmap t%d = eg_umma_tmap_2d(%s, %d, %d, %d, %d, %d, %d, %v);\n",
						j, operands[tm.Operand], tm.ElemBytes, tm.Inner, tm.Outer, tm.Stride, tm.BoxInner, tm.BoxOuter, tm.Swizzle128)
					args = append(args, fmt.Sprintf("t%d", j))
					last = max(last, tm.Operand)
				}
				args = append(append(args, operands[last+1:]...), "direct")
				grid, block := g.Grid(m)
				fmt.Fprintf(&b, "eg_emu_launch(dim3(%d, %d, %d), dim3(%d), %d, %s, %s)",
					grid[0], grid[1], grid[2], block[0], g.SharedBytes(), g.Name, strings.Join(args, ", "))
				return b.String()
			}
			launch := fmt.Sprintf("%s_launch(%d, %s, out, 0); %s", g.Name, m, strings.Join(operands, ", "), direct(maps))
			res := runEmu(t, src, launch, bufs...)
			if !bytes.Equal(res["out"], res["direct"]) {
				t.Fatalf("direct launch with TensorMaps differs from the launcher's")
			}
			// A map over the wrong rows must change the result.
			bad := slices.Clone(maps)
			bad[0].Stride *= 2
			bad[0].Outer /= 2
			res = runEmu(t, src, launch[:strings.Index(launch, ";")+1]+" "+direct(bad), bufs...)
			if bytes.Equal(res["out"], res["direct"]) {
				t.Fatalf("mutated map went undetected")
			}
			t.Logf("EMULATED: %s m=%d: %d maps from TensorMaps match the launcher's", g.Name, m, len(maps))
		})
	}
}

func randBytes(r *rand.Rand, n int) []byte {
	b := make([]byte, n)
	for i := range b {
		b[i] = byte(r.Uint32())
	}
	return b
}

// e4m3Bytes returns n finite positive E4M3 scales near 1.
func e4m3Bytes(r *rand.Rand, n int) []byte {
	b := make([]byte, n)
	for i := range b {
		b[i] = E4M3Bits(float32(0.5 + r.Float64()))
	}
	return b
}

// TestCompileForUMMA lowers a decoder-style projection, RMSNorm then a gemm
// with a residual epilogue, for several targets and row counts, checks which
// kernels the plan holds, and runs it emulated against Eval. The NVFP4 plans
// quantize x, which Eval does not, so they are held to the error of 4-bit
// activations rather than to BF16 rounding.
func TestCompileForUMMA(t *testing.T) {
	const n, k = 256, 512
	tests := []struct {
		target string
		m      int
		fp4    bool
		want   string // kernel kinds in order
	}{
		{"", 3, false, "row gemm"},
		{"sm_121a", 3, false, "row gemm"},
		{"sm_100f", 3, false, "row gemm"}, // not in cudadevicetable
		{"sm_100a", 3, false, "row umma"},
		{"sm_103a", 3, false, "row umma"},
		{"sm_103a", 130, false, "row umma"},
		{"sm_100a", 2, true, "row quant umma"},
		{"sm_103a", 129, true, "row quant umma"},
	}
	for _, tt := range tests {
		name := fmt.Sprintf("%s/m%d", tt.target, tt.m)
		if tt.fp4 {
			name += "/nvfp4"
		}
		t.Run(name, func(t *testing.T) {
			b := NewBuilder()
			x := b.Input("x", F32, tt.m, k)
			norm := b.Cast(BF16, b.RMSNorm(x, b.Input("gamma", F32, k), k, 1e-5))
			var w egraph.Id
			r := rand.New(rand.NewPCG(uint64(tt.m), 5))
			env := randEnv(r, Tensor{"x", F32, []int{tt.m, k}}, Tensor{"gamma", F32, []int{k}}, Tensor{"res", F32, []int{tt.m, n}})
			if tt.fp4 {
				w = b.NVFP4(b.Input("wq", U8, n, k/2), b.Input("wsf", U8, n, k/16), b.Input("wgs", F32, 1))
				nw := randNVFP4(r, n, k)
				env["wq"] = Value{Shape: []int{n, k / 2}, Data: u8s(nw.Packed)}
				env["wsf"] = Value{Shape: []int{n, k / 16}, Data: u8s(nw.Scales)}
				env["wgs"] = Value{Shape: []int{1}, Data: []float64{float64(nw.GlobalScal)}}
			} else {
				w = b.Input("w", BF16, n, k)
				env["w"] = randEnv(r, Tensor{"w", BF16, []int{n, k}})["w"]
			}
			root := b.Map(MustLambda("(add $0 $1)"), b.MatMul(norm, w), b.Input("res", F32, tt.m, n))
			expr, err := Optimize(b.G, root)
			if err != nil {
				t.Fatal(err)
			}
			plan, err := CompileFor(expr, "proj", F32, tt.target)
			if err != nil {
				t.Fatal(err)
			}
			var kinds []string
			for _, pk := range plan.Kernels {
				switch {
				case pk.UMMA != nil:
					kinds = append(kinds, "umma")
				case pk.Quant != nil:
					kinds = append(kinds, "quant")
				case pk.GEMM != nil, pk.Tile != nil:
					kinds = append(kinds, "gemm")
				case pk.Row != nil:
					kinds = append(kinds, "row")
				default:
					kinds = append(kinds, "other")
				}
			}
			if got := strings.Join(kinds, " "); got != tt.want {
				t.Fatalf("plan kernels %q, want %q", got, tt.want)
			}
			if _, err := plan.Source(); err != nil {
				t.Fatal(err)
			}
			if !UMMATarget(tt.target) {
				return
			}
			want, err := Eval(expr, env)
			if err != nil {
				t.Fatal(err)
			}
			got := runPlan(t, plan, env)
			// The RMS error relative to the RMS of the result: FP32
			// accumulation order for BF16, 4-bit activations for NVFP4.
			var se, sw float64
			for i := range want.Data {
				d := got[i] - want.Data[i]
				se += d * d
				sw += want.Data[i] * want.Data[i]
			}
			rel, tol := math.Sqrt(se/sw), 1e-5
			if tt.fp4 {
				tol = 0.15
			}
			t.Logf("EMULATED: relative RMS error %.3g against Eval", rel)
			if math.IsNaN(rel) || rel > tol {
				t.Fatalf("relative RMS error %.3g, want at most %g", rel, tol)
			}
		})
	}
}

func u8s(b []uint8) []float64 {
	v := make([]float64, len(b))
	for i, x := range b {
		v[i] = float64(x)
	}
	return v
}
