package tcgen05

import (
	"fmt"
	"math"
	"math/rand/v2"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudatmem"
)

func randMat(r *rand.Rand, rows, cols int, scale float64) [][]float32 {
	m := make([][]float32, rows)
	for i := range m {
		m[i] = make([]float32, cols)
		for j := range m[i] {
			m[i][j] = float32(r.NormFloat64() * scale)
		}
	}
	return m
}

func flat(m [][]float32) []float32 {
	var out []float32
	for _, r := range m {
		out = append(out, r...)
	}
	return out
}

func randSSMInput(r *rand.Rand, s SSMChunkSchedule) SSMInput {
	t := s.SeqLen()
	in := SSMInput{
		Q:     randMat(r, t, s.HeadDimK, 1/math.Sqrt(float64(s.HeadDimK))),
		K:     randMat(r, t, s.HeadDimK, 1/math.Sqrt(float64(s.HeadDimK))),
		V:     randMat(r, t, s.HeadDimV, 1),
		Alpha: make([]float32, t),
		Beta:  make([]float32, t),
	}
	for i := range in.Alpha {
		in.Alpha[i] = float32(0.85 + 0.15*r.Float64())
		in.Beta[i] = float32(0.1 + 0.4*r.Float64())
	}
	if s.ConvKernel > 0 {
		in.ConvQ = randMat(r, s.ConvKernel, s.HeadDimK, 0.5)
		in.ConvK = randMat(r, s.ConvKernel, s.HeadDimK, 0.5)
		in.ConvV = randMat(r, s.ConvKernel, s.HeadDimV, 0.5)
	}
	return in
}

func TestSSMValidate(t *testing.T) {
	ok := SSMChunkSchedule{HeadDimK: 64, HeadDimV: 64, ChunkLen: 64, NumChunks: 4, ConvKernel: 4}
	if err := ok.Validate(); err != nil {
		t.Fatalf("Validate(%+v) = %v", ok, err)
	}
	for _, tt := range []struct {
		name string
		edit func(*SSMChunkSchedule)
	}{
		{"head dim", func(s *SSMChunkSchedule) { s.HeadDimK = 48 }},
		{"chunk len", func(s *SSMChunkSchedule) { s.ChunkLen = 96 }},
		{"no chunks", func(s *SSMChunkSchedule) { s.NumChunks = 0 }},
		{"conv width", func(s *SSMChunkSchedule) { s.ConvKernel = 5 }},
		{"conv staging", func(s *SSMChunkSchedule) { s.HeadDimK, s.ChunkLen = 128, 32 }},
		{"shared memory", func(s *SSMChunkSchedule) { s.HeadDimK, s.HeadDimV, s.ChunkLen = 128, 128, 128 }},
		{"target", func(s *SSMChunkSchedule) { s.Target = "sm_121a" }},
		{"variant", func(s *SSMChunkSchedule) { s.Variant = 7 }},
		{"name", func(s *SSMChunkSchedule) { s.Name = "eg_bad" }},
	} {
		s := ok
		tt.edit(&s)
		if err := s.Validate(); err == nil {
			t.Errorf("%s: Validate(%+v) = nil, want error", tt.name, s)
		}
	}
}

// TestSSMChunkedMatchesRecurrence checks that the chunkwise algorithm the
// kernel implements equals the token-by-token recurrence.
func TestSSMChunkedMatchesRecurrence(t *testing.T) {
	r := rand.New(rand.NewPCG(2, 2))
	for _, v := range []SSMVariant{SSMMamba2SSD, SSMGatedDeltaNet} {
		for _, cw := range []int{0, 4} {
			s := SSMChunkSchedule{Variant: v, HeadDimK: 32, HeadDimV: 64, ChunkLen: 32, NumChunks: 3, ConvKernel: cw}
			in := randSSMInput(r, s)
			want, wantS, err := s.Reference(in)
			if err != nil {
				t.Fatal(err)
			}
			got, gotS, err := s.ReferenceChunked(in)
			if err != nil {
				t.Fatal(err)
			}
			if e := relL2(flat(got), flat(want)); e > 1e-6 {
				t.Errorf("%v conv %d: chunked output rel L2 %g", v, cw, e)
			}
			if e := relL2(flat(gotS), flat(wantS)); e > 1e-6 {
				t.Errorf("%v conv %d: chunked state rel L2 %g", v, cw, e)
			}
		}
	}
}

func TestSSMTraceVerifies(t *testing.T) {
	for _, v := range []SSMVariant{SSMMamba2SSD, SSMGatedDeltaNet} {
		s := SSMChunkSchedule{Variant: v, HeadDimK: 128, HeadDimV: 128, ChunkLen: 64, NumChunks: 4, ConvKernel: 4}
		for _, target := range []cudatmem.Target{cudatmem.SM100A, cudatmem.SM107A} {
			sum, err := cudatmem.VerifyTrace(target, s.TraceEvents())
			if err != nil {
				t.Fatalf("%v %s: VerifyTrace: %v", v, target, err)
			}
			if sum.MaxAllocatedColumns != 512 || sum.Shifts != 3*3*4 {
				t.Errorf("%v %s: summary %+v", v, target, sum)
			}
		}
	}
}

func runSSMKernel(t *testing.T, s SSMChunkSchedule, src string, heads int, ins []SSMInput) (out, state []float32) {
	t.Helper()
	var q, k, v, a, b, cq, ck, cv []float32
	for _, in := range ins {
		q = append(q, flat(in.Q)...)
		k = append(k, flat(in.K)...)
		v = append(v, flat(in.V)...)
		a = append(a, in.Alpha...)
		b = append(b, in.Beta...)
		cq = append(cq, flat(in.ConvQ)...)
		ck = append(ck, flat(in.ConvK)...)
		cv = append(cv, flat(in.ConvV)...)
	}
	grid, block := s.Grid(heads, 0)
	launch := fmt.Sprintf("eg_emu_launch(dim3(%d, 1, 1), dim3(%d, 1, 1), %d, %s, %d, q, k, v, alpha, beta, conv_q, conv_k, conv_v, out, state)",
		grid[0], block[0], s.SharedBytes(), s.Name, heads)
	res := runEmu(t, src, launch,
		emuBuf{"q", "float", f32Bytes(q)},
		emuBuf{"k", "float", f32Bytes(k)},
		emuBuf{"v", "float", f32Bytes(v)},
		emuBuf{"alpha", "float", f32Bytes(a)},
		emuBuf{"beta", "float", f32Bytes(b)},
		emuBuf{"conv_q", "float", f32Bytes(append(cq, 0))},
		emuBuf{"conv_k", "float", f32Bytes(append(ck, 0))},
		emuBuf{"conv_v", "float", f32Bytes(append(cv, 0))},
		emuBuf{"out", "float", poison(4 * heads * s.SeqLen() * s.HeadDimV)},
		emuBuf{"state", "float", poison(4 * heads * s.HeadDimV * s.HeadDimK)},
	)
	return bytesF32(res["out"]), bytesF32(res["state"])
}

// TestSSMEmulated runs the emitted kernel for both variants on the host
// emulator (4 chunks of 64 tokens, d_k = d_v = 64, conv width 4, 2 heads)
// and compares it with the float64 recurrence.
func TestSSMEmulated(t *testing.T) {
	if testing.Short() {
		t.Skip("emulated kernel run")
	}
	for _, v := range []SSMVariant{SSMMamba2SSD, SSMGatedDeltaNet} {
		t.Run(v.String(), func(t *testing.T) {
			s := SSMChunkSchedule{Name: "ssm_emu_" + v.String(), Variant: v, HeadDimK: 64, HeadDimV: 64, ChunkLen: 64, NumChunks: 4, ConvKernel: 4}
			r := rand.New(rand.NewPCG(7, uint64(v)))
			const heads = 2
			var ins []SSMInput
			var want, wantS []float32
			for range heads {
				in := randSSMInput(r, s)
				o, st, err := s.Reference(in)
				if err != nil {
					t.Fatal(err)
				}
				ins = append(ins, in)
				want = append(want, flat(o)...)
				wantS = append(wantS, flat(st)...)
			}
			src, err := s.CUDA()
			if err != nil {
				t.Fatal(err)
			}
			got, gotS := runSSMKernel(t, s, src, heads, ins)
			t.Logf("rel L2: output %.3g, state %.3g", relL2(got, want), relL2(gotS, wantS))
			if e := relL2(got, want); !(e <= 1e-4) {
				t.Errorf("output rel L2 = %g, want <= 1e-4", e)
			}
			if e := relL2(gotS, wantS); !(e <= 1e-4) {
				t.Errorf("state rel L2 = %g, want <= 1e-4", e)
			}
		})
	}
}

// TestSSMEmulatedMutants checks that the multi-chunk oracle rejects a
// kernel that drops the carried state (enable_input_d = 0 on the state
// update) or skips the tcgen05.shift of the convolution.
func TestSSMEmulatedMutants(t *testing.T) {
	if testing.Short() {
		t.Skip("emulated kernel run")
	}
	s := SSMChunkSchedule{Name: "ssm_mutant", Variant: SSMGatedDeltaNet, HeadDimK: 32, HeadDimV: 32, ChunkLen: 32, NumChunks: 3, ConvKernel: 3}
	r := rand.New(rand.NewPCG(9, 9))
	in := randSSMInput(r, s)
	want, _, err := s.Reference(in)
	if err != nil {
		t.Fatal(err)
	}
	src, err := s.CUDA()
	if err != nil {
		t.Fatal(err)
	}
	got, _ := runSSMKernel(t, s, src, 1, []SSMInput{in})
	if e := relL2(got, flat(want)); !(e <= 1e-4) {
		t.Fatalf("unmutated kernel rel L2 = %g", e)
	}
	for _, m := range []struct{ name, old, new string }{
		{"state enable_input_d=0", "eg_tm_mma_ss(tm, base + C_S, ut, qs, DK, L, c > 0);", "eg_tm_mma_ss(tm, base + C_S, ut, qs, DK, L, 0);"},
		{"no shift", "eg_tm_shift_down(tm, base + C_A, (D + 7) & ~7);", ""},
	} {
		t.Run(m.name, func(t *testing.T) {
			if !strings.Contains(src, m.old) {
				t.Fatalf("source lacks %q", m.old)
			}
			got, _ := runSSMKernel(t, s, strings.Replace(src, m.old, m.new, 1), 1, []SSMInput{in})
			if e := relL2(got, flat(want)); e <= 1e-2 {
				t.Errorf("mutant passed the oracle: rel L2 = %g", e)
			}
		})
	}
}
