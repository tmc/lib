package qualify

import (
	"fmt"
	"math"
	"math/rand/v2"
	"strings"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/x/cudahostemu"
	"github.com/tmc/lib/nvidia/x/egraph/tilebridge"
)

func TestVerifyAllTargetISAs(t *testing.T) {
	start := time.Now()
	certs, err := VerifyAllTargetISAs(256, 512, 256)
	if err != nil {
		t.Fatal(err)
	}
	if len(certs) != 5 {
		t.Fatalf("%d certificates, want 5", len(certs))
	}
	for i, c := range certs {
		if c.RefArch != "sm_90a" || c.CandArch != TargetISAs[i+1] {
			t.Errorf("certificate %d compares %s with %s", i, c.RefArch, c.CandArch)
		}
		if !c.BijectiveMatch || c.MismatchDetail != "" || c.CoordinatesTotal != 33554432 {
			t.Errorf("%s: %+v", c.CandArch, c)
		}
		if !(c.MaxAccumULPBound > 0 && c.MaxAccumULPBound < 1e-5) || !(c.ProbabilisticULPBound < c.MaxAccumULPBound) {
			t.Errorf("%s: bounds %g, %g", c.CandArch, c.MaxAccumULPBound, c.ProbabilisticULPBound)
		}
	}
	t.Logf("6 schedules x 33554432 MACs in %v; sm_90a vs sm_100a bound %.3g (prob %.3g)",
		time.Since(start), certs[0].MaxAccumULPBound, certs[0].ProbabilisticULPBound)
}

func TestProveCrossISAEquivalenceMutants(t *testing.T) {
	const m, n, k = 256, 64, 64
	spec := func(arch string) ArchScheduleSpec {
		s, err := DefaultSchedule(arch, m, n, k)
		if err != nil {
			t.Fatal(err)
		}
		return s
	}
	lower := func(s ArchScheduleSpec) lowering {
		l, err := lowerSchedule(s)
		if err != nil {
			t.Fatal(err)
		}
		return l
	}
	ref := spec("sm_90a")
	swapLanes := func(lane, reg int) (int, int) {
		// cudahostemu.MMAM16N8C with lane/4 and lane%4 exchanged.
		g, tt := lane&3, lane>>2
		return g + 8*((reg>>1)&1), 2*tt + reg&1
	}
	tests := []struct {
		name string
		cand func() (ArchScheduleSpec, lowering)
		want string // substring of MismatchDetail; "" for a match
	}{
		{"sm_100a", func() (ArchScheduleSpec, lowering) { s := spec("sm_100a"); return s, lower(s) }, ""},
		{"sm_121a", func() (ArchScheduleSpec, lowering) { s := spec("sm_121a"); return s, lower(s) }, ""},
		{"cta_group::2 peer row offset 129", func() (ArchScheduleSpec, lowering) {
			s := spec("sm_100a")
			l := lower(s).(*tcgen05Lowering)
			l.peerRowOffset = 129
			return s, l
		}, "candidate sm_100a: (m,n,k)=(129,0,0) accumulates A[128,0]·B[0,0]"},
		{"cta_group::2 peer row offset 127", func() (ArchScheduleSpec, lowering) {
			s := spec("sm_107a")
			l := lower(s).(*tcgen05Lowering)
			l.peerRowOffset = 127
			return s, l
		}, "candidate sm_107a: (m,n,k)=(127,0,0) accumulates A[128"},
		{"wgmma lane/4 and lane%4 swapped", func() (ArchScheduleSpec, lowering) {
			l := lower(ref).(*wgmmaLowering)
			l.cStore = swapLanes
			return ref, l
		}, "candidate sm_90a: (m,n,k)=(1,0,0) accumulates A[0,0]·B[0,2]"},
		{"scale block 32 vs 16", func() (ArchScheduleSpec, lowering) {
			s := spec("sm_100a")
			s.ScaleBlockSize = 32
			return s, lower(s)
		}, "candidate sm_100a: (m,n,k)=(0,0,16) applies scales sA[0,0]·sB[0,0], want sA[0,1]·sB[0,1]"},
		{"mma.sync scale block 32", func() (ArchScheduleSpec, lowering) {
			s := spec("sm_120a")
			s.ScaleBlockSize = 32
			return s, lower(s)
		}, "applies scales"},
		{"FP4 packed in nibble order, decoded by the bitopt program", func() (ArchScheduleSpec, lowering) {
			l := lower(ref).(*wgmmaLowering)
			l.pack = func(codes []uint8) []uint32 {
				var w uint32
				for i, c := range codes {
					w |= uint32(c&15) << (4 * i)
				}
				return []uint32{w}
			}
			return ref, l
		}, "candidate sm_90a: NVFP4 decode"},
		{"descriptor swizzle 64B over 128B TMA writes", func() (ArchScheduleSpec, lowering) {
			l := lower(ref).(*wgmmaLowering)
			l.readerS = cudahostemu.Swizzle64B
			return ref, l
		}, "MMA descriptor reads"},
		{"tcgen05 descriptor without swizzle", func() (ArchScheduleSpec, lowering) {
			s := spec("sm_103a")
			l := lower(s).(*tcgen05Lowering)
			l.readerS = 0
			return s, l
		}, "MMA descriptor reads"},
		{"mma.sync A loads with lane/4 and lane%4 swapped", func() (ArchScheduleSpec, lowering) {
			s := spec("sm_121a")
			l := lower(s).(*mmaSyncLowering)
			l.aLoad = func(lane, reg int) (int, int) {
				g, tt := lane&3, lane>>2
				return g + 8*(reg&1), 8*tt + 32*((reg>>1)&1)
			}
			return s, l
		}, "candidate sm_121a: (m,n,k)=(0,0,8) accumulates A[1,0]·B[8,0]"},
		{"problem size", func() (ArchScheduleSpec, lowering) {
			s, err := DefaultSchedule("sm_100a", m, n, 2*k)
			if err != nil {
				t.Fatal(err)
			}
			return s, lower(s)
		}, "problem 256x64x64 vs 256x64x128"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cs, cl := tt.cand()
			c := proveLowerings(ref, cs, lower(ref), cl)
			if tt.want == "" {
				if !c.BijectiveMatch {
					t.Fatalf("mismatch: %s", c.MismatchDetail)
				}
				return
			}
			if c.BijectiveMatch {
				t.Fatal("mutant certified as equivalent")
			}
			if !strings.Contains(c.MismatchDetail, tt.want) {
				t.Errorf("MismatchDetail = %q, want it to contain %q", c.MismatchDetail, tt.want)
			}
		})
	}
}

func TestProveCrossISAEquivalenceErrors(t *testing.T) {
	ok, err := DefaultSchedule("sm_120a", 128, 128, 128)
	if err != nil {
		t.Fatal(err)
	}
	for _, bad := range []ArchScheduleSpec{
		{Arch: "sm_80", ProblemM: 128, ProblemN: 128, ProblemK: 128},
		{Arch: "sm_100a", ProblemM: 128, ProblemN: 128, ProblemK: 128, TileM: 128, TileN: 128, TileK: 64, CTAGroupSize: 2, ScaleBlockSize: 16, SwizzleBytes: 128},
		{Arch: "sm_90a", ProblemM: 128, ProblemN: 128, ProblemK: 128, TileM: 96, TileN: 128, TileK: 64, ScaleBlockSize: 16, SwizzleBytes: 128},
		{Arch: "sm_121a", ProblemM: 128, ProblemN: 128, ProblemK: 128, TileM: 64, TileN: 64, TileK: 32, ScaleBlockSize: 16, SwizzleBytes: 128},
		{Arch: "sm_121a", ProblemM: 128, ProblemN: 128, ProblemK: 128, TileM: 64, TileN: 64, TileK: 64, ScaleBlockSize: 8, SwizzleBytes: 128},
	} {
		if _, err := ProveCrossISAEquivalence(ok, bad); err == nil {
			t.Errorf("ProveCrossISAEquivalence accepted %+v", bad)
		}
	}
	if _, err := VerifyAllTargetISAs(100, 128, 128); err == nil {
		t.Error("VerifyAllTargetISAs accepted M=100")
	}
}

func TestMaxAccumULPBoundHolds(t *testing.T) {
	// D accumulated in 16-wide chunks (tilebridge's Tile IR reference, as
	// on sm_90a) and in 64-wide chunks (as on sm_120a) differ by at most
	// MaxAccumULPBound·Σ|a||b| in every element.
	const m, n, k = 64, 64, 1024
	ref, err := DefaultSchedule("sm_90a", m, n, k)
	if err != nil {
		t.Fatal(err)
	}
	cand, err := DefaultSchedule("sm_120a", m, n, k)
	if err != nil {
		t.Fatal(err)
	}
	cert, err := ProveCrossISAEquivalence(ref, cand)
	if err != nil || !cert.BijectiveMatch {
		t.Fatalf("ProveCrossISAEquivalence = %+v, %v", cert, err)
	}
	r := rand.New(rand.NewPCG(34, 34))
	x := make([]float32, m*k) // A, row-major
	w := make([]float32, n*k) // B transposed: w[n][k]
	for i := range x {
		x[i] = tilebridge.BF16.Round(float32(r.NormFloat64()))
	}
	for i := range w {
		w[i] = tilebridge.BF16.Round(float32(r.NormFloat64()))
	}
	mod, err := tilebridge.GEMMToTileIR(&tilebridge.GEMM{Name: "equiv_ref", N: n, K: k, Out: tilebridge.F32})
	if err != nil {
		t.Fatal(err)
	}
	d16, err := tilebridge.EvalTileIRReference(mod, map[string][]float32{"x": x, "w": w}, m, n, k, 0)
	if err != nil {
		t.Fatal(err)
	}
	worst := 0.0
	for i := range m {
		for j := range n {
			var acc float32
			var mag float64
			for c := 0; c < k; c += 64 {
				var s float32
				for kk := c; kk < c+64; kk++ {
					s = float32(s + float32(x[i*k+kk]*w[j*k+kk]))
					mag += math.Abs(float64(x[i*k+kk]) * float64(w[j*k+kk]))
				}
				acc = float32(acc + s)
			}
			ratio := math.Abs(float64(acc)-float64(d16[i*n+j])) / mag
			if ratio > cert.MaxAccumULPBound {
				t.Errorf("D[%d,%d]: |%v - %v| = %.3g·Σ|ab|, bound %.3g", i, j, acc, d16[i*n+j], ratio, cert.MaxAccumULPBound)
			}
			worst = max(worst, ratio)
		}
	}
	if worst == 0 {
		t.Error("the two orders agreed everywhere; the test does not exercise the bound")
	}
	t.Logf("worst drift %.3g·Σ|ab|, bound %.3g, probabilistic %.3g", worst, cert.MaxAccumULPBound, cert.ProbabilisticULPBound)
}

func ExampleVerifyAllTargetISAs() {
	certs, err := VerifyAllTargetISAs(128, 128, 64)
	if err != nil {
		panic(err)
	}
	for _, c := range certs {
		fmt.Println(c.RefArch, c.CandArch, c.BijectiveMatch, c.CoordinatesTotal)
	}
	// Output:
	// sm_90a sm_100a true 1048576
	// sm_90a sm_103a true 1048576
	// sm_90a sm_107a true 1048576
	// sm_90a sm_120a true 1048576
	// sm_90a sm_121a true 1048576
}
