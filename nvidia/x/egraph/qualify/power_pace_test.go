package qualify

import (
	"fmt"
	"math"
	"slices"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudahostemu"
)

// b300 is a B300-like step (280 W to 1400 W at 0.85 V over 80 clusters)
// with an effective inductance of 1 pH, chosen so the synchronized start
// droops past 0.12 V in this model.
var b300 = PacingConfig{
	NumClusters: 80, RampBuckets: 1, IdlePowerWatts: 280, ActivePowerWatts: 1400,
	CoreVoltage: 0.85, PkgInductanceNh: 0.001, MaxAllowedDroopV: 0.035,
}

func TestPeakDiDt(t *testing.T) {
	cfg := PacingConfig{NumClusters: 8, RampBuckets: 1, IdlePowerWatts: 0, ActivePowerWatts: 80, CoreVoltage: 1, PkgInductanceNh: 1}
	const s = 80.0 / 8 / 10 // amps per ns per cluster
	tests := []struct {
		buckets int
		step    uint32
		rise    float64
		want    float64 // clusters ramping at once
	}{
		{1, 0, 10, 8},
		{4, 0, 10, 8},
		{4, 10, 10, 2},
		{4, 5, 10, 4},
		{4, 9, 10, 4},
		{3, 20, 10, 3}, // buckets of 3, 3, 2
		{8, 1, 10, 8},
		{8, 2, 10, 5},
		{4, 10, 5, 2},
	}
	for _, tt := range tests {
		c := cfg
		c.ClusterRiseNs = tt.rise
		if got := peakDiDt(8, tt.buckets, tt.step, c); math.Abs(got-tt.want*s*10/tt.rise) > 1e-9 {
			t.Errorf("peakDiDt(8, %d, %d, rise %v) = %v, want %v", tt.buckets, tt.step, tt.rise, got, tt.want*s*10/tt.rise)
		}
	}
}

func TestSolveOptimalPacing(t *testing.T) {
	r200 := b300
	r200.ActivePowerWatts = 2100
	r200.NumClusters = 74
	for _, tt := range []struct {
		name string
		cfg  PacingConfig
	}{{"b300", b300}, {"r200", r200}} {
		t.Run(tt.name, func(t *testing.T) {
			un, err := AnalyzePowerTransient(tt.cfg)
			if err != nil {
				t.Fatal(err)
			}
			if un.UnpacedVoltageDroopV <= 0.120 || !un.TriggersPowerBrake {
				t.Errorf("unpaced droop %.4f V, brake %v; want > 0.120 V and brake", un.UnpacedVoltageDroopV, un.TriggersPowerBrake)
			}
			best, r, err := SolveOptimalPacing(tt.cfg)
			if err != nil {
				t.Fatal(err)
			}
			if r.PacedVoltageDroopV > 0.035 || r.RampOverheadNs >= 400 {
				t.Errorf("paced droop %.4f V with %d ns ramp; want <= 0.035 V, < 400 ns", r.PacedVoltageDroopV, r.RampOverheadNs)
			}
			gate := best
			gate.MaxAllowedDroopV = 0.045
			if g, _ := AnalyzePowerTransient(gate); g.TriggersPowerBrake {
				t.Errorf("solved pacing %d x %d ns still brakes at 45 mV", best.RampBuckets, best.BucketStepNs)
			}
			// Brute force over the same space finds nothing cheaper.
			for g := 1; g <= tt.cfg.NumClusters; g++ {
				for step := uint32(0); step <= 10; step++ {
					cost := uint32(g-1) * step
					if cost < r.RampOverheadNs && tt.cfg.PkgInductanceNh*peakDiDt(tt.cfg.NumClusters, g, step, tt.cfg) <= tt.cfg.MaxAllowedDroopV {
						t.Errorf("%d buckets x %d ns (%d ns) also meets the limit, cheaper than %d ns", g, step, cost, r.RampOverheadNs)
					}
				}
			}
			t.Logf("%s: unpaced %.4f V; %d buckets x %d ns -> %.4f V, %d ns ramp",
				tt.name, un.UnpacedVoltageDroopV, best.RampBuckets, best.BucketStepNs, r.PacedVoltageDroopV, r.RampOverheadNs)

			// Mutants: no buckets or no step leave the full step in place.
			for _, m := range []PacingConfig{{RampBuckets: 1, BucketStepNs: best.BucketStepNs}, {RampBuckets: best.RampBuckets, BucketStepNs: 0}} {
				c := best
				c.RampBuckets, c.BucketStepNs = m.RampBuckets, m.BucketStepNs
				r, err := AnalyzePowerTransient(c)
				if err != nil {
					t.Fatal(err)
				}
				if !r.TriggersPowerBrake || r.PacedVoltageDroopV != r.UnpacedVoltageDroopV {
					t.Errorf("%d buckets x %d ns: paced %.4f V, brake %v; want the unpaced droop", c.RampBuckets, c.BucketStepNs, r.PacedVoltageDroopV, r.TriggersPowerBrake)
				}
			}
		})
	}
}

func TestSolveOptimalPacingUnreachable(t *testing.T) {
	// With the raw 0.08 nH package inductance and a 10 ns cluster rise, the
	// model's synchronized droop is about 10 V; no stagger within 500 ns
	// brings it under 45 mV.
	cfg := b300
	cfg.PkgInductanceNh = 0.08
	cfg.MaxAllowedDroopV = 0.045
	_, _, err := SolveOptimalPacing(cfg)
	if err == nil || !strings.Contains(err.Error(), "no stagger") {
		t.Errorf("SolveOptimalPacing(0.08 nH) = %v, want no-stagger error", err)
	}
	t.Log(err)
	for _, bad := range []PacingConfig{
		{},
		{NumClusters: 4, RampBuckets: 1, CoreVoltage: 0},
		{NumClusters: 4, RampBuckets: 1, CoreVoltage: 1, IdlePowerWatts: 10, ActivePowerWatts: 5},
	} {
		if _, err := AnalyzePowerTransient(bad); err == nil {
			t.Errorf("AnalyzePowerTransient(%+v) accepted an invalid config", bad)
		}
	}
}

func TestEmitPowerPacedProloguePTX(t *testing.T) {
	cfg := b300
	cfg.RampBuckets, cfg.BucketStepNs = 5, 12
	ptx, err := EmitPowerPacedProloguePTX(cfg, "qp_prologue")
	if err != nil {
		t.Fatal(err)
	}
	if vs, _ := AuditPTXDeterminism(ptx, DeterministicConfig{}); len(vs) != 0 {
		t.Errorf("AuditPTXDeterminism(prologue) = %v", vs)
	}
	body := funcBody(t, ptx, "qp_prologue")
	const nx, ny, nz = 3, 4, 2
	for z := range nz {
		for y := range ny {
			for x := range nx {
				s := &ptxSim{regs: map[string]uint64{
					"%clusterid.x": uint64(x), "%clusterid.y": uint64(y), "%clusterid.z": uint64(z),
					"%nclusterid.x": nx, "%nclusterid.y": ny,
				}}
				s.run(t, body)
				cid := (z*ny+y)*nx + x
				want := uint64(cid%5) * 12
				if len(s.sleeps) != 1 || s.sleeps[0] != want {
					t.Errorf("cluster (%d,%d,%d): nanosleep %v, want [%d]", x, y, z, s.sleeps, want)
				}
			}
		}
	}
}

func TestInterleaveSchedule(t *testing.T) {
	mk := func(p string, n int) []string {
		s := make([]string, n)
		for i := range s {
			s[i] = fmt.Sprintf("%s%d", p, i)
		}
		return s
	}
	isMMA := func(s string) bool { return strings.HasPrefix(s, "mma") }
	tests := []struct {
		mma, alu, longest int
	}{
		{16, 0, 16},
		{16, 1, 8},
		{16, 3, 4},
		{16, 15, 1},
		{16, 40, 1},
		{5, 2, 2},
		{0, 3, 0},
	}
	for _, tt := range tests {
		mma, alu := mk("mma", tt.mma), mk("alu", tt.alu)
		got := InterleaveSchedule(mma, alu)
		if len(got) != tt.mma+tt.alu {
			t.Fatalf("InterleaveSchedule(%d, %d) has %d instructions", tt.mma, tt.alu, len(got))
		}
		// Each stream keeps its order: zero numerical effect for
		// independent streams.
		var gm, ga []string
		for _, s := range got {
			if isMMA(s) {
				gm = append(gm, s)
			} else {
				ga = append(ga, s)
			}
		}
		if !slices.Equal(gm, mma) || !slices.Equal(ga, alu) {
			t.Errorf("InterleaveSchedule(%d, %d) reordered a stream: %v", tt.mma, tt.alu, got)
		}
		if l := LongestRun(got, isMMA); l != tt.longest {
			t.Errorf("InterleaveSchedule(%d, %d): longest MMA run %d, want %d", tt.mma, tt.alu, l, tt.longest)
		}
	}
}

func TestPacedPrologueCUDAHostEmu(t *testing.T) {
	if _, err := cudahostemu.Compiler(); err != nil {
		t.Skip(err)
	}
	const n = 128
	x := make([]float32, n)
	y := make([]float32, n)
	want := make([]float32, n)
	for i := range n {
		x[i] = float32(i)*0.37 - 11
		y[i] = float32(i%17) * 1.5
		want[i] = float32(float32(2.5*x[i]) + y[i])
	}
	for _, buckets := range []int{1, 8} {
		cfg := b300
		cfg.RampBuckets, cfg.BucketStepNs = buckets, 40
		src, err := PacedPrologueCUDA(cfg)
		if err != nil {
			t.Fatal(err)
		}
		out, err := cudahostemu.Run(t.Context(), t.TempDir(), cudahostemu.Program{
			Source: src,
			Launch: "eg_emu_launch(dim3(4), dim3(32), 0, qp_paced_axpy, x, y, 2.5f, 128, starts)",
			Buffers: []cudahostemu.Buffer{
				{Name: "x", CType: "float", Data: f32Bytes(x)},
				{Name: "y", CType: "float", Data: f32Bytes(y)},
				{Name: "starts", CType: "uint64_t", Data: make([]byte, 8*4)},
			},
		})
		if err != nil {
			t.Fatal(err)
		}
		got := bytesF32(out["y"])
		for i := range want {
			if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
				t.Errorf("buckets %d: y[%d] = %v, want %v", buckets, i, got[i], want[i])
			}
		}
	}
}

func ExampleSolveOptimalPacing() {
	cfg := PacingConfig{
		NumClusters: 80, RampBuckets: 1, IdlePowerWatts: 280, ActivePowerWatts: 1400,
		CoreVoltage: 0.85, PkgInductanceNh: 0.001, MaxAllowedDroopV: 0.035,
	}
	best, r, err := SolveOptimalPacing(cfg)
	if err != nil {
		panic(err)
	}
	fmt.Printf("unpaced %.3f V, %d buckets x %d ns: %.3f V, %d ns ramp\n",
		r.UnpacedVoltageDroopV, best.RampBuckets, best.BucketStepNs, r.PacedVoltageDroopV, r.RampOverheadNs)
	// Output: unpaced 0.132 V, 4 buckets x 10 ns: 0.033 V, 30 ns ramp
}
