package cudaroofline

import (
	"bytes"
	"errors"
	"flag"
	"math"
	"os"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudaerrors"
)

var update = flag.Bool("update", false, "rewrite the golden floor vector")

// goldenModel is the device the committed vector is computed against. GB10 at
// host-page-table residency is chosen because its bandwidth is the one measured
// figure in the table, so a rejection on the memory axis rests on a measurement
// rather than on a datasheet.
func goldenModel(t *testing.T) *Model {
	t.Helper()
	m, err := NewModel("sm_121", HostPageTableResident)
	if err != nil {
		t.Fatalf("NewModel: %v", err)
	}
	return m
}

// goldenOperations are the cases the vector covers. The byte counts are chosen
// so the memory floor is a whole number of nanoseconds at 162.4 GB/s, and the
// FLOP counts so the compute floor is whole at 40 TFLOPS; a vector whose
// expected values are exact is one a consumer can compare without agreeing on a
// tolerance first.
var goldenOperations = []struct {
	op         Operation
	durationNs uint64
}{
	{Operation{Name: "matvec_fp32_at_floor", FLOPs: 2_000_000, Bytes: 162_400, Precision: FP32}, 1000},
	{Operation{Name: "matvec_fp32_comfortably_above", FLOPs: 2_000_000, Bytes: 162_400, Precision: FP32}, 4200},
	{Operation{Name: "matvec_fp32_one_ns_below_floor", FLOPs: 2_000_000, Bytes: 162_400, Precision: FP32}, 999},
	{Operation{Name: "matvec_fp32_half_the_floor", FLOPs: 2_000_000, Bytes: 162_400, Precision: FP32}, 500},
	{Operation{Name: "gemm_fp32_compute_bound_accepted", FLOPs: 40_000_000, Bytes: 1_624, Precision: FP32}, 1000},
	{Operation{Name: "gemm_fp32_compute_bound_rejected", FLOPs: 40_000_000, Bytes: 1_624, Precision: FP32}, 999},
	{Operation{Name: "gemm_fp16_tensor_memory_floor_only", FLOPs: 400_000_000, Bytes: 162_400, Precision: FP16Tensor}, 1200},
	{Operation{Name: "gemm_fp16_tensor_below_memory_floor", FLOPs: 400_000_000, Bytes: 162_400, Precision: FP16Tensor}, 800},
	{Operation{Name: "zero_duration_claim", FLOPs: 2_000_000, Bytes: 162_400, Precision: FP32}, 0},
}

func goldenJudgments(t *testing.T) (*Model, []Judgment) {
	t.Helper()
	m := goldenModel(t)
	js := make([]Judgment, 0, len(goldenOperations))
	for _, c := range goldenOperations {
		js = append(js, m.Judge(c.op, c.durationNs))
	}
	return m, js
}

func TestFloorMatchesGoldenVector(t *testing.T) {
	m, js := goldenJudgments(t)
	want := NewRecord(m, js)

	if *update {
		var buf bytes.Buffer
		if err := want.Encode(&buf); err != nil {
			t.Fatalf("Encode: %v", err)
		}
		if err := os.WriteFile(GoldenVectorPath, buf.Bytes(), 0o644); err != nil {
			t.Fatalf("writing %s: %v", GoldenVectorPath, err)
		}
		t.Logf("wrote %s (%d triples)", GoldenVectorPath, len(want.Triples))
		return
	}

	f, err := os.Open(GoldenVectorPath)
	if err != nil {
		t.Fatalf("opening the golden vector: %v (regenerate with -update)", err)
	}
	defer f.Close()
	got, err := DecodeRecord(f)
	if err != nil {
		t.Fatalf("DecodeRecord: %v", err)
	}

	if got.Device != want.Device {
		t.Errorf("device record drifted:\n got %+v\nwant %+v", got.Device, want.Device)
	}
	if len(got.Triples) != len(want.Triples) {
		t.Fatalf("got %d triples, want %d (regenerate with -update)", len(got.Triples), len(want.Triples))
	}
	for i, w := range want.Triples {
		g := got.Triples[i]
		if g.Name != w.Name {
			t.Errorf("triple %d: got name %q, want %q", i, g.Name, w.Name)
			continue
		}
		// The floor is the number consumers read, so it is compared exactly
		// up to float64 round-tripping through JSON and nothing looser.
		if !nearlyEqual(g.FloorNs, w.FloorNs) {
			t.Errorf("%s: floor %v, want %v", w.Name, g.FloorNs, w.FloorNs)
		}
		if !nearlyEqual(g.MemoryFloorNs, w.MemoryFloorNs) || !nearlyEqual(g.ComputeFloorNs, w.ComputeFloorNs) {
			t.Errorf("%s: per-axis floors (%v, %v), want (%v, %v)",
				w.Name, g.MemoryFloorNs, g.ComputeFloorNs, w.MemoryFloorNs, w.ComputeFloorNs)
		}
		if g.Verdict != w.Verdict {
			t.Errorf("%s: verdict %q, want %q", w.Name, g.Verdict, w.Verdict)
		}
		if g.Basis != w.Basis || g.Axis != w.Axis {
			t.Errorf("%s: basis/axis (%q, %q), want (%q, %q)", w.Name, g.Basis, g.Axis, w.Basis, w.Axis)
		}
		if g.CeilingProvenance != w.CeilingProvenance {
			t.Errorf("%s: ceiling provenance %q, want %q", w.Name, g.CeilingProvenance, w.CeilingProvenance)
		}
	}
}

func nearlyEqual(a, b float64) bool {
	if a == b {
		return true
	}
	return math.Abs(a-b) <= 1e-9*math.Max(math.Abs(a), math.Abs(b))
}

// TestGoldenVectorContainsRejectedAndAcceptedVerdicts guards the vector against
// becoming a fixture that only proves the easy direction. A consumer that reads
// only accepted rows never exercises its own rejection path, and a floor nobody
// rejects against is indistinguishable from no floor at all.
func TestGoldenVectorContainsRejectedAndAcceptedVerdicts(t *testing.T) {
	f, err := os.Open(GoldenVectorPath)
	if err != nil {
		t.Fatalf("opening the golden vector: %v", err)
	}
	defer f.Close()
	r, err := DecodeRecord(f)
	if err != nil {
		t.Fatalf("DecodeRecord: %v", err)
	}

	counts := map[string]int{}
	axes := map[string]int{}
	bases := map[string]int{}
	for _, tr := range r.Triples {
		counts[tr.Verdict]++
		if tr.Verdict == string(VerdictRejected) {
			axes[tr.Axis]++
			if tr.Reason == "" {
				t.Errorf("%s: rejected with no reason", tr.Name)
			}
		}
		bases[tr.Basis]++
	}
	if counts[string(VerdictAccepted)] == 0 {
		t.Error("vector holds no accepted verdict")
	}
	if counts[string(VerdictRejected)] == 0 {
		t.Error("vector holds no rejected verdict")
	}
	// A rejection on one axis does not exercise the other: the compute floor
	// and the memory floor are different arithmetic against different
	// provenance.
	if axes[string(MemoryLimited)] == 0 {
		t.Error("vector holds no rejection on the memory axis")
	}
	if axes[string(ComputeLimited)] == 0 {
		t.Error("vector holds no rejection on the compute axis")
	}
	if bases[string(FloorFromMemoryOnly)] == 0 {
		t.Error("vector holds no memory-only floor, so the absent-compute-peak path is untested")
	}
}

// TestEstimateAboveRooflineRejectedWithInputs: a duration below the floor is a
// throughput above the roofline. Rejecting it is not enough -- the rejection has
// to carry the numbers, because the usual cause is a wrong byte count or a
// ceiling that needs revising, and neither can be found from a bare verdict.
func TestEstimateAboveRooflineRejectedWithInputs(t *testing.T) {
	m := goldenModel(t)
	op := Operation{Name: "impossible", FLOPs: 2_000_000, Bytes: 162_400, Precision: FP32}
	j := m.Judge(op, 100)

	if j.Verdict != VerdictRejected {
		t.Fatalf("verdict %q, want %q", j.Verdict, VerdictRejected)
	}
	if j.Operation != op {
		t.Errorf("judgment lost its operation: %+v", j.Operation)
	}
	if j.DurationNs != 100 {
		t.Errorf("judgment lost the duration it rejected: %d", j.DurationNs)
	}
	if j.FloorNs <= 0 {
		t.Errorf("rejection carries no floor: %v", j.FloorNs)
	}
	if j.CeilingProvenance != Measured {
		t.Errorf("provenance %q, want %q: GB10 bandwidth is the measured figure", j.CeilingProvenance, Measured)
	}
	for _, want := range []string{"100", "floor", string(MemoryLimited)} {
		if !strings.Contains(j.Reason, want) {
			t.Errorf("reason %q does not mention %q", j.Reason, want)
		}
	}

	// The same claim one nanosecond above the floor is accepted, so the
	// rejection is a statement about the boundary and not about the operation.
	if j := m.Judge(op, 1001); j.Verdict != VerdictAccepted {
		t.Errorf("1001 ns: verdict %q, want %q (floor %v)", j.Verdict, VerdictAccepted, j.FloorNs)
	}
}

// TestUnavailableAxisFloorIsSentinelNotZero: zero is a floor an operation can
// really have -- one that moves no bytes has a memory floor of exactly zero --
// so absence cannot be spelled zero without making the two indistinguishable.
func TestUnavailableAxisFloorIsSentinelNotZero(t *testing.T) {
	m := goldenModel(t)

	tensor := m.Floor(Operation{Name: "t", FLOPs: 1 << 40, Bytes: 4096, Precision: FP16Tensor})
	if tensor.ComputeFloorNs != FloorUnavailable {
		t.Errorf("absent compute floor is %v, want the %v sentinel", tensor.ComputeFloorNs, FloorUnavailable)
	}
	if tensor.Basis != FloorFromMemoryOnly {
		t.Errorf("basis %q, want %q", tensor.Basis, FloorFromMemoryOnly)
	}

	noBytes := m.Floor(Operation{Name: "n", FLOPs: 40_000_000, Bytes: 0, Precision: FP32})
	if noBytes.MemoryFloorNs != 0 {
		t.Errorf("an operation that moves no bytes has a memory floor of 0, got %v", noBytes.MemoryFloorNs)
	}
	if noBytes.Basis != FloorFromBothAxes {
		t.Errorf("basis %q, want %q: a zero memory floor is a floor", noBytes.Basis, FloorFromBothAxes)
	}
}

// TestFloorRestsOnMemoryAloneWhenComputePeakAbsent states the consequence of the
// missing tensor peaks: the bound is weaker, and says so, rather than being
// completed with a number nobody measured.
func TestFloorRestsOnMemoryAloneWhenComputePeakAbsent(t *testing.T) {
	m := goldenModel(t)

	// 400 MFLOP in 1200 ns is 333 TFLOPS, far above any plausible GB10 tensor
	// peak. Nothing here can reject it, because nothing here knows the peak.
	j := m.Judge(Operation{Name: "unbounded_tensor", FLOPs: 400_000_000, Bytes: 162_400, Precision: FP16Tensor}, 1200)
	if j.Verdict != VerdictAccepted {
		t.Errorf("verdict %q, want %q", j.Verdict, VerdictAccepted)
	}
	if j.Basis != FloorFromMemoryOnly {
		t.Fatalf("basis %q, want %q", j.Basis, FloorFromMemoryOnly)
	}

	// The same shape in FP32, where a peak exists, is rejected. The
	// difference between the two verdicts is the missing peak and nothing
	// else, which is what makes the absence worth encoding.
	if j := m.Judge(Operation{Name: "same_shape_fp32", FLOPs: 400_000_000, Bytes: 162_400, Precision: FP32}, 1200); j.Verdict != VerdictRejected {
		t.Errorf("FP32: verdict %q, want %q (compute floor %v)", j.Verdict, VerdictRejected, j.ComputeFloorNs)
	}
}

func TestUnjudgedWhenNoCeilingForDevice(t *testing.T) {
	m := NewModelWithCeiling(Ceiling{Architecture: "sm_999", Residency: DeviceResident})
	j := m.Judge(Operation{Name: "x", FLOPs: 1000, Bytes: 1000, Precision: FP32}, 1)
	if j.Verdict != VerdictUnjudged {
		t.Errorf("verdict %q, want %q: an unknown device is not an acceptance", j.Verdict, VerdictUnjudged)
	}
	if j.Basis != FloorAbsent || j.FloorNs != FloorUnavailable {
		t.Errorf("basis %q floor %v, want %q and the sentinel", j.Basis, j.FloorNs, FloorAbsent)
	}
	if j.Reason == "" {
		t.Error("unjudged with no reason")
	}
}

func TestAnalyticalDataMovementBounds(t *testing.T) {
	problem := map[string]int{
		"B": 1,
		"M": 1024,
		"N": 1024,
		"K": 1024,
	}
	tileFactors := map[string]int{
		"M": 128,
		"N": 128,
		"K": 64,
	}
	dtypes := map[string]int{
		"A": 2, // FP16: 2 bytes
		"B": 2, // FP16: 2 bytes
		"C": 4, // FP32: 4 bytes
	}
	bypassed := map[string]bool{}

	dm, err := AnalyticalDataMovement(problem, tileFactors, dtypes, bypassed)
	if err != nil {
		t.Fatal(err)
	}

	// Exact analytical derivation:
	// A and B are each read once while shared or L2 storage retains them.
	// Each is 1024*1024*2 = 2,097,152 bytes.
	// C: 1024*1024 * 4 = 4,194,304 bytes
	// ReadBytes = 4,194,304 bytes
	// WriteBytes = 4,194,304 bytes
	// TotalBytes = 8,388,608 bytes
	// FLOPs = 2 * 1024^3 = 2,147,483,648
	// OpIntensity = 2147483648 / 8388608 = 256

	wantRead := uint64(4194304)
	wantWrite := uint64(4194304)
	wantTotal := uint64(8388608)

	if dm.ReadBytes != wantRead {
		t.Errorf("ReadBytes = %d, want %d", dm.ReadBytes, wantRead)
	}
	if dm.WriteBytes != wantWrite {
		t.Errorf("WriteBytes = %d, want %d", dm.WriteBytes, wantWrite)
	}
	if dm.TotalBytes != wantTotal {
		t.Errorf("TotalBytes = %d, want %d", dm.TotalBytes, wantTotal)
	}

	expectedIntensity := float64(2147483648) / float64(wantTotal)
	if math.Abs(dm.OperationalIntensity-expectedIntensity) > 1e-6 {
		t.Errorf("OperationalIntensity = %v, want %v", dm.OperationalIntensity, expectedIntensity)
	}
}

func TestHardwareVerificationPendingWithoutGPU(t *testing.T) {
	probed := false
	cal, err := VerifyHardwareExecution("sm_121", func(target string) (bool, error) {
		probed = target == "sm_121"
		return false, nil
	})
	if err == nil {
		t.Fatalf("expected error from VerifyHardwareExecution without hardware, got nil")
	}
	if !errors.Is(err, cudaerrors.ErrHardwarePending) {
		t.Errorf("err = %v, want ErrHardwarePending", err)
	}
	if !probed {
		t.Fatal("device availability probe was not called with the target")
	}
	if cal.Status != VerificationPending {
		t.Errorf("cal.Status = %q, want %q", cal.Status, VerificationPending)
	}
	if cal.LatencyNs != FloorUnavailable {
		t.Errorf("cal.LatencyNs = %v, want %v (FloorUnavailable)", cal.LatencyNs, FloorUnavailable)
	}
}

func TestHardwareVerificationPendingWithAvailableDevice(t *testing.T) {
	cal, err := VerifyHardwareExecution("sm_121", func(target string) (bool, error) {
		if target != "sm_121" {
			t.Errorf("probe target = %q, want sm_121", target)
		}
		return true, nil
	})
	if cal.Status != VerificationPending {
		t.Errorf("cal.Status = %q, want %q", cal.Status, VerificationPending)
	}
	if cal.LatencyNs != FloorUnavailable {
		t.Errorf("cal.LatencyNs = %v, want %v", cal.LatencyNs, FloorUnavailable)
	}
	if !errors.Is(err, cudaerrors.ErrHardwarePending) {
		t.Errorf("err = %v, want ErrHardwarePending", err)
	}
}

func TestAnalyticalDataMovementBoundsRectangular(t *testing.T) {
	problem := map[string]int{"B": 2, "M": 5, "N": 7, "K": 3}
	tiles := map[string]int{"M": 2, "N": 4}
	dtypes := map[string]int{"A": 1, "B": 2, "C": 4}
	got, err := AnalyticalDataMovement(problem, tiles, dtypes, nil)
	if err != nil {
		t.Fatal(err)
	}
	if got.TensorTransfers["A"] != 30 || got.TensorTransfers["B"] != 84 {
		t.Fatalf("A/B transfers = %v, want A=30 B=84", got.TensorTransfers)
	}
	if got.TensorTransfers["C"] != 280 {
		t.Fatalf("C transfers = %d, want 280", got.TensorTransfers["C"])
	}
	bypassed, err := AnalyticalDataMovement(problem, tiles, dtypes, map[string]bool{"A": true, "B": true})
	if err != nil {
		t.Fatal(err)
	}
	if bypassed.TensorTransfers["A"] != 60 || bypassed.TensorTransfers["B"] != 252 {
		t.Fatalf("bypassed A/B transfers = %v, want A=60 B=252", bypassed.TensorTransfers)
	}
}

func TestAnalyticalDataMovementBypassNeverBelowCompulsory(t *testing.T) {
	problem := map[string]int{"B": 2, "M": 5, "N": 7, "K": 3}
	tiles := map[string]int{"M": 2, "N": 4}
	dtypes := map[string]int{"A": 1, "B": 2, "C": 4}
	minimum := uint64(30 + 84 + 280)
	for mask := 0; mask < 8; mask++ {
		bypassed := map[string]bool{
			"A": mask&1 != 0,
			"B": mask&2 != 0,
			"C": mask&4 != 0,
		}
		got, err := AnalyticalDataMovement(problem, tiles, dtypes, bypassed)
		if err != nil {
			t.Fatal(err)
		}
		if got.TotalBytes < minimum {
			t.Fatalf("mask %03b total %d is below compulsory %d", mask, got.TotalBytes, minimum)
		}
	}
}

func TestAnalyticalDataMovementRejectsInvalidExtents(t *testing.T) {
	problem := map[string]int{"B": 1, "M": 4, "N": 8, "K": 2}
	tiles := map[string]int{"M": 2, "N": 4}
	dtypes := map[string]int{"A": 2, "B": 2, "C": 4}
	tests := []struct {
		name                   string
		problem, tiles, dtypes map[string]int
	}{
		{name: "negative extent", problem: map[string]int{"B": 1, "M": -1, "N": 8, "K": 2}, tiles: tiles, dtypes: dtypes},
		{name: "missing extent", problem: map[string]int{"B": 1, "M": 4, "N": 8}, tiles: tiles, dtypes: dtypes},
		{name: "unknown extent", problem: map[string]int{"B": 1, "M": 4, "N": 8, "K": 2, "P": 3}, tiles: tiles, dtypes: dtypes},
		{name: "missing tile", problem: problem, tiles: map[string]int{"M": 2}, dtypes: dtypes},
		{name: "missing dtype", problem: problem, tiles: tiles, dtypes: map[string]int{"A": 2, "B": 2}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got, err := AnalyticalDataMovement(tt.problem, tt.tiles, tt.dtypes, nil); err == nil {
				t.Fatalf("AnalyticalDataMovement = %+v, nil; want invalid-input error", got)
			}
		})
	}
}
