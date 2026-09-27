package cudaroofline

import (
	"fmt"
	"math"

	"github.com/tmc/lib/nvidia/x/cudaerrors"
)

// Operation is a unit of work whose cost is claimed rather than observed: a
// candidate kernel a search proposes, an entry in a plan, a row in a report.
// FLOPs and Bytes are what the operation must do to be correct, not what some
// implementation of it was seen to do.
type Operation struct {
	Name      string
	FLOPs     uint64
	Bytes     uint64    // DRAM traffic on the model's memory path
	Precision Precision // format and pipe the FLOPs are carried out in
}

// Verdict is what judging a claimed duration against the floor concludes.
type Verdict string

const (
	// VerdictAccepted means the claimed duration is at or above the floor.
	// This is not a statement that the duration is real, or good, or that the
	// kernel is correct. It says only that no hardware limit in the table
	// forbids it.
	VerdictAccepted Verdict = "accepted"

	// VerdictRejected means the claimed duration is below the floor: the
	// operation would have had to move its bytes faster than the memory path
	// can carry them, or issue its FLOPs faster than the pipe can retire
	// them. A timer misread, a byte count that does not describe the traffic,
	// a kernel that skipped the work, or a ceiling that is too low.
	VerdictRejected Verdict = "rejected_below_floor"

	// VerdictUnjudged means no floor could be computed, because the table has
	// no ceiling for this architecture and residency. It is not acceptance:
	// nothing was checked.
	VerdictUnjudged Verdict = "unjudged"
)

// FloorBasis says which axes the floor rests on, and so how strong a bound it
// is. A floor is the largest of the per-axis minimum durations, so a missing
// axis can only make it lower: a bound resting on memory alone rejects strictly
// less than one resting on both, and a duration it accepts may still be
// unphysical on the axis nobody could check.
type FloorBasis string

const (
	// FloorFromBothAxes means a memory floor and a compute floor were both
	// available and the larger was taken. This is the strongest bound the
	// package produces.
	FloorFromBothAxes FloorBasis = "memory_and_compute"

	// FloorFromMemoryOnly means no compute peak exists for the operation's
	// precision, so the floor is the memory floor alone and is weaker than it
	// would be with one. See PeakTFLOPSFor: every tensor format lands here.
	FloorFromMemoryOnly FloorBasis = "memory_only"

	// FloorFromComputeOnly means no memory ceiling was available, so only the
	// compute floor applies. No entry in the table lands here -- every one
	// carries a bandwidth -- so this is reachable only through
	// NewModelWithCeiling with an unstated bandwidth.
	FloorFromComputeOnly FloorBasis = "compute_only"

	// FloorAbsent means neither axis produced a floor: the model has no memory
	// peak and no compute peak for the operation's precision. Zero work has a
	// zero floor when a peak is available.
	FloorAbsent FloorBasis = "absent"
)

// Judgment records a floor, how it was derived, and what it concluded about one
// claimed duration. Every input travels with it: a rejection that does not
// carry the numbers it rejected cannot be argued with, and the argument is the
// point -- an operation below the floor is as often a wrong byte count or a
// ceiling that needs revising as it is a bad measurement.
type Judgment struct {
	Operation  Operation
	DurationNs uint64 // the claimed duration being judged; zero when judging the floor alone

	FloorNs float64    // smallest duration the ceilings permit, in nanoseconds
	Basis   FloorBasis // which axes FloorNs rests on
	Axis    LimitingAxis
	Verdict Verdict

	// MemoryFloorNs and ComputeFloorNs are the per-axis minima. An axis that
	// produced no floor is -1, never zero: zero is a real floor, held by an
	// operation that moves no bytes or does no FLOPs.
	MemoryFloorNs  float64
	ComputeFloorNs float64

	// CeilingProvenance is the provenance of the limit named by Axis, so a
	// rejection against an unverified datasheet figure is distinguishable
	// from one against a measured bandwidth. On GB10 a memory-axis rejection
	// is Measured; every compute-axis rejection in this repository is
	// Unverified.
	CeilingProvenance Provenance

	// Reason states, for a rejection or an unjudged verdict, what happened.
	// It is empty for VerdictAccepted.
	Reason string
}

// FloorUnavailable marks a per-axis floor that could not be computed. It is not
// zero, because zero is a floor an operation can really have.
const FloorUnavailable = -1.0

// Floor returns the smallest duration the model's ceilings permit for op,
// together with how that bound was derived. It judges nothing; Judge does.
//
// The memory floor is Bytes / PeakMemoryGBps, which is exact in these units
// because one GB/s is one byte per nanosecond. The compute floor is
// FLOPs / (PeakTFLOPS * 1000), because one TFLOPS is a thousand FLOPs per
// nanosecond.
func (m *Model) Floor(op Operation) Judgment {
	j := Judgment{
		Operation:      op,
		MemoryFloorNs:  FloorUnavailable,
		ComputeFloorNs: FloorUnavailable,
		Basis:          FloorAbsent,
		FloorNs:        FloorUnavailable,
	}

	if m.Ceiling.PeakMemoryGBps > 0 {
		j.MemoryFloorNs = float64(op.Bytes) / m.Ceiling.PeakMemoryGBps
	}
	// The model may carry a caller-supplied peak for a part or partition the
	// architecture table does not describe. Use that same peak and provenance
	// for the floor and the encoded record. It describes only the FP32 pipe.
	if peak := m.Ceiling.PeakTFLOPS; op.Precision == FP32 && peak > 0 {
		j.ComputeFloorNs = float64(op.FLOPs) / (peak * 1000.0)
	}

	switch {
	case j.MemoryFloorNs >= 0 && j.ComputeFloorNs >= 0:
		j.Basis = FloorFromBothAxes
		if j.ComputeFloorNs > j.MemoryFloorNs {
			j.FloorNs, j.Axis = j.ComputeFloorNs, ComputeLimited
		} else {
			j.FloorNs, j.Axis = j.MemoryFloorNs, MemoryLimited
		}
	case j.MemoryFloorNs >= 0:
		j.Basis = FloorFromMemoryOnly
		j.FloorNs, j.Axis = j.MemoryFloorNs, MemoryLimited
	case j.ComputeFloorNs >= 0:
		j.Basis = FloorFromComputeOnly
		j.FloorNs, j.Axis = j.ComputeFloorNs, ComputeLimited
	}
	j.CeilingProvenance = m.Ceiling.provenanceFor(j.Axis)
	return j
}

// Judge bounds op from below and says whether durationNs is physically possible
// against that bound.
//
// This is the rejection rule a search uses to throw out a candidate whose
// reported time cannot have happened, and it is the single source of that rule:
// a consumer that cannot import this package reads the floor from the golden
// vector in testdata rather than recomputing it. See WriteRecord.
func (m *Model) Judge(op Operation, durationNs uint64) Judgment {
	j := m.Floor(op)
	j.DurationNs = durationNs

	switch {
	case j.Basis == FloorAbsent:
		j.Verdict = VerdictUnjudged
		j.Reason = fmt.Sprintf("no floor for %q at %q residency", m.Ceiling.Architecture, m.Ceiling.Residency)
	case durationNs == 0:
		j.Verdict = VerdictRejected
		j.Reason = "claimed duration is zero"
	case float64(durationNs) < j.FloorNs:
		j.Verdict = VerdictRejected
		j.Reason = fmt.Sprintf("%d ns is below the %s floor of %.3f ns (%s, %s ceiling)",
			durationNs, j.Axis, j.FloorNs, j.Basis, j.CeilingProvenance)
	default:
		j.Verdict = VerdictAccepted
	}
	return j
}

// CompoundDataMovement captures theoretical data movement counts and
// operational intensity. Byte fields are DRAM bytes; TensorTransfers records
// the corresponding per-tensor DRAM traffic under keys A, B and C.
type CompoundDataMovement struct {
	ReadBytes            uint64            // input bytes read from DRAM
	WriteBytes           uint64            // output bytes written to DRAM
	TotalBytes           uint64            // ReadBytes + WriteBytes
	OperationalIntensity float64           // FLOPs per DRAM byte
	TensorTransfers      map[string]uint64 // DRAM bytes by tensor role
}

// AnalyticalDataMovement estimates DRAM traffic for a GEMM workload with
// extents B, M, N and K. The tile sizes M and N control reuse: without bypass,
// A and B are each read once; bypassing shared or L2 retention reloads A for
// each N tile and B for each M tile. C's compulsory output write is never
// removed by bypass. All sizes must be positive and byte arithmetic is checked.
func AnalyticalDataMovement(problem, tileFactors, dtypes map[string]int, bypassed map[string]bool) (CompoundDataMovement, error) {
	for name := range problem {
		if name != "B" && name != "M" && name != "N" && name != "K" {
			return CompoundDataMovement{}, fmt.Errorf("unknown problem dimension %q", name)
		}
	}
	for _, name := range []string{"B", "M", "N", "K"} {
		if problem[name] <= 0 {
			return CompoundDataMovement{}, fmt.Errorf("missing or non-positive problem extent %s", name)
		}
	}
	for name := range tileFactors {
		if name != "B" && name != "M" && name != "N" && name != "K" {
			return CompoundDataMovement{}, fmt.Errorf("unknown tile dimension %q", name)
		}
	}
	for _, name := range []string{"M", "N"} {
		if tileFactors[name] <= 0 {
			return CompoundDataMovement{}, fmt.Errorf("missing or non-positive tile extent %s", name)
		}
	}
	for name := range dtypes {
		if name != "A" && name != "B" && name != "C" {
			return CompoundDataMovement{}, fmt.Errorf("unknown tensor dtype %q", name)
		}
	}
	for _, name := range []string{"A", "B", "C"} {
		if dtypes[name] <= 0 {
			return CompoundDataMovement{}, fmt.Errorf("missing or non-positive dtype size for tensor %s", name)
		}
	}
	for name := range bypassed {
		if name != "A" && name != "B" && name != "C" {
			return CompoundDataMovement{}, fmt.Errorf("unknown bypass tensor %q", name)
		}
	}

	b, m, n, k := uint64(problem["B"]), uint64(problem["M"]), uint64(problem["N"]), uint64(problem["K"])
	mt, nt := uint64(tileFactors["M"]), uint64(tileFactors["N"])
	dtA, dtB, dtC := uint64(dtypes["A"]), uint64(dtypes["B"]), uint64(dtypes["C"])

	wholeA, err := checkedProduct(b, m, k, dtA)
	if err != nil {
		return CompoundDataMovement{}, fmt.Errorf("tensor A bytes: %w", err)
	}
	wholeB, err := checkedProduct(b, k, n, dtB)
	if err != nil {
		return CompoundDataMovement{}, fmt.Errorf("tensor B bytes: %w", err)
	}
	writesC, err := checkedProduct(b, m, n, dtC)
	if err != nil {
		return CompoundDataMovement{}, fmt.Errorf("tensor C bytes: %w", err)
	}
	aBytes, bBytes := wholeA, wholeB
	if bypassed["A"] {
		aBytes, err = checkedProduct(wholeA, ceilDivU64(n, nt))
		if err != nil {
			return CompoundDataMovement{}, fmt.Errorf("tensor A reload bytes: %w", err)
		}
	}
	if bypassed["B"] {
		bBytes, err = checkedProduct(wholeB, ceilDivU64(m, mt))
		if err != nil {
			return CompoundDataMovement{}, fmt.Errorf("tensor B reload bytes: %w", err)
		}
	}
	readBytes, err := checkedSum(aBytes, bBytes)
	if err != nil {
		return CompoundDataMovement{}, err
	}
	totalBytes, err := checkedSum(readBytes, writesC)
	if err != nil {
		return CompoundDataMovement{}, err
	}
	flops, err := checkedProduct(2, b, m, n, k)
	if err != nil {
		return CompoundDataMovement{}, fmt.Errorf("GEMM FLOPs: %w", err)
	}
	intensity := float64(flops) / float64(totalBytes)
	return CompoundDataMovement{
		ReadBytes: readBytes, WriteBytes: writesC, TotalBytes: totalBytes,
		OperationalIntensity: intensity,
		TensorTransfers:      map[string]uint64{"A": aBytes, "B": bBytes, "C": writesC},
	}, nil
}

func checkedProduct(values ...uint64) (uint64, error) {
	product := uint64(1)
	for _, value := range values {
		if value != 0 && product > math.MaxUint64/value {
			return 0, fmt.Errorf("byte or FLOP count overflows uint64")
		}
		product *= value
	}
	return product, nil
}

func checkedSum(values ...uint64) (uint64, error) {
	total := uint64(0)
	for _, value := range values {
		if total > math.MaxUint64-value {
			return 0, fmt.Errorf("byte count overflows uint64")
		}
		total += value
	}
	return total, nil
}

func ceilDivU64(n, d uint64) uint64 {
	q := n / d
	if n%d != 0 {
		q++
	}
	return q
}

// HardwareVerificationStatus represents the result of checking whether
// calibration qualification is available.
type HardwareVerificationStatus string

const (
	// VerificationPending means the workload has not been physically measured.
	VerificationPending HardwareVerificationStatus = "pending"
)

// HardwareCalibration holds the calibration status and latency. LatencyNs is
// [FloorUnavailable] whenever Status is VerificationPending; this lane does
// not measure device execution.
type HardwareCalibration struct {
	Status    HardwareVerificationStatus // current qualification state
	LatencyNs float64                    // nanoseconds, or FloorUnavailable
}

// HardwareProbe reports whether target has an available device. It does not
// execute a workload or establish performance qualification.
type HardwareProbe func(target string) (available bool, err error)

// VerifyHardwareExecution checks device availability with probe. This host-side
// package does not measure latency, so even an available device remains
// VerificationPending until a native calibration path is added.
func VerifyHardwareExecution(target string, probe HardwareProbe) (HardwareCalibration, error) {
	if probe == nil {
		return HardwareCalibration{Status: VerificationPending, LatencyNs: FloorUnavailable},
			fmt.Errorf("hardware verification on %s has no device probe: %w", target, cudaerrors.ErrHardwarePending)
	}
	available, err := probe(target)
	if err != nil {
		return HardwareCalibration{Status: VerificationPending, LatencyNs: FloorUnavailable},
			fmt.Errorf("hardware probe on %s: %w", target, err)
	}
	if !available {
		return HardwareCalibration{Status: VerificationPending, LatencyNs: FloorUnavailable},
			fmt.Errorf("hardware verification on %s: %w", target, cudaerrors.ErrHardwarePending)
	}
	return HardwareCalibration{
		Status:    VerificationPending,
		LatencyNs: FloorUnavailable,
	}, fmt.Errorf("device is available on %s but no calibration measurement exists: %w", target, cudaerrors.ErrHardwarePending)
}
