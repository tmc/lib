package cudaroofline

import (
	"fmt"

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

// CompoundDataMovement captures theoretical data movement counts and operational intensity.
type CompoundDataMovement struct {
	ReadBytes            uint64
	WriteBytes           uint64
	TotalBytes           uint64
	OperationalIntensity float64
	TensorTransfers      map[string]uint64
}

// AnalyticalDataMovement computes exact theoretical DRAM transfer counts and
// operational intensity for a tensor contraction (e.g. GEMM) with given problem
// extents, tile factors, element byte sizes, and datatype bypass settings.
func AnalyticalDataMovement(problem map[string]int, tileFactors map[string]int, dtypes map[string]int, bypassed map[string]bool) CompoundDataMovement {
	b := uint64(problem["B"])
	if b <= 0 {
		b = 1
	}
	m := uint64(problem["M"])
	n := uint64(problem["N"])
	k := uint64(problem["K"])

	mt := uint64(tileFactors["M"])
	if mt <= 0 {
		mt = 1
	}
	nt := uint64(tileFactors["N"])
	if nt <= 0 {
		nt = 1
	}

	dtA := uint64(dtypes["A"])
	if dtA <= 0 {
		dtA = 2
	}
	dtB := uint64(dtypes["B"])
	if dtB <= 0 {
		dtB = 2
	}
	dtC := uint64(dtypes["C"])
	if dtC <= 0 {
		dtC = 4
	}

	// FLOPs = 2 * B * M * N * K
	flops := 2 * b * m * n * k

	transfers := make(map[string]uint64)
	var readBytes, writeBytes uint64

	// Tensor A: B x M x K. Reused across N / N_t outer tiles.
	if !bypassed["A"] {
		nTiles := (n + nt - 1) / nt
		transfersA := b * m * k * nTiles * dtA
		transfers["A"] = transfersA
		readBytes += transfersA
	}

	// Tensor B: B x K x N. Reused across M / M_t outer tiles.
	if !bypassed["B"] {
		mTiles := (m + mt - 1) / mt
		transfersB := b * k * n * mTiles * dtB
		transfers["B"] = transfersB
		readBytes += transfersB
	}

	// Tensor C: B x M x N. Accumulated and written to DRAM.
	if !bypassed["C"] {
		writesC := b * m * n * dtC
		transfers["C"] = writesC
		writeBytes += writesC
	}

	totalBytes := readBytes + writeBytes
	var opIntensity float64
	if totalBytes > 0 {
		opIntensity = float64(flops) / float64(totalBytes)
	}

	return CompoundDataMovement{
		ReadBytes:            readBytes,
		WriteBytes:           writeBytes,
		TotalBytes:           totalBytes,
		OperationalIntensity: opIntensity,
		TensorTransfers:      transfers,
	}
}

// HardwareVerificationStatus represents calibration qualification status.
type HardwareVerificationStatus string

const (
	VerificationPending  HardwareVerificationStatus = "pending"
	VerificationMeasured HardwareVerificationStatus = "measured"
)

// HardwareCalibration holds hardware verification outcome.
type HardwareCalibration struct {
	Status    HardwareVerificationStatus
	LatencyNs float64
}

// VerifyHardwareExecution gates execution latency measurement on physical hardware availability.
// Without an accessible, probed GPU, it returns status VerificationPending and wraps
// cudaerrors.ErrHardwarePending rather than emitting synthesized execution times.
func VerifyHardwareExecution(target string) (HardwareCalibration, error) {
	return HardwareCalibration{
		Status:    VerificationPending,
		LatencyNs: FloorUnavailable,
	}, fmt.Errorf("hardware verification on %s: %w", target, cudaerrors.ErrHardwarePending)
}
