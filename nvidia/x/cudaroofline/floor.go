package cudaroofline

import (
	"fmt"
	"math"

	"github.com/tmc/lib/nvidia/x/cudaerrors"
	"github.com/tmc/lib/nvidia/x/cudalayout"
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

// TensorMovement contains a tensor's resident footprint, bytes read by lower
// consumers, and bytes written back by those consumers. A bypassed tensor has
// no footprint or accesses at that level.
type TensorMovement struct {
	ResidentBytes uint64
	ReadBytes     uint64
	WriteBytes    uint64
}

// HierarchyLevel contains the footprint and accesses at one mapped level.
// ResidentBytes is the sum of its retained tensor footprints, not GPU occupancy.
type HierarchyLevel struct {
	Level         cudalayout.StorageLevel
	ResidentBytes uint64
	Tensors       map[cudalayout.TensorRole]TensorMovement
}

// HierarchyMovement contains logical temporal GEMM traffic. Levels are ordered
// from the lowest retained storage level to DRAM. DRAM contains the total
// transfers at the DRAM boundary and the corresponding FLOPs per byte.
type HierarchyMovement struct {
	Levels []HierarchyLevel
	DRAM   CompoundDataMovement
}

// AnalyticalGEMMHierarchy computes footprints and transfers for a temporal
// GEMM mapping with C initially zero. Levels must increase and end at DRAM;
// orders at each level are outermost first and include every non-unit factor.
// All four B/M/N/K extents and positive A/B/C element byte sizes are required.
// Nonidentity spatial meshes are unsupported.
//
// Each retained buffer holds one projected tile until its outer-loop key
// changes. A/B fill on each tile visit. C initializes on its first visit,
// writes back on eviction or termination, and reloads on subsequent visits.
// Bypassed buffers consume zero capacity; transfers use the nearest retained
// upper level. Scalar computation accesses operands and accumulates C below
// the lowest listed buffer. Counts are algebraic, without enumerating extents.
//
// These are exact logical counts for this tile-retention contract. They do not
// model partial overlapping tiles, physical cache replacement, transactions,
// padding, spatial sharing, measured traffic, residency or execution latency.
func AnalyticalGEMMHierarchy(mapping cudalayout.MapSpace, dtypes map[cudalayout.TensorRole]int) (HierarchyMovement, error) {
	factors := mapping.Factorization
	levels := factors.Levels
	roles := []cudalayout.TensorRole{cudalayout.TensorA, cudalayout.TensorB, cudalayout.TensorC}
	dims := []string{"B", "M", "N", "K"}
	unsupported := func(reason string) (HierarchyMovement, error) {
		return HierarchyMovement{}, fmt.Errorf("GEMM hierarchy: %s: %w", reason, cudalayout.ErrUnsupported)
	}
	if len(mapping.Problem) != len(dims) {
		return unsupported("expected exactly B/M/N/K problem dimensions")
	}
	for _, dim := range dims {
		if mapping.Problem[dim] <= 0 {
			return unsupported("missing or non-positive problem dimension " + dim)
		}
	}
	if len(levels) == 0 || levels[len(levels)-1] != cudalayout.LevelDRAM {
		return unsupported("storage levels must end at DRAM")
	}
	listed := make(map[cudalayout.StorageLevel]bool, len(levels))
	for i, level := range levels {
		if level < cudalayout.LevelRegister || level > cudalayout.LevelDRAM || (i > 0 && level <= levels[i-1]) {
			return unsupported("storage levels must be known and strictly increasing")
		}
		listed[level] = true
	}
	for level := range factors.Factors {
		if !listed[level] {
			return unsupported("factors reference an unlisted storage level")
		}
	}
	if err := factors.Validate(mapping.Problem); err != nil {
		return HierarchyMovement{}, fmt.Errorf("GEMM hierarchy factors: %w", err)
	}
	for _, mesh := range []map[string]int{mapping.Spatial.CTAMesh, mapping.Spatial.WarpMesh} {
		for dim, count := range mesh {
			if mapping.Problem[dim] <= 0 || count != 1 {
				return unsupported("only identity spatial meshes are modeled")
			}
		}
	}
	if len(dtypes) != len(roles) {
		return unsupported("expected exactly A/B/C element byte sizes")
	}
	for _, role := range roles {
		if dtypes[role] <= 0 {
			return unsupported("missing or non-positive tensor byte size")
		}
	}
	for level, tensors := range mapping.Bypass.Bypassed {
		if !listed[level] {
			return unsupported("bypass references an unlisted storage level")
		}
		for role, bypassed := range tensors {
			if role < cudalayout.TensorA || role > cudalayout.TensorC || (level == cudalayout.LevelDRAM && bypassed) {
				return unsupported("unknown tensor role or bypassed DRAM tensor")
			}
		}
	}
	type loop struct {
		level int
		dim   string
		size  uint64
	}
	var loops []loop
	for level := range mapping.Permutation.Orders {
		if !listed[level] {
			return unsupported("permutation references an unlisted storage level")
		}
	}
	for i := len(levels) - 1; i >= 0; i-- {
		level := levels[i]
		seen := make(map[string]bool)
		for _, dim := range mapping.Permutation.Orders[level] {
			if mapping.Problem[dim] <= 0 || seen[dim] {
				return unsupported("unknown or repeated permutation dimension")
			}
			seen[dim] = true
			size := factors.Factors[level][dim]
			if size > 1 {
				loops = append(loops, loop{i, dim, uint64(size)})
			}
		}
		for _, dim := range dims {
			if factors.Factors[level][dim] > 1 && !seen[dim] {
				return unsupported("permutation omits a non-unit factor")
			}
		}
	}
	projects := func(role cudalayout.TensorRole, dim string) bool {
		return dim == "B" || (role == cudalayout.TensorA && dim != "N") ||
			(role == cudalayout.TensorB && dim != "M") || (role == cudalayout.TensorC && dim != "K")
	}
	report := HierarchyMovement{Levels: make([]HierarchyLevel, len(levels))}
	tile := map[string]uint64{"B": 1, "M": 1, "N": 1, "K": 1}
	for i, level := range levels {
		for dim, factor := range factors.Factors[level] {
			tile[dim] *= uint64(factor) // factors.Validate checked the full int product.
		}
		entry := HierarchyLevel{Level: level, Tensors: make(map[cudalayout.TensorRole]TensorMovement, len(roles))}
		for _, role := range roles {
			var bytes uint64
			if !mapping.Bypass.IsBypassed(level, role) {
				bytes = uint64(dtypes[role])
				for _, dim := range dims {
					if projects(role, dim) {
						var err error
						bytes, err = checkedProduct(bytes, tile[dim])
						if err != nil {
							return HierarchyMovement{}, fmt.Errorf("GEMM %s %s footprint: %w", level, role, err)
						}
					}
				}
			}
			entry.Tensors[role] = TensorMovement{ResidentBytes: bytes}
			var err error
			entry.ResidentBytes, err = checkedSum(entry.ResidentBytes, bytes)
			if err != nil {
				return HierarchyMovement{}, err
			}
		}
		report.Levels[i] = entry
	}
	charge := func(child int, role cudalayout.TensorRole, read, write uint64) error {
		for parent := child + 1; parent < len(levels); parent++ {
			if mapping.Bypass.IsBypassed(levels[parent], role) {
				continue
			}
			entry := report.Levels[parent].Tensors[role]
			var err error
			entry.ReadBytes, err = checkedSum(entry.ReadBytes, read)
			if err != nil {
				return err
			}
			entry.WriteBytes, err = checkedSum(entry.WriteBytes, write)
			if err != nil {
				return err
			}
			report.Levels[parent].Tensors[role] = entry
			return nil
		}
		return fmt.Errorf("GEMM hierarchy has no retained source for %s", role)
	}
	for i, level := range levels[:len(levels)-1] {
		for _, role := range roles {
			if mapping.Bypass.IsBypassed(level, role) {
				continue
			}
			last := -1
			distinct := uint64(1)
			for j, loop := range loops {
				if loop.level <= i {
					break
				}
				if projects(role, loop.dim) {
					last = j
					var err error
					distinct, err = checkedProduct(distinct, loop.size)
					if err != nil {
						return HierarchyMovement{}, err
					}
				}
			}
			runs := uint64(1)
			for j := 0; j <= last; j++ {
				var err error
				runs, err = checkedProduct(runs, loops[j].size)
				if err != nil {
					return HierarchyMovement{}, err
				}
			}
			bytes := report.Levels[i].Tensors[role].ResidentBytes
			read, err := checkedProduct(runs, bytes)
			if err != nil {
				return HierarchyMovement{}, err
			}
			var write uint64
			if role == cudalayout.TensorC {
				write = read
				read, err = checkedProduct(runs-distinct, bytes)
				if err != nil {
					return HierarchyMovement{}, err
				}
			}
			if err := charge(i, role, read, write); err != nil {
				return HierarchyMovement{}, err
			}
		}
	}
	macs, err := checkedProduct(uint64(mapping.Problem["B"]), uint64(mapping.Problem["M"]), uint64(mapping.Problem["N"]), uint64(mapping.Problem["K"]))
	if err != nil {
		return HierarchyMovement{}, err
	}
	outputs, err := checkedProduct(uint64(mapping.Problem["B"]), uint64(mapping.Problem["M"]), uint64(mapping.Problem["N"]))
	if err != nil {
		return HierarchyMovement{}, err
	}
	for _, role := range roles {
		read, err := checkedProduct(macs, uint64(dtypes[role]))
		if err != nil {
			return HierarchyMovement{}, err
		}
		var write uint64
		if role == cudalayout.TensorC {
			write = read
			read, err = checkedProduct(macs-outputs, uint64(dtypes[role]))
			if err != nil {
				return HierarchyMovement{}, err
			}
		}
		if err := charge(-1, role, read, write); err != nil {
			return HierarchyMovement{}, err
		}
	}
	dram := report.Levels[len(levels)-1]
	report.DRAM.TensorTransfers = make(map[string]uint64, len(roles))
	for i, role := range roles {
		entry := dram.Tensors[role]
		total, err := checkedSum(entry.ReadBytes, entry.WriteBytes)
		if err != nil {
			return HierarchyMovement{}, err
		}
		report.DRAM.TensorTransfers[[]string{"A", "B", "C"}[i]] = total
		report.DRAM.ReadBytes, err = checkedSum(report.DRAM.ReadBytes, entry.ReadBytes)
		if err != nil {
			return HierarchyMovement{}, err
		}
		report.DRAM.WriteBytes, err = checkedSum(report.DRAM.WriteBytes, entry.WriteBytes)
		if err != nil {
			return HierarchyMovement{}, err
		}
	}
	report.DRAM.TotalBytes, err = checkedSum(report.DRAM.ReadBytes, report.DRAM.WriteBytes)
	if err != nil {
		return HierarchyMovement{}, err
	}
	flops, err := checkedProduct(2, macs)
	if err != nil {
		return HierarchyMovement{}, err
	}
	report.DRAM.OperationalIntensity = float64(flops) / float64(report.DRAM.TotalBytes)
	return report, nil
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
