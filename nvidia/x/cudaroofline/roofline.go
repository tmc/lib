package cudaroofline

import (
	"errors"
	"fmt"
	"sort"
	"strings"
)

// Classification labels for evaluated kernel launches.
const (
	ClassMemoryBound   = "memory_bound"
	ClassComputeBound  = "compute_bound"
	ClassUnconstrained = "unconstrained"

	// ClassAboveCeiling marks a launch that attained more than the ceiling
	// allows. It is a statement about the ceiling, not about the kernel: the
	// ceiling is too low, describes memory the kernel does not touch, or the
	// byte count fed to Evaluate is wrong.
	ClassAboveCeiling = "above_ceiling"
)

// Residency says where the bytes a kernel reads live, and so which memory path
// the read takes. Bandwidth is a property of the path, not of the architecture:
// on a GB10 the same kernel reading the same bytes attains 1.43x more from
// device-resident memory than from memory read through host page tables.
type Residency string

const (
	// DeviceResident is memory the driver owns on the device: cudaMalloc,
	// cudaMallocPitch, a device-side cudaMemcpy destination.
	DeviceResident Residency = "device"

	// HostPageTableResident is memory mapped into the process address space
	// that the GPU reads through host page tables: an mmap of a file, or a
	// host allocation reached via pageableMemoryAccess. Weights mapped from a
	// model file are host-page-table resident unless something copied them.
	HostPageTableResident Residency = "host_page_table"
)

// Provenance records where a ceiling's numbers came from. It exists because a
// division by a number nobody measured produces a headroom nobody can collect.
type Provenance string

const (
	// Measured means someone ran a microbenchmark on the part named in the
	// ceiling's comment and this is what it reported.
	Measured Provenance = "measured"

	// Unverified means the number was carried from vendor documentation or
	// from prior code and has not been measured in this repository. Treat a
	// ratio computed against it as a hypothesis.
	Unverified Provenance = "unverified"
)

// LimitingAxis names which of a ceiling's two limits bound a launch, and so
// which of the ceiling's two numbers a classification was computed against. The
// axis is not a property of the hardware alone: it follows from the kernel's
// arithmetic intensity, because the roofline is the lower of a memory slope and
// a compute plateau and they cross at the knee.
type LimitingAxis string

const (
	// NoLimit means no ceiling was consulted: the launch had no duration, no
	// bytes, or no FLOPs, so Efficiency is zero and the classification is not
	// a reading of the hardware. It is the zero LimitingAxis.
	NoLimit LimitingAxis = ""

	// MemoryLimited means PeakMemoryGBps bound the launch: at this intensity
	// the memory slope lies below the compute plateau.
	MemoryLimited LimitingAxis = "memory"

	// ComputeLimited means PeakTFLOPS bound the launch.
	ComputeLimited LimitingAxis = "compute"
)

// Ceiling is the set of hardware limits a kernel is judged against. A ceiling
// is specific to both an architecture and a Residency, and records how its
// numbers were obtained.
//
// The zero Ceiling is not usable; build one with LookupCeiling or fill every
// field and check it with ValidateCeiling.
type Ceiling struct {
	Architecture string    // e.g. "sm_90", "sm_121"
	Residency    Residency // memory path the bandwidth figures describe

	PeakTFLOPS      float64 // Peak single-precision or target precision compute throughput in TFLOPS
	PeakMemoryGBps  float64 // Peak DRAM bandwidth in GB/s for this residency
	L2BandwidthGBps float64 // Peak L2 cache bandwidth in GB/s; zero if unknown

	// ComputeProvenance describes PeakTFLOPS. MemoryProvenance describes
	// PeakMemoryGBps and, when it is nonzero, L2BandwidthGBps.
	ComputeProvenance Provenance
	MemoryProvenance  Provenance
}

// provenanceFor returns the provenance of the limit on the given axis. A
// classification rests on one of the two limits, not on both, so this is the
// marking that has to travel with it.
func (c Ceiling) provenanceFor(axis LimitingAxis) Provenance {
	switch axis {
	case MemoryLimited:
		return c.MemoryProvenance
	case ComputeLimited:
		return c.ComputeProvenance
	}
	return ""
}

// ceilingKey identifies a table entry. Architecture alone cannot: it does not
// distinguish a cudaMalloc buffer from an mmap'd file.
type ceilingKey struct {
	arch      string
	residency Residency
}

// knownCeilings holds the ceilings this repository can name. Entries exist only
// for residencies that were actually characterized: there is deliberately no
// host-page-table entry for the discrete parts, because nobody here has
// measured what a host-page-table read costs on them, and guessing one is the
// defect this table exists to prevent.
//
// The GB10 entries are measured on an NVIDIA GB10 (48 SMs, 256-bit bus,
// 24 MiB L2, compute capability 12.1, pageableMemoryAccess=1,
// usesHostPageTables=1) with a grid-stride float4 read over a 4 GiB buffer,
// best-of-5 per run:
//
//	four unmodified control runs: 239.6, 239.0, 239.4, 238.8 GB/s (spread 0.33%, no drift)
//
//	2x2 {allocation} x {access pattern}:
//	                                    float4 grid-stride   Q8_0 34-byte narrow
//	  cudaMalloc                             232.4 GB/s           218.9 GB/s
//	  mmap(file, MAP_PRIVATE|MAP_POPULATE)   161.8 GB/s           162.4 GB/s
//
// Cross-validated end to end: a real inference workload ablated to
// host-page-table residency ran at 162.2 GB/s aggregate, inside the synthetic
// 161.8-162.4 GB/s band.
//
// Each entry below carries the highest bandwidth measured for its residency, so
// a kernel that exceeds it refutes the measurement rather than the rounding.
// The 273 GB/s figure quoted for this part is 14% above anything measured here
// and 68% above what host-page-table reads face; it is not used.
//
// The two residencies did not get equal search depth, and "highest measured"
// therefore means something stronger for one than the other:
//
//	device (239.6 GB/s)  best of a sweep of 32 launch geometries
//	                     (48..2304 blocks x 128..1024 threads), best-of-5 each.
//	mmap   (162.4 GB/s)  one geometry, best-of-5. No geometry sweep.
//
// The device ceiling survived a geometry search; the mmap ceiling did not. What
// warrants the mmap row is a second, independent line of evidence rather than
// search depth: the 162.2 GB/s end-to-end agreement above. A synthetic figure
// that reproduces a full workload's aggregate to 0.2% has earned the number by
// a different route.
//
// The consequence, which is the thing to carry away: if the mmap ceiling is
// ever revised, it will move UP, because an untried geometry can only read
// faster than the one tried. Efficiencies computed against it are therefore
// biased high, and revising it would move above-ceiling readings back toward
// normal. The asymmetry runs in the direction that manufactures refutations,
// not the direction that hides them.
//
// The GB10 compute peaks are unverified and the GB10 L2 bandwidth was not
// measured at all, so it is left zero.
var knownCeilings = map[ceilingKey]Ceiling{
	{"sm_70", DeviceResident}: {
		Architecture: "sm_70", Residency: DeviceResident,
		PeakTFLOPS: 15.7, PeakMemoryGBps: 900.0, L2BandwidthGBps: 2500.0,
		ComputeProvenance: Unverified, MemoryProvenance: Unverified,
	},
	{"sm_75", DeviceResident}: {
		Architecture: "sm_75", Residency: DeviceResident,
		PeakTFLOPS: 14.2, PeakMemoryGBps: 672.0, L2BandwidthGBps: 1800.0,
		ComputeProvenance: Unverified, MemoryProvenance: Unverified,
	},
	{"sm_80", DeviceResident}: {
		Architecture: "sm_80", Residency: DeviceResident,
		PeakTFLOPS: 19.5, PeakMemoryGBps: 1935.0, L2BandwidthGBps: 5120.0,
		ComputeProvenance: Unverified, MemoryProvenance: Unverified,
	},
	{"sm_86", DeviceResident}: {
		Architecture: "sm_86", Residency: DeviceResident,
		PeakTFLOPS: 35.6, PeakMemoryGBps: 912.0, L2BandwidthGBps: 2800.0,
		ComputeProvenance: Unverified, MemoryProvenance: Unverified,
	},
	{"sm_89", DeviceResident}: {
		Architecture: "sm_89", Residency: DeviceResident,
		PeakTFLOPS: 82.6, PeakMemoryGBps: 1008.0, L2BandwidthGBps: 3600.0,
		ComputeProvenance: Unverified, MemoryProvenance: Unverified,
	},
	{"sm_90", DeviceResident}: {
		Architecture: "sm_90", Residency: DeviceResident,
		PeakTFLOPS: 67.0, PeakMemoryGBps: 3350.0, L2BandwidthGBps: 12000.0,
		ComputeProvenance: Unverified, MemoryProvenance: Unverified,
	},
	{"sm_121", DeviceResident}: {
		Architecture: "sm_121", Residency: DeviceResident,
		PeakTFLOPS: 40.0, PeakMemoryGBps: 239.6,
		ComputeProvenance: Unverified, MemoryProvenance: Measured,
	},
	{"sm_121", HostPageTableResident}: {
		Architecture: "sm_121", Residency: HostPageTableResident,
		PeakTFLOPS: 40.0, PeakMemoryGBps: 162.4,
		ComputeProvenance: Unverified, MemoryProvenance: Measured,
	},
	{"sm_121a", DeviceResident}: {
		Architecture: "sm_121a", Residency: DeviceResident,
		PeakTFLOPS: 40.0, PeakMemoryGBps: 239.6,
		ComputeProvenance: Unverified, MemoryProvenance: Measured,
	},
	{"sm_121a", HostPageTableResident}: {
		Architecture: "sm_121a", Residency: HostPageTableResident,
		PeakTFLOPS: 40.0, PeakMemoryGBps: 162.4,
		ComputeProvenance: Unverified, MemoryProvenance: Measured,
	},
}

// LookupCeiling returns the known ceiling for arch (e.g. "sm_90", "sm_121") at
// the given residency. It fails rather than substituting another residency's
// number: on GB10 those differ by 1.43x.
func LookupCeiling(arch string, residency Residency) (Ceiling, error) {
	key := ceilingKey{strings.ToLower(strings.TrimSpace(arch)), residency}
	if c, ok := knownCeilings[key]; ok {
		return c, nil
	}
	if known := residenciesFor(key.arch); len(known) > 0 {
		return Ceiling{}, fmt.Errorf("cudaroofline: no ceiling for %q at %q residency (have: %s)",
			arch, residency, strings.Join(known, ", "))
	}
	return Ceiling{}, fmt.Errorf("cudaroofline: unknown architecture %q", arch)
}

// residenciesFor returns the residencies the table knows for arch, sorted.
func residenciesFor(arch string) []string {
	var out []string
	for k := range knownCeilings {
		if k.arch == arch {
			out = append(out, string(k.residency))
		}
	}
	sort.Strings(out)
	return out
}

// Model evaluates kernel operational points against one ceiling.
type Model struct {
	Ceiling Ceiling
}

// NewModel creates a roofline model for arch at the given residency.
func NewModel(arch string, residency Residency) (*Model, error) {
	c, err := LookupCeiling(arch, residency)
	if err != nil {
		return nil, err
	}
	return NewModelWithCeiling(c), nil
}

// NewModelWithCeiling creates a roofline model with a caller-supplied ceiling,
// for a part or a memory path the table does not cover. The caller is
// responsible for the ceiling's provenance; ValidateCeiling checks that it is
// at least stated.
func NewModelWithCeiling(c Ceiling) *Model {
	return &Model{Ceiling: c}
}

// KernelPoint records the arithmetic intensity, throughput, and roofline
// classification of a single kernel launch.
type KernelPoint struct {
	Name           string  // Kernel identifier or symbol name
	FLOPs          uint64  // Floating-point operations performed
	Bytes          uint64  // Bytes transferred (DRAM traffic)
	DurationNs     uint64  // Device execution span in nanoseconds
	Intensity      float64 // Operational arithmetic intensity (FLOPs / Byte)
	AttainedTFLOPS float64 // Attained throughput in TFLOPS
	AttainedGBps   float64 // Attained bandwidth in GB/s

	// Classification is one of the Class constants.
	Classification string

	// Efficiency is the ratio of attained throughput to the ceiling at this
	// intensity. It is not clamped: a value above 1.0 means the launch did
	// something the ceiling says is impossible, which is the most informative
	// reading this package produces, and Classification is ClassAboveCeiling.
	Efficiency float64

	// LimitingAxis names the single limit Efficiency was divided by: the
	// memory slope for a memory-bound launch, the compute plateau for a
	// compute-bound one, and for ClassAboveCeiling the limit that was
	// exceeded. Classification rests on that limit and on no other number.
	LimitingAxis LimitingAxis

	// CeilingProvenance is the provenance of the limit named by LimitingAxis.
	// It travels here so that a consumer holding only a KernelPoint can tell a
	// ratio against a measured ceiling from a ratio against a datasheet
	// figure: Unverified means Efficiency and Classification are hypotheses
	// about the hardware, not readings of it. On GB10 a compute-bound verdict
	// is always Unverified, because nobody has measured that part's FLOPS
	// ceiling, while a memory-bound verdict is Measured.
	//
	// It is empty exactly when LimitingAxis is NoLimit.
	CeilingProvenance Provenance
}

// KneeIntensity returns the operational arithmetic intensity where the memory
// ceiling meets the peak compute ceiling (TFLOPS / (PeakMemoryGBps / 1000)).
func (m *Model) KneeIntensity() float64 {
	if m.Ceiling.PeakMemoryGBps <= 0 {
		return 0
	}
	// PeakTFLOPS / (PeakMemoryGBps / 1000) = (PeakTFLOPS * 1000) / PeakMemoryGBps
	return (m.Ceiling.PeakTFLOPS * 1000.0) / m.Ceiling.PeakMemoryGBps
}

// CeilingAt reports the maximum attainable TFLOPS at the given arithmetic intensity.
func (m *Model) CeilingAt(intensity float64) float64 {
	c, _ := m.ceilingAt(intensity)
	return c
}

// ceilingAt reports the maximum attainable TFLOPS at the given intensity and
// which of the ceiling's two limits produced it. The two are returned together
// because a ratio against the roofline is a ratio against exactly one of them,
// and which one decides what the ratio means.
func (m *Model) ceilingAt(intensity float64) (float64, LimitingAxis) {
	if intensity <= 0 {
		return 0, NoLimit
	}
	if memCeiling := intensity * (m.Ceiling.PeakMemoryGBps / 1000.0); memCeiling < m.Ceiling.PeakTFLOPS {
		return memCeiling, MemoryLimited
	}
	return m.Ceiling.PeakTFLOPS, ComputeLimited
}

// Evaluate analyzes a kernel execution given its FLOP count, memory bytes
// transferred, and execution duration in nanoseconds.
//
// The bytes must be counted on the same memory path the model's ceiling
// describes; a launch judged against the wrong residency reports a headroom
// that cannot be collected.
func (m *Model) Evaluate(kernelName string, flops uint64, bytesTransferred uint64, durationNs uint64) KernelPoint {
	pt := KernelPoint{
		Name:           kernelName,
		FLOPs:          flops,
		Bytes:          bytesTransferred,
		DurationNs:     durationNs,
		Classification: ClassUnconstrained,
	}

	if durationNs == 0 || bytesTransferred == 0 {
		return pt
	}

	// Intensity = FLOPs / Byte
	pt.Intensity = float64(flops) / float64(bytesTransferred)

	// Attained TFLOPS = (FLOPs / 1e12) / (durationNs / 1e9) = FLOPs / (durationNs * 1000)
	pt.AttainedTFLOPS = float64(flops) / (float64(durationNs) * 1000.0)

	// Attained GB/s = (Bytes / 1e9) / (durationNs / 1e9) = Bytes / durationNs
	pt.AttainedGBps = float64(bytesTransferred) / float64(durationNs)

	ceilingTFLOPS, axis := m.ceilingAt(pt.Intensity)
	if ceilingTFLOPS > 0 {
		pt.Efficiency = pt.AttainedTFLOPS / ceilingTFLOPS
		pt.LimitingAxis = axis
		pt.CeilingProvenance = m.Ceiling.provenanceFor(axis)
	}

	switch {
	case pt.Efficiency > 1.0:
		// The ceiling is refuted. Say so instead of naming a bottleneck
		// derived from it.
		pt.Classification = ClassAboveCeiling
	case pt.Efficiency < unconstrainedEfficiency:
		// Not saturating either bottleneck: latency-bound, or the byte and
		// FLOP counts do not describe what the kernel did.
		pt.Classification = ClassUnconstrained
	case pt.LimitingAxis == MemoryLimited:
		pt.Classification = ClassMemoryBound
	default:
		pt.Classification = ClassComputeBound
	}
	return pt
}

// unconstrainedEfficiency is the fraction of the binding ceiling below which
// Evaluate reports ClassUnconstrained instead of naming a bottleneck.
//
// The value is chosen, not derived. Nothing in this repository measures where a
// kernel stops saturating its bottleneck and starts being latency-bound, and
// this package has no such measurement to point at; 0.35 is a reporting cutoff
// picked to keep obviously idle launches from being filed as bound by a limit
// they are nowhere near. Moving it moves the boundary between ClassUnconstrained
// and the two bound classes, and changes nothing else. Efficiency,
// LimitingAxis, and CeilingProvenance travel on every KernelPoint, so a caller
// who wants a different cutoff can reclassify from the ratio.
const unconstrainedEfficiency = 0.35

// ValidateCeiling returns an error if a ceiling is unusable: a non-positive
// limit, or a limit whose provenance is not stated. An unstated provenance is
// an error because the number's kind determines what a ratio against it means.
func ValidateCeiling(c Ceiling) error {
	if c.PeakTFLOPS <= 0 {
		return errors.New("cudaroofline: PeakTFLOPS must be positive")
	}
	if c.PeakMemoryGBps <= 0 {
		return errors.New("cudaroofline: PeakMemoryGBps must be positive")
	}
	switch c.Residency {
	case DeviceResident, HostPageTableResident:
	default:
		return fmt.Errorf("cudaroofline: unknown residency %q", c.Residency)
	}
	if err := validProvenance(c.ComputeProvenance); err != nil {
		return fmt.Errorf("cudaroofline: ComputeProvenance: %w", err)
	}
	if err := validProvenance(c.MemoryProvenance); err != nil {
		return fmt.Errorf("cudaroofline: MemoryProvenance: %w", err)
	}
	return nil
}

func validProvenance(p Provenance) error {
	switch p {
	case Measured, Unverified:
		return nil
	}
	return fmt.Errorf("unknown provenance %q", p)
}
