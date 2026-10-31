package cudaroofline

import (
	"errors"
	"fmt"
	"maps"
	"math"
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

// PrecisionPeak is a caller-supplied dense floating-point compute ceiling for
// one Precision on the Architecture, Residency, DeviceName, and MaxSMClockMHz
// named by its enclosing Ceiling.
//
// Provenance and Citation are caller-supplied assertions recorded for
// auditability; they do not authenticate hardware measurements. Sparse (2:4
// structured sparsity) peaks are rejected so a 2x sparse datasheet figure
// cannot be applied to dense arithmetic.
type PrecisionPeak struct {
	PeakTFLOPS float64    `json:"peak_tflops"`
	Provenance Provenance `json:"provenance"`
	Citation   string     `json:"citation,omitempty"`
	Sparse     bool       `json:"sparse,omitempty"`
}

// Ceiling is the set of hardware limits a kernel is judged against. A ceiling
// is specific to both an architecture and a Residency, and records how its
// numbers were obtained.
//
// The zero Ceiling is not usable; build one with LookupCeiling or fill every
// field and check it with ValidateCeiling.
type Ceiling struct {
	Architecture string    // e.g. "sm_90", "sm_121"
	Residency    Residency // memory path the bandwidth figures describe

	// Device is the product name the numbers were measured on, as NVML and
	// nvidia-smi report it (e.g. "NVIDIA H200"), for an entry from
	// LookupDeviceCeiling. It is empty for an architecture-wide entry.
	// It records where a table entry came from; DeviceName below is the
	// caller's assertion that qualifies PrecisionPeaks.
	Device string

	// DeviceName and MaxSMClockMHz name the specific SKU and maximum SM clock
	// (matching cudacensus.Identity) that qualify PrecisionPeaks. They are
	// caller-supplied assertions checked for consistency, not hardware
	// authentication.
	DeviceName    string `json:"device_name,omitempty"`
	MaxSMClockMHz uint32 `json:"max_sm_clock_mhz,omitempty"`

	PeakTFLOPS      float64 // Peak FP32 non-tensor compute throughput in TFLOPS
	PeakMemoryGBps  float64 // Peak DRAM bandwidth in GB/s for this residency
	L2BandwidthGBps float64 // Peak L2 cache bandwidth in GB/s; zero if unknown

	// ComputeProvenance describes PeakTFLOPS. MemoryProvenance describes
	// PeakMemoryGBps and, when it is nonzero, L2BandwidthGBps.
	ComputeProvenance Provenance
	MemoryProvenance  Provenance

	// PrecisionPeaks holds optional caller-supplied dense floating-point
	// peaks keyed by Precision for the single Architecture, Residency,
	// DeviceName, and MaxSMClockMHz of this Ceiling.
	PrecisionPeaks map[Precision]PrecisionPeak `json:"precision_peaks,omitempty"`
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
		c.PrecisionPeaks = maps.Clone(c.PrecisionPeaks)
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

// knownDeviceCeilings holds ceilings for one product rather than one
// architecture. An architecture does not name a memory system: the H100 SXM
// and the H200 are both sm_90 and differ by 1.4x in datasheet bandwidth, so
// the sm_90 row above is right for at most one of them. Keys are product
// names as NVML reports them, compared by normalizeDevice.
//
// NVIDIA H200 (SXM, 141 GB HBM3e, 132 SMs; 4.8 TB/s datasheet), measured
// 2026-10-22 in the Open Kernel Challenge campaign
// (h200-challenge-groundwork/methods/workload.py:67-83,
// ATTENTION-CAMPAIGN.md:27): torch copy_ of 128 Mi fp32 elements, counted as
// read plus write (1 GiB moved), timed as a CUDA-graph replay between events
// with a 512 MiB cold-L2 flush before each sample; median of 20 samples:
//
//	2740.48 GB/s effective copy
//
// This is a weak ceiling in three ways, all of which make it low: it is one
// kernel at one geometry, a median rather than a best, and the span includes
// the graph replay intercept (about 4 us of about 392 us). It is 57% of the
// datasheet figure. Expect it to move up when a read-only sweep is run, and
// read an efficiency above 1.0 against it as a statement about the ceiling.
// The highest native DRAM rate profiled in the same campaign was 1501.5 GB/s
// (ncu dram__bytes, ATTENTION-CAMPAIGN.md). The same campaign timed a 4096^3
// BF16 torch.mm at about 778 TFLOP/s; that is a tensor-core figure and has no
// place in the FP32 PeakTFLOPS below, which is the datasheet's 67 TFLOPS.
//
// NVIDIA GB10 carries the sm_121 measurements documented on knownCeilings.
// deviceKey identifies a knownDeviceCeilings entry by normalized product name.
type deviceKey struct {
	name      string
	residency Residency
}

var knownDeviceCeilings = map[deviceKey]Ceiling{
	{"nvidia h200", DeviceResident}: {
		Architecture: "sm_90", Residency: DeviceResident, Device: "NVIDIA H200",
		PeakTFLOPS: 67.0, PeakMemoryGBps: 2740.48,
		ComputeProvenance: Unverified, MemoryProvenance: Measured,
	},
	{"nvidia gb10", DeviceResident}: {
		Architecture: "sm_121", Residency: DeviceResident, Device: "NVIDIA GB10",
		PeakTFLOPS: 40.0, PeakMemoryGBps: 239.6,
		ComputeProvenance: Unverified, MemoryProvenance: Measured,
	},
	{"nvidia gb10", HostPageTableResident}: {
		Architecture: "sm_121", Residency: HostPageTableResident, Device: "NVIDIA GB10",
		PeakTFLOPS: 40.0, PeakMemoryGBps: 162.4,
		ComputeProvenance: Unverified, MemoryProvenance: Measured,
	},
}

// normalizeDevice folds case and runs of spaces, so "NVIDIA  H200" and
// "nvidia h200" name the same product. It does not drop words: "NVIDIA H200
// NVL" is a different part with a different power limit.
func normalizeDevice(name string) string {
	return strings.ToLower(strings.Join(strings.Fields(name), " "))
}

// LookupDeviceCeiling returns the ceiling measured on the named product (as
// NVML, nvidia-smi or cudaGetDeviceProperties report it) at the given
// residency. It never falls back to the architecture-wide table: a part the
// table does not name has no measured ceiling here, and the caller should
// measure one or choose LookupCeiling knowingly.
func LookupDeviceCeiling(device string, residency Residency) (Ceiling, error) {
	key := deviceKey{normalizeDevice(device), residency}
	if c, ok := knownDeviceCeilings[key]; ok {
		return c, nil
	}
	var residencies []string
	devices := map[string]bool{}
	for k, c := range knownDeviceCeilings {
		if k.name == key.name {
			residencies = append(residencies, string(k.residency))
		}
		devices[c.Device] = true
	}
	if len(residencies) > 0 {
		sort.Strings(residencies)
		return Ceiling{}, fmt.Errorf("cudaroofline: no ceiling for %q at %q residency (have: %s)",
			device, residency, strings.Join(residencies, ", "))
	}
	names := make([]string, 0, len(devices))
	for d := range devices {
		names = append(names, d)
	}
	sort.Strings(names)
	return Ceiling{}, fmt.Errorf("cudaroofline: no measured ceiling for device %q (have: %s)", device, strings.Join(names, ", "))
}

// Model evaluates kernel operational points against one ceiling.
//
// Ceiling is the single authoritative state for the model. NewModelWithCeiling
// defensively clones Ceiling.PrecisionPeaks so mutating the caller's input map
// does not alter the model. Explicit mutations to Model.Ceiling or
// Model.Ceiling.PrecisionPeaks affect future reads consistently across all
// Model methods; Model is not safe for concurrent mutation and read.
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
// for a part or a memory path the table does not cover. It defensively clones
// c.PrecisionPeaks so mutating the caller's map after construction does not
// mutate the model. The caller is responsible for the ceiling's provenance;
// ValidateCeiling checks that it is at least stated.
func NewModelWithCeiling(c Ceiling) *Model {
	c.PrecisionPeaks = maps.Clone(c.PrecisionPeaks)
	return &Model{Ceiling: c}
}

// PeakTFLOPSFor returns the validated floating-point compute peak and
// provenance for p from m.Ceiling.
//
// For FP32 it returns m.Ceiling.PeakTFLOPS and m.Ceiling.ComputeProvenance
// (and verifies that m.Ceiling.PrecisionPeaks, if populated, is coherent and
// does not contradict Ceiling.PeakTFLOPS). For other floating-point precisions
// (FP64, FP16, TF32Tensor, FP16Tensor, BF16Tensor, FP8Tensor) it requires a
// validated coherent m.Ceiling.PrecisionPeaks entry qualified by
// m.Ceiling.Architecture, m.Ceiling.DeviceName, and m.Ceiling.MaxSMClockMHz,
// and never falls back to Ceiling.PeakTFLOPS. INT8Tensor is an integer TOPS
// format and is always refused here.
func (m *Model) PeakTFLOPSFor(p Precision) (float64, Provenance, error) {
	if m == nil {
		return 0, "", errors.New("cudaroofline: nil model")
	}
	if err := validPrecision(p); err != nil {
		return 0, "", err
	}
	if !p.IsFloatingPoint() {
		return 0, "", fmt.Errorf("cudaroofline: no %s peak: %s measures integer TOPS, not floating-point TFLOPS (the model holds an FP32 non-tensor peak only)", p, p)
	}
	if p == FP32 {
		if m.Ceiling.PeakTFLOPS <= 0 || math.IsNaN(m.Ceiling.PeakTFLOPS) || math.IsInf(m.Ceiling.PeakTFLOPS, 0) {
			return 0, "", errors.New("cudaroofline: no finite positive FP32 peak in the model")
		}
		if err := validProvenance(m.Ceiling.ComputeProvenance); err != nil {
			return 0, "", errors.New("cudaroofline: FP32 peak has no recognized provenance")
		}
		if len(m.Ceiling.PrecisionPeaks) > 0 {
			if err := validatePrecisionPeaks(m.Ceiling); err != nil {
				return 0, "", err
			}
		}
		return m.Ceiling.PeakTFLOPS, m.Ceiling.ComputeProvenance, nil
	}
	if len(m.Ceiling.PrecisionPeaks) == 0 {
		return 0, "", fmt.Errorf("cudaroofline: no %s peak: the model holds an FP32 non-tensor peak only", p)
	}
	if err := validatePrecisionPeaks(m.Ceiling); err != nil {
		return 0, "", err
	}
	pk, ok := m.Ceiling.PrecisionPeaks[p]
	if !ok {
		return 0, "", fmt.Errorf("cudaroofline: no %s peak in model for %s (%s at %d MHz)",
			p, m.Ceiling.DeviceName, m.Ceiling.Architecture, m.Ceiling.MaxSMClockMHz)
	}
	return pk.PeakTFLOPS, pk.Provenance, nil
}

// CeilingAtFor reports the maximum attainable floating-point TFLOPS for
// precision p at the given arithmetic intensity (FLOPs / Byte). It bounds the
// memory slope against p's own validated compute peak from m.PeakTFLOPSFor(p),
// never substituting m.Ceiling.PeakTFLOPS for a non-FP32 precision.
func (m *Model) CeilingAtFor(p Precision, intensity float64) (float64, error) {
	peak, _, err := m.PeakTFLOPSFor(p)
	if err != nil {
		return 0, err
	}
	if m.Ceiling.PeakMemoryGBps <= 0 || math.IsNaN(m.Ceiling.PeakMemoryGBps) || math.IsInf(m.Ceiling.PeakMemoryGBps, 0) {
		return 0, errors.New("cudaroofline: PeakMemoryGBps must be finite and positive")
	}
	if err := validProvenance(m.Ceiling.MemoryProvenance); err != nil {
		return 0, fmt.Errorf("cudaroofline: MemoryProvenance: %w", err)
	}
	if intensity <= 0 || math.IsNaN(intensity) || math.IsInf(intensity, 0) {
		return 0, nil
	}
	if memCeiling := intensity * (m.Ceiling.PeakMemoryGBps / 1000.0); memCeiling < peak {
		return memCeiling, nil
	}
	return peak, nil
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

// ValidateCeiling returns an error if a ceiling is unusable: a nonfinite or non-positive
// limit, or a limit whose provenance is not stated. An unstated provenance is
// an error because the number's kind determines what a ratio against it means.
// When PrecisionPeaks is populated, it also checks that Architecture,
// DeviceName, and MaxSMClockMHz are stated and that every entry is a dense
// floating-point TFLOPS peak with recognized provenance and non-empty citation.
func ValidateCeiling(c Ceiling) error {
	if c.PeakTFLOPS <= 0 || math.IsNaN(c.PeakTFLOPS) || math.IsInf(c.PeakTFLOPS, 0) {
		return errors.New("cudaroofline: PeakTFLOPS must be finite and positive")
	}
	if c.PeakMemoryGBps <= 0 || math.IsNaN(c.PeakMemoryGBps) || math.IsInf(c.PeakMemoryGBps, 0) {
		return errors.New("cudaroofline: PeakMemoryGBps must be finite and positive")
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
	if len(c.PrecisionPeaks) > 0 {
		if err := validatePrecisionPeaks(c); err != nil {
			return err
		}
	}
	return nil
}

func validatePrecisionPeaks(c Ceiling) error {
	if !isCanonicalSMArch(c.Architecture) {
		return fmt.Errorf("cudaroofline: PrecisionPeaks requires canonical sm_XY architecture, got %q", c.Architecture)
	}
	if strings.TrimSpace(c.DeviceName) == "" {
		return errors.New("cudaroofline: DeviceName is required when PrecisionPeaks are set")
	}
	if c.MaxSMClockMHz == 0 {
		return errors.New("cudaroofline: MaxSMClockMHz must be positive when PrecisionPeaks are set")
	}
	keys := make([]string, 0, len(c.PrecisionPeaks))
	for p := range c.PrecisionPeaks {
		keys = append(keys, string(p))
	}
	sort.Strings(keys)
	for _, k := range keys {
		p := Precision(k)
		pk := c.PrecisionPeaks[p]
		if err := validPrecision(p); err != nil {
			return err
		}
		if !p.IsFloatingPoint() {
			return fmt.Errorf("cudaroofline: PrecisionPeaks[%q]: %s measures integer TOPS, not floating-point TFLOPS", p, p)
		}
		if pk.Sparse {
			return fmt.Errorf("cudaroofline: PrecisionPeaks[%q]: sparse (2:4) peaks are unsupported; ceiling requires dense (1:1) peaks", p)
		}
		if pk.PeakTFLOPS <= 0 || math.IsNaN(pk.PeakTFLOPS) || math.IsInf(pk.PeakTFLOPS, 0) {
			return fmt.Errorf("cudaroofline: PrecisionPeaks[%q]: PeakTFLOPS must be finite and positive", p)
		}
		if err := validProvenance(pk.Provenance); err != nil {
			return fmt.Errorf("cudaroofline: PrecisionPeaks[%q]: Provenance: %w", p, err)
		}
		if strings.TrimSpace(pk.Citation) == "" {
			return fmt.Errorf("cudaroofline: PrecisionPeaks[%q]: Citation is required", p)
		}
		if p == FP32 {
			if pk.PeakTFLOPS != c.PeakTFLOPS || pk.Provenance != c.ComputeProvenance {
				return fmt.Errorf("cudaroofline: PrecisionPeaks[%q] (%v, %q) disagrees with Ceiling.PeakTFLOPS (%v, %q)",
					p, pk.PeakTFLOPS, pk.Provenance, c.PeakTFLOPS, c.ComputeProvenance)
			}
		}
	}
	return nil
}

func isCanonicalSMArch(arch string) bool {
	if !strings.HasPrefix(arch, "sm_") {
		return false
	}
	body := arch[len("sm_"):]
	if strings.HasSuffix(body, "a") {
		body = body[:len(body)-1]
	}
	if len(body) < 2 || body[0] == '0' {
		return false
	}
	for i := 0; i < len(body); i++ {
		if body[i] < '0' || body[i] > '9' {
			return false
		}
	}
	return true
}

func validProvenance(p Provenance) error {
	switch p {
	case Measured, Unverified:
		return nil
	}
	return fmt.Errorf("unknown provenance %q", p)
}
