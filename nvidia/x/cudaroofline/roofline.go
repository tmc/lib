package cudaroofline

import (
	"errors"
	"fmt"
	"strings"
)

// Classification labels for evaluated kernel launches.
const (
	ClassMemoryBound   = "memory_bound"
	ClassComputeBound  = "compute_bound"
	ClassUnconstrained = "unconstrained"
)

// Ceiling represents the theoretical or empirical hardware ceilings of a GPU device.
type Ceiling struct {
	Architecture    string  // e.g. "sm_90", "sm_121"
	PeakTFLOPS      float64 // Peak single-precision or target precision compute throughput in TFLOPS
	PeakMemoryGBps  float64 // Peak DRAM bandwidth in GB/s
	L2BandwidthGBps float64 // Peak L2 cache bandwidth in GB/s
}

// Known hardware ceilings for standard NVIDIA architectures.
// Values represent theoretical device limits for typical configurations.
var knownCeilings = map[string]Ceiling{
	"sm_70": {
		Architecture:    "sm_70",
		PeakTFLOPS:      15.7,
		PeakMemoryGBps:  900.0,
		L2BandwidthGBps: 2500.0,
	},
	"sm_75": {
		Architecture:    "sm_75",
		PeakTFLOPS:      14.2,
		PeakMemoryGBps:  672.0,
		L2BandwidthGBps: 1800.0,
	},
	"sm_80": {
		Architecture:    "sm_80",
		PeakTFLOPS:      19.5,
		PeakMemoryGBps:  1935.0,
		L2BandwidthGBps: 5120.0,
	},
	"sm_86": {
		Architecture:    "sm_86",
		PeakTFLOPS:      35.6,
		PeakMemoryGBps:  912.0,
		L2BandwidthGBps: 2800.0,
	},
	"sm_89": {
		Architecture:    "sm_89",
		PeakTFLOPS:      82.6,
		PeakMemoryGBps:  1008.0,
		L2BandwidthGBps: 3600.0,
	},
	"sm_90": {
		Architecture:    "sm_90",
		PeakTFLOPS:      67.0,
		PeakMemoryGBps:  3350.0,
		L2BandwidthGBps: 12000.0,
	},
	"sm_121": {
		Architecture:    "sm_121",
		PeakTFLOPS:      40.0,
		PeakMemoryGBps:  273.0,
		L2BandwidthGBps: 2000.0,
	},
	"sm_121a": {
		Architecture:    "sm_121a",
		PeakTFLOPS:      40.0,
		PeakMemoryGBps:  273.0,
		L2BandwidthGBps: 2000.0,
	},
}

// LookupCeiling returns the known device ceilings for arch (e.g. "sm_90", "sm_121").
func LookupCeiling(arch string) (Ceiling, error) {
	key := strings.ToLower(strings.TrimSpace(arch))
	if c, ok := knownCeilings[key]; ok {
		return c, nil
	}
	return Ceiling{}, fmt.Errorf("cudaroofline: unknown architecture %q", arch)
}

// Model evaluates kernel operational points against hardware ceilings.
type Model struct {
	Ceiling Ceiling
}

// NewModel creates a roofline model configured with the known ceilings for arch.
func NewModel(arch string) (*Model, error) {
	c, err := LookupCeiling(arch)
	if err != nil {
		return nil, err
	}
	return NewModelWithCeiling(c), nil
}

// NewModelWithCeiling creates a roofline model with caller-supplied ceilings.
func NewModelWithCeiling(c Ceiling) *Model {
	return &Model{Ceiling: c}
}

// KernelPoint records the arithmetic intensity, throughput, and roofline classification
// of a single kernel launch.
type KernelPoint struct {
	Name           string  // Kernel identifier or symbol name
	FLOPs          uint64  // Floating-point operations performed
	Bytes          uint64  // Bytes transferred (DRAM traffic)
	DurationNs     uint64  // Device execution span in nanoseconds
	Intensity      float64 // Operational arithmetic intensity (FLOPs / Byte)
	AttainedTFLOPS float64 // Attained throughput in TFLOPS
	AttainedGBps   float64 // Attained bandwidth in GB/s
	Classification string  // "memory_bound", "compute_bound", or "unconstrained"
	Efficiency     float64 // Ratio of attained throughput to ceiling at this intensity (0.0 to 1.0)
}

// KneeIntensity returns the operational arithmetic intensity where the memory ceiling
// meets the peak compute ceiling (TFLOPS / (PeakMemoryGBps / 1000)).
func (m *Model) KneeIntensity() float64 {
	if m.Ceiling.PeakMemoryGBps <= 0 {
		return 0
	}
	// PeakTFLOPS / (PeakMemoryGBps / 1000) = (PeakTFLOPS * 1000) / PeakMemoryGBps
	return (m.Ceiling.PeakTFLOPS * 1000.0) / m.Ceiling.PeakMemoryGBps
}

// CeilingAt reports the maximum attainable TFLOPS at the given arithmetic intensity.
func (m *Model) CeilingAt(intensity float64) float64 {
	if intensity <= 0 {
		return 0
	}
	memCeiling := intensity * (m.Ceiling.PeakMemoryGBps / 1000.0)
	if memCeiling < m.Ceiling.PeakTFLOPS {
		return memCeiling
	}
	return m.Ceiling.PeakTFLOPS
}

// Evaluate analyzes a kernel execution given its FLOP count, memory bytes transferred,
// and execution duration in nanoseconds.
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

	ceilingTFLOPS := m.CeilingAt(pt.Intensity)
	if ceilingTFLOPS > 0 {
		pt.Efficiency = pt.AttainedTFLOPS / ceilingTFLOPS
		if pt.Efficiency > 1.0 {
			pt.Efficiency = 1.0
		}
	}

	// If the kernel achieves less than 35% of the theoretical roofline ceiling,
	// it is not saturating the bottleneck and is labeled unconstrained (or latency-bound).
	if pt.Efficiency < 0.35 {
		pt.Classification = ClassUnconstrained
		return pt
	}

	knee := m.KneeIntensity()
	if pt.Intensity < knee {
		pt.Classification = ClassMemoryBound
	} else {
		pt.Classification = ClassComputeBound
	}

	return pt
}

// ValidateCeiling returns an error if any required ceiling parameter is non-positive.
func ValidateCeiling(c Ceiling) error {
	if c.PeakTFLOPS <= 0 {
		return errors.New("cudaroofline: PeakTFLOPS must be positive")
	}
	if c.PeakMemoryGBps <= 0 {
		return errors.New("cudaroofline: PeakMemoryGBps must be positive")
	}
	return nil
}
