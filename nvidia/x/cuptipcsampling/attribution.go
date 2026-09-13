package cuptipcsampling

import (
	"strings"

	"github.com/tmc/lib/nvidia/cupti"
	"github.com/tmc/lib/nvidia/x/cuptiactivity"
)

// AttributedKernelSASS represents a GPU kernel execution activity record
// correlated with instruction-level SASS metrics, dominant stall reasons, and warp counts.
type AttributedKernelSASS struct {
	Record        cuptiactivity.Record // Traced kernel execution activity record
	KernelName    string               // Symbol or function kernel name
	CorrelationID uint32               // Native CUPTI correlation ID
	DurationNs    uint64               // Device execution interval (End - Start)
	TotalWarps    uint64               // Total warp execution samples / cycles
	IssuedWarps   uint64               // Total selected & issued warp instructions
	StallWarps    uint64               // Total stalled warp cycles
	DominantStall string               // Primary pipeline bottleneck across kernel
	DominantPct   float64              // Percentage of stall cycles caused by dominant reason
	SASSMetrics   []SASSMetric         // Instruction-level SASS metrics in this kernel
}

// StallBreakdown returns a map of canonical stall reason names to their percentage
// of total stall samples across the entire kernel.
func (ak AttributedKernelSASS) StallBreakdown() map[string]float64 {
	if ak.StallWarps == 0 {
		return nil
	}
	counts := make(map[string]uint64)
	for _, sm := range ak.SASSMetrics {
		for _, s := range sm.Stalls {
			counts[s.Name] += s.Samples
		}
	}
	breakdown := make(map[string]float64, len(counts))
	for k, v := range counts {
		breakdown[k] = float64(v) / float64(ak.StallWarps) * 100.0
	}
	return breakdown
}

// AttributePCSamples joins PC sampling SASS metrics to kernel execution records
// by matching CorrelationID or kernel name. Non-kernel records are ignored.
func AttributePCSamples(records []cuptiactivity.Record, metrics []SASSMetric) []AttributedKernelSASS {
	if len(records) == 0 {
		return nil
	}

	byName := make(map[string][]SASSMetric)
	for _, m := range metrics {
		byName[m.KernelName] = append(byName[m.KernelName], m)
	}

	var attributed []AttributedKernelSASS
	for _, r := range records {
		if r.Kind != cupti.ActivityKindConcurrentKernel && r.Kind != cupti.ActivityKindKernel {
			continue
		}

		kName := r.Name
		kMetrics := byName[kName]
		if len(kMetrics) == 0 {
			for name, ms := range byName {
				if strings.HasPrefix(name, kName) || strings.HasPrefix(kName, name) {
					kMetrics = ms
					break
				}
			}
		}

		var duration uint64
		if r.Complete && r.End >= r.Start {
			duration = r.End - r.Start
		}

		ak := AttributedKernelSASS{
			Record:        r,
			KernelName:    kName,
			CorrelationID: r.CorrelationID,
			DurationNs:    duration,
			SASSMetrics:   append([]SASSMetric(nil), kMetrics...),
		}

		stallCounts := make(map[string]uint64)
		for _, sm := range kMetrics {
			ak.TotalWarps += sm.TotalSamples
			ak.IssuedWarps += sm.Selected
			ak.StallWarps += sm.StallSamples
			for _, s := range sm.Stalls {
				stallCounts[s.Name] += s.Samples
			}
		}

		var maxCount uint64
		var dominant string
		for name, count := range stallCounts {
			if count > maxCount {
				maxCount = count
				dominant = name
			}
		}
		ak.DominantStall = dominant
		if ak.StallWarps > 0 && maxCount > 0 {
			ak.DominantPct = float64(maxCount) / float64(ak.StallWarps) * 100.0
		}

		attributed = append(attributed, ak)
	}

	return attributed
}
