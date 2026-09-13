package cuptiactivity

import (
	"github.com/tmc/lib/nvidia/cupti"
	"github.com/tmc/lib/nvidia/x/cuptipmsampling"
)

// PMSample represents one hardware Performance Monitor observation.
type PMSample = cuptipmsampling.Sample

// PMReader provides a streaming reader interface over decoded PM samples.
type PMReader = cuptipmsampling.Reader

// PMSamplerConfig specifies high-level configuration for hardware PM counter sampling.
type PMSamplerConfig = cuptipmsampling.SamplerConfig

// NewPMReader creates a streaming PM counter reader over decoded samples.
func NewPMReader(samples []PMSample) *PMReader {
	return cuptipmsampling.NewReader(samples)
}

// AttributedKernel represents a GPU kernel activity record enriched with correlated
// hardware PM performance counter samples.
type AttributedKernel struct {
	Record        Record                   // Traced kernel execution activity record
	KernelName    string                   // Symbol or function kernel name
	CorrelationID uint32                   // Native CUPTI correlation ID
	DurationNs    uint64                   // Device execution interval (End - Start)
	PMSamples     []cuptipmsampling.Sample // Hardware PM counter samples overlapping kernel execution
	PMAverages    map[string]float64       // Mean hardware counter values during kernel execution
}

// AttributePMSamples correlates hardware PM counter samples to kernel execution
// intervals by finding PM samples overlapping each kernel's [Start, End] time window.
func AttributePMSamples(records []Record, pmSamples []cuptipmsampling.Sample) []AttributedKernel {
	if len(records) == 0 {
		return nil
	}

	var out []AttributedKernel
	for _, r := range records {
		if r.Kind != cupti.ActivityKindConcurrentKernel && r.Kind != cupti.ActivityKindKernel {
			continue
		}

		var duration uint64
		if r.Complete && r.End >= r.Start {
			duration = r.End - r.Start
		}

		ak := AttributedKernel{
			Record:        r,
			KernelName:    r.Name,
			CorrelationID: r.CorrelationID,
			DurationNs:    duration,
		}

		if len(pmSamples) > 0 && r.Start > 0 && r.End >= r.Start {
			var matching []cuptipmsampling.Sample
			sums := make(map[string]float64)
			counts := make(map[string]int)

			for _, ps := range pmSamples {
				// Check timestamp overlap: [ps.StartTimestamp, ps.EndTimestamp] overlaps [r.Start, r.End]
				if ps.EndTimestamp < r.Start || ps.StartTimestamp > r.End {
					continue
				}
				matching = append(matching, ps)
				for mName, val := range ps.Values {
					sums[mName] += val
					counts[mName]++
				}
			}

			ak.PMSamples = matching
			if len(sums) > 0 {
				ak.PMAverages = make(map[string]float64, len(sums))
				for mName, sum := range sums {
					cnt := counts[mName]
					if cnt > 0 {
						ak.PMAverages[mName] = sum / float64(cnt)
					}
				}
			}
		}

		out = append(out, ak)
	}

	return out
}
