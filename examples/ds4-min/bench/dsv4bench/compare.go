package dsv4bench

import "fmt"

// MetricComparison compares one performance metric in two aligned runs.
// Delta is target minus base, computed pair by pair before its range is
// summarized.  Keeping the paired delta separate avoids presenting a
// difference of independent medians as an ABBA result.
type MetricComparison struct {
	Base   Range `json:"base"`
	Target Range `json:"target"`
	Delta  Range `json:"delta"`
}

// Comparison is a model-free comparison of two benchmark CSVs. Rows must be
// in the same order and describe the same frontier. It reports the aligned
// pairwise change, but does not qualify performance; callers still need the
// campaign's warmup, ordering, and host-condition gates.
type Comparison struct {
	Samples      int              `json:"samples"`
	CtxTokens    []int64          `json:"ctx_tokens"`
	PrefillTPS   MetricComparison `json:"prefill_tps"`
	GenTPS       MetricComparison `json:"gen_tps"`
	GenFirstMS   MetricComparison `json:"gen_first_ms"`
	GenSteadyTPS MetricComparison `json:"gen_steady_tps"`
}

// Compare returns paired summaries for base and target benchmark rows. It
// rejects mismatched frontiers and workload sizes rather than silently
// comparing rows from different experiments.
func Compare(base, target []Row) (Comparison, error) {
	if len(base) == 0 || len(target) == 0 {
		return Comparison{}, fmt.Errorf("compare benchmark: empty input")
	}
	if len(base) != len(target) {
		return Comparison{}, fmt.Errorf("compare benchmark: %d rows versus %d", len(base), len(target))
	}
	ctx := make([]int64, len(base))
	metrics := [4][][]float64{{}, {}, {}, {}}
	for i := range base {
		if err := validate(base[i]); err != nil {
			return Comparison{}, fmt.Errorf("compare benchmark base row %d: %w", i, err)
		}
		if err := validate(target[i]); err != nil {
			return Comparison{}, fmt.Errorf("compare benchmark target row %d: %w", i, err)
		}
		if !sameWorkload(base[i], target[i]) {
			return Comparison{}, fmt.Errorf("compare benchmark row %d: workload differs", i)
		}
		ctx[i] = base[i].CtxTokens
		metrics[0] = append(metrics[0], []float64{base[i].PrefillTPS, target[i].PrefillTPS, target[i].PrefillTPS - base[i].PrefillTPS})
		metrics[1] = append(metrics[1], []float64{base[i].GenTPS, target[i].GenTPS, target[i].GenTPS - base[i].GenTPS})
		metrics[2] = append(metrics[2], []float64{base[i].GenFirstMS, target[i].GenFirstMS, target[i].GenFirstMS - base[i].GenFirstMS})
		metrics[3] = append(metrics[3], []float64{base[i].GenSteadyTPS, target[i].GenSteadyTPS, target[i].GenSteadyTPS - base[i].GenSteadyTPS})
	}
	return Comparison{
		Samples:      len(base),
		CtxTokens:    ctx,
		PrefillTPS:   compareMetric(metrics[0]),
		GenTPS:       compareMetric(metrics[1]),
		GenFirstMS:   compareMetric(metrics[2]),
		GenSteadyTPS: compareMetric(metrics[3]),
	}, nil
}

func sameWorkload(a, b Row) bool {
	return a.CtxTokens == b.CtxTokens && a.PrefillTokens == b.PrefillTokens &&
		a.GenTokens == b.GenTokens && a.GenSteadyTokens == b.GenSteadyTokens
}

func compareMetric(values [][]float64) MetricComparison {
	base := make([]float64, len(values))
	target := make([]float64, len(values))
	delta := make([]float64, len(values))
	for i, value := range values {
		base[i], target[i], delta[i] = value[0], value[1], value[2]
	}
	return MetricComparison{Base: summarizeValues(base), Target: summarizeValues(target), Delta: summarizeValues(delta)}
}
