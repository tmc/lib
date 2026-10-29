package autotune

import (
	"fmt"
	"slices"
	"strings"

	"github.com/tmc/lib/nvidia/x/cuptipcsampling"
)

// StallHistogram aggregates CUPTI PC-sampling stall counts attributed to a
// SASS PC, an e-graph op kind or a whole kernel.
type StallHistogram struct {
	TotalSamples   uint64 `json:"total_samples"`
	WaitTMEM       uint64 `json:"wait_tmem"`       // tensor memory waits (tcgen05.wait::ld)
	WaitMBarrier   uint64 `json:"wait_mbarrier"`   // barrier and mbarrier waits
	MIOThrottle    uint64 `json:"mio_throttle"`    // shared memory / TMA issue queue, short scoreboard
	MathThrottle   uint64 `json:"math_throttle"`   // tensor or math pipe busy
	LongScoreboard uint64 `json:"long_scoreboard"` // global or C2C load dependency
}

// Add returns the sum of h and o.
func (h StallHistogram) Add(o StallHistogram) StallHistogram {
	return StallHistogram{
		TotalSamples:   h.TotalSamples + o.TotalSamples,
		WaitTMEM:       h.WaitTMEM + o.WaitTMEM,
		WaitMBarrier:   h.WaitMBarrier + o.WaitMBarrier,
		MIOThrottle:    h.MIOThrottle + o.MIOThrottle,
		MathThrottle:   h.MathThrottle + o.MathThrottle,
		LongScoreboard: h.LongScoreboard + o.LongScoreboard,
	}
}

func (h StallHistogram) total() uint64 {
	sum := h.WaitTMEM + h.WaitMBarrier + h.MIOThrottle + h.MathThrottle + h.LongScoreboard
	return max(sum, h.TotalSamples)
}

// Ratio returns n as a fraction of h's samples: TotalSamples, or the sum
// of the classified counts if that is larger.
func (h StallHistogram) Ratio(n uint64) float64 {
	t := h.total()
	if t == 0 {
		return 0
	}
	return float64(n) / float64(t)
}

// StallProfile converts h to the [StallProfile] used by [OnlineCostModel]:
// WaitMBarrier counts as barrier stalls and WaitTMEM as short scoreboard
// stalls.
func (h StallHistogram) StallProfile() StallProfile {
	return StallProfile{
		TotalSamples:          h.TotalSamples,
		StallMIOThrottle:      h.MIOThrottle,
		StallMathPipeThrottle: h.MathThrottle,
		StallLongScoreboard:   h.LongScoreboard,
		StallShortScoreboard:  h.WaitTMEM,
		StallBarrier:          h.WaitMBarrier,
	}
}

// A PCRange attributes the SASS offsets [Start, End) of one function to
// Label: an e-graph op kind, PTX block or source site. Callers derive
// ranges from line tables, for example with
// [github.com/tmc/lib/nvidia/x/cudalineinfo].
type PCRange struct {
	Start, End uint64
	Label      string
}

// Unattributed is the [AttributionMap] key for samples at offsets no range
// covers.
const Unattributed = ""

// AttributionMap sums per-PC stall histograms into per-label histograms.
// Ranges must be nonempty and must not overlap. Samples at offsets outside
// every range are summed under [Unattributed] rather than dropped.
func AttributionMap(ranges []PCRange, stalls map[uint64]StallHistogram) (map[string]StallHistogram, error) {
	rs := slices.Clone(ranges)
	slices.SortFunc(rs, func(a, b PCRange) int {
		switch {
		case a.Start < b.Start:
			return -1
		case a.Start > b.Start:
			return 1
		}
		return 0
	})
	for i, r := range rs {
		if r.End <= r.Start {
			return nil, fmt.Errorf("attribution map: empty range [%#x, %#x) for %q", r.Start, r.End, r.Label)
		}
		if r.Label == Unattributed {
			return nil, fmt.Errorf("attribution map: range [%#x, %#x) has no label", r.Start, r.End)
		}
		if i > 0 && r.Start < rs[i-1].End {
			return nil, fmt.Errorf("attribution map: ranges %q and %q overlap at %#x", rs[i-1].Label, r.Label, r.Start)
		}
	}
	out := make(map[string]StallHistogram)
	for pc, h := range stalls {
		i, _ := slices.BinarySearchFunc(rs, pc, func(r PCRange, pc uint64) int {
			switch {
			case r.End <= pc:
				return -1
			case r.Start > pc:
				return 1
			}
			return 0
		})
		label := Unattributed
		if i < len(rs) && rs[i].Start <= pc && pc < rs[i].End {
			label = rs[i].Label
		}
		out[label] = out[label].Add(h)
	}
	return out, nil
}

// classifyReason returns the [StallHistogram] field a native CUPTI stall
// reason name counts toward, or nil. Routing is by name substring and has
// not been checked against every CUPTI version's inventory. Reasons ending
// in _not_issued repeat their issued counterpart and are skipped.
func classifyReason(h *StallHistogram, name string) *uint64 {
	switch {
	case name == "" || strings.HasSuffix(name, "_not_issued"):
		return nil
	case strings.Contains(name, "tmem"):
		return &h.WaitTMEM
	case strings.Contains(name, "long_scoreboard") || strings.Contains(name, "lg_throttle"):
		return &h.LongScoreboard
	case strings.Contains(name, "mio_throttle") || strings.Contains(name, "short_scoreboard"):
		return &h.MIOThrottle
	case strings.Contains(name, "math_pipe_throttle") || strings.Contains(name, "tensor"):
		return &h.MathThrottle
	case strings.Contains(name, "barrier") || strings.Contains(name, "membar") || strings.Contains(name, "sleeping"):
		return &h.WaitMBarrier
	}
	return nil
}

// HistogramsFromBatch converts a CUPTI PC-sampling batch into per-offset
// stall histograms for the function named fn ("" for every function),
// naming native reason indexes through inv. TotalSamples counts every
// issued-reason sample at the PC, classified or not.
func HistogramsFromBatch(b cuptipcsampling.Batch, inv cuptipcsampling.ReasonInventory, fn string) map[uint64]StallHistogram {
	out := make(map[uint64]StallHistogram)
	for _, pc := range b.PCs {
		if fn != "" && pc.Name != fn {
			continue
		}
		h := out[pc.Offset]
		for _, r := range pc.Reasons {
			name := inv.Lookup(r.Index)
			if strings.HasSuffix(name, "_not_issued") {
				continue
			}
			h.TotalSamples += uint64(r.Samples)
			if f := classifyReason(&h, name); f != nil {
				*f += uint64(r.Samples)
			}
		}
		out[pc.Offset] = h
	}
	return out
}
