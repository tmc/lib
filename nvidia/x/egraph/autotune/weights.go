package autotune

import (
	"fmt"
	"math"
	"strings"
)

// Stall-ratio thresholds at which [ScheduleWeights.Refine] changes a
// preference. They are heuristics, not measured break-even points.
const (
	MBarrierThreshold       = 0.40
	MIOThreshold            = 0.25
	TMEMThreshold           = 0.30
	LongScoreboardThreshold = 0.35
	MathThreshold           = 0.35
)

// MaxStages is the deepest TMA prefetch pipeline the autotuner proposes.
const MaxStages = 6

// ScheduleWeights holds stall-directed schedule preferences learned from
// profiles. The zero value has no preferences and ranks candidates by
// analytical cost.
type ScheduleWeights struct {
	PreferStages           int  `json:"prefer_stages,omitempty"`
	PreferCTAGroup         int  `json:"prefer_cta_group,omitempty"`
	PreferSwizzle128B      bool `json:"prefer_swizzle_128b,omitempty"`
	PreferDoubleBufferTMEM bool `json:"prefer_double_buffer_tmem,omitempty"`
	PreferSparse2to4       bool `json:"prefer_sparse_2to4,omitempty"`
}

// Refine returns w updated by the stall histogram h observed for candidate
// c on twin:
//
//   - barrier/mbarrier waits above [MBarrierThreshold] or long scoreboard
//     stalls above [LongScoreboardThreshold]: prefer one more TMA stage
//     than c, up to [MaxStages];
//   - MIO throttle above [MIOThreshold]: prefer 128-byte swizzle, and
//     cta_group::2 multicast on targets with tcgen05;
//   - TMEM waits above [TMEMThreshold]: prefer double-buffered TMEM
//     accumulators;
//   - math pipe throttle above [MathThreshold]: prefer 2:4 sparsity.
func (w ScheduleWeights) Refine(twin HardwareTwin, c ScheduleCandidate, h StallHistogram) ScheduleWeights {
	if h.Ratio(h.WaitMBarrier) > MBarrierThreshold || h.Ratio(h.LongScoreboard) > LongScoreboardThreshold {
		w.PreferStages = max(w.PreferStages, min(c.Stages+1, MaxStages))
	}
	if h.Ratio(h.MIOThrottle) > MIOThreshold {
		w.PreferSwizzle128B = true
		if twin.SupportsTCGen05 {
			w.PreferCTAGroup = 2
		}
	}
	if h.Ratio(h.WaitTMEM) > TMEMThreshold && twin.TMEMColumns >= 512 {
		w.PreferDoubleBufferTMEM = true
	}
	if h.Ratio(h.MathThrottle) > MathThreshold {
		w.PreferSparse2to4 = true
	}
	return w
}

// Penalties applied by [ScheduleWeights.Predict] per unmet preference, as
// fractions of analytical cost.
const (
	StagePenalty      = 0.35 // per stage of distance from PreferStages
	PreferencePenalty = 0.10 // per other unmet preference
)

// Predict returns c's analytical cost scaled up for each preference c does
// not meet. It ranks candidates that have not been measured.
func (w ScheduleWeights) Predict(c ScheduleCandidate) float64 {
	cost := c.AnalyticalCostUS
	if cost <= 0 {
		cost = 1
	}
	m := 1.0
	if w.PreferStages > 0 {
		m += StagePenalty * math.Abs(float64(c.Stages-w.PreferStages))
	}
	unmet := func(prefer, has bool) {
		if prefer && !has {
			m += PreferencePenalty
		}
	}
	unmet(w.PreferCTAGroup > 0, c.CTAGroup == w.PreferCTAGroup)
	unmet(w.PreferSwizzle128B, c.Swizzle128B)
	unmet(w.PreferDoubleBufferTMEM, c.DoubleBufferTMEM)
	unmet(w.PreferSparse2to4, c.UseSparse2to4)
	return cost * m
}

// supportsSetMaxNReg reports whether the autotuner emits setmaxnreg for
// twin. It is limited to the sm_90a and sm_10x targets; sm_12x is left out
// because the emitted code has not been checked there.
func supportsSetMaxNReg(twin HardwareTwin) bool {
	return twin.ComputeMajor == 9 || twin.ComputeMajor == 10
}

func checkMaxNReg(name string, n, lo, hi int) error {
	if n == 0 {
		return nil
	}
	if n%8 != 0 || n < lo || n > hi {
		return fmt.Errorf("%s %d: want a multiple of 8 in [%d, %d]", name, n, lo, hi)
	}
	return nil
}

// Validate reports whether c is a schedule the autotuner can emit for
// twin: 2 to [MaxStages] stages, cta_group 1 or 2 (2 only with tcgen05),
// TMEM double buffering only with 512 TMEM columns, and setmaxnreg targets
// that are multiples of 8 with the producer in [24, 256] and the consumer
// in [the producer, 256].
func (c ScheduleCandidate) Validate(twin HardwareTwin) error {
	if c.Stages < 2 || c.Stages > MaxStages {
		return fmt.Errorf("schedule %s: %d stages, want 2 to %d", c.Name, c.Stages, MaxStages)
	}
	switch c.CTAGroup {
	case 1:
	case 2:
		if !twin.SupportsTCGen05 {
			return fmt.Errorf("schedule %s: cta_group::2 needs tcgen05, %s has none", c.Name, twin.Target)
		}
	default:
		return fmt.Errorf("schedule %s: cta_group %d, want 1 or 2", c.Name, c.CTAGroup)
	}
	if c.DoubleBufferTMEM && twin.TMEMColumns < 512 {
		return fmt.Errorf("schedule %s: double-buffered TMEM needs 512 columns, %s has %d", c.Name, twin.Target, twin.TMEMColumns)
	}
	if (c.ProducerMaxRegs != 0 || c.ConsumerMaxRegs != 0) && !supportsSetMaxNReg(twin) {
		return fmt.Errorf("schedule %s: setmaxnreg not emitted for %s", c.Name, twin.Target)
	}
	if err := checkMaxNReg("producer setmaxnreg", c.ProducerMaxRegs, 24, 256); err != nil {
		return fmt.Errorf("schedule %s: %w", c.Name, err)
	}
	if err := checkMaxNReg("consumer setmaxnreg", c.ConsumerMaxRegs, max(24, c.ProducerMaxRegs), 256); err != nil {
		return fmt.Errorf("schedule %s: %w", c.Name, err)
	}
	return nil
}

// KnobPreamble returns CUDA C++ that fixes c's schedule knobs for a kernel
// compiled for twin: macros for the TMA stage count, cta_group, swizzle,
// TMEM buffering and any nonzero tile sizes, and, when c sets register
// targets, AUTOTUNE_PRODUCER_MAXNREG and AUTOTUNE_CONSUMER_MAXNREG with
// device functions autotune_producer_regs and autotune_consumer_regs that
// producer and consumer warpgroups call to run setmaxnreg.dec and
// setmaxnreg.inc. The CUDA_EGRAPH_AUTOTUNE_NATIVE=1 test compiles it with
// NVRTC for sm_121a and sm_90a; ordinary tests only check the text.
func KnobPreamble(twin HardwareTwin, c ScheduleCandidate) (string, error) {
	if err := c.Validate(twin); err != nil {
		return "", fmt.Errorf("knob preamble: %w", err)
	}
	b2i := func(b bool) int {
		if b {
			return 1
		}
		return 0
	}
	var b strings.Builder
	fmt.Fprintf(&b, "// autotune schedule %s for %s\n", c.Name, twin.Target)
	fmt.Fprintf(&b, "#define AUTOTUNE_TMA_STAGES %d\n", c.Stages)
	fmt.Fprintf(&b, "#define AUTOTUNE_CTA_GROUP %d\n", c.CTAGroup)
	fmt.Fprintf(&b, "#define AUTOTUNE_SWIZZLE_128B %d\n", b2i(c.Swizzle128B))
	fmt.Fprintf(&b, "#define AUTOTUNE_DOUBLE_BUFFER_TMEM %d\n", b2i(c.DoubleBufferTMEM))
	fmt.Fprintf(&b, "#define AUTOTUNE_SPARSE_2TO4 %d\n", b2i(c.UseSparse2to4))
	for _, t := range []struct {
		name string
		n    int
	}{{"M", c.TileM}, {"N", c.TileN}, {"K", c.TileK}} {
		if t.n > 0 {
			fmt.Fprintf(&b, "#define AUTOTUNE_TILE_%s %d\n", t.name, t.n)
		}
	}
	if c.ProducerMaxRegs != 0 {
		fmt.Fprintf(&b, "#define AUTOTUNE_PRODUCER_MAXNREG %d\n", c.ProducerMaxRegs)
		fmt.Fprintf(&b, "__device__ __forceinline__ void autotune_producer_regs() { asm volatile(\"setmaxnreg.dec.sync.aligned.u32 %d;\\n\" ::: \"memory\"); }\n", c.ProducerMaxRegs)
	}
	if c.ConsumerMaxRegs != 0 {
		fmt.Fprintf(&b, "#define AUTOTUNE_CONSUMER_MAXNREG %d\n", c.ConsumerMaxRegs)
		fmt.Fprintf(&b, "__device__ __forceinline__ void autotune_consumer_regs() { asm volatile(\"setmaxnreg.inc.sync.aligned.u32 %d;\\n\" ::: \"memory\"); }\n", c.ConsumerMaxRegs)
	}
	return b.String(), nil
}
