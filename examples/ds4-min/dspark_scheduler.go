package main

import (
	"math"
	"strconv"
	"strings"
)

// dsparkSchedule is the deterministic CUDA schedule at upstream 11e8cad.
// It counts only attempted cycles. Skips still require the target feature
// journal to advance; predicted support rows never become trusted history.
// A new request owns a fresh value. Timing controls are opt-in.
type dsparkSchedule struct {
	policy                    dsparkSchedulePolicy
	cycles, accepted, noDraft uint64
	lifetime                  uint32
	extra, saved              float64
	skip                      uint32
	skipped, longAccept       bool
}

type dsparkSchedulePolicy struct {
	enabled, seedBatch, spark                            bool
	window, baseSkip, longSkip, slowSkip                 uint32
	noDraftSkip, shortSkip, coldSkip                     uint32
	minAverage, maxMSPerAccept, maxExtraSaved, breakEven uint32
	tail                                                 uint32
	coldThreshold                                        float64
}

// dsparkEnvUint copies strtoul's accepted numeric prefix (including trailing
// text), but rejects values outside uint32. Source: ds4.c57045.
func dsparkEnvUint(value string, fallback uint32) uint32 {
	value = strings.TrimLeft(value, " \t\n\r\v\f")
	negative := strings.HasPrefix(value, "-")
	if strings.HasPrefix(value, "+") || negative {
		value = value[1:]
	}
	i := 0
	for i < len(value) && value[i] >= '0' && value[i] <= '9' {
		i++
	}
	if i == 0 {
		return fallback
	}
	v, err := strconv.ParseUint(value[:i], 10, 32)
	if err != nil || negative && v != 0 {
		return fallback
	}
	return uint32(v)
}

func newDSparkSchedulePolicy(seedBatch, spark bool, getenv func(string) string) dsparkSchedulePolicy {
	u := func(name string, fallback uint32) uint32 { return dsparkEnvUint(getenv(name), fallback) }
	p := dsparkSchedulePolicy{enabled: getenv("DS4_DSPARK_SCHEDULER") != "0", seedBatch: seedBatch, spark: spark}
	p.window = u("DS4_DSPARK_SCHEDULER_WINDOW", 4)
	if p.window == 0 {
		p.window = 4
	}
	skip := uint32(2)
	if seedBatch && spark {
		skip = 32
	}
	p.baseSkip = u("DS4_DSPARK_SCHEDULER_SKIP", skip)
	p.longSkip = u("DS4_DSPARK_SCHEDULER_SKIP", 8)
	p.slowSkip = u("DS4_DSPARK_SCHEDULER_SLOW_SKIP", 4)
	p.noDraftSkip = u("DS4_DSPARK_SCHEDULER_NO_DRAFT_SKIP", 3)
	p.shortSkip = u("DS4_DSPARK_SCHEDULER_SHORT_ACCEPT_NO_DRAFT_SKIP", 4)
	p.coldSkip = u("DS4_DSPARK_SCHEDULER_COLD_LOW_CONFIDENCE_SKIP", 7)
	p.minAverage = u("DS4_DSPARK_SCHEDULER_MIN_AVG_MILLI", 1500)
	p.maxMSPerAccept = u("DS4_DSPARK_SCHEDULER_MAX_MS_PER_ACCEPT_MILLI", 0)
	p.maxExtraSaved = u("DS4_DSPARK_SCHEDULER_MAX_EXTRA_SAVED_RATIO_MILLI", 0)
	p.breakEven = u("DS4_DSPARK_SCHEDULER_BREAK_EVEN_WINDOW", 0)
	p.tail = u("DS4_DSPARK_SCHEDULER_TAIL_MIN_TOKENS", 10)
	p.coldThreshold = float64(u("DS4_DSPARK_SCHEDULER_COLD_LOW_CONFIDENCE_MILLI", 500)) / 1000
	return p
}

func (s *dsparkSchedule) shouldSkip() bool {
	if !s.policy.enabled {
		return false
	}
	s.skipped = false
	if s.skip == 0 {
		return false
	}
	s.skip--
	s.skipped = true
	return true
}

func (s *dsparkSchedule) resetWindow() {
	s.cycles, s.accepted, s.noDraft = 0, 0, 0
	s.extra, s.saved = 0, 0
}

func dsparkFinitePositive(v float64) bool { return v > 0 && !math.IsNaN(v) && !math.IsInf(v, 0) }

// note must be called once per attempt or skipped cycle. accepted counts
// draft suffix tokens only; targetMS is the ordinary target seed's duration.
func (s *dsparkSchedule) note(accepted uint32, noDraft bool, extraMS, targetMS float64, confidence0 float64, confidenceValid bool) {
	p := s.policy
	if !p.enabled {
		return
	}
	if s.skipped {
		s.skipped = false
		return
	}
	s.cycles++
	s.accepted += uint64(accepted)
	if accepted != 0 {
		if accepted > math.MaxUint32-s.lifetime {
			s.lifetime = math.MaxUint32
		} else {
			s.lifetime += accepted
		}
		if accepted > 2 {
			s.longAccept = true
		}
	}
	if noDraft {
		s.noDraft++
	}
	if dsparkFinitePositive(extraMS) {
		s.extra += extraMS
	}
	if accepted != 0 && dsparkFinitePositive(targetMS) {
		s.saved += targetMS * float64(accepted)
	}
	if noDraft && p.noDraftSkip != 0 {
		skip := p.noDraftSkip
		if s.lifetime != 0 && !s.longAccept {
			skip = max(skip, p.shortSkip)
		} else if s.lifetime == 0 && confidenceValid && confidence0 <= p.coldThreshold {
			skip = max(skip, p.coldSkip)
		}
		s.skip = max(s.skip, skip)
	}
	unprofitable := p.maxExtraSaved != 0 && s.accepted != 0 && s.saved > 0 && s.extra*1000 > s.saved*float64(p.maxExtraSaved)
	if p.breakEven != 0 && s.cycles >= uint64(p.breakEven) && unprofitable {
		s.skip = p.slowSkip
		s.resetWindow()
		return
	}
	if s.cycles < uint64(p.window) {
		return
	}
	low := s.accepted*1000/s.cycles < uint64(p.minAverage)
	many := s.noDraft*2 >= s.cycles
	slow := p.maxMSPerAccept != 0 && s.accepted != 0 && s.extra/float64(s.accepted)*1000 > float64(p.maxMSPerAccept)
	if low || many || slow || unprofitable {
		s.skip = p.baseSkip
		if p.seedBatch && p.spark && s.longAccept {
			s.skip = p.longSkip
		}
		if many || slow || unprofitable {
			s.skip = max(s.skip, p.slowSkip)
		}
	}
	s.resetWindow()
}
