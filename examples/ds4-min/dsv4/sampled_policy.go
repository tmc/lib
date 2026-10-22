package dsv4

// SampledMode selects the sampled verifier. OpportunisticSampling accepts
// greedy suffixes after a sampled boundary and is not distribution preserving.
type SampledMode uint8

const (
	ExactSampling SampledMode = iota
	OpportunisticSampling
)

// SampledPolicy is a request-local deterministic scheduler. Strict suppresses
// drafting. Timing and hardware-specific scheduling remain the native owner's
// responsibility. The zero value uses the ordinary 11e8cad defaults.
type SampledPolicy struct {
	Strict    bool
	SeedBatch bool
	// Spark is set only by a qualified native owner. Metal seed batching
	// retains the 32-cycle pause after long accepts; Spark rechecks after eight.
	Spark                     bool
	Skip                      int
	cycles, accepted, noDraft int
	lifetime, longAccept      bool
}

// Allow consumes one backoff cycle. remaining includes the boundary token.
// The pinned dispatch skips drafting when fewer than ten tokens remain.
func (p *SampledPolicy) Allow(remaining int) bool {
	if p.Strict || remaining < 10 {
		return false
	}
	if p.Skip > 0 {
		p.Skip--
		return false
	}
	return true
}

// Note records one proposed cycle, excluding strict, tail and backoff cycles.
// confidence is optional; nil means no first-confidence readout was available.
// It is the pinned upstream raw first-confidence readout, not its sigmoid.
func (p *SampledPolicy) Note(accepted int, noDraft bool, confidence *float32) {
	p.cycles++
	p.accepted += accepted
	if accepted > 0 {
		p.lifetime = true
	}
	if accepted > 2 {
		p.longAccept = true
	}
	if noDraft {
		p.noDraft++
		n := 3
		if p.lifetime && !p.longAccept {
			n = 4
		}
		if !p.lifetime && confidence != nil && *confidence <= .5 {
			n = 7
		}
		p.Skip = max(p.Skip, n)
	}
	if p.cycles < 4 {
		return
	}
	if p.accepted*1000/p.cycles < 1500 || p.noDraft*2 >= p.cycles {
		n := 2
		if p.SeedBatch {
			n = 32
			if p.longAccept && p.Spark {
				n = 8
			}
		}
		if p.noDraft*2 >= p.cycles {
			n = max(n, 4)
		}
		p.Skip = n
	}
	p.cycles, p.accepted, p.noDraft = 0, 0, 0
}
