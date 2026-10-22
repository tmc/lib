package sampler

// ThinkBudget controls when an evaluator closes a model's thinking section.
// Close contains the token sequence for "</think>". The zero value never
// forces a token.
type ThinkBudget struct {
	Soft, Hard int
	Rank       int
	Close      []int
	pending    []int
	forced     bool
}

// Force returns the next forced close token, if any. A pending multi-token
// close is emitted before evaluating the limits again. Hard limits force the
// close at or below Hard; soft limits force a one-token close only when that
// token is within Rank according to rank. rank returns a one-based rank and
// zero when the token is absent.
func (b *ThinkBudget) Force(remaining int, inThink bool, rank func(id, max int) int) (id int, ok bool) {
	if !inThink {
		b.pending = nil
		b.forced = false
		return 0, false
	}
	if len(b.pending) != 0 {
		id = b.pending[0]
		b.pending = b.pending[1:]
		return id, true
	}
	if b.forced {
		return 0, false
	}
	if len(b.Close) == 0 {
		return 0, false
	}
	if b.Hard >= 0 && remaining <= b.Hard {
		b.pending = append([]int(nil), b.Close[1:]...)
		b.forced = true
		return b.Close[0], true
	}
	if b.Soft >= 0 && remaining <= b.Soft && len(b.Close) == 1 && b.Rank > 0 && rank != nil {
		if n := rank(b.Close[0], b.Rank); n > 0 && n <= b.Rank {
			b.forced = true
			return b.Close[0], true
		}
	}
	return 0, false
}
