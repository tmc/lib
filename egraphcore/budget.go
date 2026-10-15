package egraphcore

import "math"

// WorkBudget tracks work units used by a search or rewrite pass. A positive
// limit is enforced; a non-positive limit means unlimited work.
type WorkBudget struct {
	limit     int
	used      int
	exhausted bool
	memory    *MemoryScope
}

// NewWorkBudget creates a work budget. A positive limit bounds TrySpend;
// zero or a negative limit means unlimited work.
func NewWorkBudget(limit int) *WorkBudget {
	return &WorkBudget{limit: limit}
}

// TrySpend records units of work and reports whether the budget permits them.
// Negative units are invalid and return false. In unlimited mode the used
// counter saturates at the largest int rather than overflowing.
func (b *WorkBudget) TrySpend(units int) bool {
	if b == nil || units < 0 {
		return false
	}
	if units == 0 {
		return !b.exhausted
	}
	if b.limit > 0 {
		if b.exhausted || b.used >= b.limit || units > b.limit-b.used {
			b.exhausted = true
			return false
		}
		b.used += units
		return true
	}
	if units > math.MaxInt-b.used {
		b.used = math.MaxInt
	} else {
		b.used += units
	}
	return true
}

// Used returns the recorded work count. In unlimited mode this count saturates
// at the largest int if the exact total cannot be represented.
func (b *WorkBudget) Used() int {
	if b == nil {
		return 0
	}
	return b.used
}

// Remaining returns the remaining bounded work. math.MaxInt denotes an
// unlimited budget; a nil budget has no remaining work.
func (b *WorkBudget) Remaining() int {
	if b == nil {
		return 0
	}
	if b.limit <= 0 {
		return math.MaxInt
	}
	if b.exhausted || b.used >= b.limit {
		return 0
	}
	return b.limit - b.used
}

// Exhausted reports whether a bounded budget has been fully spent.
func (b *WorkBudget) Exhausted() bool {
	return b != nil && b.limit > 0 && (b.exhausted || b.used >= b.limit)
}
