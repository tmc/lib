package speculative

// NgramDrafter drafts candidate tokens by matching recent sequence context against an n-gram table.
type NgramDrafter struct {
	n       int
	table   map[uint64][]int32
	history []int32
	indexed int
}

// NewNgramDrafter creates an n-gram drafter with context window size n (typically 2..4).
func NewNgramDrafter(n int) *NgramDrafter {
	if n < 2 {
		n = 3
	}
	return &NgramDrafter{
		n:     n,
		table: make(map[uint64][]int32),
	}
}

// Feed updates the internal n-gram index with new token observations.
func (d *NgramDrafter) Feed(tokens []int32) {
	d.history = append(d.history, tokens...)
	for d.indexed+d.n < len(d.history) {
		i := d.indexed
		key := d.hashNgram(d.history[i : i+d.n])
		nextTok := d.history[i+d.n]
		d.table[key] = append(d.table[key], nextTok)
		d.indexed++
	}
}

// Draft predicts up to maxDraft next candidate tokens from the current context.
func (d *NgramDrafter) Draft(context []int32, maxDraft int) []int32 {
	if len(context) < d.n || maxDraft <= 0 {
		return nil
	}

	draft := make([]int32, 0, maxDraft)
	currContext := make([]int32, len(context))
	copy(currContext, context)

	for len(draft) < maxDraft {
		if len(currContext) < d.n {
			break
		}
		window := currContext[len(currContext)-d.n:]
		key := d.hashNgram(window)
		candidates, exists := d.table[key]
		if !exists || len(candidates) == 0 {
			break
		}

		// Pick the most recent token following this n-gram
		pred := candidates[len(candidates)-1]
		draft = append(draft, pred)
		currContext = append(currContext, pred)
	}

	return draft
}

func (d *NgramDrafter) hashNgram(tokens []int32) uint64 {
	h := uint64(14695981039346656037)
	for _, tok := range tokens {
		h ^= uint64(uint32(tok))
		h *= 1099511628211
	}
	return h
}
