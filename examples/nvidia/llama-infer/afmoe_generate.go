package main

import (
	"fmt"
	"time"
)

// generateAfmoe measures CPU wall time. forward includes input staging,
// forward computation, optional greedy sampling, device synchronization and
// the scalar host ID read. drain is outside each recorded interval. Prefill
// is the complete sequential prompt; only its last row is sampled.
func generateAfmoe(prompt []int32, n, maxContext int,
	drain func() error, forward func(token int32, pos int, sample bool) (int32, error)) (ids []int32, prefill time.Duration, decode []time.Duration, err error) {
	if len(prompt) == 0 || n < 1 || maxContext < len(prompt) || n-1 > maxContext-len(prompt) || drain == nil || forward == nil {
		return nil, 0, nil, fmt.Errorf("afmoe generation: invalid prompt, output count, context or callback")
	}
	if err := drain(); err != nil {
		return nil, 0, nil, fmt.Errorf("afmoe prefill drain: %w", err)
	}
	t0 := time.Now()
	var next int32
	for pos, token := range prompt {
		if next, err = forward(token, pos, pos == len(prompt)-1); err != nil {
			return nil, 0, nil, fmt.Errorf("afmoe prefill at %d: %w", pos, err)
		}
	}
	prefill = time.Since(t0)
	ids = append(ids, next)
	for i := 1; i < n; i++ {
		if err := drain(); err != nil {
			return ids, prefill, decode, fmt.Errorf("afmoe decode drain %d: %w", i, err)
		}
		t0 := time.Now()
		next, err = forward(next, len(prompt)+i-1, true)
		if err != nil {
			return ids, prefill, decode, fmt.Errorf("afmoe decode %d: %w", i, err)
		}
		decode = append(decode, time.Since(t0))
		ids = append(ids, next)
	}
	return ids, prefill, decode, nil
}
