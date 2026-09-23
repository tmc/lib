package main

import (
	"context"
	"fmt"
	"slices"
)

// generate starts a fresh greedy request. The callback sees each generated
// token, including EOS; false stops without another device execution. The
// prompt and at most maxTokens-1 generated tokens occupy cache positions.
func (p *pjrtModel) generate(ctx context.Context, prompt []int32, maxTokens int, eos []int32, emit func(int32) bool) error {
	var callback func(int32, bool) bool
	if emit != nil {
		callback = func(id int32, final bool) bool { return emit(id) }
	}
	return p.generateFinal(ctx, prompt, maxTokens, eos, callback)
}

// generateFinal marks the last prediction on normal completion, including EOS
// and a capacity-clamped limit. A callback stop or cancellation may end a request
// before a final prediction; callers must discard any incomplete text then.
func (p *pjrtModel) generateFinal(ctx context.Context, prompt []int32, maxTokens int, eos []int32, emit func(int32, bool) bool) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	if p == nil || p.model == nil {
		return fmt.Errorf("pjrt: model is closed")
	}
	if emit == nil {
		return fmt.Errorf("pjrt: token callback is required")
	}
	if maxTokens < 0 {
		return fmt.Errorf("pjrt: negative token limit")
	}
	if len(prompt) == 0 {
		return fmt.Errorf("pjrt: empty prompt")
	}
	if len(prompt) > p.capacity {
		return fmt.Errorf("pjrt: prompt exceeds capacity")
	}
	for _, id := range prompt {
		if id < 0 || int64(id) >= int64(p.vocab) {
			return fmt.Errorf("pjrt: prompt token outside vocabulary")
		}
	}
	for _, id := range eos {
		if id < 0 || int64(id) >= int64(p.vocab) {
			return fmt.Errorf("pjrt: EOS outside vocabulary")
		}
	}
	if maxTokens == 0 {
		return nil
	}
	// Clamp without overflowing len(prompt)+maxTokens. A full prompt can still
	// predict one token; that prediction need not be appended to device history.
	maxTokens = min(maxTokens, p.capacity-len(prompt)+1)
	if err := p.reset(); err != nil {
		return err
	}
	logits, err := p.forward(ctx, prompt, 0)
	if err != nil {
		return err
	}
	position := len(prompt)
	for step := 0; step < maxTokens; step++ {
		if err := ctx.Err(); err != nil {
			return err
		}
		id := sampleArgmax(logits)
		final := slices.Contains(eos, id) || step+1 == maxTokens
		if !emit(id, final) {
			return nil
		}
		if err := ctx.Err(); err != nil {
			return err
		}
		if final {
			return nil
		}
		logits, err = p.forward(ctx, []int32{id}, position)
		if err != nil {
			return err
		}
		position++
	}
	return nil
}
