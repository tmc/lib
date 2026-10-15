package main

import (
	"fmt"
	"time"
)

// generate runs runGPUInference's prefill and then its decode loop. It
// feeds each token of prompt to prefill at its position, in order, and then
// calls decode with st, forward, plain, show, maxSeq, steps and draftN from
// position len(prompt), with the token prefill predicted for the last
// prompt token as the pending token. It returns how long each phase took.
//
// generate stops at the first error. It wraps an error from prefill with
// its position and runs no decode after it; it returns an error from decode
// unchanged.
func generate(st *decodeState, prompt []int32, prefill, forward func(tok int32, pos int) (int32, error), plain func(step int, tok int32, pos int) (int32, error), show func(tok int32) error, maxSeq, steps, draftN int) (prefillTime, decodeTime time.Duration, err error) {
	var next int32
	t0 := time.Now()
	for pos, tok := range prompt {
		if next, err = prefill(tok, pos); err != nil {
			return 0, 0, fmt.Errorf("gpu prefill step %d: %w", pos, err)
		}
	}
	prefillTime = time.Since(t0)

	t0 = time.Now()
	if _, _, err := decode(st, forward, plain, show, next, len(prompt), maxSeq, steps, draftN); err != nil {
		return prefillTime, 0, err
	}
	return prefillTime, time.Since(t0), nil
}

// decode runs runGPUInference's decode loop from the pending token next at
// position pos until steps tokens are emitted or the KV cache of maxSeq
// positions is full. It records every emitted token in st, passes each to
// show in order, and returns the final pending token and position.
//
// If st has a drafter, each step first calls st.speculate with forward,
// drafting up to draftN tokens. If that emits nothing, or st has no
// drafter, the step emits next, shows it, and calls plain(step, next, pos)
// to predict the token after it, where step is the number of tokens emitted
// before next.
//
// Only plain forwards go through plain. In runGPUInference, plain times its
// forward with the step timer and labels it for cudaprof; the speculative
// forwards call forward directly and are neither timed nor labelled. Routing
// them through plain would change what the timer and profiler record, so
// decode keeps the two paths separate.
//
// decode returns an error from plain or show unchanged and wraps an error
// from a speculative step with its position. It stops at the first error:
// no token is shown and no forward runs after it. On error, the returned
// pending token and position are those from before the failed step; st has
// already recorded the tokens of that step that were emitted, and if plain
// failed, show has been called with its pending token.
func decode(st *decodeState, forward func(tok int32, pos int) (int32, error), plain func(step int, tok int32, pos int) (int32, error), show func(tok int32) error, next int32, pos, maxSeq, steps, draftN int) (int32, int, error) {
	for step := 0; step < steps; {
		if pos >= maxSeq {
			break
		}

		if st.drafter != nil {
			emitted, n, p, err := st.speculate(forward, next, draftN, pos, maxSeq, steps-step)
			if err != nil {
				return next, pos, fmt.Errorf("speculative step at position %d: %w", pos, err)
			}
			if len(emitted) > 0 {
				for _, tok := range emitted {
					if err := show(tok); err != nil {
						return next, pos, err
					}
					step++
				}
				next, pos = n, p
				continue
			}
		}

		st.emit(next)
		if err := show(next); err != nil {
			return next, pos, err
		}
		tok, err := plain(step, next, pos)
		if err != nil {
			return next, pos, err
		}
		next = tok
		pos++
		step++
	}
	return next, pos, nil
}
