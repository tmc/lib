package main

import (
	"fmt"
	"io"
	"time"
)

// runBenchmark runs runGPUInference's benchmark suite: one warmup prefill
// of prompt and then benchCount timed iterations of generate, resetting the
// KV cache before each and writing TTFT, TPOT and decode throughput lines
// to out. It refuses steps <= 0 before running any forward pass.
func runBenchmark(out io.Writer, dtype weightDType, prompt []int32, steps, benchCount, maxSeq int, reset func(), forward func(tok int32, pos int) (int32, error)) error {
	if steps <= 0 {
		return fmt.Errorf("benchmark requires -steps > 0, got %d", steps)
	}
	prefix := "BenchmarkLlama32_1B/backend=cuda/weights=" + string(dtype) + "/gen_tokens=16/prompt=neural_network/search_graphs=100"
	for pos, tokID := range prompt {
		if _, err := forward(tokID, pos); err != nil {
			return fmt.Errorf("warmup step %d: %w", pos, err)
		}
	}
	plain := func(step int, tok int32, pos int) (int32, error) {
		outTok, err := forward(tok, pos)
		if err != nil {
			return 0, fmt.Errorf("decode step %d: %w", pos, err)
		}
		return outTok, nil
	}
	noopShow := func(int32) error { return nil }
	for iter := 0; iter < benchCount; iter++ {
		reset()
		st := newDecodeState(prompt, steps, nil)
		ttftDur, decDur, err := generate(st, prompt, forward, forward, plain, noopShow, maxSeq, steps, 0)
		if err != nil {
			return err
		}
		if len(st.generated) == 0 {
			return fmt.Errorf("benchmark generated 0 tokens (prompt len %d, maxSeq %d)", len(prompt), maxSeq)
		}
		ttftNs := ttftDur.Nanoseconds()
		tpotNs := (decDur / time.Duration(len(st.generated))).Nanoseconds()
		tokPerSec := float64(len(st.generated)) / decDur.Seconds()
		if _, err := fmt.Fprintf(out, "%s/TTFT-1 1 %d ns/op\n%s/TPOT-1 1 %d ns/op\n%s/decode-1 1 %.6f tok/s\n",
			prefix, ttftNs, prefix, tpotNs, prefix, tokPerSec); err != nil {
			return err
		}
	}
	return nil
}

// achievedBandwidthGBps returns the achieved weight-streaming bandwidth in
// GB/s (1e9 bytes/s) for a decode step that streams streamedBytes in tpot.
// Since 1 byte/ns == 1e9 bytes/s == 1 GB/s, this is streamedBytes / tpot.Nanoseconds().
// It returns 0 when streamedBytes <= 0 or tpot <= 0.
func achievedBandwidthGBps(streamedBytes int64, tpot time.Duration) float64 {
	if streamedBytes <= 0 || tpot <= 0 {
		return 0
	}
	ns := tpot.Nanoseconds()
	if ns <= 0 {
		return 0
	}
	return float64(streamedBytes) / float64(ns)
}

// formatBenchmarkBandwidth formats the streamed weight bytes per token and
// achieved streaming bandwidth in GB/s for a decode TPOT duration. It returns
// an empty string when streamedBytes <= 0 or tpot <= 0.
func formatBenchmarkBandwidth(streamedBytes int64, tpot time.Duration) string {
	bw := achievedBandwidthGBps(streamedBytes, tpot)
	if bw <= 0 {
		return ""
	}
	return fmt.Sprintf("%d bytes/tok (%.2f GB/s)", streamedBytes, bw)
}

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
