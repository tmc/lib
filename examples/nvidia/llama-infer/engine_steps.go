package main

import (
	"encoding/json"
	"fmt"
	"io"
	"time"
)

const engineStepBoundary = "host_input_to_host_token_v1"

// engineStep measures one prediction, including input staging and delivery of
// its sampled token to the host. The prior-stream drain is outside WallNS.
type engineStep struct {
	Boundary    string `json:"boundary"`
	Phase       string `json:"phase"`
	OutputIndex int    `json:"output_index"`
	ContextLen  int    `json:"context_len"`
	WallNS      int64  `json:"wall_ns"`
	TokenID     int32  `json:"token_id"`
}

// generateEngineSteps predicts exactly steps tokens. Unlike generate, it does
// not predict an unused token after emitting the last requested output.
func generateEngineSteps(prompt []int32, steps, maxSeq int, drain func() error, prefill func(int32, int) (int32, error), plain func(int, int32, int) (int32, error), emit func(int32) error, now func() time.Time) ([]engineStep, error) {
	if len(prompt) == 0 || steps <= 0 || len(prompt) > maxSeq || steps-1 > maxSeq-len(prompt) {
		return nil, fmt.Errorf("engine steps require a non-empty prompt and room for all predictions")
	}
	packets := make([]engineStep, 0, steps)
	var next int32
	for index := 0; index < steps; index++ {
		if err := drain(); err != nil {
			return nil, fmt.Errorf("engine step %d drain: %w", index, err)
		}
		start := now()
		var err error
		if index == 0 {
			for pos, tok := range prompt {
				next, err = prefill(tok, pos)
				if err != nil {
					return nil, fmt.Errorf("engine prefill position %d: %w", pos, err)
				}
			}
		} else {
			next, err = plain(index-1, next, len(prompt)+index-1)
			if err != nil {
				return nil, fmt.Errorf("engine decode output %d: %w", index, err)
			}
		}
		elapsed := now().Sub(start)
		if elapsed <= 0 {
			return nil, fmt.Errorf("engine output %d has non-positive duration", index)
		}
		phase := "decode"
		if index == 0 {
			phase = "prefill"
		}
		packets = append(packets, engineStep{engineStepBoundary, phase, index, len(prompt) + index, elapsed.Nanoseconds(), next})
		if err := emit(next); err != nil {
			return nil, fmt.Errorf("emit engine output %d: %w", index, err)
		}
	}
	return packets, nil
}

func writeEngineSteps(w io.Writer, packets []engineStep) error {
	for _, packet := range packets {
		data, err := json.Marshal(packet)
		if err != nil {
			return err
		}
		if _, err = fmt.Fprintf(w, "ENGINE_STEP %s\n", data); err != nil {
			return err
		}
	}
	return nil
}

// engineStepOptions names modes whose different generation paths are not part
// of the registered engine-step boundary.
type engineStepOptions struct {
	Steps                                          int
	Draft, TP                                      int
	Bench, StepTimes, Trace, Profile, Disasm, Emit bool
	Greedy                                         bool
	ExplicitTokens                                 bool
}

func validateEngineStepOptions(o engineStepOptions) error {
	if o.Steps != 128 || !o.ExplicitTokens {
		return fmt.Errorf("-engine-step-times requires -steps=128 and explicit -token-ids")
	}
	if o.Draft != 0 || o.TP != 0 || o.Bench || o.StepTimes || o.Trace || o.Profile || o.Disasm || o.Emit || !o.Greedy {
		return fmt.Errorf("-engine-step-times requires greedy single-device generation without draft, bench, step-times, trace, profile, disasm or egraph-emit")
	}
	return nil
}

// warmupEngineSteps runs exactly one full-shape throwaway request, retaining
// its IDs. Reset rewinds sequence state; drain completes it before measurement.
func warmupEngineSteps(prompt []int32, steps, maxSeq int, reset func() error, drain func() error, prefill func(int32, int) (int32, error), plain func(int, int32, int) (int32, error), now func() time.Time) ([]int32, error) {
	if err := reset(); err != nil {
		return nil, fmt.Errorf("reset before engine warmup: %w", err)
	}
	ids := make([]int32, 0, steps)
	emit := func(tok int32) error { ids = append(ids, tok); return nil }
	if _, err := generateEngineSteps(prompt, steps, maxSeq, drain, prefill, plain, emit, now); err != nil {
		return nil, fmt.Errorf("engine warmup: %w", err)
	}
	if err := drain(); err != nil {
		return nil, fmt.Errorf("drain engine warmup: %w", err)
	}
	if err := reset(); err != nil {
		return nil, fmt.Errorf("reset after engine warmup: %w", err)
	}
	if err := drain(); err != nil {
		return nil, fmt.Errorf("drain reset engine warmup: %w", err)
	}
	return ids, nil
}
