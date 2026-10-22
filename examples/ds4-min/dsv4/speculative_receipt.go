package dsv4

import (
	"crypto/sha256"
	"encoding/binary"
	"fmt"
)

// SpeculativeStep records the input history for one generated token.
// HistorySHA256 hashes int32 little-endian token IDs, including the prompt
// and previously generated tokens, but not Token itself.
type SpeculativeStep struct {
	Position      int
	Token         int
	HistorySHA256 string
}

// SpeculativeCaseReceipt supplements an identity trace with its prompt and
// per-step provenance. Name identifies a preregistered case, not a result.
type SpeculativeCaseReceipt struct {
	Name             string
	Prompt           []int
	Trace            SpeculativeTrace
	StrictSteps      []SpeculativeStep
	SpeculativeSteps []SpeculativeStep
}

// SpeculativeCaseContract pins the prompt and generation count before a run.
type SpeculativeCaseContract struct {
	PromptSHA256 string
	Generated    int
}

// CheckSpeculativeReceipts requires exactly the eight named preregistered
// prompts and validates both histories before evaluating greedy identity.
// This is a receipt checker, not a cross-implementation numerical oracle or
// a device snapshot correctness check.
func CheckSpeculativeReceipts(names []string, contracts map[string]SpeculativeCaseContract, receipts []SpeculativeCaseReceipt, threshold float64) (SpeculativeIdentityReport, error) {
	var empty SpeculativeIdentityReport
	if len(names) != 8 || len(contracts) != 8 || len(receipts) != 8 {
		return empty, fmt.Errorf("speculative receipts: require eight cases")
	}
	want := make(map[string]bool)
	for _, name := range names {
		if name == "" || want[name] {
			return empty, fmt.Errorf("speculative receipts: empty or duplicate case name")
		}
		want[name] = true
	}
	traces := make([]SpeculativeTrace, 0, 8)
	for _, r := range receipts {
		if !want[r.Name] {
			return empty, fmt.Errorf("speculative receipts: unexpected or duplicate case %q", r.Name)
		}
		delete(want, r.Name)
		contract, ok := contracts[r.Name]
		promptHash, err := SpeculativeHistorySHA256(r.Prompt)
		if !ok || err != nil || contract.Generated <= 0 || contract.PromptSHA256 != promptHash || len(r.Trace.Strict) != contract.Generated || len(r.Trace.Speculative) != contract.Generated {
			return empty, fmt.Errorf("speculative receipts: case %q violates pinned prompt or count", r.Name)
		}
		if len(r.Prompt) == 0 {
			return empty, fmt.Errorf("speculative receipts: case %q has no prompt", r.Name)
		}
		for _, arm := range []struct {
			tokens []int
			steps  []SpeculativeStep
		}{{r.Trace.Strict, r.StrictSteps}, {r.Trace.Speculative, r.SpeculativeSteps}} {
			if len(arm.tokens) == 0 || len(arm.steps) != len(arm.tokens) {
				return empty, fmt.Errorf("speculative receipts: case %q missing steps", r.Name)
			}
			history := append([]int(nil), r.Prompt...)
			for i, step := range arm.steps {
				hash, err := SpeculativeHistorySHA256(history)
				if err != nil {
					return empty, err
				}
				if step.Position != len(history) || step.Token != arm.tokens[i] || step.HistorySHA256 != hash {
					return empty, fmt.Errorf("speculative receipts: case %q step %d provenance mismatch", r.Name, i)
				}
				if step.Token < 0 || int64(step.Token) > 2147483647 {
					return empty, fmt.Errorf("speculative receipts: invalid generated token")
				}
				history = append(history, step.Token)
			}
		}
		traces = append(traces, r.Trace)
	}
	return CheckSpeculativeIdentity(traces, threshold)
}

// SpeculativeHistorySHA256 returns the canonical hash of a token history.
func SpeculativeHistorySHA256(tokens []int) (string, error) {
	h := sha256.New()
	var b [4]byte
	for _, token := range tokens {
		if token < 0 || int64(token) > 2147483647 {
			return "", fmt.Errorf("speculative history: invalid token %d", token)
		}
		binary.LittleEndian.PutUint32(b[:], uint32(token))
		h.Write(b[:])
	}
	return fmt.Sprintf("%x", h.Sum(nil)), nil
}
