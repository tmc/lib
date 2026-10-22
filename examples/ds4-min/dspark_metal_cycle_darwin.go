//go:build darwin && arm64

package main

import (
	"fmt"
	"time"

	"github.com/tmc/lib/examples/ds4-min/dsv4"
)

// metalDSparkCycle owns the adaptive schedule for one request. Support
// proposals are produced once, before runtime caps; only verified target
// rows become history. The bounded verifier admits at most eight rows.
type metalDSparkCycle struct {
	drafter  *metalDSparkDrafter
	verifier *dsv4.Speculator
	policy   metalDSparkPolicy
	schedule dsparkSchedule
	targetMS float64
	failed   error
}

func newMetalDSparkCycle(backend *metalStateAdapter, target *metalRequestState, w *metalDSparkWeights, threshold float32, policy metalDSparkPolicy, getenv func(string) string) (*metalDSparkCycle, error) {
	if w == nil || w.model == nil || w.model.block > 7 || getenv == nil {
		return nil, fmt.Errorf("metal dspark cycle: seed-plus-seven support and schedule configuration required")
	}
	d, verifier, err := newMetalDSparkSpeculator(backend, target, w, threshold)
	if err != nil {
		return nil, err
	}
	return &metalDSparkCycle{drafter: d, verifier: verifier, policy: policy, schedule: dsparkSchedule{policy: newMetalDSparkSchedulePolicy(policy.seedBatchEnabled(), getenv)}}, nil
}

func (c *metalDSparkCycle) evalSeed(seed int) error {
	d := c.drafter
	start := time.Now()
	err := d.verifier.adapter.Eval(d.forward.target, seed, d.forward.target.runtime.pos)
	if err == nil {
		c.targetMS = float64(time.Since(start)) / float64(time.Millisecond)
	}
	return err
}

func (c *metalDSparkCycle) note(accepted int, start time.Time, seedMS float64) {
	extra := max(0, float64(time.Since(start))/float64(time.Millisecond)-seedMS)
	c.noteConfidence(accepted, extra, c.drafter.forward.confidenceLogits.HostSlice()[0])
}

// The scheduler compares the raw head logit, not its sigmoid probability.
func (c *metalDSparkCycle) noteConfidence(accepted int, extra float64, confidenceLogit float32) {
	c.schedule.note(uint32(accepted), accepted == 0, extra, c.targetMS, float64(confidenceLogit), true)
}

func (c *metalDSparkCycle) run(history []int, maxTokens, eos int, ignoreEOS bool) ([]int, error) {
	if c == nil {
		return nil, fmt.Errorf("metal dspark cycle: missing owner")
	}
	if c.failed != nil {
		return nil, c.failed
	}
	accepted, err := c.runCycle(history, maxTokens, eos, ignoreEOS)
	if err != nil {
		c.failed = err
	}
	return accepted, err
}

func (c *metalDSparkCycle) runCycle(history []int, maxTokens, eos int, ignoreEOS bool) ([]int, error) {
	if c == nil || c.drafter == nil || c.verifier == nil || maxTokens < 1 || len(history) < 2 {
		return nil, fmt.Errorf("metal dspark cycle: invalid request frontier")
	}
	d := c.drafter
	if d.forward == nil || d.forward.weights.model.block > 7 {
		return nil, fmt.Errorf("metal dspark cycle: support block exceeds admitted seed-plus-seven verifier")
	}
	if err := d.prepare(history); err != nil {
		return nil, err
	}
	seed, err := metalSpeculativeFrontier(d.verifier.CurrentLogits())
	if err != nil {
		return nil, err
	}
	start := time.Now()
	seedMS := float64(0)
	skip := c.schedule.shouldSkip()
	tail := c.schedule.policy.enabled && c.schedule.policy.tail != 0 && uint32(maxTokens) < c.schedule.policy.tail
	if skip || tail {
		if err := c.evalSeed(seed); err != nil {
			return nil, err
		}
		seedMS = c.targetMS
		if skip {
			c.schedule.note(0, true, 0, 0, 0, false)
		}
		return []int{seed}, nil
	}
	incoming := c.policy.seedBatchEnabled()
	if incoming {
		err = d.prepareIncoming(history, seed)
	} else {
		if err := c.evalSeed(seed); err != nil {
			return nil, err
		}
		seedMS = c.targetMS
		history = append(append([]int(nil), history...), seed)
		err = d.prepare(history)
	}
	if err != nil {
		return nil, err
	}
	proposal, err := d.forward.proposal(seed, d.forward.weights.model.block, d.threshold, incoming)
	if err != nil {
		return nil, err
	}
	// Metal's source short-fallback is based on the uncapped proposal.
	if incoming && len(proposal) < 3 {
		if err := c.evalSeed(seed); err != nil {
			return nil, err
		}
		seedMS = c.targetMS
		history = append(append([]int(nil), history...), seed)
		d.prepared = false
		if err := d.prepare(history); err != nil {
			return nil, err
		}
		incoming = false
	}
	if incoming {
		proposal = append([]int{seed}, proposal...)
	}
	budget := maxTokens
	prefix := []int(nil)
	if !incoming {
		prefix, budget = []int{seed}, budget-1
	}
	if budget < 1 || !ignoreEOS && seed == eos || len(proposal) == 0 {
		if incoming {
			if err := c.evalSeed(seed); err != nil {
				return nil, err
			}
			prefix = []int{seed}
			seedMS = c.targetMS
		}
		if err := d.DiscardDrafts(); err != nil {
			return nil, err
		}
		c.note(0, start, seedMS)
		return prefix, nil
	}
	proposal = proposal[:min(len(proposal), budget)]
	if !ignoreEOS {
		for i, token := range proposal {
			if token == eos {
				proposal = proposal[:i+1]
				break
			}
		}
	}
	d.pending, d.ready = append([]int(nil), proposal...), true
	result, err := c.verifier.VerifyPrepared(d.seed, len(d.history), len(proposal))
	if err != nil {
		return nil, err
	}
	accepted := append(prefix, proposal[:result.Accepted]...)
	if incoming && result.Accepted == 0 {
		if err := c.evalSeed(seed); err != nil {
			return nil, err
		}
		accepted = []int{seed}
	}
	n := len(accepted) - 1
	c.note(n, start, seedMS)
	return accepted, nil
}
