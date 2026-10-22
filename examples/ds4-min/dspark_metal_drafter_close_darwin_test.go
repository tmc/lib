//go:build darwin && arm64

package main

import (
	"errors"
	"testing"

	"github.com/tmc/lib/examples/ds4-min/dsv4"
)

func TestMetalDSparkDrafterVerifierClose(t *testing.T) {
	for _, fail := range []bool{false, true} {
		frontier := &metalLifecycleSnapshot{fail: fail}
		v := &metalSpeculativeVerifier{frontier: frontier}
		d := &metalDSparkDrafter{verifier: v, prepared: true, drafted: true, ready: true, history: []int{1}, pending: []int{2}}
		err := d.Close()
		if !v.closed || frontier.calls != 1 || d.prepared || d.drafted || d.ready || len(d.history) != 0 || len(d.pending) != 0 {
			t.Errorf("owner close bypassed verifier: closed=%v calls=%d", v.closed, frontier.calls)
		}
		if fail {
			if !errors.Is(err, dsv4.ErrDesynchronized) || d.verifier != v || v.frontier != frontier {
				t.Errorf("failed verifier close lost retry owner: %v", err)
			}
			frontier.fail = false
			if err := d.Close(); err != nil || d.verifier != nil || v.frontier != nil || frontier.calls != 2 {
				t.Errorf("owner retry failed: %v", err)
			}
		} else if err != nil || d.verifier != nil || v.frontier != nil {
			t.Errorf("successful owner close retained verifier: %v", err)
		}
		calls := frontier.calls
		if err := d.Close(); err != nil || frontier.calls != calls {
			t.Errorf("owner double closed verifier: %v", err)
		}
	}
}
