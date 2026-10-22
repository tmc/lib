//go:build darwin && arm64

package dsv4expertcache

import (
	"errors"
	"testing"
)

// The physical gate retains the entire resource graph on nonterminal waits.
// No output, pipeline, slab or device may close while a submitted read is live.
type metalDeviceGateOwner struct {
	submitted bool
	wait      func() error
	terminal  func() bool
	cleanup   []func() error
}

var retainedMetalDeviceGates []*metalDeviceGateOwner

func (o *metalDeviceGateOwner) close() error {
	if o.submitted {
		if err := o.wait(); err != nil {
			return err
		}
		if !o.terminal() {
			return errors.New("nonterminal device gate")
		}
	}
	for len(o.cleanup) > 0 {
		i := len(o.cleanup) - 1
		if err := o.cleanup[i](); err != nil {
			return err
		}
		o.cleanup = o.cleanup[:i]
	}
	return nil
}

func TestMetalDeviceGateOwner(t *testing.T) {
	t.Run("release-error", func(t *testing.T) {
		failure := errors.New("release failed")
		o := &metalDeviceGateOwner{cleanup: []func() error{func() error { return failure }}}
		if err := o.close(); !errors.Is(err, failure) || len(o.cleanup) != 1 {
			t.Fatal("release error discarded", err)
		}
	})
	for _, scenario := range []string{"failed-wait", "nonterminal", "unsubmitted", "partial-constructor"} {
		t.Run(scenario, func(t *testing.T) {
			freed := 0
			pending := scenario == "failed-wait"
			terminal := scenario != "nonterminal"
			o := &metalDeviceGateOwner{submitted: scenario == "failed-wait" || scenario == "nonterminal", wait: func() error {
				if pending {
					return errors.New("wait failed")
				}
				return nil
			}, terminal: func() bool { return terminal }, cleanup: []func() error{func() error { freed++; return nil }}}
			err := o.close()
			if o.submitted {
				if err == nil || freed != 0 {
					t.Fatal("live gate freed resources", err, freed)
				}
				pending = false
				terminal = true
				if err := o.close(); err != nil || freed != 1 {
					t.Fatal("terminal retry failed", err, freed)
				}
			} else if err != nil || freed != 1 {
				t.Fatal("unsubmitted cleanup failed", err, freed)
			}
		})
	}
}
