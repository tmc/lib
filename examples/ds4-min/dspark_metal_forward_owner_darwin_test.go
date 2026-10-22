//go:build darwin && arm64

package main

import (
	"strings"
	"testing"
)

// These cases exercise the production full-forward entry before any buffer
// is inspected. The request scratch has a real stage count; no device exists.
func TestMetalDSparkForwardOwnerAdmission(t *testing.T) {
	for _, name := range []string{"nil model", "closed weights", "closed attention", "zero stages", "zero block", "oversized block"} {
		t.Run(name, func(t *testing.T) {
			w := &metalDSparkWeights{model: &dsparkModel{block: 1, stages: 1}}
			d := &metalDSparkDense{weights: w}
			a := &metalDSparkAttention{weights: w}
			s := metalDSparkForwardRows{attention: make([]metalDSparkAttentionRows, 1), ffn: make([]metalDSparkFFNRows, 1)}
			switch name {
			case "nil model":
				w.model = nil
			case "closed weights":
				w.closed = true
			case "closed attention":
				a.closed = true
			case "zero stages":
				w.model.stages = 0
				s.attention, s.ffn = nil, nil
			case "zero block":
				w.model.block = 0
			case "oversized block":
				w.model.block = 17
			}
			defer func() {
				if r := recover(); r != nil {
					t.Errorf("full-forward owner admission panicked: %v", r)
				}
			}()
			err := d.EncodeForward(nil, a, 0, 130, 0, 0, 2, s)
			if err == nil || !strings.Contains(err.Error(), "invalid stage owners") {
				t.Fatalf("full-forward owner not refused first: %v", err)
			}
			if len(d.batches) != 0 || len(a.batches) != 0 {
				t.Fatal("rejected owner retained a producer batch")
			}
		})
	}
	t.Run("valid owners invalid batch", func(t *testing.T) {
		w := &metalDSparkWeights{model: &dsparkModel{block: 1, stages: 1}}
		d := &metalDSparkDense{weights: w}
		a := &metalDSparkAttention{weights: w}
		s := metalDSparkForwardRows{attention: make([]metalDSparkAttentionRows, 1), ffn: make([]metalDSparkFFNRows, 1)}
		defer func() {
			if r := recover(); r != nil {
				t.Errorf("full-forward batch admission panicked: %v", r)
			}
		}()
		err := d.EncodeForward(nil, a, 0, 130, 0, 0, 2, s)
		if err == nil || !strings.Contains(err.Error(), "stage batch") {
			t.Fatalf("full-forward batch not refused first: %v", err)
		}
		if len(d.batches) != 0 || len(a.batches) != 0 {
			t.Fatal("invalid batch retained a producer")
		}
	})
}
