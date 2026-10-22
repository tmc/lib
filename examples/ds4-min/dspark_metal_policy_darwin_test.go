//go:build darwin && arm64

package main

import "testing"

func TestMetalDSparkPolicy(t *testing.T) {
	for _, test := range []struct {
		name string
		p    metalDSparkPolicy
		want bool
	}{
		{"m5-iq2-q2", metalDSparkPolicy{m5: true, gateType: 16, downType: 10}, true},
		{"other-apple", metalDSparkPolicy{gateType: 16, downType: 10}, false},
		{"streaming", metalDSparkPolicy{m5: true, streaming: true, gateType: 16, downType: 10}, false},
		{"exact", metalDSparkPolicy{m5: true, exactSampling: true, gateType: 16, downType: 10}, false},
		{"tp-mxfp4", metalDSparkPolicy{m5: true, tensorParallel: true, gateType: 39, downType: 39}, true},
		{"non-tp-mxfp4", metalDSparkPolicy{m5: true, gateType: 39, downType: 39}, false},
		{"override-disable", metalDSparkPolicy{m5: true, gateType: 16, downType: 10, seedOverride: "0yes"}, false},
		{"override-enable", metalDSparkPolicy{seedOverride: "false"}, true},
	} {
		t.Run(test.name, func(t *testing.T) {
			if got := test.p.seedBatchEnabled(); got != test.want {
				t.Fatalf("enabled=%v want=%v", got, test.want)
			}
		})
	}
	for _, n := range []int{0, 1, 2, 3, 16} {
		got, err := metalDSparkNeedsShortFallback(2, n)
		if err != nil || got != (n < 3) {
			t.Fatalf("proposal %d fallback=%v err=%v", n, got, err)
		}
	}
	if _, err := metalDSparkNeedsShortFallback(1, 3); err == nil {
		t.Fatal("minimum-two preparation must be admitted first")
	}
}
