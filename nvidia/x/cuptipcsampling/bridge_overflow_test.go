package cuptipcsampling_test

import (
	"bytes"
	"context"
	"github.com/tmc/lib/nvidia/x/cudaprof"
	"github.com/tmc/lib/nvidia/x/cuptipcsampling"
	"math"
	"strings"
	"testing"
)

func TestBridgeOverflowDoesNotRecordBatch(t *testing.T) {
	for _, tc := range []struct {
		name string
		set  func(*cuptipcsampling.ProfileBridge)
		want string
	}{
		{"total", func(b *cuptipcsampling.ProfileBridge) { b.TotalSamples = math.MaxInt64 }, "total sample count overflow"},
		{"reason", func(b *cuptipcsampling.ProfileBridge) { b.ReasonSamples = map[string]int64{"selected": math.MaxInt64} }, "reason sample count overflow"},
		{"unmapped", func(b *cuptipcsampling.ProfileBridge) { b.UnmappedStalls = map[string]int64{"unknown": math.MaxInt64} }, "unmapped sample count overflow"},
		{"denominator", func(b *cuptipcsampling.ProfileBridge) { b.DroppedSamples = math.MaxUint64 }, "coverage sample count overflow"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var out bytes.Buffer
			b := &cuptipcsampling.ProfileBridge{Session: cudaprof.NewSession(&out), Inventory: cuptipcsampling.ReasonInventory{Reasons: []cuptipcsampling.StallReason{{Index: 1, Name: "selected"}, {Index: 2, Name: "unknown"}}}}
			tc.set(b)
			total, lost, other := b.TotalSamples, b.DroppedSamples, b.StallOther
			batch := cuptipcsampling.Batch{Valid: true, PCs: []cuptipcsampling.PC{{Name: "kernel", Reasons: []cuptipcsampling.StallSamples{{Index: 1, Samples: 1}, {Index: 2, Samples: 1}}}}}
			if err := b.RecordBatch(context.Background(), batch); err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("error = %v, want %s", err, tc.want)
			}
			if b.TotalSamples != total || b.DroppedSamples != lost || b.StallOther != other || b.StallCoverage != nil {
				t.Fatal("rejected batch changed accumulated accounting")
			}
		})
	}
}
