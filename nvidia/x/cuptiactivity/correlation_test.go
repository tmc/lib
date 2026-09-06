package cuptiactivity_test

import (
	"fmt"
	"os"
	"testing"

	"github.com/tmc/lib/nvidia/cupti"
	"github.com/tmc/lib/nvidia/x/cuptiactivity"
)

type copyPair struct{ api, copy cuptiactivity.Record }

// correlateCopies checks identities and a bracket measured by cuptiGetTimestamp.
// It never treats API duration as device duration or requires their ends to nest.
func correlateCopies(records []cuptiactivity.Record, before, after, bytes uint64) ([]copyPair, error) {
	if before == 0 || after <= before {
		return nil, fmt.Errorf("invalid CUPTI clock bracket [%d,%d]", before, after)
	}
	apis := make(map[uint32]cuptiactivity.Record)
	for _, r := range records {
		if r.Kind != cupti.ActivityKindDriver {
			continue
		}
		if _, exists := apis[r.CorrelationID]; exists {
			return nil, fmt.Errorf("duplicate API correlation %d", r.CorrelationID)
		}
		apis[r.CorrelationID] = r
	}
	var pairs []copyPair
	seen := make(map[uint32]bool)
	for _, r := range records {
		if r.Kind != cupti.ActivityKindMemcpy {
			continue
		}
		a, ok := apis[r.CorrelationID]
		if !ok || r.CorrelationID == 0 || seen[r.CorrelationID] {
			return nil, fmt.Errorf("copy correlation %d is missing or reused", r.CorrelationID)
		}
		seen[r.CorrelationID] = true
		if !a.Complete || !r.Complete || a.Start < before || a.End > after || r.Start < before || r.End > after || a.Start > r.Start {
			return nil, fmt.Errorf("correlation %d outside CUPTI clock bracket or unordered", r.CorrelationID)
		}
		if a.ReturnValue != 0 || a.ProcessID != uint32(os.Getpid()) || r.Bytes != bytes || r.DeviceID != 0 {
			return nil, fmt.Errorf("correlation %d identity/status/size mismatch", r.CorrelationID)
		}
		if len(pairs) > 0 && (r.ContextID != pairs[0].copy.ContextID || r.StreamID != pairs[0].copy.StreamID) {
			return nil, fmt.Errorf("copy context/stream changed")
		}
		pairs = append(pairs, copyPair{a, r})
	}
	if len(pairs) != 2 {
		return nil, fmt.Errorf("correlated copies=%d, want 2", len(pairs))
	}
	return pairs, nil
}

func TestCopyCorrelationOracle(t *testing.T) {
	records := []cuptiactivity.Record{
		{Kind: cupti.ActivityKindDriver, CorrelationID: 1, Start: 20, End: 25, Complete: true, ProcessID: uint32(os.Getpid())},
		{Kind: cupti.ActivityKindMemcpy, CorrelationID: 1, Start: 22, End: 30, Complete: true, Bytes: 64},
		{Kind: cupti.ActivityKindDriver, CorrelationID: 2, Start: 40, End: 60, Complete: true, ProcessID: uint32(os.Getpid())},
		{Kind: cupti.ActivityKindMemcpy, CorrelationID: 2, Start: 42, End: 50, Complete: true, Bytes: 64},
	}
	if _, err := correlateCopies(records, 10, 70, 64); err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct {
		name   string
		change func([]cuptiactivity.Record)
	}{
		{"wrong correlation", func(r []cuptiactivity.Record) { r[1].CorrelationID = 99 }},
		{"wrong clock", func(r []cuptiactivity.Record) { r[1].Start = 1 }},
		{"wrong process", func(r []cuptiactivity.Record) { r[0].ProcessID++ }},
		{"wrong bytes", func(r []cuptiactivity.Record) { r[1].Bytes++ }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			r := append([]cuptiactivity.Record(nil), records...)
			tc.change(r)
			if _, err := correlateCopies(r, 10, 70, 64); err == nil {
				t.Fatal("bad trace accepted")
			}
		})
	}
}
