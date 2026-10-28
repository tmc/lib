package cudatrial

import (
	"context"
	"errors"
	"math"
	"testing"
	"time"
)

func TestReceiptCloneMemory(t *testing.T) {
	m, _ := NewThreadEdgeMap(16, 1)
	m.Record(0, 3)
	failure := errors.New("producer failed")
	r := Receipt{Before: Conditions{Values: map[ConditionKind]int64{ConditionHostAvailableBytes: 0}}, After: Conditions{Values: map[ConditionKind]int64{}}, Timings: map[TimingKind]time.Duration{TimingGPUBusy: 0}, Provenance: map[string]string{"producer": "original"}, Coverage: m.Snapshot(), Err: failure}
	var charged int64
	c, err := r.CloneWithMemory(func(n int64) error { charged += n; return nil })
	if err != nil || charged <= int64(len(r.Coverage.counts)) {
		t.Fatal(charged, err)
	}
	r.Before.Values[ConditionHostAvailableBytes] = 9
	r.Timings[TimingGPUBusy] = time.Hour
	r.Provenance["producer"] = "changed"
	r.Coverage.Reset()
	if c.Before.Values[ConditionHostAvailableBytes] != 0 || !c.Before.Has(ConditionHostAvailableBytes) || !c.HasTiming(TimingGPUBusy) || c.Timings[TimingGPUBusy] != 0 || c.Provenance["producer"] != "original" || c.Coverage.counts[11] != 1 || !errors.Is(c.Err, failure) {
		t.Fatalf("clone=%+v coverage=%v", c, c.Coverage.Bytes())
	}
	denied := errors.New("denied")
	z, err := r.CloneWithMemory(func(int64) error { return denied })
	if !errors.Is(err, denied) || z.Timings != nil || z.Coverage != nil {
		t.Fatal(z, err)
	}
}

func TestInterleaveMemoryPreflight(t *testing.T) {
	denied := errors.New("denied")
	calls := 0
	arms := []Arm{{Name: "a", Run: func(context.Context) (Receipt, error) { calls++; return Receipt{Complete: true}, nil }}}
	r, err := InterleaveWithOptions(t.Context(), arms, 2, Options{ReserveMemory: func(int64) error { return denied }})
	if !errors.Is(err, denied) || r != nil || calls != 0 {
		t.Fatal(r, err, calls)
	}
	_, err = InterleaveWithOptions(t.Context(), append(arms, arms...), math.MaxInt, Options{})
	if err == nil || calls != 0 {
		t.Fatal("overflow admitted", err, calls)
	}
}
