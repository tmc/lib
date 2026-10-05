package cudarepro

import (
	"context"
	"errors"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudaschedule"
)

func measureSequence(t *testing.T, indexes ...int) Measurement {
	t.Helper()
	n := 0
	m, err := Measure(context.Background(), Identity{Policy: "exact-uint64"}, len(indexes), func(context.Context) (cudaschedule.RunRecord, error) {
		r := mismatchRecord(uint64(indexes[n]))
		if indexes[n] < 0 {
			r.Stdout = "PASS"
		}
		n++
		return r, nil
	})
	if err != nil || n != len(indexes) {
		t.Fatalf("Measure: calls=%d, error=%v", n, err)
	}
	return m
}

func TestRatePreservesIdentityAndFrequency(t *testing.T) {
	i := Identity{Policy: "exact-uint64"}
	baseline := measureSequence(t, 0, 0, 0, -1)
	for _, tt := range []struct {
		name    string
		indexes []int
		want    bool
	}{
		{"equal", []int{0, 0, 0, -1}, true},
		{"better", []int{0, 0, 0, 0}, true},
		{"rate drift", []int{0, -1, -1, -1}, false},
		{"identity drift", []int{1, 1, 1, 1}, false},
		{"healthy", []int{-1, -1, -1, -1}, false},
	} {
		t.Run(tt.name, func(t *testing.T) {
			got, err := AcceptRate(i, baseline, measureSequence(t, tt.indexes...), false)
			if err != nil || got != tt.want {
				t.Fatalf("AcceptRate=%v, %v; want %v", got, err, tt.want)
			}
		})
	}
	one := measureSequence(t, 0, -1, -1, -1)
	if !one.OneOff() {
		t.Fatal("one-off not reported")
	}
	for _, allow := range []bool{false, true} {
		got, err := AcceptRate(i, one, one, allow)
		if err != nil || got != allow {
			t.Fatalf("one-off acceptance=%v, %v", got, err)
		}
	}
	if _, err := AcceptRate(i, baseline, one, true); err != nil {
		t.Fatal(err)
	}
	one.Reproductions++
	if _, err := AcceptRate(i, baseline, one, true); err == nil {
		t.Fatal("accepted invented count")
	}
}

func TestMeasureRejectsInvalidAndStops(t *testing.T) {
	i := Identity{Policy: "exact-uint64"}
	calls := 0
	replay := func(context.Context) (cudaschedule.RunRecord, error) { calls++; return mismatchRecord(0), nil }
	for _, count := range []int{0, -1} {
		if _, err := Measure(context.Background(), i, count, replay); err == nil {
			t.Fatal("accepted invalid count")
		}
	}
	if calls != 0 {
		t.Fatal("invalid count executed runner")
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := Measure(ctx, i, 4, replay); !errors.Is(err, context.Canceled) {
		t.Fatalf("cancellation: %v", err)
	}
	if calls != 0 {
		t.Fatal("cancelled measurement executed runner")
	}
	m, err := Measure(context.Background(), i, 4, func(context.Context) (cudaschedule.RunRecord, error) {
		calls++
		r := mismatchRecord(0)
		if calls == 2 {
			r.Mismatch = nil
		}
		return r, nil
	})
	if err == nil || calls != 2 || len(m.Runs) != 2 || m.Reproductions != 1 {
		t.Fatalf("invalid evidence retried: %+v, calls=%d, err=%v", m, calls, err)
	}
}

func TestMeasureSnapshotsReusedReceipt(t *testing.T) {
	r := mismatchRecord(0)
	n := 0
	m, err := Measure(context.Background(), Identity{Policy: "exact-uint64"}, 3, func(context.Context) (cudaschedule.RunRecord, error) {
		r.Mismatch.Index = uint64(n)
		n++
		return r, nil
	})
	if err != nil || m.Reproductions != 1 {
		t.Fatalf("Measure=%+v, %v", m, err)
	}
	for n, run := range m.Runs {
		if run.Mismatch.Index != uint64(n) {
			t.Fatal("earlier receipt mutated")
		}
	}
	r.Baseline = &r
	if _, err := snapshot(r); err == nil {
		t.Fatal("accepted cyclic receipt")
	}
}

func TestAcceptRateRefusesInvalidMeasurements(t *testing.T) {
	i := Identity{Policy: "exact-uint64"}
	b := measureSequence(t, 0, 0)
	for _, candidate := range []Measurement{
		{},
		measureSequence(t, 0),
		{Reproductions: -1, Runs: b.Runs},
		{Reproductions: 3, Runs: b.Runs},
	} {
		if _, err := AcceptRate(i, b, candidate, true); err == nil {
			t.Fatal("accepted invalid measurement")
		}
	}
	healthy := measureSequence(t, -1, -1)
	if _, err := AcceptRate(i, healthy, b, true); err == nil {
		t.Fatal("accepted unreproduced baseline")
	}
}
