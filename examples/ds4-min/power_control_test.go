package main

import (
	"errors"
	"math"
	"reflect"
	"testing"
	"time"
)

func TestPowerAdmission(t *testing.T) {
	for _, tt := range []struct {
		percent int
		family  string
		want    bool
	}{
		{0, "flash", false}, {1, "flash", true}, {100, "flash", true}, {101, "flash", false},
		{50, "v41", false}, {100, "v41", true}, {50, "glm52", false}, {100, "glm52", true},
		{50, "pro", true},
	} {
		if got := validatePower(tt.percent, tt.family) == nil; got != tt.want {
			t.Errorf("power %d family %s: admitted %v, want %v", tt.percent, tt.family, got, tt.want)
		}
	}
}

func TestPowerMeasuredBoundaries(t *testing.T) {
	p := newPowerControl(50, 2)
	current := time.Unix(0, 0)
	var events []string
	var sleeps []time.Duration
	p.now = func() time.Time { events = append(events, "clock"); return current }
	p.sleep = func(d time.Duration) {
		events = append(events, "sleep")
		sleeps = append(sleeps, d)
		current = current.Add(d)
	}
	p.prefill = true
	for _, sample := range []time.Duration{8 * time.Millisecond, 16 * time.Millisecond} {
		start := p.start()
		if err := p.finishLayer(0, start, func() error { events = append(events, "wait"); current = current.Add(sample); return nil }); err != nil {
			t.Fatal(err)
		}
	}
	if !reflect.DeepEqual(sleeps, []time.Duration{8 * time.Millisecond, 9 * time.Millisecond}) {
		t.Fatalf("measured sleeps %v", sleeps)
	}
	if !reflect.DeepEqual(events, []string{"clock", "wait", "clock", "sleep", "clock", "wait", "clock", "sleep"}) {
		t.Fatalf("completion order %v", events)
	}
	if p.layers[1] != 0 {
		t.Fatal("layer averages alias")
	}
	start := p.start()
	boom := errors.New("device completion")
	if err := p.finishLayer(0, start, func() error { return boom }); !errors.Is(err, boom) {
		t.Fatalf("failure %v", err)
	}
	if len(sleeps) != 2 || p.layers[0] != powerAverage(.008, .016) {
		t.Fatal("failed work changed pacing")
	}
	p.prefill = false
	start = p.start()
	current = current.Add(4 * time.Millisecond)
	p.finishToken(start)
	if p.decode != 0.004 || sleeps[2] != 4*time.Millisecond {
		t.Fatalf("decode average %g sleeps %v", p.decode, sleeps)
	}
	child := p.newRequest()
	if child == p || child.decode != 0 || child.layers[0] != 0 || child.prefill {
		t.Fatal("request shares pacing history")
	}
}

func TestPowerDisabledAndSamples(t *testing.T) {
	p := newPowerControl(100, 43)
	if p != nil {
		t.Fatal("full power allocated pacing state")
	}
	if err := p.finishLayer(0, p.start(), func() error { t.Fatal("disabled pacing waited"); return nil }); err != nil {
		t.Fatal(err)
	}
	p.finishToken(p.start())
	for _, sample := range []float64{0, -1, math.NaN(), math.Inf(1)} {
		if got := powerAverage(2, sample); got != 2 {
			t.Fatalf("invalid sample %g changed average to %g", sample, got)
		}
	}
	for _, avg := range []float64{0, -1, math.NaN(), math.Inf(1)} {
		if got := powerAverage(avg, 3); got != 3 {
			t.Fatalf("invalid average %g became %g", avg, got)
		}
	}
}

func TestPowerOneShotSleep(t *testing.T) {
	calls := 0
	oneShotPowerSleep(2500*time.Millisecond+37*time.Nanosecond, func(sec, nsec int64) int32 {
		calls++
		if sec != 2 || nsec != 500000037 {
			t.Fatalf("timespec %d/%d", sec, nsec)
		}
		return -1 // EINTR must not cause a retry.
	})
	if calls != 1 {
		t.Fatalf("interrupted sleep calls %d", calls)
	}
}
