package cudahotswap

import (
	"context"
	"math"
	"testing"
)

func TestServiceSetCheckSnapshot(t *testing.T) {
	s, _, _, config, leases := serviceFixture(t)
	defer s.Close(context.Background())
	f, err := s.RecordFixture(context.Background(), "kernel", config)
	if err != nil {
		t.Fatal(err)
	}
	before, _ := s.Status("kernel")
	p := DefaultReplayPolicy()
	p.Rounds = 7
	calls := 0
	c := CheckConfig{Fixture: f, Policy: &p, Checker: func(context.Context, State) error { calls++; return nil }, KeepOnFailure: true}
	state, err := s.SetCheck("kernel", c, "replay and app")
	if err != nil || state.Generation != before.Generation || state.Version != before.Version || state.ConfigRevision != before.ConfigRevision+1 || state.Label != "replay and app" {
		t.Fatalf("check config: %+v, %v", state, err)
	}
	p.Rounds = 99
	c.KeepOnFailure = false
	got := s.settings["kernel"].check
	if got.Policy.Rounds != 7 || !got.KeepOnFailure || got.Fixture != f || got.Checker == nil || calls != 0 || len(*leases) != 2 {
		t.Fatal("settings alias caller, run checks or acquire a lease")
	}
	if f.replay.policy.Rounds != 5 {
		t.Fatal("settings changed the recorded fixture")
	}
	if _, err := s.SetCheck("kernel", CheckConfig{Checker: c.Checker}, "app only"); err != nil {
		t.Fatal(err)
	}
	if s.settings["kernel"].check.Fixture != nil || s.settings["kernel"].check.Policy != nil {
		t.Fatal("application-only configuration retained replay")
	}
}

func TestServiceSetCheckRefusal(t *testing.T) {
	s, _, _, config, leases := serviceFixture(t)
	defer s.Close(context.Background())
	f, err := s.RecordFixture(context.Background(), "kernel", config)
	if err != nil {
		t.Fatal(err)
	}
	other, _, _, _, _ := serviceFixture(t)
	defer other.Close(context.Background())
	p := DefaultReplayPolicy()
	for _, tc := range []struct {
		name    string
		service *Service
		kernel  string
		config  CheckConfig
	}{
		{"nil service", nil, "kernel", CheckConfig{Fixture: f}},
		{"no check", s, "kernel", CheckConfig{}},
		{"policy alone", s, "kernel", CheckConfig{Policy: &p}},
		{"foreign service", other, "kernel", CheckConfig{Fixture: f}},
		{"foreign kernel", s, "missing", CheckConfig{Fixture: f}},
		{"unrecorded", s, "kernel", CheckConfig{Fixture: new(Fixture)}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if _, err := tc.service.SetCheck(tc.kernel, tc.config, "refused"); err == nil {
				t.Fatal("invalid check admitted")
			}
		})
	}
	for _, bad := range []ReplayPolicy{{}, {Rounds: 65}, {Rounds: 1, MaxAbs: math.NaN()}, {Rounds: 1, RMS: math.Inf(1)}, {Rounds: 1, NoiseRMS: -1}} {
		if _, err := s.SetCheck("kernel", CheckConfig{Fixture: f, Policy: &bad}, "bad policy"); err == nil {
			t.Fatal("invalid replay policy admitted")
		}
	}
	f.replay.noise[1] = ReplayMetrics{MaxAbs: replayFloatPointer(1), RMS: replayFloatPointer(1)}
	if _, err := s.SetCheck("kernel", CheckConfig{Fixture: f}, "noise"); err == nil {
		t.Fatal("recorded excess noise admitted")
	}
	state, _ := s.Status("kernel")
	if state.ConfigRevision != 0 || s.settings["kernel"].check != nil || len(*leases) != 2 {
		t.Fatal("refusal changed revision, settings or leases")
	}
	if err := f.Close(context.Background()); err != nil {
		t.Fatal(err)
	}
	if _, err := s.SetCheck("kernel", CheckConfig{Fixture: f}, "closed"); err == nil {
		t.Fatal("closed fixture admitted")
	}
}

func replayFloatPointer(v float64) *float64 { return &v }
