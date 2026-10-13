package nvtelemetry_test

import (
	"bytes"
	"errors"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/nvtelemetry"
)

// budgetBoundaryStream uses the public writer so the reader sees canonical
// envelopes, sequence numbers, and completion counters.
func budgetBoundaryStream(t *testing.T, samples, phases int, complete bool, wrongRun bool, phasesFirst bool) []byte {
	t.Helper()
	var buf bytes.Buffer
	w := nvtelemetry.NewWriter(&buf, nvtelemetry.CollectorConfig{})
	m := nvtelemetry.Manifest{RunID: "budget-boundary", SchemaVersion: nvtelemetry.SchemaVersionV5}
	if err := w.WriteManifest(&m); err != nil {
		t.Fatal(err)
	}
	writeSamples := func() {
		for i := 0; i < samples; i++ {
			s := nvtelemetry.Sample{Seq: uint64(i + 1), QueryStartNanos: int64(i), QueryEndNanos: int64(i + 1)}
			if err := w.WriteSample(&s); err != nil {
				t.Fatal(err)
			}
		}
	}
	writePhases := func() {
		for i := 0; i < phases; i++ {
			p := nvtelemetry.PhaseMarker{Name: "phase", StartMonotonicNanos: int64(i), EndMonotonicNanos: int64(i + 1)}
			if err := w.WritePhase(&p); err != nil {
				t.Fatal(err)
			}
		}
	}
	if phasesFirst {
		writePhases()
		writeSamples()
	} else {
		writeSamples()
		writePhases()
	}
	if complete {
		m.IsComplete = true
		m.TotalSamples = uint64(samples)
		m.EndTimeMonotonic = int64(samples + phases)
		m.LostRecordsByFamily = w.FamilyLoss()
		if wrongRun {
			m.RunID = "other-run"
		}
		if err := w.WriteManifest(&m); err != nil {
			t.Fatal(err)
		}
	}
	return buf.Bytes()
}

func TestStreamBudgetBoundaryRoundTrip(t *testing.T) {
	for _, tc := range []struct {
		name            string
		samples, phases int
	}{
		{"samples-below", 65535, 0},
		{"samples-at", 65536, 0},
		{"phases-below", 0, 65535},
		{"phases-at", 0, 65536},
		{"sample-cap-then-phase", 65536, 1},
		{"phase-cap-then-sample", 1, 65536},
	} {
		t.Run(tc.name, func(t *testing.T) {
			raw := budgetBoundaryStream(t, tc.samples, tc.phases, true, false, tc.phases == 65536 && tc.samples == 1)
			m, phases, samples, err := nvtelemetry.ReadStreamStrict(bytes.NewReader(raw), 0)
			if err != nil {
				t.Fatal(err)
			}
			if !m.IsComplete || m.TotalSamples != uint64(tc.samples) || len(samples) != tc.samples || len(phases) != tc.phases {
				t.Fatalf("completion: samples %d/%d phases %d/%d manifest %+v", len(samples), tc.samples, len(phases), tc.phases, m)
			}
			for i, s := range samples {
				if s.Seq != uint64(i+1) || s.QueryStartNanos != int64(i) || s.QueryEndNanos != int64(i+1) || len(s.Readings) != 0 {
					t.Fatalf("sample %d: %+v", i, s)
				}
			}
			for i, p := range phases {
				if p.Name != "phase" || p.StartMonotonicNanos != int64(i) || p.EndMonotonicNanos != int64(i+1) {
					t.Fatalf("phase %d: %+v", i, p)
				}
			}
		})
	}
}

func TestStreamBudgetBoundaryOverflow(t *testing.T) {
	for _, tc := range []struct {
		name            string
		samples, phases int
	}{
		{"samples", 65537, 0},
		{"phases", 0, 65537},
	} {
		t.Run(tc.name, func(t *testing.T) {
			raw := budgetBoundaryStream(t, tc.samples, tc.phases, false, false, false)
			m, phases, samples, err := nvtelemetry.ReadStream(bytes.NewReader(raw), 0)
			if !errors.Is(err, nvtelemetry.ErrBudgetExceeded) || m != nil || phases != nil || samples != nil {
				t.Fatalf("overflow returned partial output: manifest %v phases %d samples %d error %v", m != nil, len(phases), len(samples), err)
			}
		})
	}
}

func TestStreamBudgetBoundaryTerminalIdentity(t *testing.T) {
	raw := budgetBoundaryStream(t, 65536, 0, true, true, false)
	m, phases, samples, err := nvtelemetry.ReadStreamStrict(bytes.NewReader(raw), 0)
	if !errors.Is(err, nvtelemetry.ErrIdentityMismatch) || m != nil || phases != nil || samples != nil {
		t.Fatalf("identity refusal: manifest %v phases %d samples %d error %v", m != nil, len(phases), len(samples), err)
	}
}

func TestStreamBudgetBoundaryBytes(t *testing.T) {
	raw := budgetBoundaryStream(t, 0, 0, true, false, false)
	if _, _, _, err := nvtelemetry.ReadStreamStrict(bytes.NewReader(raw), int64(len(raw))); err != nil {
		t.Fatal(err)
	}
	for _, maxBytes := range []int64{int64(len(raw) - 1), (64 << 20) + 1} {
		m, phases, samples, err := nvtelemetry.ReadStream(bytes.NewReader(raw), maxBytes)
		if !errors.Is(err, nvtelemetry.ErrBudgetExceeded) || m != nil || phases != nil || samples != nil {
			t.Fatalf("byte budget %d: %v", maxBytes, err)
		}
	}
	// The existing scanner line limit still refuses a record at 4 MiB.
	m, phases, samples, err := nvtelemetry.ReadStream(strings.NewReader(strings.Repeat(" ", 4<<20)+"\n"), 0)
	if err == nil || m != nil || phases != nil || samples != nil {
		t.Fatalf("line budget returned partial output: %v", err)
	}
}
