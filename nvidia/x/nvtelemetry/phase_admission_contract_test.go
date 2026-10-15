package nvtelemetry

import (
	"context"
	"encoding/json"
	"errors"
	"math"
	"strings"
	"testing"
	"time"
)

type phaseContractSampler struct {
	sample func(context.Context) ([]Reading, error)
}

func (s *phaseContractSampler) Sample(ctx context.Context) ([]Reading, error) {
	if s.sample != nil {
		return s.sample(ctx)
	}
	return nil, nil
}
func (*phaseContractSampler) Devices() []DeviceMetadata { return nil }

func phaseCollector(cfg CollectorConfig, w *Writer) (*Collector, *FakeClock) {
	clock := NewFakeClock(1, 1)
	return NewCollector("phase-admission", &phaseContractSampler{}, clock, cfg, w), clock
}

func TestPhaseRecordSizeContract(t *testing.T) {
	allBytes := make([]byte, 256)
	for i := range allBytes {
		allBytes[i] = byte(i)
	}
	for _, name := range []string{"", "simple", "\"\\\b\f\n\r\t<>&", "é😀\u2028\u2029", string(allBytes)} {
		for _, timestamp := range []int64{1, math.MinInt64, math.MaxInt64} {
			p := PhaseMarker{Name: name, StartMonotonicNanos: timestamp, EndMonotonicNanos: timestamp}
			b, err := json.Marshal(StreamRecord{Phase: &p})
			if err != nil {
				t.Fatal(err)
			}
			want := len(b) + 1
			got, err := phaseRecordSize(p, want)
			if err != nil || got != want {
				t.Fatalf("name %q: size = %d, %v, want %d", name, got, err, want)
			}
			if _, err := phaseRecordSize(p, want-1); !errors.Is(err, ErrBudgetExceeded) {
				t.Fatalf("name %q: size-1 err = %v", name, err)
			}
		}
	}
	// The retained input already exists; admission must reject it without
	// creating a proportional escaped JSON buffer.
	name := strings.Repeat("<", 4096)
	if allocations := testing.AllocsPerRun(10, func() {
		if _, err := phaseRecordSize(PhaseMarker{Name: name}, 100); !errors.Is(err, ErrBudgetExceeded) {
			t.Fatal(err)
		}
	}); allocations != 0 {
		t.Fatalf("oversize preflight allocated: %v", allocations)
	}
}

func TestPhaseAdmissionContract(t *testing.T) {
	for _, tt := range []struct {
		name string
		cfg  CollectorConfig
	}{
		{"record cap", CollectorConfig{MaxRecordBudget: 2}},
		{"buffer cap", CollectorConfig{MaxBufferedRecords: 2}},
	} {
		t.Run(tt.name, func(t *testing.T) {
			c, _ := phaseCollector(tt.cfg, nil)
			for _, phase := range []string{"", "", "a", "b"} {
				if err := c.SetPhase(phase); err != nil {
					t.Fatal(err)
				}
			}
			for range 8 {
				if err := c.SetPhase("refused"); !errors.Is(err, ErrBudgetExceeded) {
					t.Fatalf("transition past cap = %v", err)
				}
				if c.currentPhase != "b" || len(c.phases) != 1 {
					t.Fatalf("refusal changed active/history: %q %+v", c.currentPhase, c.phases)
				}
			}
			m, err := c.Stop()
			if err != nil || !m.IsComplete || len(m.Phases) != 2 || m.Phases[0].Name != "a" || m.Phases[1].Name != "b" {
				t.Fatalf("final marker double-counted or lost: %+v, %v", m, err)
			}
			m2, err := c.Stop()
			if err != nil || len(m2.Phases) != 2 {
				t.Fatalf("second Stop = %+v, %v", m2, err)
			}
		})
	}
	t.Run("shared buffer", func(t *testing.T) {
		c, _ := phaseCollector(CollectorConfig{MaxBufferedRecords: 2}, nil)
		if err := c.SetPhase("a"); err != nil {
			t.Fatal(err)
		}
		if err := c.Step(t.Context()); err != nil {
			t.Fatal(err)
		}
		if err := c.SetPhase("b"); !errors.Is(err, ErrBudgetExceeded) {
			t.Fatalf("sample plus completed and active phases escaped cap: %v", err)
		}
		if err := c.SetPhase(""); err != nil {
			t.Fatalf("closing existing active phase charged twice: %v", err)
		}
		m, err := c.Stop()
		if err != nil || !m.IsComplete || m.TotalSamples != 1 || len(m.Phases) != 1 {
			t.Fatalf("shared cap boundary: %+v %v", m, err)
		}
	})
	t.Run("sample admission counts active phase", func(t *testing.T) {
		c, _ := phaseCollector(CollectorConfig{MaxBufferedRecords: 1}, nil)
		if err := c.SetPhase("a"); err != nil {
			t.Fatal(err)
		}
		if err := c.Step(t.Context()); !errors.Is(err, ErrBudgetExceeded) {
			t.Fatalf("sample ignored active reservation: %v", err)
		}
		m, err := c.Stop()
		if !errors.Is(err, ErrBudgetExceeded) || m.IsComplete || len(m.Phases) != 1 || m.TotalSamples != 0 {
			t.Fatalf("buffer refusal: %+v %v", m, err)
		}
	})
	t.Run("no active reservation", func(t *testing.T) {
		c, _ := phaseCollector(CollectorConfig{MaxBufferedRecords: 1}, nil)
		if err := c.Step(t.Context()); err != nil {
			t.Fatalf("nonexistent phase consumed capacity: %v", err)
		}
		if err := c.SetPhase("a"); !errors.Is(err, ErrBudgetExceeded) {
			t.Fatalf("active phase exceeded sample-full buffer: %v", err)
		}
		m, err := c.Stop()
		if err != nil || !m.IsComplete || len(m.Phases) != 0 || m.TotalSamples != 1 {
			t.Fatalf("empty phase state: %+v %v", m, err)
		}
	})
	t.Run("zero limits stay unlimited", func(t *testing.T) {
		c, _ := phaseCollector(CollectorConfig{}, nil)
		for range 8 {
			if err := c.SetPhase("a"); err != nil {
				t.Fatal(err)
			}
		}
		m, err := c.Stop()
		if err != nil || len(m.Phases) != 8 {
			t.Fatalf("zero limits: %+v %v", m, err)
		}
	})
}

type phaseFailureWriter struct {
	err   error
	calls int
}

func (w *phaseFailureWriter) Write(p []byte) (int, error) {
	w.calls++
	if w.err != nil {
		return 0, w.err
	}
	return len(p), nil
}

func TestPhaseFailureAndByteAdmission(t *testing.T) {
	t.Run("write retry retains prior state", func(t *testing.T) {
		cause := errors.New("phase sink failure")
		sink := &phaseFailureWriter{err: cause}
		c, _ := phaseCollector(CollectorConfig{MaxRecordBudget: 2}, NewWriter(sink, CollectorConfig{}))
		if err := c.SetPhase("a"); err != nil {
			t.Fatal(err)
		}
		for range 8 {
			if err := c.SetPhase("b"); !errors.Is(err, cause) {
				t.Fatalf("write error identity lost: %v", err)
			}
			if len(c.phases) != 0 || c.phaseBytes != 0 || c.currentPhase != "a" || c.phaseStart != 1 {
				t.Fatalf("failed retry grew or changed state: %+v %d %q", c.phases, c.phaseBytes, c.currentPhase)
			}
		}
		sink.err = nil
		if err := c.SetPhase("b"); err != nil {
			t.Fatal(err)
		}
		m, err := c.Stop()
		if err != nil || len(m.Phases) != 2 || m.Phases[0].Name != "a" || m.Phases[1].Name != "b" {
			t.Fatalf("retry commit: %+v %v", m, err)
		}
	})
	for _, name := range []string{"a", "<>&\"\\\n", "é\u2028\xff"} {
		t.Run("escaped boundary "+name, func(t *testing.T) {
			p := PhaseMarker{Name: name, StartMonotonicNanos: 1, EndMonotonicNanos: 1}
			b, _ := json.Marshal(StreamRecord{Phase: &p})
			for _, delta := range []int{-1, 0} {
				c, _ := phaseCollector(CollectorConfig{MaxByteBudget: len(b) + 1 + delta}, nil)
				err := c.SetPhase(name)
				if delta < 0 {
					if !errors.Is(err, ErrBudgetExceeded) || c.currentPhase != "" || len(c.phases) != 0 {
						t.Fatalf("size-1 input admitted: %v", err)
					}
				} else {
					if err != nil {
						t.Fatal(err)
					}
					m, err := c.Stop()
					if err != nil || !m.IsComplete || len(m.Phases) != 1 || c.phaseBytes != len(b)+1 {
						t.Fatalf("exact byte boundary: %+v %v %d", m, err, c.phaseBytes)
					}
				}
			}
		})
	}
	t.Run("preflight before writer", func(t *testing.T) {
		sink := &phaseFailureWriter{}
		c, _ := phaseCollector(CollectorConfig{MaxByteBudget: 100}, NewWriter(sink, CollectorConfig{}))
		if err := c.SetPhase(strings.Repeat("<", 4096)); !errors.Is(err, ErrBudgetExceeded) || sink.calls != 0 || c.currentPhase != "" {
			t.Fatalf("oversize retained or written: %v, writes %d", err, sink.calls)
		}
	})
	t.Run("history byte sum", func(t *testing.T) {
		p := PhaseMarker{Name: "a", StartMonotonicNanos: 1, EndMonotonicNanos: 1}
		b, _ := json.Marshal(StreamRecord{Phase: &p})
		c, _ := phaseCollector(CollectorConfig{MaxByteBudget: 2 * (len(b) + 1)}, nil)
		_ = c.SetPhase("a")
		if err := c.SetPhase("b"); err != nil {
			t.Fatal(err)
		}
		if err := c.SetPhase("c"); !errors.Is(err, ErrBudgetExceeded) || len(c.phases) != 1 || c.currentPhase != "b" {
			t.Fatalf("history escaped bytes: %v %+v", err, c.phases)
		}
		m, err := c.Stop()
		if err != nil || len(m.Phases) != 2 || c.phaseBytes != 2*(len(b)+1) {
			t.Fatalf("final history boundary: %+v %v %d", m, err, c.phaseBytes)
		}
	})
	t.Run("final marker refusal", func(t *testing.T) {
		p := PhaseMarker{Name: "a", StartMonotonicNanos: 1, EndMonotonicNanos: 1}
		b, _ := json.Marshal(StreamRecord{Phase: &p})
		c, clock := phaseCollector(CollectorConfig{MaxByteBudget: len(b) + 1}, nil)
		if err := c.SetPhase("a"); err != nil {
			t.Fatal(err)
		}
		clock.Advance(9 * time.Nanosecond)
		m, err := c.Stop()
		if !errors.Is(err, ErrBudgetExceeded) || m.IsComplete || !strings.Contains(m.IncompleteReason, "finish_error") || len(m.Phases) != 0 || c.currentPhase != "a" || c.phaseBytes != 0 {
			t.Fatalf("final refusal claimed completion or retained history: %+v %v", m, err)
		}
		m2, err := c.Stop()
		if !errors.Is(err, ErrBudgetExceeded) || len(m2.Phases) != 0 {
			t.Fatalf("final refusal retry: %+v %v", m2, err)
		}
	})
}
