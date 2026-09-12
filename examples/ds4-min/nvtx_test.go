package main

import (
	"bytes"
	"encoding/json"
	"errors"
	"io"
	"strings"
	"testing"
	"time"
)

func TestCalibrationName(t *testing.T) {
	// The range text is the only key the Nsight export offers, so two runs
	// or two labels must never collide in one capture.
	a := calibrationName("run-a", "start")
	for _, other := range []string{
		calibrationName("run-a", "end"),
		calibrationName("run-b", "start"),
	} {
		if a == other {
			t.Fatalf("calibration names collide: %q", a)
		}
	}
	if !strings.Contains(a, "run-a") || !strings.Contains(a, "start") {
		t.Fatalf("name %q drops the run or the label", a)
	}
}

func TestCalibrationBracket(t *testing.T) {
	for _, tt := range []struct {
		name          string
		before, after uint64
		want          bool
	}{
		{"ordered", 10, 20, true},
		{"equal is the tightest bracket, not a broken one", 10, 10, true},
		{"reversed", 20, 10, false},
	} {
		c := calibration{Before: tt.before, After: tt.after}
		if got := c.valid(); got != tt.want {
			t.Errorf("%s: valid() = %v, want %v", tt.name, got, tt.want)
		}
	}
}

func TestWriteCalibrations(t *testing.T) {
	if err := writeCalibrations(io.Discard, nil); err == nil {
		t.Fatal("accepted an empty calibration set")
	}
	good := []calibration{
		{Schema: "ds4-min.calibration/v1", RunID: "r", Label: "start", Name: calibrationName("r", "start"), Before: 1, After: 2},
		{Schema: "ds4-min.calibration/v1", RunID: "r", Label: "end", Name: calibrationName("r", "end"), Before: 900, After: 901},
	}
	var out bytes.Buffer
	if err := writeCalibrations(&out, good); err != nil {
		t.Fatal(err)
	}
	dec := json.NewDecoder(&out)
	for i := range good {
		var got calibration
		if err := dec.Decode(&got); err != nil {
			t.Fatalf("record %d: %v", i, err)
		}
		if got != good[i] {
			t.Fatalf("record %d round-trip: %+v, want %+v", i, got, good[i])
		}
	}
	if err := dec.Decode(new(calibration)); err == nil {
		t.Fatal("extra record after the set")
	}
	// A reversed bracket must fail the write rather than land in a receipt
	// that a later reader cannot tell from a measured one.
	bad := []calibration{good[0], {Label: "end", Before: 900, After: 899}}
	if err := writeCalibrations(io.Discard, bad); err == nil {
		t.Fatal("wrote a backwards bracket")
	}
}

// calibrate must produce a usable bracket even with no provider attached,
// where RangeStart is a no-op and the two clock reads can be equal.
func TestCalibrateWithoutProvider(t *testing.T) {
	c := calibrate("unit", "start")
	if !c.valid() {
		t.Fatalf("bracket %d..%d is unusable", c.Before, c.After)
	}
	if c.Name != calibrationName("unit", "start") || c.Schema == "" {
		t.Fatalf("calibration = %+v", c)
	}
}

// A phase span is closed both by a defer, for the error paths, and at the
// phase's real boundary. end must therefore be exactly-once: a second call
// would restamp the span's duration with the time spent after the phase,
// and would unbalance the NVTX range.
func TestTracedEndIsExactlyOnce(t *testing.T) {
	spans = newSpanRecorder(levelPhase, "exactly-once")
	defer func() { spans = nil }()

	e := beginTraced(levelPhase, "prefill", nil)
	e.end()
	first := spans.spans[0].durationN
	if first == 0 {
		t.Fatal("end did not stamp a duration")
	}
	// Sleep long enough that a second stamp could not equal the first.
	time.Sleep(2 * time.Millisecond)
	e.end()
	if got := spans.spans[0].durationN; got != first {
		t.Fatalf("second end restamped the duration: %d, then %d", first, got)
	}
	if !*e.done {
		t.Fatal("end did not record that it ran")
	}
	// The zero value must stay a no-op rather than panicking on nil done.
	tracedEnd{}.end()
}

// The phase spans in run are closed by a deferred end, so a failure between
// begin and the normal boundary still closes the interval. This models that
// control flow: it fails if the defer is dropped.
func TestTracedEndClosesOnErrorPath(t *testing.T) {
	spans = newSpanRecorder(levelPhase, "error-path")
	defer func() { spans = nil }()

	phase := func(fail bool) error {
		e := beginTraced(levelPhase, "prefill", nil)
		defer e.end()
		if fail {
			return errors.New("sync failed")
		}
		e.end()
		return nil
	}
	if err := phase(true); err == nil {
		t.Fatal("want an error")
	}
	if err := phase(false); err != nil {
		t.Fatal(err)
	}
	if len(spans.spans) != 2 {
		t.Fatalf("recorded %d spans, want 2", len(spans.spans))
	}
	for i, s := range spans.spans {
		if s.durationN == 0 {
			t.Errorf("span %d (%s) was left open", i, s.name)
		}
	}
}
