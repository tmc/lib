package cudacensus

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

const fixture = "testdata/capture"

func TestReadDir(t *testing.T) {
	c, err := ReadDir(fixture)
	if err != nil {
		t.Fatal(err)
	}
	// clock_sync in the fixture: cupti_ns 1788203906159596148,
	// unix_ns 1788203906159596420.
	if got, want := c.ClockOffsetNs, int64(-272); got != want {
		t.Errorf("ClockOffsetNs = %d, want %d", got, want)
	}
	if got, want := len(c.Launches), 132; got != want {
		t.Errorf("launches = %d, want %d", got, want)
	}
	if got, want := len(c.Transfers), 2; got != want {
		t.Errorf("transfers = %d, want %d", got, want)
	}
	if got, want := len(c.Spans), 3; got != want {
		t.Fatalf("spans = %d, want %d", got, want)
	}
	for i := 1; i < len(c.Launches); i++ {
		if c.Launches[i-1].StartNs > c.Launches[i].StartNs {
			t.Fatalf("launches not sorted at %d", i)
		}
	}
}

func TestTransfersCarryBytes(t *testing.T) {
	// The point of the test is the asymmetry: a transfer records bytes and a
	// kernel does not. If a future capture format adds bytes to kernels, this
	// test is where to notice.
	c, err := ReadDir(fixture)
	if err != nil {
		t.Fatal(err)
	}
	for _, tr := range c.Transfers {
		if tr.Bytes == 0 {
			t.Errorf("%s transfer has no bytes", tr.Kind)
		}
	}
}

func TestSpanOnDeviceClock(t *testing.T) {
	c, err := ReadDir(fixture)
	if err != nil {
		t.Fatal(err)
	}
	tok, ok := c.Span("token")
	if !ok {
		t.Fatal("no token span")
	}
	// Host timestamps in events.app.trimmed.jsonl, plus the -272 ns offset.
	if got, want := tok.StartNs, int64(1788203908417456492-272); got != want {
		t.Errorf("token start = %d, want %d", got, want)
	}
	if got, want := tok.EndNs, int64(1788203908420635694-272); got != want {
		t.Errorf("token end = %d, want %d", got, want)
	}
	if got := tok.Labels["phase"]; got != "decode" {
		t.Errorf("phase label = %q, want %q", got, "decode")
	}
	if got, want := tok.Seq, int64(1); got != want {
		t.Errorf("seq = %d, want %d", got, want)
	}
}

func TestWindowSelectsTheToken(t *testing.T) {
	c, err := ReadDir(fixture)
	if err != nil {
		t.Fatal(err)
	}
	tok, _ := c.Span("token")
	// Every kernel in the fixture was trimmed to lie inside this span, so the
	// window must recover all of them. That is what makes the span-to-device
	// clock conversion load-bearing: without the offset it would not.
	if got, want := len(c.Window(tok.StartNs, tok.EndNs)), 132; got != want {
		t.Errorf("window = %d launches, want %d", got, want)
	}
	if got := len(c.Window(tok.EndNs, tok.EndNs+1)); got != 0 {
		t.Errorf("empty window = %d launches, want 0", got)
	}
}

func TestSpansNamed(t *testing.T) {
	c, err := ReadDir(fixture)
	if err != nil {
		t.Fatal(err)
	}
	for _, tt := range []struct {
		name string
		want int
	}{
		{"prefill", 1},
		{"token", 1},
		{"decode", 1},
		{"generation", 0}, // this workload does not use that name
	} {
		if got := len(c.SpansNamed(tt.name)); got != tt.want {
			t.Errorf("SpansNamed(%q) = %d, want %d", tt.name, got, tt.want)
		}
	}
}

func TestReadDirErrors(t *testing.T) {
	dir := t.TempDir()
	if _, err := ReadDir(dir); err == nil {
		t.Error("empty dir: want error")
	}

	// Spans with no clock_sync cannot be put on the device clock.
	d2 := t.TempDir()
	write(t, filepath.Join(d2, "events.app.1.jsonl"), `{"kind":"span","name":"x","start_ns":1,"end_ns":2}`)
	write(t, filepath.Join(d2, "events.1.jsonl"), `{"kind":"capture_meta","pid":1}`)
	_, err := ReadDir(d2)
	if err == nil || !strings.Contains(err.Error(), "clock_sync") {
		t.Errorf("missing clock_sync: err = %v, want a clock_sync error", err)
	}

	d3 := t.TempDir()
	write(t, filepath.Join(d3, "events.1.jsonl"), "{not json}")
	if _, err := ReadDir(d3); err == nil {
		t.Error("malformed line: want error")
	}
}

func write(t *testing.T, path, content string) {
	t.Helper()
	if err := os.WriteFile(path, []byte(content+"\n"), 0o644); err != nil {
		t.Fatal(err)
	}
}
