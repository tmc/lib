//go:build darwin || linux

package hosttest

import (
	"strings"
	"testing"
)

// runObserver runs one observer fixture mode with CUDAINTERCEPT_OBSERVE set
// to observe and returns the fixture's summary line.
func runObserver(t *testing.T, binary, mode, observe string) string {
	t.Helper()
	t.Setenv("CUDAINTERCEPT_OBSERVE", observe)
	timedOut, err, out, elapsed := runFixture(t, binary, mode)
	t.Logf("elapsed=%s timeout=%v result=%v output=%s", elapsed, timedOut, err, out)
	if timedOut || err != nil {
		t.Fatal("observer fixture did not exit cleanly")
	}
	// The concurrent mode reports its drained/dropped split on stderr,
	// which is interleaved into out; the summary is the line with bad=.
	for _, line := range strings.Split(out, "\n") {
		if strings.Contains(line, " bad=") {
			return line
		}
	}
	t.Fatal("observer fixture printed no summary")
	return ""
}

func TestBoundedObservation(t *testing.T) {
	binary := buildFixture(t, "observer", shimSource(t))
	for _, tt := range []struct {
		name, mode, observe, want string
	}{
		{"default_off", "off", "", "forwarded=5 drain=-1 dropped=0 head=0 slots=0 bad=none"},
		{"not_one_is_off", "off", "true", "forwarded=5 drain=-1 dropped=0 head=0 slots=0 bad=none"},
		{"bound_drops_newest", "bound", "1", "forwarded=1032 records=1024 dropped=7 again=0 after=1 dropped=7 bad=none"},
		{"drain_consumes_in_order", "consume", "1", "drains=8,8,4,0 bad=none"},
		{"concurrent_accounting", "concurrent", "1", "forwarded=64000 total=64000 bad=none"},
		{"sequence_exhaustion", "exhaust", "1", "forwarded=8 records=3 dropped=2 after=0 dropped=5 slots=3 bad=none"},
	} {
		t.Run(tt.name, func(t *testing.T) {
			if got := runObserver(t, binary, tt.mode, tt.observe); got != tt.want {
				t.Fatalf("got %q, want %q", got, tt.want)
			}
		})
	}
}

func TestBoundedObservationMutants(t *testing.T) {
	for _, tt := range []struct {
		name     string
		old, new string
		mode     string
		observe  string
		want     string
	}{
		{
			name:    "no_enable_check",
			old:     "    if (observing)\n        observe(",
			new:     "    if (1)\n        observe(",
			mode:    "off",
			observe: "",
			want:    "forwarded=5 drain=-1 dropped=0 head=5 slots=5 bad=none",
		},
		{
			name:    "no_capacity_check",
			old:     "h - t >= CUDAINTERCEPT_CAPACITY) {",
			new:     "h - t >= UINT64_MAX) {",
			mode:    "bound",
			observe: "1",
			want:    "forwarded=1032 records=0 dropped=0 again=0 after=0 dropped=0 bad=record after drain",
		},
		{
			name:    "no_sequence_limit",
			old:     "h == UINT64_MAX || ",
			new:     "",
			mode:    "exhaust",
			observe: "1",
			want:    "forwarded=8 records=5 dropped=0 after=3 dropped=0 slots=7 bad=head tail",
		},
		{
			name:    "no_drain_head_bound",
			old:     " && t != h;",
			new:     " && t != h + 1;",
			mode:    "exhaust",
			observe: "1",
			want:    "forwarded=8 records=4 dropped=2 after=0 dropped=5 slots=3 bad=version",
		},
	} {
		t.Run(tt.name, func(t *testing.T) {
			binary := buildFixture(t, "observer", shimSource(t, [2]string{tt.old, tt.new}))
			if got := runObserver(t, binary, tt.mode, tt.observe); got != tt.want {
				t.Fatalf("got %q, want %q", got, tt.want)
			}
		})
	}
}
