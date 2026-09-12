package metalprof_test

import (
	"bytes"
	"strings"
	"testing"
	"time"

	"github.com/google/pprof/profile"
	"github.com/tmc/lib/apple/x/metal/metalprof"
)

//go:noinline
func recordFromCaller(fr *metalprof.FlightRecorder) {
	fr.Record("kernel", time.Microsecond, [3]int{1}, [3]int{1}, map[string]string{"request": "original"})
}

func TestFlightRecorderRecordingStack(t *testing.T) {
	fr := metalprof.NewFlightRecorder(1)
	recordFromCaller(fr)
	// The caller's copy must not mutate the retained event.
	events := fr.Events()
	events[0].Labels["request"] = "changed"
	var buf bytes.Buffer
	if err := fr.SnapshotPprof(&buf); err != nil {
		t.Fatal(err)
	}
	p, err := profile.ParseData(buf.Bytes())
	if err != nil {
		t.Fatal(err)
	}
	if err := p.CheckValid(); err != nil {
		t.Fatal(err)
	}
	s := p.Sample[0]
	if s.Label["request"][0] != "original" || s.Label["stack_origin"][0] != "record" {
		t.Fatalf("labels: %v", s.Label)
	}
	found := false
	for _, loc := range s.Location {
		for _, line := range loc.Line {
			fn := line.Function
			if strings.HasSuffix(fn.Name, ".recordFromCaller") {
				found = true
				if !strings.HasSuffix(fn.Filename, "flightrecorder_stack_test.go") || line.Line <= 0 {
					t.Fatalf("missing source: %+v", fn)
				}
			}
			if fn.Filename == "MetalShaders.metal" {
				t.Fatal("fabricated shader source")
			}
		}
	}
	if !found {
		t.Fatal("recording caller missing")
	}
	if s.Value[0] != 1 || s.Value[1] != 1000 {
		t.Fatalf("values: %v", s.Value)
	}
}
