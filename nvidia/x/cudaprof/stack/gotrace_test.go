package stack

import (
	"context"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"runtime/pprof"
	"runtime/trace"
	"strings"
	"testing"
	"time"
)

const goDebugFixture = `M=-1 P=-1 G=-1 Sync Time=90 N=1 Trace=100 Mono=110 Wall=now
M=12 P=0 G=7 StackSample Time=150
Stack=
	main.leaf @ 0x123
		/work.go:7
	main.main @ 0x100
		/main.go:2

M=-1 P=-1 G=-1 Sync Time=190 N=2 Trace=200 Mono=210 Wall=now
`

func TestGoTraceDebug(t *testing.T) {
	identity := Record{Process: "p", PID: 1, Generation: 2, Session: "s"}
	got, err := ReadGoTraceDebug(strings.NewReader(goDebugFixture), identity, Limits{})
	if err != nil {
		t.Fatal(err)
	}
	if len(got.Samples) != 1 || got.Samples[0].TraceTime != 150 || got.Samples[0].Time != 150 || got.Samples[0].Frames[0].Function != "main.leaf" || got.ClockStatus != "unavailable" {
		t.Fatalf("got %+v", got)
	}
	if err := QualifyGoMonotonic(got, "linux", "go1.27.1"); err != nil {
		t.Fatal(err)
	}
	if got.ClockStatus != "linux_monotonic" || got.ClockUncertaintyNs != 64 || got.Samples[0].Time != 150 {
		t.Fatal("invalid clock qualification")
	}
	for _, tc := range []struct {
		name, input string
		limits      Limits
	}{
		{"missing frame", strings.Replace(goDebugFixture, "\t\t/main.go:2", "", 1), Limits{}},
		{"bad line", strings.Replace(goDebugFixture, "work.go:7", "work.go:x", 1), Limits{}},
		{"bytes", goDebugFixture, Limits{Bytes: 20}},
		{"frames", goDebugFixture, Limits{Frames: 1}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if _, err := ReadGoTraceDebug(strings.NewReader(tc.input), identity, tc.limits); err == nil {
				t.Fatal("accepted malformed sample")
			}
		})
	}
	got, err = ReadGoTraceDebug(strings.NewReader(strings.Replace(goDebugFixture, "Mono=210", "Mono=211", 1)), identity, Limits{})
	if err != nil {
		t.Fatal(err)
	}
	if got.ClockStatus != "unavailable" || got.Samples[0].Clock != "go_trace" {
		t.Fatal("invented constant offset")
	}
}
func TestAnalyzeGoTrace(t *testing.T) {
	if runtime.Version() != "go1.27.1" {
		t.Skip("backend pinned to Go 1.27.1")
	}
	tool, err := exec.LookPath("go")
	if err != nil {
		t.Skip("no local Go tool")
	}
	dir := t.TempDir()
	path := filepath.Join(dir, "runtime.trace")
	f, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	cpu, err := os.Create(filepath.Join(dir, "cpu.pprof"))
	if err != nil {
		t.Fatal(err)
	}
	if err := trace.Start(f); err != nil {
		t.Fatal(err)
	}
	if err := pprof.StartCPUProfile(cpu); err != nil {
		trace.Stop()
		t.Fatal(err)
	}
	deadline := time.Now().Add(100 * time.Millisecond)
	for time.Now().Before(deadline) {
	}
	pprof.StopCPUProfile()
	trace.Stop()
	f.Close()
	cpu.Close()
	got, err := AnalyzeGoTrace(context.Background(), tool, path, Record{Process: "test", PID: os.Getpid()}, Limits{})
	if err != nil {
		t.Fatal(err)
	}
	if len(got.Samples) == 0 {
		t.Fatal("missing CPU samples")
	}
	if len(got.ClockSnapshots) == 0 {
		t.Fatal("missing trace calibration evidence")
	}
}
