package main

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cudaprof/timeline"
)

func TestNativeMemoryEvidence(t *testing.T) {
	records := timeline.Records{UnifiedMemory: []timeline.UnifiedMemory{{Journal: "p", Sequence: 3, Start: 10, End: 20, Valid: true, CounterKind: 1, Value: 4096}, {Journal: "p", Sequence: 4, Start: 30, End: 30, Valid: true, CounterKind: 3, Value: 0xfeed}}, MemoryPolicies: []timeline.MemoryPolicy{{Journal: "p", Event: "prefetch_request", ResultKnown: true}}}
	r := summarizeMemory(records, json.RawMessage(`{"unified_memory":{"state":"collected","native_configuration_results":[0,0,0,0]}}`))
	if r.Migration.State != "observed" || r.PageFaults.State != "observed" || r.Residency.State != "unsupported" || r.Placement.State != "observed" {
		t.Fatalf("evidence = %+v", r)
	}
	if !strings.Contains(r.UnifiedMemory[1].ValueMeaning, "instruction") {
		t.Fatalf("CPU fault value = %s", r.UnifiedMemory[1].ValueMeaning)
	}
	empty := summarizeMemory(timeline.Records{}, json.RawMessage(`{"unified_memory":{"state":"collected"}}`))
	if empty.Migration.State == "observed" || empty.PageFaults.State == "observed" {
		t.Fatal("empty collection proves zero events")
	}
}
func TestMemoryTraceAlignment(t *testing.T) {
	records := timeline.Records{UnifiedMemory: []timeline.UnifiedMemory{{Valid: true, Start: 10, End: 10, CounterKind: 3, RawEnd: 1, Value: 0xfeed}}, MemoryPolicies: []timeline.MemoryPolicy{{Time: 1, TimestampDomain: "unknown", Event: "attribute"}, {Time: 5, TimestampDomain: "monotonic", Event: "prefetch_request"}}}
	if got := memoryTraceOrigin(records, 100); got != 5 {
		t.Fatalf("origin=%d", got)
	}
	var b bytes.Buffer
	if err := writeMemoryTrace(&b, records, 5, 100, 0, 1); err != nil {
		t.Fatal(err)
	}
	if b.Len() == 0 {
		t.Fatal("empty trace")
	}
	if err := writeMemoryTrace(&b, records, 11, 100, 0, 1); err == nil {
		t.Fatal("accepted unalignable timestamp")
	}
}
func TestMemoryProbeNativeResults(t *testing.T) {
	for _, tt := range []struct {
		err  error
		want *int32
	}{{nil, new(int32)}, {driver.Result(1), func() *int32 { v := int32(1); return &v }()}, {errors.New("symbol missing"), nil}} {
		got := memoryDriverResult(tt.err)
		if (got == nil) != (tt.want == nil) || (got != nil && *got != *tt.want) {
			t.Fatalf("result(%v)=%v", tt.err, got)
		}
	}
	for _, args := range [][]string{{"-placement", "-um"}, {"-prefetch=device"}, {"-advice=bad"}, {"-placement", "-prefetch=bad"}} {
		if err := runProbeCommand(args); err == nil {
			t.Fatalf("accepted %v", args)
		}
	}
}

func TestMemoryCollectionEnvironment(t *testing.T) {
	for _, enabled := range []bool{false, true} {
		options, err := parseCaptureArgs([]string{fmt.Sprintf("--unified-memory=%t", enabled), fmt.Sprintf("--memory-policy=%t", enabled), "--", "true"})
		if err != nil {
			t.Fatal(err)
		}
		options.Env = []string{"CUDAPROF_UNIFIED_MEMORY=1", "CUDAPROF_MEMORY_POLICY=1"}
		environment := captureEnvironment(options, "session", "dir")
		want := "0"
		if enabled {
			want = "1"
		}
		for _, key := range []string{"CUDAPROF_UNIFIED_MEMORY", "CUDAPROF_MEMORY_POLICY"} {
			if got := lookupEnv(environment, key); got != want {
				t.Fatalf("%s=%s want %s", key, got, want)
			}
		}
	}
}

func TestMemoryWindowsRequireClockScopeAndOverlap(t *testing.T) {
	records := timeline.Records{UnifiedMemory: []timeline.UnifiedMemory{{Journal: "p", Process: 7, Valid: true, CounterKind: 3, Start: 20, End: 20}, {Journal: "p", Process: 7, Valid: true, CounterKind: 1, Start: 10, End: 20, EndValid: true}}, Ranges: []timeline.Range{{Journal: "p", Process: 7, Start: 0, End: 20, Name: `cudaprof:v1 {"phase":"prefill","operation":"load"}`}, {Journal: "q", Process: 7, Start: 0, End: 30, Name: `cudaprof:v1 {"phase":"prefill","operation":"load"}`}, {Journal: "p", Process: 8, Start: 0, End: 30, Name: `cudaprof:v1 {"phase":"prefill","operation":"load"}`}, {Journal: "p", Process: 7, Start: 0, End: 30, Name: "unstructured marker"}, {Journal: "p", Process: 7, Start: 20, End: 30, Name: `cudaprof:v1 {"phase":"decode","operation":"compute"}`}}}
	noClock := summarizeMemory(records, json.RawMessage(`{"clocks_unified":false}`))
	if len(noClock.UnifiedMemory[0].OperationWindows) != 0 {
		t.Fatal("guessed shared clock")
	}
	unified := summarizeMemory(records, json.RawMessage(`{"clocks_unified":true}`))
	if got := unified.UnifiedMemory[0].OperationWindows; len(got) != 1 || got[0] != 4 {
		t.Fatalf("instant windows=%v", got)
	}
	if got := unified.UnifiedMemory[1].OperationWindows; len(got) != 1 || got[0] != 0 {
		t.Fatalf("interval windows=%v", got)
	}
}
