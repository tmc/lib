package cudacapture_test

import (
	"bytes"
	"context"
	"errors"
	"strings"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/cupti"
	"github.com/tmc/lib/nvidia/x/cudacapture"
	"github.com/tmc/lib/nvidia/x/cudaprof"
)

func TestAmbiguousCorrelation(t *testing.T) {
	tracker := cudacapture.NewCorrelationTracker()

	// Normal 1-to-1 match
	if err := tracker.RecordRuntime(cudacapture.RuntimeLaunch{
		CorrelationID: 101,
		KernelName:    "kernel_unambiguous",
		HostThreadID:  1,
		Callstack:     []string{"main", "launch"},
	}); err != nil {
		t.Fatal(err)
	}
	if err := tracker.RecordActivity(cudacapture.ActivityRecord{
		Correlation: 101,
		Name:        "kernel_unambiguous",
		Start:       1000,
		End:         2000,
	}); err != nil {
		t.Fatal(err)
	}

	// Negative control: Scrambled / duplicate correlation IDs
	// Two distinct runtime launches receive the same correlation ID 202
	if err := tracker.RecordRuntime(cudacapture.RuntimeLaunch{
		CorrelationID: 202,
		KernelName:    "kernel_scrambled_a",
		HostThreadID:  2,
		Callstack:     []string{"streamA", "launch"},
	}); err != nil {
		t.Fatal(err)
	}
	if err := tracker.RecordRuntime(cudacapture.RuntimeLaunch{
		CorrelationID: 202,
		KernelName:    "kernel_scrambled_b",
		HostThreadID:  3,
		Callstack:     []string{"streamB", "launch"},
	}); err != nil {
		t.Fatal(err)
	}
	if err := tracker.RecordActivity(cudacapture.ActivityRecord{
		Correlation: 202,
		Name:        "kernel_scrambled_a",
		Start:       3000,
		End:         4000,
	}); err != nil {
		t.Fatal(err)
	}

	matched, ambiguous, unmappedR, unmappedA := tracker.Join()

	// 1. Verify unambiguous match was joined correctly
	if len(matched) != 1 {
		t.Fatalf("len(matched) = %d, want 1", len(matched))
	}
	if matched[0].CorrelationID != 101 || matched[0].KernelName != "kernel_unambiguous" {
		t.Errorf("unexpected matched join: %+v", matched[0])
	}

	// 2. Verify ambiguous correlation was detected and preserved rather than arbitrarily joined
	if len(ambiguous) != 1 {
		t.Fatalf("len(ambiguous) = %d, want 1", len(ambiguous))
	}
	amb := ambiguous[0]
	if amb.CorrelationID != 202 {
		t.Errorf("ambiguous CorrelationID = %d, want 202", amb.CorrelationID)
	}
	if amb.RuntimeCount != 2 {
		t.Errorf("ambiguous RuntimeCount = %d, want 2", amb.RuntimeCount)
	}
	if len(amb.KernelNames) < 2 {
		t.Errorf("expected multiple kernel names in ambiguous record, got: %v", amb.KernelNames)
	}

	if len(unmappedR) != 0 || len(unmappedA) != 0 {
		t.Errorf("expected 0 unmapped, got unmappedR=%d, unmappedA=%d", len(unmappedR), len(unmappedA))
	}
}

func TestModuleGenerationReuse(t *testing.T) {
	tracker := cudacapture.NewModuleTracker()

	baseAddr := uint64(0x7fff_1000)
	digestV1 := "1111111111111111111111111111111111111111111111111111111111111111"
	digestV2 := "2222222222222222222222222222222222222222222222222222222222222222"

	// 1. Load module at baseAddr for generation 1
	codeV1 := cudacapture.CodeFiles{Source: "k.cu", PTX: "k.ptx", CUBIN: "k1.cubin", SASS: "k1.sass"}
	gen1, err := tracker.Load("mod-1", baseAddr, digestV1, codeV1)
	if err != nil {
		t.Fatalf("Load gen1 failed: %v", err)
	}
	if gen1 != 1 {
		t.Fatalf("gen1 = %d, want 1", gen1)
	}

	// Loading same digest at same address returns same generation
	gen1Again, err := tracker.Load("mod-1", baseAddr, digestV1, codeV1)
	if err != nil {
		t.Fatalf("Load same digest failed: %v", err)
	}
	if gen1Again != 1 {
		t.Fatalf("gen1Again = %d, want 1", gen1Again)
	}

	// Verify gen1 matches digestV1
	if err := tracker.Verify(baseAddr, 1, digestV1); err != nil {
		t.Errorf("Verify gen1 failed: %v", err)
	}

	// 2. Negative control: Module address reused with a DIFFERENT digest
	// Recompilation or JIT reload reuses baseAddr with digestV2
	codeV2 := cudacapture.CodeFiles{Source: "k.cu", PTX: "k.ptx", CUBIN: "k2.cubin", SASS: "k2.sass"}
	gen2, err := tracker.Load("mod-1", baseAddr, digestV2, codeV2)
	if err != nil {
		t.Fatalf("Load gen2 failed: %v", err)
	}
	if gen2 != 2 {
		t.Fatalf("gen2 = %d, want 2", gen2)
	}

	// Stale attribution attempt: querying gen1 with digestV2 must fail
	err = tracker.Verify(baseAddr, 1, digestV2)
	if !errors.Is(err, cudacapture.ErrModuleGenerationReuse) {
		t.Errorf("expected ErrModuleGenerationReuse verifying gen1 with digestV2, got: %v", err)
	}

	// Correct lookup: gen2 matches digestV2
	if err := tracker.Verify(baseAddr, 2, digestV2); err != nil {
		t.Errorf("Verify gen2 failed: %v", err)
	}

	// Verify lookup retrieves correct generation artifacts
	entry1, err := tracker.Lookup(baseAddr, 1)
	if err != nil || entry1.ImageDigest != digestV1 {
		t.Errorf("Lookup gen1 failed: %v", err)
	}

	entry2, err := tracker.Lookup(baseAddr, 2)
	if err != nil || entry2.ImageDigest != digestV2 {
		t.Errorf("Lookup gen2 failed: %v", err)
	}
}

func TestCallstackJoinedByExternalID(t *testing.T) {
	cudaprof.ResetExternalStacks()
	cudaprof.Correlate(cupti.ExternalCorrelationKindCustom0)
	t.Cleanup(func() {
		cudaprof.Correlate(cupti.ExternalCorrelationKindInvalid)
		cudaprof.ResetExternalStacks()
	})

	pushedExternalID := cudaprof.Do(context.Background(), cudaprof.Labels("phase", "test_join"), func(context.Context) {})

	if pushedExternalID == 0 {
		t.Skip("CUPTI external correlation push is unavailable on this host")
	}

	kind := cupti.ExternalCorrelationKindCustom0
	stack, ok := cudaprof.ExternalStack(kind, pushedExternalID)
	if !ok || len(stack) == 0 {
		t.Fatalf("ExternalStack(%d) = %v, ok=%v", pushedExternalID, stack, ok)
	}

	// Verify stack contains this test function at the Do call site
	var foundCaller bool
	for _, frame := range stack {
		if strings.Contains(frame, "TestCallstackJoinedByExternalID") {
			foundCaller = true
			break
		}
	}
	if !foundCaller {
		t.Errorf("expected stack to contain TestCallstackJoinedByExternalID, got: %v", stack)
	}

	tracker := cudacapture.NewCorrelationTracker()

	// 1. Exact ID join: agent external_correlation -> CUDA correlation ID -> kernel activity
	cudaCorrID := uint64(101)
	if err := tracker.RecordExternalStack(uint32(kind), pushedExternalID, stack); err != nil {
		t.Fatal(err)
	}
	// Agent records external_correlation record (external_id -> cudaCorrID)
	if err := tracker.RecordActivity(cudacapture.ActivityRecord{
		Subtype:      "external_correlation",
		ExternalKind: uint32(cupti.ExternalCorrelationKindCustom0),
		ExternalID:   pushedExternalID,
		Correlation:  uint32(cudaCorrID),
	}); err != nil {
		t.Fatal(err)
	}
	// Runtime launch recorded without stack
	if err := tracker.RecordRuntime(cudacapture.RuntimeLaunch{
		CorrelationID: cudaCorrID,
		KernelName:    "fused_attention",
		HostThreadID:  10,
	}); err != nil {
		t.Fatal(err)
	}
	// Device activity recorded for kernel
	if err := tracker.RecordActivity(cudacapture.ActivityRecord{
		Correlation: uint32(cudaCorrID),
		Name:        "fused_attention",
		Start:       1000,
		End:         2500,
	}); err != nil {
		t.Fatal(err)
	}

	matched, ambiguous, unmappedR, unmappedA := tracker.Join()
	if len(matched) != 1 {
		t.Fatalf("matched count = %d, want 1", len(matched))
	}
	if len(ambiguous) != 0 || len(unmappedR) != 0 || len(unmappedA) != 0 {
		t.Errorf("unexpected drops/ambiguities: amb=%d, unmappedR=%d, unmappedA=%d", len(ambiguous), len(unmappedR), len(unmappedA))
	}

	join := matched[0]
	if join.CorrelationID != cudaCorrID {
		t.Errorf("CorrelationID = %d, want %d", join.CorrelationID, cudaCorrID)
	}
	if join.KernelName != "fused_attention" {
		t.Errorf("KernelName = %q, want fused_attention", join.KernelName)
	}
	if join.Callstack == nil || len(join.Callstack) == 0 {
		t.Fatal("expected joined Callstack to be populated, got nil or empty")
	}
	if join.Callstack[len(join.Callstack)-1] != stack[len(stack)-1] {
		t.Errorf("leaf frame = %q, want %q", join.Callstack[len(join.Callstack)-1], stack[len(stack)-1])
	}

	// 2. Mutant / negative control: Scrambled external IDs must not give kernel the wrong stack
	// Register conflicting stacks for external ID 999
	tracker2 := cudacapture.NewCorrelationTracker()
	scrambledExtID := uint64(999)
	_ = tracker2.RecordExternalStack(0, scrambledExtID, []string{"pkg.FuncA"})
	_ = tracker2.RecordExternalStack(0, scrambledExtID, []string{"pkg.FuncB"}) // Conflicting stack
	_ = tracker2.RecordExternalCorrelation(0, scrambledExtID, 202)
	_ = tracker2.RecordRuntime(cudacapture.RuntimeLaunch{CorrelationID: 202, KernelName: "scrambled_kernel"})
	_ = tracker2.RecordActivity(cudacapture.ActivityRecord{Correlation: 202, Name: "scrambled_kernel", Start: 3000, End: 4000})

	m2, amb2, _, _ := tracker2.Join()
	if len(m2) != 0 {
		t.Errorf("expected 0 matches for scrambled external ID with conflicting stacks, got %d", len(m2))
	}
	if len(amb2) != 1 {
		t.Fatalf("expected 1 ambiguous record for scrambled external ID, got %d", len(amb2))
	}
	if amb2[0].CorrelationID != 202 {
		t.Errorf("ambiguous CorrelationID = %d, want 202", amb2[0].CorrelationID)
	}
}

func TestCallstackAbsentWithoutExternalID(t *testing.T) {
	tracker := cudacapture.NewCorrelationTracker()

	// Ingest runtime launch and device activity with no external correlation record
	cudaCorrID := uint64(505)
	if err := tracker.RecordRuntime(cudacapture.RuntimeLaunch{
		CorrelationID: cudaCorrID,
		KernelName:    "unscoped_kernel",
		HostThreadID:  1,
	}); err != nil {
		t.Fatal(err)
	}
	if err := tracker.RecordActivity(cudacapture.ActivityRecord{
		Correlation: uint32(cudaCorrID),
		Name:        "unscoped_kernel",
		Start:       1000,
		End:         2000,
	}); err != nil {
		t.Fatal(err)
	}

	matched, ambiguous, unmappedR, unmappedA := tracker.Join()
	if len(matched) != 1 {
		t.Fatalf("matched count = %d, want 1", len(matched))
	}
	if len(ambiguous) != 0 || len(unmappedR) != 0 || len(unmappedA) != 0 {
		t.Errorf("unexpected discrepancies: amb=%d, unmappedR=%d, unmappedA=%d", len(ambiguous), len(unmappedR), len(unmappedA))
	}

	join := matched[0]
	// Missing links: a kernel with no external ID has no stack (report as having no stack, not empty stack)
	if join.Callstack != nil {
		t.Fatalf("expected Callstack == nil (no stack), got non-nil: %#v", join.Callstack)
	}
}

func TestCallstackNotJoinedByTimestamp(t *testing.T) {
	tracker := cudacapture.NewCorrelationTracker()

	// Simulate a Do region running on host from t = 10_000 to t = 50_000 with external ID 77
	extID := uint64(77)
	doStack := []string{"main.Workflow", "cudaprof.Do"}
	if err := tracker.RecordExternalStack(0, extID, doStack); err != nil {
		t.Fatal(err)
	}
	// External ID 77 was associated with a DIFFERENT CUDA correlation ID 999
	if err := tracker.RecordExternalCorrelation(0, extID, 999); err != nil {
		t.Fatal(err)
	}

	// Kernel 707 executes on device at t = 20_000 to t = 30_000
	// This timestamp falls strictly INSIDE the host Do region interval [10_000, 50_000],
	// but has NO external correlation record mapping to external ID 77!
	kernelCorrID := uint64(707)
	if err := tracker.RecordRuntime(cudacapture.RuntimeLaunch{
		CorrelationID: kernelCorrID,
		KernelName:    "concurrent_unrelated_kernel",
		HostThreadID:  2,
	}); err != nil {
		t.Fatal(err)
	}
	if err := tracker.RecordActivity(cudacapture.ActivityRecord{
		Correlation: uint32(kernelCorrID),
		Name:        "concurrent_unrelated_kernel",
		Start:       20_000,
		End:         30_000,
	}); err != nil {
		t.Fatal(err)
	}

	matched, _, _, _ := tracker.Join()
	if len(matched) != 1 {
		t.Fatalf("matched count = %d, want 1", len(matched))
	}

	// Slice 7 control: Timestamps do not join.
	// If a join fell back to timestamp proximity, it would have falsely attached doStack.
	if matched[0].Callstack != nil {
		t.Fatalf("timestamp joined host call stack unexpectedly: got %#v, want nil", matched[0].Callstack)
	}
}

func TestFoldedStackExport(t *testing.T) {
	// 1. Slice 7 control: Host time is not kernel time.
	// Region sleeps for 10ms on host and launches a short 500ns kernel on device.
	// Region weight in folded stack must be 500, not ~10_000_000.
	hostSleepStart := time.Now()
	time.Sleep(10 * time.Millisecond)
	_ = time.Since(hostSleepStart)

	joins := []cudacapture.CorrelationJoin{
		{
			CorrelationID:    1,
			KernelName:       "short_kernel",
			Callstack:        []string{"main.main", "main.runBenchmark"},
			DeviceStart:      100_000,
			DeviceEnd:        100_500, // 500 ns device duration
			DeviceDurationNs: 500,
		},
		// 2. Overlapping intervals for another kernel: [200_000, 201_000) and [200_500, 201_500)
		// Union duration is 1500 ns (not naive sum 2000 ns)
		{
			CorrelationID:    2,
			KernelName:       "stream_kernel",
			Callstack:        []string{"main.main", "main.runStream"},
			DeviceStart:      200_000,
			DeviceEnd:        201_000,
			DeviceDurationNs: 1000,
		},
		{
			CorrelationID:    3,
			KernelName:       "stream_kernel",
			Callstack:        []string{"main.main", "main.runStream"},
			DeviceStart:      200_500,
			DeviceEnd:        201_500,
			DeviceDurationNs: 1000,
		},
		// 3. Kernel with no host stack (nil)
		{
			CorrelationID:    4,
			KernelName:       "driver_kernel",
			Callstack:        nil,
			DeviceStart:      300_000,
			DeviceEnd:        300_800,
			DeviceDurationNs: 800,
		},
	}

	var buf bytes.Buffer
	if err := cudacapture.ExportFoldedStacks(&buf, joins); err != nil {
		t.Fatalf("ExportFoldedStacks failed: %v", err)
	}

	out := buf.String()
	lines := strings.Split(strings.TrimSpace(out), "\n")
	if len(lines) != 3 {
		t.Fatalf("expected 3 folded stack lines, got %d: %q", len(lines), out)
	}

	// Verify line 1: driver_kernel (no host stack) -> [GPU]driver_kernel 800
	if lines[0] != "[GPU]driver_kernel 800" {
		t.Errorf("line 0 = %q, want %q", lines[0], "[GPU]driver_kernel 800")
	}

	// Verify line 2: short_kernel has weight 500 (device time), NOT 10ms host sleep
	if lines[1] != "main.main;main.runBenchmark;[GPU]short_kernel 500" {
		t.Errorf("line 1 = %q, want %q", lines[1], "main.main;main.runBenchmark;[GPU]short_kernel 500")
	}

	// Verify line 3: stream_kernel interval union = 1500 (not 2000)
	if lines[2] != "main.main;main.runStream;[GPU]stream_kernel 1500" {
		t.Errorf("line 2 = %q, want %q", lines[2], "main.main;main.runStream;[GPU]stream_kernel 1500")
	}
}

func TestCallstackEndToEnd(t *testing.T) {
	cudaprof.ResetExternalStacks()
	t.Cleanup(cudaprof.ResetExternalStacks)
	const kind = cupti.ExternalCorrelationKindCustom0
	const correlationID = uint64(701)
	cudaprof.Correlate(kind)
	externalID := cudaprof.Do(context.Background(), cudaprof.Labels("phase", "end-to-end"), func(context.Context) {})
	cudaprof.Correlate(cupti.ExternalCorrelationKindInvalid)
	stack := []string{"main.run", "main.call"}
	if externalID == 0 {
		// Host-only test double: model the sidecar produced after a successful
		// CUPTI push, while the cudaprof package test exercises that push path.
		externalID = 71
		if !cudaprof.RecordExternalStack(kind, externalID, stack) {
			t.Fatal("RecordExternalStack did not retain test stack")
		}
	} else {
		var ok bool
		stack, ok = cudaprof.ExternalStack(kind, externalID)
		if !ok {
			t.Fatalf("Do returned external ID %d without stack", externalID)
		}
	}
	capture := &cudacapture.Capture{}
	if err := capture.AttachExternalStackTable(); err != nil {
		t.Fatal(err)
	}
	tracker := cudacapture.NewCorrelationTracker()
	if err := tracker.ReadCaptureExternalStackTable(capture); err != nil {
		t.Fatal(err)
	}
	if err := tracker.RecordActivity(cudacapture.ActivityRecord{
		Subtype: "external_correlation", ExternalKind: uint32(kind), ExternalID: externalID,
		Correlation: uint32(correlationID),
	}); err != nil {
		t.Fatal(err)
	}
	if err := tracker.RecordRuntime(cudacapture.RuntimeLaunch{CorrelationID: correlationID, KernelName: "kernel"}); err != nil {
		t.Fatal(err)
	}
	if err := tracker.RecordActivity(cudacapture.ActivityRecord{Correlation: uint32(correlationID), Name: "kernel", Start: 1, End: 501}); err != nil {
		t.Fatal(err)
	}
	matched, ambiguous, _, _ := tracker.Join()
	if len(ambiguous) != 0 || len(matched) != 1 || strings.Join(matched[0].Callstack, ";") != strings.Join(stack, ";") {
		t.Fatalf("joined=%+v ambiguous=%+v", matched, ambiguous)
	}
	withLosses := &cudacapture.Capture{Files: map[string][]byte{cudacapture.ExternalStacksName: []byte(`{"version":1,"losses":3,"stacks":[]}`)}}
	if err := tracker.ReadCaptureExternalStackTable(withLosses); err != nil || tracker.ExternalStackLosses() != 3 {
		t.Fatalf("read losses: err=%v losses=%d", err, tracker.ExternalStackLosses())
	}
}

func TestCallstackKindIsolated(t *testing.T) {
	tracker := cudacapture.NewCorrelationTracker()
	_ = tracker.RecordExternalStack(uint32(cupti.ExternalCorrelationKindCustom0), 5, []string{"wrong"})
	_ = tracker.RecordActivity(cudacapture.ActivityRecord{Subtype: "external_correlation", ExternalKind: uint32(cupti.ExternalCorrelationKindCustom1), ExternalID: 5, Correlation: 10})
	_ = tracker.RecordRuntime(cudacapture.RuntimeLaunch{CorrelationID: 10, KernelName: "K"})
	_ = tracker.RecordActivity(cudacapture.ActivityRecord{Correlation: 10, Name: "K", Start: 1, End: 2})
	matched, ambiguous, _, _ := tracker.Join()
	if len(ambiguous) != 0 || len(matched) != 1 || matched[0].Callstack != nil {
		t.Fatalf("wrong namespace stack attached: matched=%+v ambiguous=%+v", matched, ambiguous)
	}
}

func TestCallstackNestedFrameworkKinds(t *testing.T) {
	tracker := cudacapture.NewCorrelationTracker()
	_ = tracker.RecordExternalStack(uint32(cupti.ExternalCorrelationKindCustom0), 1, []string{"custom0"})
	_ = tracker.RecordActivity(cudacapture.ActivityRecord{Subtype: "external_correlation", ExternalKind: uint32(cupti.ExternalCorrelationKindCustom0), ExternalID: 1, Correlation: 10})
	_ = tracker.RecordActivity(cudacapture.ActivityRecord{Subtype: "external_correlation", ExternalKind: uint32(cupti.ExternalCorrelationKindCustom1), ExternalID: 99, Correlation: 10})
	_ = tracker.RecordRuntime(cudacapture.RuntimeLaunch{CorrelationID: 10, KernelName: "K"})
	_ = tracker.RecordActivity(cudacapture.ActivityRecord{Correlation: 10, Name: "K", Start: 1, End: 2})
	matched, ambiguous, _, _ := tracker.Join()
	if len(ambiguous) != 0 || len(matched) != 1 || strings.Join(matched[0].Callstack, ";") != "custom0" {
		t.Fatalf("matched=%+v ambiguous=%+v", matched, ambiguous)
	}
}

func TestJoinCallstackIsCopy(t *testing.T) {
	tracker := cudacapture.NewCorrelationTracker()
	_ = tracker.RecordExternalStack(0, 1, []string{"original"})
	_ = tracker.RecordExternalCorrelation(0, 1, 10)
	_ = tracker.RecordRuntime(cudacapture.RuntimeLaunch{CorrelationID: 10, KernelName: "K"})
	_ = tracker.RecordActivity(cudacapture.ActivityRecord{Correlation: 10, Name: "K", Start: 1, End: 2})
	first, _, _, _ := tracker.Join()
	first[0].Callstack[0] = "mutated"
	second, _, _, _ := tracker.Join()
	if second[0].Callstack[0] != "original" {
		t.Fatalf("second Join stack = %v, want original", second[0].Callstack)
	}
}

func TestRuntimeCallstackRuleIsLocal(t *testing.T) {
	tracker := cudacapture.NewCorrelationTracker()
	_ = tracker.RecordExternalStack(0, 1, []string{"external"})
	_ = tracker.RecordExternalCorrelation(0, 1, 10)
	_ = tracker.RecordRuntime(cudacapture.RuntimeLaunch{CorrelationID: 10, KernelName: "external-kernel"})
	_ = tracker.RecordActivity(cudacapture.ActivityRecord{Correlation: 10, Name: "external-kernel", Start: 1, End: 2})
	_ = tracker.RecordRuntime(cudacapture.RuntimeLaunch{CorrelationID: 20, KernelName: "runtime-kernel", Callstack: []string{"runtime"}})
	_ = tracker.RecordActivity(cudacapture.ActivityRecord{Correlation: 20, Name: "runtime-kernel", Start: 3, End: 4})
	matched, _, _, _ := tracker.Join()
	if len(matched) != 2 || strings.Join(matched[0].Callstack, ";") != "external" || strings.Join(matched[1].Callstack, ";") != "runtime" {
		t.Fatalf("unexpected local stack handling: %+v", matched)
	}
}

func TestCallstackMissingForExternalID(t *testing.T) {
	tracker := cudacapture.NewCorrelationTracker()
	_ = tracker.RecordExternalCorrelation(0, 9, 10)
	_ = tracker.RecordRuntime(cudacapture.RuntimeLaunch{CorrelationID: 10, KernelName: "K"})
	_ = tracker.RecordActivity(cudacapture.ActivityRecord{Correlation: 10, Name: "K", Start: 1, End: 2})
	matched, _, _, _ := tracker.Join()
	if len(matched) != 1 || !matched[0].HasExternalID || matched[0].ExternalID != 9 || !matched[0].CallstackMissing || matched[0].Callstack != nil {
		t.Fatalf("missing stack not distinguished: %+v", matched)
	}
}

func TestAmbiguousExternalIDs(t *testing.T) {
	tracker := cudacapture.NewCorrelationTracker()
	_ = tracker.RecordExternalCorrelation(0, 9, 10)
	_ = tracker.RecordExternalCorrelation(0, 11, 10)
	_ = tracker.RecordRuntime(cudacapture.RuntimeLaunch{CorrelationID: 10, KernelName: "K"})
	_ = tracker.RecordActivity(cudacapture.ActivityRecord{Correlation: 10, Name: "K", Start: 1, End: 2})
	matched, ambiguous, _, _ := tracker.Join()
	if len(matched) != 0 || len(ambiguous) != 1 {
		t.Fatalf("matched=%+v ambiguous=%+v", matched, ambiguous)
	}
}

func TestSameStackTwiceNotAmbiguous(t *testing.T) {
	tracker := cudacapture.NewCorrelationTracker()
	_ = tracker.RecordExternalStack(0, 9, []string{"same"})
	_ = tracker.RecordExternalStack(0, 9, []string{"same"})
	_ = tracker.RecordExternalCorrelation(0, 9, 10)
	_ = tracker.RecordRuntime(cudacapture.RuntimeLaunch{CorrelationID: 10, KernelName: "K"})
	_ = tracker.RecordActivity(cudacapture.ActivityRecord{Correlation: 10, Name: "K", Start: 1, End: 2})
	matched, ambiguous, _, _ := tracker.Join()
	if len(matched) != 1 || len(ambiguous) != 0 {
		t.Fatalf("matched=%+v ambiguous=%+v", matched, ambiguous)
	}
}

func TestScrambledExternalIDsMoveStacks(t *testing.T) {
	tracker := cudacapture.NewCorrelationTracker()
	for id, stack := range map[uint64]string{1: "SA", 2: "SB"} {
		_ = tracker.RecordExternalStack(0, id, []string{stack})
	}
	_ = tracker.RecordExternalCorrelation(0, 2, 10)
	_ = tracker.RecordExternalCorrelation(0, 1, 20)
	_ = tracker.RecordExternalCorrelation(0, 99, 30)
	for id, name := range map[uint64]string{10: "A", 20: "B", 30: "C"} {
		_ = tracker.RecordRuntime(cudacapture.RuntimeLaunch{CorrelationID: id, KernelName: name})
		_ = tracker.RecordActivity(cudacapture.ActivityRecord{Correlation: uint32(id), Name: name, Start: id, End: id + 1})
	}
	matched, _, _, _ := tracker.Join()
	got := map[string]string{}
	for _, join := range matched {
		if len(join.Callstack) != 0 {
			got[join.KernelName] = join.Callstack[0]
		}
	}
	if got["A"] != "SB" || got["B"] != "SA" || got["C"] != "" {
		t.Fatalf("kernel stacks = %v, want A=SB B=SA C=<none>", got)
	}
}

func TestFoldedWeightIsDeviceBusyNotSpan(t *testing.T) {
	var out bytes.Buffer
	joins := []cudacapture.CorrelationJoin{
		{KernelName: "K", Callstack: []string{"site"}, DeviceStart: 1, DeviceEnd: 501},
		{KernelName: "K", Callstack: []string{"site"}, DeviceStart: 10_000_001, DeviceEnd: 10_000_501},
	}
	if err := cudacapture.ExportFoldedStacks(&out, joins); err != nil {
		t.Fatal(err)
	}
	if got := strings.TrimSpace(strings.TrimPrefix(out.String(), "site;[GPU]K ")); got != "1000" {
		t.Fatalf("folded weight = %q, want 1000", got)
	}
}

func TestFoldedRejectsUnavailableTiming(t *testing.T) {
	for _, exec := range []cudaprof.KernelExecution{{KernelName: "K"}, {KernelName: "K", StartNs: 4, EndNs: 3}} {
		var out bytes.Buffer
		if err := cudaprof.ExportFoldedStacks(&out, []cudaprof.KernelExecution{exec}); err == nil {
			t.Fatalf("ExportFoldedStacks(%+v) succeeded", exec)
		}
	}
}

func TestFoldedEscapesNames(t *testing.T) {
	var out bytes.Buffer
	err := cudaprof.ExportFoldedStacks(&out, []cudaprof.KernelExecution{{KernelName: "bad\nname", StartNs: 1, EndNs: 2}})
	if err == nil || strings.Count(out.String(), "\n") != 0 {
		t.Fatalf("invalid name error=%v output=%q", err, out.String())
	}
}

func TestFoldedTwoDevices(t *testing.T) {
	var out bytes.Buffer
	execs := []cudaprof.KernelExecution{
		{KernelName: "K", Callstack: []string{"site"}, StartNs: 1, EndNs: 101, Device: 0},
		{KernelName: "K", Callstack: []string{"site"}, StartNs: 1, EndNs: 101, Device: 1},
	}
	if err := cudaprof.ExportFoldedStacks(&out, execs); err != nil {
		t.Fatal(err)
	}
	if got := strings.TrimSpace(strings.TrimPrefix(out.String(), "site;[GPU]K ")); got != "200" {
		t.Fatalf("folded weight = %q, want 200", got)
	}
}
