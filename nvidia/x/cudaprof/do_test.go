package cudaprof

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"runtime/pprof"
	"strings"
	"sync"
	"sync/atomic"
	"testing"

	"github.com/tmc/lib/nvidia/cupti"
	"github.com/tmc/lib/nvidia/x/nvtx"
)

var sidecarTestPushedID uint64

func TestMain(m *testing.M) {
	if os.Getenv("CUDAPROF_TEST_STACK_CHILD") == "1" {
		pushExternalID = func(_ cupti.ExternalCorrelationKind, id uint64) error { sidecarTestPushedID = id; return nil }
		popExternalID = func(cupti.ExternalCorrelationKind) (uint64, error) { return sidecarTestPushedID, nil }
	}
	os.Exit(m.Run())
}

func TestExternalStackSidecarWrittenByTarget(t *testing.T) {
	if os.Getenv("CUDAPROF_TEST_STACK_CHILD") == "1" {
		if sidecarTargetCallSite() == 0 {
			os.Exit(2)
		}
		os.Exit(0)
	}
	path := filepath.Join(t.TempDir(), "external-stacks.json")
	cmd := exec.Command(os.Args[0], "-test.run=^TestExternalStackSidecarWrittenByTarget$")
	cmd.Env = sidecarTestEnvironment(path)
	if output, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("child: %v\n%s", err, output)
	}
	f, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	scanner := bufio.NewScanner(f)
	scanner.Buffer(make([]byte, 1024), 1<<20)
	if !scanner.Scan() {
		t.Fatal("sidecar has no header")
	}
	var header struct {
		Version int    `json:"version"`
		Kind    string `json:"kind"`
		Losses  string `json:"losses"`
	}
	if err := json.Unmarshal(scanner.Bytes(), &header); err != nil {
		t.Fatal(err)
	}
	if header.Version != 1 || header.Kind != "header" {
		t.Fatalf("header = %+v", header)
	}
	if !scanner.Scan() {
		t.Fatal("sidecar has no stack record")
	}
	var record ExternalStackRecord
	if err := json.Unmarshal(scanner.Bytes(), &record); err != nil {
		t.Fatal(err)
	}
	for _, frame := range record.Frames {
		if strings.Contains(frame, "sidecarTargetCallSite") {
			return
		}
	}
	t.Fatalf("sidecar stack has no target call site: %v", record.Frames)
}

func sidecarTestEnvironment(path string) []string {
	var env []string
	for _, item := range os.Environ() {
		if strings.HasPrefix(item, "CUDAPROF_TEST_STACK_CHILD=") || strings.HasPrefix(item, "CUDAPROF_CORRELATE=") || strings.HasPrefix(item, ExternalStacksEnv+"=") {
			continue
		}
		env = append(env, item)
	}
	return append(env, "CUDAPROF_TEST_STACK_CHILD=1", "CUDAPROF_CORRELATE=1", ExternalStacksEnv+"="+path)
}

//go:noinline
func sidecarTargetCallSite() uint64 {
	return Do(context.Background(), Labels(), func(context.Context) {})
}

// resetCorrelation restores the package default and undoes it after the test.
// The disposition is sticky by design, so a test that leaves a failure in place
// would report it as the next test's.
func resetCorrelation(t *testing.T) {
	t.Helper()
	kind := correlationKind.Load()
	disp := CorrelationDisposition()
	t.Cleanup(func() {
		correlationKind.Store(kind)
		correlationDisposition.Store(disp)
	})
	Correlate(cupti.ExternalCorrelationKindInvalid)
}

func withFakeCorrelation(t *testing.T, push func(cupti.ExternalCorrelationKind, uint64) error) {
	t.Helper()
	oldPush, oldPop := pushExternalID, popExternalID
	var lastID uint64
	pushExternalID = func(kind cupti.ExternalCorrelationKind, id uint64) error {
		err := push(kind, id)
		if err == nil {
			lastID = id
		}
		return err
	}
	popExternalID = func(cupti.ExternalCorrelationKind) (uint64, error) { return lastID, nil }
	Correlate(cupti.ExternalCorrelationKindCustom0)
	ResetExternalStacks()
	t.Cleanup(func() {
		pushExternalID, popExternalID = oldPush, oldPop
		ResetExternalStacks()
		Correlate(cupti.ExternalCorrelationKindInvalid)
	})
}

func TestExternalStackBudget(t *testing.T) {
	withFakeCorrelation(t, func(cupti.ExternalCorrelationKind, uint64) error { return nil })
	externalStackBudget = 2
	t.Cleanup(func() { externalStackBudget = 4096 })
	for id := uint64(1); id <= 5; id++ {
		Do(context.Background(), Labels(), func(context.Context) {})
	}
	var buf bytes.Buffer
	if err := WriteExternalStackTable(&buf); err != nil {
		t.Fatal(err)
	}
	var table ExternalStackTable
	if err := json.Unmarshal(buf.Bytes(), &table); err != nil {
		t.Fatal(err)
	}
	if len(table.Stacks) != 2 || table.Losses != 3 {
		t.Fatalf("table has %d stacks and %d losses, want 2 and 3", len(table.Stacks), table.Losses)
	}
}

func TestExternalStackVariantsAreRetained(t *testing.T) {
	ResetExternalStacks()
	t.Cleanup(ResetExternalStacks)
	kind := cupti.ExternalCorrelationKindCustom0
	RecordExternalStack(kind, 7, []string{"b"})
	RecordExternalStack(kind, 7, []string{"a"})
	RecordExternalStack(kind, 7, []string{"a"})
	var buf bytes.Buffer
	if err := WriteExternalStackTable(&buf); err != nil {
		t.Fatal(err)
	}
	var table ExternalStackTable
	if err := json.Unmarshal(buf.Bytes(), &table); err != nil {
		t.Fatal(err)
	}
	if len(table.Stacks) != 2 || table.Stacks[0].Frames[0] != "a" || table.Stacks[1].Frames[0] != "b" {
		t.Fatalf("stack variants = %+v, want deterministic a then b", table.Stacks)
	}
}

func TestNoStackWhenPushFails(t *testing.T) {
	withFakeCorrelation(t, func(cupti.ExternalCorrelationKind, uint64) error { return errors.New("push failed") })
	if id := Do(context.Background(), Labels(), func(context.Context) {}); id != 0 {
		t.Fatalf("Do ID = %d after failed push, want 0", id)
	}
	if got := ExternalStackLosses(); got != 0 {
		t.Fatalf("losses = %d, want 0", got)
	}
	var buf bytes.Buffer
	if err := WriteExternalStackTable(&buf); err != nil {
		t.Fatal(err)
	}
	var table ExternalStackTable
	if err := json.Unmarshal(buf.Bytes(), &table); err != nil {
		t.Fatal(err)
	}
	if len(table.Stacks) != 0 {
		t.Fatalf("failed push recorded %d stacks", len(table.Stacks))
	}
}

func TestDoStackRootFirst(t *testing.T) {
	withFakeCorrelation(t, func(cupti.ExternalCorrelationKind, uint64) error { return nil })
	id := callstackTestCaller()
	stack, ok := ExternalStack(cupti.ExternalCorrelationKindCustom0, id)
	if !ok || len(stack) < 2 {
		t.Fatalf("ExternalStack = %v, ok=%v", stack, ok)
	}
	if !strings.Contains(stack[len(stack)-1], "callstackTestCaller") {
		t.Fatalf("leaf = %q, want callstackTestCaller", stack[len(stack)-1])
	}
	if !strings.Contains(stack[len(stack)-2], "TestDoStackRootFirst") {
		t.Fatalf("caller frame = %q, want TestDoStackRootFirst", stack[len(stack)-2])
	}
	for _, frame := range stack {
		if strings.Contains(frame, "cudaprof.Do") || strings.Contains(frame, "runtime/pprof.Do") {
			t.Fatalf("internal frame remains in stack: %q", frame)
		}
	}
}

//go:noinline
func callstackTestCaller() uint64 {
	return Do(context.Background(), Labels(), func(context.Context) {})
}

func TestDoStackTruncationMarked(t *testing.T) {
	withFakeCorrelation(t, func(cupti.ExternalCorrelationKind, uint64) error { return nil })
	var descend func(int) uint64
	descend = func(n int) uint64 {
		if n == 0 {
			return Do(context.Background(), Labels(), func(context.Context) {})
		}
		return descend(n - 1)
	}
	id := descend(100)
	stack, ok := ExternalStack(cupti.ExternalCorrelationKindCustom0, id)
	if !ok || len(stack) == 0 || stack[0] != "[truncated]" {
		t.Fatalf("stack root = %v, ok=%v; want truncation marker", stack, ok)
	}
}

// TestDoPushesNothingUntilCorrelateIsCalled is the reason correlation is
// opt-in: the first push loads libcupti, and a labelling helper must not do
// that to a process that never asked to be profiled. A Do with correlation off
// must leave the library untouched.
func TestDoPushesNothingUntilCorrelateIsCalled(t *testing.T) {
	resetCorrelation(t)
	if cupti.Loaded() {
		t.Skip("libcupti already loaded in this process; this test can only prove the negative from a clean start")
	}
	var ran bool
	Do(context.Background(), Labels("phase", "off"), func(context.Context) { ran = true })
	if !ran {
		t.Fatal("Do did not call f")
	}
	if cupti.Loaded() {
		t.Error("Do loaded libcupti with correlation off; it must not touch the library uninvited")
	}
	if got := CorrelationDisposition(); got != CorrelationOff {
		t.Errorf("disposition = %q, want %q", got, CorrelationOff)
	}
}

// TestCorrelateReportsWhatActuallyHappened is the control that fails against
// the version of this package that generated a correlation ID and never pushed
// it. That version left the disposition saying "pushing" forever, because
// nothing ever called CUPTI and so nothing could ever contradict it.
//
// After Correlate, one Do must move the disposition off CorrelationPushing on
// any host where libcupti is not callable -- which is every host that is not
// running a capture. A disposition still reading "pushing" here means the push
// was never attempted.
func TestCorrelateReportsWhatActuallyHappened(t *testing.T) {
	resetCorrelation(t)
	Correlate(cupti.ExternalCorrelationKindCustom0)
	if got := CorrelationDisposition(); got != CorrelationPushing {
		t.Fatalf("disposition after Correlate = %q, want %q", got, CorrelationPushing)
	}
	Do(context.Background(), Labels("phase", "on"), func(context.Context) {})

	got := CorrelationDisposition()
	if cupti.Loaded() {
		// On a capture host the push is expected to succeed.
		if got != CorrelationPushing {
			t.Errorf("disposition = %q with libcupti loaded, want %q", got, CorrelationPushing)
		}
		return
	}
	if got == CorrelationPushing {
		t.Errorf("disposition = %q with no libcupti in the process: the push was never attempted", got)
	}
	if got != CorrelationUnavailable {
		t.Errorf("disposition = %q, want %q", got, CorrelationUnavailable)
	}
}

// TestCorrelateInvalidTurnsItOff checks the documented off switch, so a caller
// that enabled correlation for one phase can disable it for the next.
func TestCorrelateInvalidTurnsItOff(t *testing.T) {
	resetCorrelation(t)
	Correlate(cupti.ExternalCorrelationKindCustom0)
	Correlate(cupti.ExternalCorrelationKindInvalid)
	if got := CorrelationDisposition(); got != CorrelationOff {
		t.Errorf("disposition = %q, want %q", got, CorrelationOff)
	}
	if got := pushCorrelation(context.Background(), 1); got.cupti {
		t.Error("pushCorrelation pushed a CUPTI ID after correlation was turned off")
	}
}

// TestPushPinsWhenItEntersAThreadLocalBracket is what keeps the pop on the
// thread the push ran on. It asserts both directions: a region that enters no
// thread-local bracket must not pin, because pinning every labelled region
// would consume an OS thread per concurrent Do for no reason.
func TestPushPinsWhenItEntersAThreadLocalBracket(t *testing.T) {
	resetCorrelation(t)
	if nvtx.Enabled() {
		t.Skip("an NVTX provider is attached, so the unpinned case cannot be reached here")
	}

	entered := pushCorrelation(context.Background(), 1)
	if entered.pinned {
		popCorrelation(1, entered)
		runtime.UnlockOSThread()
		t.Fatal("pushCorrelation pinned the thread with no bracket to enter")
	}
	popCorrelation(1, entered)

	Correlate(cupti.ExternalCorrelationKindCustom0)
	entered = pushCorrelation(context.Background(), 2)
	if !entered.pinned {
		t.Error("pushCorrelation did not pin the thread for a region that pushes a per-thread CUPTI ID")
	}
	popCorrelation(2, entered)
	if entered.pinned {
		runtime.UnlockOSThread()
	}
}

// TestPopUnwindsOnlyWhatWasEntered checks that a failed push is not unwound. A
// pop of a bracket that was never entered pops its caller's, which is worse
// than not popping at all: the error lands on an unrelated region.
func TestPopUnwindsOnlyWhatWasEntered(t *testing.T) {
	resetCorrelation(t)
	// A zero regionBrackets is what a push returns when it entered nothing.
	// popCorrelation must be a no-op on it rather than calling CUPTI or NVTX.
	popCorrelation(99, regionBrackets{})
	if got := CorrelationDisposition(); got != CorrelationOff {
		t.Errorf("popping an unentered region changed the disposition to %q", got)
	}
}

// TestDoAssignsOneCorrelationIDPerRegion checks that concurrent and nested
// regions never share an ID. Two regions sharing one would be reported by the
// consumer as an ambiguous match rather than a wrong one, but it still costs
// the attribution the ID exists to carry.
func TestDoAssignsOneCorrelationIDPerRegion(t *testing.T) {
	resetCorrelation(t)
	const goroutines, depth = 8, 4

	before := atomic.LoadUint64(&nextCorrID)
	var wg sync.WaitGroup
	for range goroutines {
		wg.Add(1)
		go func() {
			defer wg.Done()
			var nest func(ctx context.Context, n int)
			nest = func(ctx context.Context, n int) {
				if n == 0 {
					return
				}
				Do(ctx, Labels("depth", "x"), func(ctx context.Context) { nest(ctx, n-1) })
			}
			nest(context.Background(), depth)
		}()
	}
	wg.Wait()

	// The counter is the only issuer, and it only ever increments, so an
	// advance of exactly one per region is what says no two regions could
	// have been handed the same ID.
	if got, want := atomic.LoadUint64(&nextCorrID)-before, uint64(goroutines*depth); got != want {
		t.Errorf("issued %d correlation IDs for %d regions", got, want)
	}
}

// TestDoKeepsLabelsReachableInsideTheRegion guards the property every other
// consumer depends on: the context f receives carries the labels, so a profile
// taken inside the region is attributed to it.
func TestDoKeepsLabelsReachableInsideTheRegion(t *testing.T) {
	resetCorrelation(t)
	Do(context.Background(), Labels("phase", "prefill", "model", "ds4"), func(ctx context.Context) {
		got := map[string]string{}
		pprof.ForLabels(ctx, func(k, v string) bool {
			got[k] = v
			return true
		})
		if got["phase"] != "prefill" || got["model"] != "ds4" {
			t.Errorf("labels inside region = %v, want phase=prefill model=ds4", got)
		}
	})
}

func TestDoRetainsEnteredNamespace(t *testing.T) {
	resetCorrelation(t)
	oldPush, oldPop := pushExternalID, popExternalID
	defer func() { pushExternalID, popExternalID = oldPush, oldPop }()
	var pushed uint64
	pushExternalID = func(kind cupti.ExternalCorrelationKind, id uint64) error {
		if kind != cupti.ExternalCorrelationKindCustom0 {
			t.Fatalf("push kind = %v", kind)
		}
		pushed = id
		return nil
	}
	popExternalID = func(kind cupti.ExternalCorrelationKind) (uint64, error) {
		if kind != cupti.ExternalCorrelationKindCustom0 {
			t.Fatalf("pop kind = %v, want entered Custom0", kind)
		}
		return pushed, nil
	}
	Correlate(cupti.ExternalCorrelationKindCustom0)
	Do(context.Background(), Labels(), func(context.Context) {
		Correlate(cupti.ExternalCorrelationKindCustom1)
	})
}
