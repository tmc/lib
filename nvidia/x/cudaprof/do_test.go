package cudaprof

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
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
	if os.Getenv("CUDAPROF_TEST_STACK_CHILD") != "" {
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
	if os.Getenv("CUDAPROF_TEST_STACK_CHILD") == "writer" {
		externalStackBudget = 1
		sidecarTargetCallSite()
		externalStackBudget = 0
		for i := 0; i < 128; i++ {
			sidecarTargetCallSite()
		}
		os.Exit(0)
	}
	path := filepath.Join(t.TempDir(), "external-stacks.json")
	cmd := exec.Command(os.Args[0], "-test.run=^TestExternalStackSidecarWrittenByTarget$")
	cmd.Env = sidecarTestEnvironmentMode(path, "writer")
	if output, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("child: %v\n%s", err, output)
	}
	f, err := os.Open(fmt.Sprintf("%s.%d", path, cmd.Process.Pid))
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
		Version    int    `json:"version"`
		Kind       string `json:"kind"`
		Losses     string `json:"losses"`
		LowerBound bool   `json:"losses_lower_bound"`
	}
	if err := json.Unmarshal(scanner.Bytes(), &header); err != nil {
		t.Fatalf("header line %q: %v", scanner.Bytes(), err)
	}
	if header.Version != 1 || header.Kind != "header" {
		t.Fatalf("header = %+v", header)
	}
	if !scanner.Scan() {
		t.Fatal("sidecar has no stack record")
	}
	var record ExternalStackRecord
	if err := json.Unmarshal(scanner.Bytes(), &record); err != nil {
		all, _ := os.ReadFile(fmt.Sprintf("%s.%d", path, cmd.Process.Pid))
		t.Fatalf("stack line %q in %q: %v", scanner.Bytes(), all, err)
	}
	if record.Process != uint64(cmd.Process.Pid) {
		t.Fatalf("record process = %d, want %d", record.Process, cmd.Process.Pid)
	}
	if len(record.Frames) == 0 || !strings.Contains(record.Frames[len(record.Frames)-1], "sidecarTargetCallSite") {
		t.Fatalf("leaf frame = %v, want sidecarTargetCallSite", record.Frames)
	}
	var lastLoss string
	for scanner.Scan() {
		var row struct {
			Kind       string `json:"kind"`
			Losses     string `json:"losses"`
			LowerBound bool   `json:"losses_lower_bound"`
		}
		if err := json.Unmarshal(scanner.Bytes(), &row); err != nil {
			t.Fatal(err)
		}
		if row.Kind == "losses" {
			lastLoss = row.Losses
			if !row.LowerBound {
				t.Fatal("loss checkpoint is not a lower bound")
			}
		}
	}
	if lastLoss != "00000000000000000065" {
		t.Fatalf("loss checkpoint = %q, want lower-bound checkpoint 65", lastLoss)
	}
}

func TestDoCostAfterStackBudgetFull(t *testing.T) {
	path := filepath.Join(t.TempDir(), "external-stacks.json")
	old, had := os.LookupEnv(ExternalStacksEnv)
	if err := os.Setenv(ExternalStacksEnv, path); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if had {
			_ = os.Setenv(ExternalStacksEnv, old)
		} else {
			_ = os.Unsetenv(ExternalStacksEnv)
		}
	})
	withFakeCorrelation(t, func(cupti.ExternalCorrelationKind, uint64) error { return nil })
	externalStackBudget = 1
	t.Cleanup(func() { externalStackBudget = 4096 })
	sidecarTargetCallSite() // retained stack
	sidecarTargetCallSite() // first loss checkpoints the header
	before, err := os.ReadFile(fmt.Sprintf("%s.%d", path, os.Getpid()))
	if err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 32; i++ {
		sidecarTargetCallSite()
	}
	after, err := os.ReadFile(fmt.Sprintf("%s.%d", path, os.Getpid()))
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(before, after) {
		t.Fatalf("sidecar changed during full-budget calls: size %d -> %d", len(before), len(after))
	}
	var checkpoint struct {
		Kind       string `json:"kind"`
		Losses     string `json:"losses"`
		LowerBound bool   `json:"losses_lower_bound"`
	}
	for _, line := range bytes.Split(after, []byte{'\n'}) {
		var row struct {
			Kind       string `json:"kind"`
			Losses     string `json:"losses"`
			LowerBound bool   `json:"losses_lower_bound"`
		}
		if len(line) == 0 {
			continue
		}
		if err := json.Unmarshal(line, &row); err != nil {
			t.Fatal(err)
		}
		if row.Kind == "losses" {
			checkpoint = row
		}
	}
	if checkpoint.Kind != "losses" || checkpoint.Losses != "00000000000000000001" || !checkpoint.LowerBound {
		t.Fatalf("loss checkpoint = %+v", checkpoint)
	}
}

func BenchmarkDoCostAfterStackBudgetFull(b *testing.B) {
	base := filepath.Join(b.TempDir(), "external-stacks.json")
	oldPath, hadPath := os.LookupEnv(ExternalStacksEnv)
	_ = os.Setenv(ExternalStacksEnv, base)
	oldPush, oldPop := pushExternalID, popExternalID
	oldBudget, oldKind := externalStackBudget, correlationKind.Load()
	var last uint64
	pushExternalID = func(_ cupti.ExternalCorrelationKind, id uint64) error { last = id; return nil }
	popExternalID = func(cupti.ExternalCorrelationKind) (uint64, error) { return last, nil }
	ResetExternalStacks()
	externalStackBudget = 1
	Correlate(cupti.ExternalCorrelationKindCustom0)
	sidecarTargetCallSite()
	sidecarTargetCallSite()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sidecarTargetCallSite()
	}
	b.StopTimer()
	pushExternalID, popExternalID = oldPush, oldPop
	externalStackBudget = oldBudget
	correlationKind.Store(oldKind)
	ResetExternalStacks()
	if hadPath {
		_ = os.Setenv(ExternalStacksEnv, oldPath)
	} else {
		_ = os.Unsetenv(ExternalStacksEnv)
	}
}

func BenchmarkDoCostWithinStackBudget(b *testing.B) {
	base := filepath.Join(b.TempDir(), "external-stacks.json")
	oldPath, hadPath := os.LookupEnv(ExternalStacksEnv)
	_ = os.Setenv(ExternalStacksEnv, base)
	oldPush, oldPop := pushExternalID, popExternalID
	oldBudget, oldKind := externalStackBudget, correlationKind.Load()
	var last uint64
	pushExternalID = func(_ cupti.ExternalCorrelationKind, id uint64) error { last = id; return nil }
	popExternalID = func(cupti.ExternalCorrelationKind) (uint64, error) { return last, nil }
	ResetExternalStacks()
	externalStackBudget = b.N + 10
	Correlate(cupti.ExternalCorrelationKindCustom0)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sidecarTargetCallSite()
	}
	b.StopTimer()
	pushExternalID, popExternalID = oldPush, oldPop
	externalStackBudget = oldBudget
	correlationKind.Store(oldKind)
	ResetExternalStacks()
	if hadPath {
		_ = os.Setenv(ExternalStacksEnv, oldPath)
	} else {
		_ = os.Unsetenv(ExternalStacksEnv)
	}
}

func TestExternalStackSidecarConcurrentChildren(t *testing.T) {
	if os.Getenv("CUDAPROF_TEST_STACK_CHILD") == "concurrent" {
		externalStackBudget = 4
		for i := 0; i < 4; i++ {
			RecordExternalStackKind(cupti.ExternalCorrelationKindCustom0, uint64(i+1), []string{"concurrent-child", fmt.Sprint(i)})
		}
		os.Exit(0)
	}
	base := filepath.Join(t.TempDir(), "external-stacks.json")
	commands := make([]*exec.Cmd, 2)
	for i := range commands {
		commands[i] = exec.Command(os.Args[0], "-test.run=^TestExternalStackSidecarConcurrentChildren$")
		commands[i].Env = sidecarTestEnvironmentMode(base, "concurrent")
	}
	var wg sync.WaitGroup
	errs := make([]error, len(commands))
	for i, cmd := range commands {
		wg.Add(1)
		go func(i int, cmd *exec.Cmd) { defer wg.Done(); _, errs[i] = cmd.CombinedOutput() }(i, cmd)
	}
	wg.Wait()
	for i, err := range errs {
		if err != nil {
			t.Fatalf("child %d: %v", i, err)
		}
	}
	for _, cmd := range commands {
		data, err := os.ReadFile(fmt.Sprintf("%s.%d", base, cmd.Process.Pid))
		if err != nil {
			t.Fatal(err)
		}
		lines := bytes.Split(bytes.TrimSpace(data), []byte{'\n'})
		if len(lines) != 5 {
			t.Fatalf("process %d has %d complete lines, want header + 4 stacks", cmd.Process.Pid, len(lines))
		}
		seen := make(map[uint64]bool)
		for n, line := range lines {
			var row struct {
				Kind       string `json:"kind"`
				Process    uint64 `json:"process"`
				ExternalID uint64 `json:"external_id"`
			}
			if err := json.Unmarshal(line, &row); err != nil {
				t.Fatalf("process %d torn record: %v", cmd.Process.Pid, err)
			}
			if row.Process != uint64(cmd.Process.Pid) {
				t.Fatalf("sidecar record process=%d, want %d", row.Process, cmd.Process.Pid)
			}
			if n > 0 {
				if row.Kind != "stack" || row.ExternalID == 0 || seen[row.ExternalID] {
					t.Fatalf("invalid or duplicate stack row: %+v", row)
				}
				seen[row.ExternalID] = true
			}
		}
		if len(seen) != 4 {
			t.Fatalf("process %d retained %d distinct records, want 4", cmd.Process.Pid, len(seen))
		}
	}
}

func sidecarTestEnvironment(path string) []string {
	return sidecarTestEnvironmentMode(path, "1")
}

func sidecarTestEnvironmentMode(path, mode string) []string {
	var env []string
	for _, item := range os.Environ() {
		if strings.HasPrefix(item, "CUDAPROF_TEST_STACK_CHILD=") || strings.HasPrefix(item, "CUDAPROF_CORRELATE=") || strings.HasPrefix(item, ExternalStacksEnv+"=") {
			continue
		}
		env = append(env, item)
	}
	return append(env, "CUDAPROF_TEST_STACK_CHILD="+mode, "CUDAPROF_CORRELATE=1", ExternalStacksEnv+"="+path)
}

//go:noinline
func sidecarTargetCallSite() uint64 {
	return DoID(context.Background(), Labels(), func(context.Context) {})
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
		DoID(context.Background(), Labels(), func(context.Context) {})
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

func TestExternalStackSidecarByteCap(t *testing.T) {
	path := filepath.Join(t.TempDir(), "external-stacks.json")
	old, had := os.LookupEnv(ExternalStacksEnv)
	_ = os.Setenv(ExternalStacksEnv, path)
	ResetExternalStacks()
	externalStackBudget = 10
	t.Cleanup(func() {
		ResetExternalStacks()
		externalStackBudget = 4096
		if had {
			_ = os.Setenv(ExternalStacksEnv, old)
		} else {
			_ = os.Unsetenv(ExternalStacksEnv)
		}
	})
	if RecordExternalStackKind(cupti.ExternalCorrelationKindCustom0, 1, []string{strings.Repeat("frame", maxExternalStackSidecarBytes/len("frame")+1)}) {
		t.Fatal("oversized sidecar record was retained")
	}
	if _, ok := ExternalStackKind(cupti.ExternalCorrelationKindCustom0, 1); ok {
		t.Fatal("failed sidecar write remained in memory")
	}
	if ExternalStackLosses() != 1 {
		t.Fatalf("losses = %d, want 1", ExternalStackLosses())
	}
}

func TestExternalStackFailedWriteIsLoss(t *testing.T) {
	base := filepath.Join(t.TempDir(), "external-stacks.json")
	if err := os.Mkdir(fmt.Sprintf("%s.%d", base, os.Getpid()), 0700); err != nil {
		t.Fatal(err)
	}
	old, had := os.LookupEnv(ExternalStacksEnv)
	_ = os.Setenv(ExternalStacksEnv, base)
	ResetExternalStacks()
	externalStackBudget = 10
	t.Cleanup(func() {
		ResetExternalStacks()
		externalStackBudget = 4096
		if had {
			_ = os.Setenv(ExternalStacksEnv, old)
		} else {
			_ = os.Unsetenv(ExternalStacksEnv)
		}
	})
	if RecordExternalStackKind(cupti.ExternalCorrelationKindCustom0, 2, []string{"failed write"}) {
		t.Fatal("failed write was reported retained")
	}
	if _, ok := ExternalStackKind(cupti.ExternalCorrelationKindCustom0, 2); ok {
		t.Fatal("failed write remained in memory")
	}
	if ExternalStackLosses() != 1 {
		t.Fatalf("losses = %d, want 1", ExternalStackLosses())
	}
}

func TestExternalStackVariantsAreRetained(t *testing.T) {
	ResetExternalStacks()
	t.Cleanup(ResetExternalStacks)
	kind := cupti.ExternalCorrelationKindCustom0
	RecordExternalStackKind(kind, 7, []string{"b"})
	RecordExternalStackKind(kind, 7, []string{"a"})
	RecordExternalStackKind(kind, 7, []string{"a"})
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
	if id := DoID(context.Background(), Labels(), func(context.Context) {}); id != 0 {
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
	stack, ok := ExternalStackKind(cupti.ExternalCorrelationKindCustom0, id)
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
	return DoID(context.Background(), Labels(), func(context.Context) {})
}

func TestDoStackTruncationMarked(t *testing.T) {
	withFakeCorrelation(t, func(cupti.ExternalCorrelationKind, uint64) error { return nil })
	var descend func(int) uint64
	descend = func(n int) uint64 {
		if n == 0 {
			return DoID(context.Background(), Labels(), func(context.Context) {})
		}
		return descend(n - 1)
	}
	id := descend(100)
	stack, ok := ExternalStackKind(cupti.ExternalCorrelationKindCustom0, id)
	if !ok || len(stack) == 0 || stack[0] != "[truncated]" {
		t.Fatalf("stack root = %v, ok=%v; want truncation marker", stack, ok)
	}
}

func TestDoStackExactly64FramesMarked(t *testing.T) {
	var pcs [maxCallstackFrames + 1]uintptr
	var caller [1]uintptr
	if n := runtime.Callers(0, caller[:]); n != 1 {
		t.Fatalf("runtime.Callers = %d, want 1", n)
	}
	for i := range pcs {
		pcs[i] = caller[0]
	}
	stack := formatCallstack(pcs[:maxCallstackFrames])
	if len(stack) == 0 || stack[0] == "[truncated]" {
		t.Fatalf("64-frame stack root = %v, want complete stack", stack)
	}
	stack = formatCallstack(pcs[:])
	if len(stack) == 0 || stack[0] != "[truncated]" {
		t.Fatalf("65-frame stack root = %v, want [truncated]", stack)
	}
}

type countedSidecarFile struct {
	externalStackSidecarFile
	syncs *int
}

func (f countedSidecarFile) Sync() error { *f.syncs++; return f.externalStackSidecarFile.Sync() }

func TestExternalStackSidecarKeepsOpenAndDoesNotSync(t *testing.T) {
	ResetExternalStacks()
	base := filepath.Join(t.TempDir(), "stacks")
	t.Setenv(ExternalStacksEnv, base)
	oldOpen := openExternalStackSidecarFile
	opens, syncs := 0, 0
	openExternalStackSidecarFile = func(path string) (externalStackSidecarFile, error) {
		opens++
		f, err := oldOpen(path)
		if err != nil {
			return nil, err
		}
		return countedSidecarFile{f, &syncs}, nil
	}
	t.Cleanup(func() { ResetExternalStacks(); openExternalStackSidecarFile = oldOpen })
	for i := uint64(1); i <= 8; i++ {
		if !RecordExternalStackKind(cupti.ExternalCorrelationKindCustom0, i, []string{"frame"}) {
			t.Fatal("record was rejected")
		}
	}
	if opens != 1 || syncs != 0 {
		t.Fatalf("retained stacks: opens=%d syncs=%d, want 1 and 0", opens, syncs)
	}
	ResetExternalStacks()
	for i := uint64(2); i <= externalStackLossCheckpointInterval; i++ {
		_ = appendExternalStackLoss(i)
	}
	if opens != 1 || syncs != 0 {
		t.Fatalf("losses between checkpoints: opens=%d syncs=%d, want unchanged", opens, syncs)
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
	DoID(context.Background(), Labels("phase", "off"), func(context.Context) { ran = true })
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
	DoID(context.Background(), Labels("phase", "on"), func(context.Context) {})

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
				DoID(ctx, Labels("depth", "x"), func(ctx context.Context) { nest(ctx, n-1) })
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
	DoID(context.Background(), Labels("phase", "prefill", "model", "ds4"), func(ctx context.Context) {
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
	DoID(context.Background(), Labels(), func(context.Context) {
		Correlate(cupti.ExternalCorrelationKindCustom1)
	})
}
