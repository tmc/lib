package cudaprof

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/cupti"
	"github.com/tmc/lib/nvidia/x/cudaprof/stack"
)

func TestSubmitDisabled(t *testing.T) {
	old := languageStacks.Swap(nil)
	defer languageStacks.Store(old)
	loaded := cupti.Loaded()
	want := errors.New("submission failed")
	if got := Submit(func() error { return want }); got != want {
		t.Fatalf("error = %v, want %v", got, want)
	}
	if cupti.Loaded() != loaded {
		t.Fatal("disabled submission loaded CUPTI")
	}
}

func TestSubmitNil(t *testing.T) {
	if err := Submit(nil); err == nil {
		t.Fatal("accepted nil function")
	}
}

func ExampleSetStackJournal() {
	var output bytes.Buffer
	journal := stack.NewJournal(&output, stack.Limits{})
	SetStackJournal(journal)
	// CUDA submission scopes can now record into output.
	SetStackJournal(nil)
	fmt.Println(output.Len())
	// Output: 0
}

func ExampleSubmit() {
	// Stack recording is disabled until SetStackJournal is called.
	err := Submit(func() error { return nil })
	fmt.Println(err)
	// Output: <nil>
}

func ExampleStackFailures() {
	before := StackFailures()
	SetStackJournal(nil)
	Submit(func() error { return nil })
	fmt.Println(StackFailures() == before)
	// Output: true
}

func TestGoStackAttribution(t *testing.T) {
	oldJournal := languageStacks.Load()
	oldFile := stackFile
	attempted, recorded, dropped, truncated := stackAttempts.Load(), stackRecorded.Load(), stackDropped.Load(), stackTruncated.Load()
	stackAttempts.Store(0)
	stackRecorded.Store(0)
	stackDropped.Store(0)
	stackTruncated.Store(0)
	defer func() {
		stackFile = oldFile
		stackAttempts.Store(attempted)
		stackRecorded.Store(recorded)
		stackDropped.Store(dropped)
		stackTruncated.Store(truncated)
	}()
	oldPush, oldPop := pushExternalID, popExternalID
	defer func() { SetStackJournal(oldJournal); pushExternalID, popExternalID = oldPush, oldPop }()
	var ids []uint64
	pushExternalID = func(kind cupti.ExternalCorrelationKind, id uint64) error {
		if kind != cupti.ExternalCorrelationKindCustom2 {
			t.Fatalf("kind = %v", kind)
		}
		ids = append(ids, id)
		return nil
	}
	popExternalID = func(kind cupti.ExternalCorrelationKind) (uint64, error) {
		if kind != cupti.ExternalCorrelationKindCustom2 {
			t.Fatalf("pop kind = %v", kind)
		}
		id := ids[len(ids)-1]
		ids = ids[:len(ids)-1]
		return id, nil
	}
	resetCorrelation(t)
	t.Setenv("CUDAPROF_SESSION", "test-session")
	transport := t.TempDir()
	t.Setenv("CUDAPROF_TRANSPORT", transport)
	dir := filepath.Join(transport, fmt.Sprintf("process-%d-123", os.Getpid()))
	if err := os.Mkdir(dir, 0700); err != nil {
		t.Fatal(err)
	}
	handshake := fmt.Sprintf(`{"protocol":1,"session":"test-session","process":%d,"generation":123,"transport_ready":true}`, os.Getpid())
	if err := os.WriteFile(filepath.Join(dir, "handshake.json"), []byte(handshake), 0600); err != nil {
		t.Fatal(err)
	}
	journalPath := filepath.Join(t.TempDir(), "journal.jsonl")
	f, err := os.Create(journalPath)
	if err != nil {
		t.Fatal(err)
	}
	stackFile = f
	SetStackJournal(stack.NewJournal(f, stack.Limits{}))
	Do(context.Background(), Labels(), func(context.Context) {
		if err := Submit(func() error { return nil }); err != nil {
			t.Fatal(err)
		}
	})
	if err := FinishStackRecording(); err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile(journalPath)
	if err != nil {
		t.Fatal(err)
	}
	records, err := stack.Read(bytes.NewReader(data), stack.Limits{})
	if err != nil {
		t.Fatal(err)
	}
	var attrs []string
	for _, r := range records {
		if r.Type == "coverage" {
			if !r.Final || r.Attempted != 2 || r.Recorded != 2 || r.Dropped != 0 {
				t.Fatalf("coverage = %+v", r)
			}
			continue
		}
		attrs = append(attrs, r.Attribution)
		if r.Status != "attached" || r.Generation != 123 || r.Session != "test-session" {
			t.Fatalf("identity/status = %+v", r)
		}
		if len(r.Frames) == 0 || !strings.Contains(r.Frames[0].Function, "TestGoStackAttribution") {
			t.Fatalf("caller frames = %+v", r.Frames)
		}
	}
	if fmt.Sprint(attrs) != "[submission scope_entry]" {
		t.Fatalf("attributions = %v", attrs)
	}
	if len(ids) != 0 {
		t.Fatal("left external IDs pushed")
	}
}

func ExampleFinishStackRecording() {
	SetStackJournal(nil)
	fmt.Println(FinishStackRecording())
	// Output: <nil>
}

func ExampleWriteStackCoverage() {
	SetStackJournal(nil)
	FinishStackRecording()
	var output bytes.Buffer
	err := WriteStackCoverage(&output)
	fmt.Println(err == nil, output.Len() > 0)
	// Output: true true
}

func ExampleStartCaptureHostProfile() {
	// The capture launcher supplies the directory. A standalone program can
	// supply one explicitly instead.
	home, _ := os.UserHomeDir()
	dir, _ := os.MkdirTemp(filepath.Join(home, "tmp"), "cudaprof-host-example-")
	defer os.RemoveAll(dir)
	old := os.Getenv("CUDAPROF_STACK_DIR")
	os.Setenv("CUDAPROF_STACK_DIR", dir)
	defer os.Setenv("CUDAPROF_STACK_DIR", old)
	p, err := StartCaptureHostProfile()
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(p.Stop())
	// Output: <nil>
}

func TestStackProcessFilename(t *testing.T) {
	process := newStackProcess()
	if process == "" {
		t.Fatal("missing process identity")
	}
	for _, suffix := range []string{".jsonl", ".trace", ".pprof", "-host.json"} {
		name := "stacks-go-" + process + suffix
		for _, r := range name {
			if r >= 'a' && r <= 'z' || r >= '0' && r <= '9' || r == '-' || r == '.' {
				continue
			}
			t.Fatalf("attachment name %q contains unsafe character %q", name, r)
		}
	}
}
