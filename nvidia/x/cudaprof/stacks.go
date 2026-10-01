package cudaprof

import (
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"runtime"
	"sync"
	"sync/atomic"

	"github.com/tmc/lib/nvidia/cupti"
	"github.com/tmc/lib/nvidia/x/cudaprof/stack"
)

var languageStacks atomic.Pointer[stack.Journal]
var stackStateMu sync.Mutex
var stackActive sync.WaitGroup
var stackAttempts, stackRecorded, stackDropped, stackTruncated atomic.Uint64
var stackFile *os.File
var stackFailures atomic.Uint64
var stackProcess = newStackProcess()

func newStackProcess() string {
	var id [16]byte
	if _, err := rand.Read(id[:]); err != nil {
		return ""
	}
	return fmt.Sprintf("%d-%s", os.Getpid(), hex.EncodeToString(id[:]))
}

// SetStackJournal enables Go stack recording into j. Passing nil disables it.
// Recording is opt-in and loads CUPTI when a region is entered. The journal's
// writer must remain open until all entered regions have completed.
// Do records scope-entry frames; Submit records frames at its call site.
// Go stacks use CUPTI Custom2, which the application must reserve for them.
func SetStackJournal(j *stack.Journal) {
	stackStateMu.Lock()
	defer stackStateMu.Unlock()
	languageStacks.Store(j)
}

// StackFailures returns the number of failed correlation operations or journal
// writes since process start. Failed pushes are also retained in the journal.
func StackFailures() uint64 { return stackFailures.Load() }

func init() {
	if v := os.Getenv("CUDAPROF_GO_STACKS"); v == "" || v == "0" {
		return
	}
	dir := os.Getenv("CUDAPROF_STACK_DIR")
	if dir == "" || stackProcess == "" {
		stackFailures.Add(1)
		return
	}
	name := "go-" + stackProcess + ".jsonl"
	// A process owns one file. A reused PID cannot append to a previous process.
	f, err := os.OpenFile(filepath.Join(dir, name), os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0600)
	if err != nil {
		stackFailures.Add(1)
		return
	}
	stackFile = f
	SetStackJournal(stack.NewJournal(f, stack.Limits{}))
}

// Submit calls f on one OS thread with an opt-in submission stack. Call it
// immediately around a synchronous CUDA submission, on the submitting thread.
// Work dispatched to another thread does not inherit this correlation ID.
// This records a submission call site, not a CPU sampling observation.
func Submit(f func() error) error {
	if f == nil {
		return fmt.Errorf("submit: nil function")
	}
	leave := enterStack("submission", 3)
	defer leave()
	return f()
}

func enterStack(attribution string, skip int) func() {
	if languageStacks.Load() == nil {
		return func() {}
	}
	stackStateMu.Lock()
	j := languageStacks.Load()
	if j == nil {
		stackStateMu.Unlock()
		return func() {}
	}
	stackActive.Add(1)
	stackStateMu.Unlock()
	stackAttempts.Add(1)
	id := atomic.AddUint64(&nextCorrID, 1)
	const kind = cupti.ExternalCorrelationKindCustom2
	pcs := make([]uintptr, 129)
	n := runtime.Callers(skip, pcs)
	truncated := n > 128
	if truncated {
		n = 128
	}
	frames := runtime.CallersFrames(pcs[:n])
	r := stack.Record{Version: stack.Version, Process: stackProcess, PID: os.Getpid(), Language: "go", Attribution: attribution, ExternalKind: uint32(kind), ExternalID: id, Truncated: truncated, Status: "attached"}
	for {
		frame, more := frames.Next()
		r.Frames = append(r.Frames, stack.Frame{Function: frame.Function, File: frame.File, Line: uint32(frame.Line)})
		if !more {
			break
		}
		if len(r.Frames) == 128 {
			r.Truncated = true
			break
		}
	}
	runtime.LockOSThread()
	err := pushExternalID(kind, id)
	if err != nil {
		stackFailures.Add(1)
		r.Status = "push_failed"
		if !cupti.Loaded() {
			r.Status = "unavailable"
		}
	}
	r.Session = os.Getenv("CUDAPROF_SESSION")
	r.Generation = stackGeneration(r.Session)
	if r.Generation == 0 && r.Status == "attached" {
		r.Status = "unavailable"
	}
	if r.Truncated {
		stackTruncated.Add(1)
	}
	return func() {
		defer stackActive.Done()
		defer runtime.UnlockOSThread()
		if err == nil {
			got, popErr := popExternalID(kind)
			if popErr != nil || got != id {
				stackFailures.Add(1)
				r.Status = "unavailable"
			}
		}
		if writeErr := j.Record(r); writeErr != nil {
			stackFailures.Add(1)
			stackDropped.Add(1)
		} else if r.Status == "attached" {
			stackRecorded.Add(1)
		} else {
			stackDropped.Add(1)
		}
	}
}

// stackGeneration joins only an unambiguous, ready native handshake. The
// process nonce is useful for filenames but is not a native capture identity.
func stackGeneration(session string) uint64 {
	if session == "" {
		return 0
	}
	transport := os.Getenv("CUDAPROF_TRANSPORT")
	if transport == "" {
		return 0
	}
	paths, err := filepath.Glob(filepath.Join(transport, fmt.Sprintf("process-%d-*", os.Getpid()), "handshake.json"))
	if err != nil {
		return 0
	}
	var generation uint64
	for _, path := range paths {
		f, err := os.Open(path)
		if err != nil {
			return 0
		}
		var h struct {
			Session    string `json:"session"`
			Process    uint64 `json:"process"`
			Generation uint64 `json:"generation"`
			Ready      bool   `json:"transport_ready"`
		}
		err = json.NewDecoder(io.LimitReader(f, 64<<10)).Decode(&h)
		f.Close()
		if err != nil {
			return 0
		}
		if h.Session != session || h.Process != uint64(os.Getpid()) || !h.Ready || h.Generation == 0 {
			continue
		}
		if generation != 0 {
			return 0
		}
		generation = h.Generation
	}
	return generation
}

// WriteStackCoverage writes final Go producer counts to w after all stack
// scopes have completed. Call FinishStackRecording first to prevent new scopes.
// Counts cover the process lifetime, including failed journal writes.
func WriteStackCoverage(w io.Writer) error {
	return stack.WriteCoverage(w, stack.Record{Process: stackProcess, PID: os.Getpid(), Session: os.Getenv("CUDAPROF_SESSION"), Generation: stackGeneration(os.Getenv("CUDAPROF_SESSION")), Language: "go", Attempted: stackAttempts.Load(), Recorded: stackRecorded.Load(), Dropped: stackDropped.Load(), TruncatedCount: stackTruncated.Load()})
}

// FinishStackRecording disables new stack scopes and waits for entered scopes
// to finish. Environment-enabled recording also writes final coverage and
// closes its owned file. Call it after work completes, never inside Do or
// Submit. Applications using SetStackJournal own and close their own writers.
// Do not change the journal concurrently with FinishStackRecording.
func FinishStackRecording() error {
	stackStateMu.Lock()
	languageStacks.Store(nil)
	stackStateMu.Unlock()
	stackActive.Wait()
	stackStateMu.Lock()
	defer stackStateMu.Unlock()
	if stackFile == nil {
		return nil
	}
	owned := stackFile
	stackFile = nil
	writeErr := WriteStackCoverage(owned)
	closeErr := owned.Close()
	if writeErr != nil {
		return writeErr
	}
	return closeErr
}
