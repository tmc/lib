package main

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"syscall"
	"time"

	"github.com/tmc/lib/nvidia/x/cudacapture"
)

const (
	captureProtocolVersion    = 1
	captureStatePlanned       = "planned"
	captureStateSpawned       = "spawned"
	captureStateFinalized     = "finalized"
	captureOutcomeNoCUDA      = "no-cuda-unverified"
	captureOutcomeObserved    = "no_cuda_observed"
	captureOutcomeCollected   = "collected"
	captureOutcomeUnsupported = "unsupported"
	captureOutcomeInjection   = "injection-failed"
	captureOutcomeTransport   = "transport-failed"
	// captureOutcomePartialRecovered reports totals that include records read
	// from generations that never wrote a finalized record. Such a generation
	// decoded completely and passed every identity and ordering check, but its
	// own outcome is unknown, so the capture is explicitly partial and never
	// successful. See observeOptions.recoverUnfinalized.
	captureOutcomePartialRecovered = "partial-recovered"
)

// Whole-run completeness. CaptureSuccess qualifies the target generation only:
// a target can produce a complete profile while a worker generation fails to
// inject, to finalize or to admit its records. These values report the run as a
// whole so target success can never imply that every generation qualified.
//
// A generation qualifies only when it finalized and reported no failure. Zero
// retained records is not a fault: a finalized generation that ran no GPU work
// is complete, and nothing here infers lost or unobserved work from a count.
const (
	// runComplete means every observed generation qualified.
	runComplete = "complete"
	// runPartial means the target finalized but at least one generation did
	// not qualify. Totals are a floor, never a run total.
	runPartial = "partial"
	// runFailed means the target generation itself never finalized, so no
	// generation in the run can be qualified from this capture.
	runFailed = "failed"
	// runUnknown means no transport was established, so the capture carries no
	// evidence either way. It is not a claim that work was lost.
	runUnknown = "unknown"
)

// captureOptions is the policy for one capture invocation. command contains
// the target argv after the -- separator and is never interpreted by a shell.
type captureOptions struct {
	Trace, Deep, Exhaustive bool
	Collect, Exclude        []string
	Replay                  string
	FollowChildren          bool
	RequireComplete         bool
	// CaptureFileBytes is the per-file budget for artifacts attached to the
	// written capture. Zero means the cudacapture package default. It applies
	// only to the capture this process is producing from its own scratch
	// directory; it never relaxes what a reader will decode.
	CaptureFileBytes int64
	// CaptureTotalBytes is the total budget for raw attachments plus generated
	// content. Zero means the cudacapture package default. Raising it produces
	// an artifact that default readers reject, which is why the read side has
	// a matching explicit option rather than a raised default.
	CaptureTotalBytes int64
	// RecoverUnfinalized admits records from readable worker generations that
	// carry no finalized record, as an explicitly labelled partial. It never
	// makes a capture successful and never affects a finalized generation.
	RecoverUnfinalized    bool
	Timeout, DrainTimeout time.Duration
	Output                string
	Inject                string
	Dir                   string
	Env                   []string
	Stdin                 io.Reader
	Stdout, Stderr        io.Writer
	command               []string
}

// captureResult keeps target termination independent from capture status.
// A target may fail while the capture itself is complete, or vice versa.
type captureResult struct {
	Protocol           int             `json:"protocol"`
	Session            string          `json:"session"`
	State              string          `json:"state"`
	Outcome            string          `json:"outcome"`
	CaptureSuccess     bool            `json:"capture_success"`
	TargetPID          int             `json:"target_pid"`
	TargetExitCode     int             `json:"target_exit_code"`
	TargetSignal       string          `json:"target_signal,omitempty"`
	Cancelled          bool            `json:"cancelled,omitempty"`
	TransportReady     bool            `json:"transport_ready"`
	BootstrapSeen      bool            `json:"bootstrap_seen"`
	CollectorFinalized bool            `json:"collector_finalized"`
	ActivityRecords    uint64          `json:"activity_records"`
	ExternalRecords    uint64          `json:"external_records,omitempty"`
	APIRecords         []captureAPI    `json:"api_records,omitempty"`
	WorkerGenerations  []captureWorker `json:"worker_generations,omitempty"`
	// RunCompleteness reports whole-run qualification independently of
	// CaptureSuccess, and UnqualifiedGenerations counts the generations that
	// cost it. Per-generation detail stays in WorkerGenerations.
	RunCompleteness        string `json:"run_completeness"`
	UnqualifiedGenerations uint64 `json:"unqualified_generations,omitempty"`
	// OmittedFiles records artifacts left out of the written capture because
	// they exceeded the per-file budget. Their exact size and digest are kept
	// so an omission is always visible and the original stays identifiable.
	OmittedFiles []captureOmittedFile `json:"omitted_files,omitempty"`
	// RecoveredGenerations and RecoveredActivityRecords count only what came
	// from unfinalized generations, so a reader can subtract the recovered
	// contribution from the totals above. They are zero unless recovery was
	// explicitly requested and at least one generation was recovered.
	RecoveredGenerations     uint64            `json:"recovered_generations,omitempty"`
	RecoveredActivityRecords uint64            `json:"recovered_activity_records,omitempty"`
	RecoveredDroppedRecords  uint64            `json:"recovered_dropped_records,omitempty"`
	DroppedRecords           uint64            `json:"dropped_records"`
	ActivityNanos            uint64            `json:"activity_nanos"`
	CopyBytes                uint64            `json:"copy_bytes"`
	Activities               []captureActivity `json:"activities,omitempty"`
	Capability               json.RawMessage   `json:"capability,omitempty"`
	CaptureError             string            `json:"capture_error,omitempty"`
	ScratchDir               string            `json:"scratch_dir,omitempty"`
}

type captureActivity struct {
	Process            uint64 `json:"process,omitempty"`
	Subtype            string `json:"subtype"`
	Name               string `json:"name,omitempty"`
	Valid              bool   `json:"valid"`
	Start              uint64 `json:"start"`
	End                uint64 `json:"end"`
	Bytes              uint64 `json:"bytes,omitempty"`
	Queued             uint64 `json:"queued,omitempty"`
	Submitted          uint64 `json:"submitted,omitempty"`
	CopyKind           uint8  `json:"copy_kind,omitempty"`
	Device             uint32 `json:"device"`
	Context            uint32 `json:"context,omitempty"`
	Stream             uint32 `json:"stream,omitempty"`
	Correlation        uint32 `json:"correlation,omitempty"`
	RuntimeCorrelation uint32 `json:"runtime_correlation,omitempty"`
}

// captureAPI retains driver/runtime launch context separately from GPU
// activity totals. Its timestamps describe the host API call, not GPU work.
type captureAPI struct {
	Domain      string `json:"domain"`
	CallbackID  uint32 `json:"callback_id"`
	Start       uint64 `json:"start"`
	End         uint64 `json:"end"`
	ProcessID   uint32 `json:"process_id"`
	ThreadID    uint32 `json:"thread_id"`
	Correlation uint32 `json:"correlation"`
	ReturnValue uint32 `json:"return_value"`
}

// captureArtifactBudget bounds the artifact a capture writes. It is a producer
// policy, applied before anything is embedded, not a limit discovered at the
// end: an attachment that does not fit is omitted with its exact size and
// digest so the capture is still written. A zero field means the corresponding
// cudacapture package default.
type captureArtifactBudget struct {
	FileBytes  int64
	TotalBytes int64
}

// effective substitutes the package default for every unset field.
func (b captureArtifactBudget) effective() captureArtifactBudget {
	if b.FileBytes <= 0 {
		b.FileBytes = cudacapture.MaxCaptureFileBytes
	}
	if b.TotalBytes <= 0 {
		b.TotalBytes = cudacapture.MaxCaptureTotalBytes
	}
	return b
}

// captureOmittedFile identifies one artifact that was not attached. The
// digest is of the complete file on disk, so a separately retained copy can be
// matched to the capture that omitted it.
type captureOmittedFile struct {
	Name   string `json:"name"`
	Bytes  int64  `json:"bytes"`
	SHA256 string `json:"sha256"`
	Reason string `json:"reason"`
}

// captureWorker records child process generations without merging them into
// the target's identity-qualified profile.
type captureWorker struct {
	Process         uint64 `json:"process"`
	Generation      uint64 `json:"generation"`
	Records         uint64 `json:"records"`
	ActivityRecords uint64 `json:"activity_records"`
	Finalized       bool   `json:"finalized"`
	Failure         string `json:"failure,omitempty"`
	Disposition     string `json:"disposition"`
}

// parseCaptureArgs parses only the launcher options before --. This preserves
// every target argument, including strings beginning with a dash.
func parseCaptureArgs(args []string) (captureOptions, error) {
	o := captureOptions{Trace: true, Replay: "off", Timeout: 0, DrainTimeout: 2 * time.Second}
	var i int
	for i < len(args) {
		arg := args[i]
		if arg == "--" {
			i++
			break
		}
		if !strings.HasPrefix(arg, "-") {
			return o, fmt.Errorf("capture option %q must appear before --", arg)
		}
		name, value, hasValue := strings.Cut(arg, "=")
		switch name {
		case "--trace":
			o.Trace = true
		case "--deep":
			o.Deep, o.Trace = true, true
		case "--exhaustive":
			o.Exhaustive, o.Deep, o.Trace = true, true, true
		case "--follow-children":
			o.FollowChildren = true
		case "--require-complete":
			o.RequireComplete = true
		case "--recover-unfinalized":
			o.RecoverUnfinalized = true
		case "--capture-total-bytes":
			v, next, err := captureValue(args, i, value, hasValue)
			if err != nil {
				return o, err
			}
			i = next
			n, err := strconv.ParseInt(v, 10, 64)
			if err != nil || n <= 0 {
				return o, fmt.Errorf("invalid %s %q", name, v)
			}
			o.CaptureTotalBytes = n
		case "--capture-file-bytes":
			v, next, err := captureValue(args, i, value, hasValue)
			if err != nil {
				return o, err
			}
			i = next
			n, err := strconv.ParseInt(v, 10, 64)
			if err != nil || n <= 0 {
				return o, fmt.Errorf("invalid %s %q", name, v)
			}
			o.CaptureFileBytes = n
		case "--collect":
			v, next, err := captureValue(args, i, value, hasValue)
			if err != nil {
				return o, err
			}
			i = next
			o.Collect = append(o.Collect, v)
		case "--exclude":
			v, next, err := captureValue(args, i, value, hasValue)
			if err != nil {
				return o, err
			}
			i = next
			o.Exclude = append(o.Exclude, v)
		case "--replay":
			v, next, err := captureValue(args, i, value, hasValue)
			if err != nil {
				return o, err
			}
			i = next
			if v != "off" && v != "application" && v != "kernel" {
				return o, fmt.Errorf("invalid replay mode %q", v)
			}
			o.Replay = v
		case "--timeout", "--drain-timeout":
			v, next, err := captureValue(args, i, value, hasValue)
			if err != nil {
				return o, err
			}
			i = next
			d, err := time.ParseDuration(v)
			if err != nil || d <= 0 {
				return o, fmt.Errorf("invalid %s %q", name, v)
			}
			if name == "--timeout" {
				o.Timeout = d
			} else {
				o.DrainTimeout = d
			}
		case "--output", "-o":
			v, next, err := captureValue(args, i, value, hasValue)
			if err != nil {
				return o, err
			}
			i = next
			o.Output = v
		case "--inject":
			v, next, err := captureValue(args, i, value, hasValue)
			if err != nil {
				return o, err
			}
			i = next
			o.Inject = v
		default:
			return o, fmt.Errorf("unknown capture option %q", name)
		}
		i++
	}
	if i == 0 || i >= len(args) {
		return o, errors.New("capture requires -- followed by a target command")
	}
	o.command = append([]string(nil), args[i:]...)
	if o.Output != "" {
		if _, err := os.Stat(o.Output); err == nil {
			return o, fmt.Errorf("capture output already exists: %s", o.Output)
		} else if !errors.Is(err, os.ErrNotExist) {
			return o, fmt.Errorf("check capture output: %w", err)
		}
	}
	if o.Inject != "" {
		if fi, err := os.Stat(o.Inject); err != nil {
			return o, fmt.Errorf("check injection library: %w", err)
		} else if fi.IsDir() {
			return o, fmt.Errorf("injection library is a directory: %s", o.Inject)
		}
	}
	return o, nil
}

func captureValue(args []string, i int, value string, hasValue bool) (string, int, error) {
	if !hasValue {
		i++
		if i >= len(args) {
			return "", i, errors.New("capture option requires a value")
		}
		value = args[i]
	}
	if value == "" {
		return "", i, errors.New("capture option requires a non-empty value")
	}
	return value, i, nil
}

func newCaptureSession() (string, string, error) {
	home, err := os.UserHomeDir()
	if err != nil {
		return "", "", fmt.Errorf("find home directory: %w", err)
	}
	root := filepath.Join(home, "tmp", "cudaprof-capture")
	if err := os.MkdirAll(root, 0700); err != nil {
		return "", "", fmt.Errorf("create capture scratch: %w", err)
	}
	dir, err := os.MkdirTemp(root, "session-")
	if err != nil {
		return "", "", fmt.Errorf("create capture session: %w", err)
	}
	var nonce [16]byte
	if _, err := rand.Read(nonce[:]); err != nil {
		os.RemoveAll(dir)
		return "", "", fmt.Errorf("create capture nonce: %w", err)
	}
	return hex.EncodeToString(nonce[:]), dir, nil
}

func captureEnvironment(o captureOptions, session, dir string) []string {
	env := append([]string(nil), o.Env...)
	if env == nil {
		env = os.Environ()
	}
	env = setCaptureEnv(env, "CUDAPROF_PROTOCOL", strconv.Itoa(captureProtocolVersion))
	env = setCaptureEnv(env, "CUDAPROF_SESSION", session)
	env = setCaptureEnv(env, "CUDAPROF_TRANSPORT", dir)
	if o.Inject != "" {
		key := "LD_PRELOAD"
		if runtime.GOOS == "darwin" {
			key = "DYLD_INSERT_LIBRARIES"
		}
		old := lookupEnv(env, key)
		if old != "" {
			o.Inject += ":" + old
		}
		env = setCaptureEnv(env, key, o.Inject)
	}
	return env
}

func setCaptureEnv(env []string, key, value string) []string {
	prefix := key + "="
	out := env[:0]
	for _, item := range env {
		if !strings.HasPrefix(item, prefix) {
			out = append(out, item)
		}
	}
	return append(out, prefix+value)
}

func lookupEnv(env []string, key string) string {
	prefix := key + "="
	for _, item := range env {
		if strings.HasPrefix(item, prefix) {
			return strings.TrimPrefix(item, prefix)
		}
	}
	return ""
}

// runCapture launches exactly one owned target. It reports target failures in
// captureResult; errors are reserved for launcher/preflight failures.
func runCapture(ctx context.Context, o captureOptions) (captureResult, error) {
	if len(o.command) == 0 {
		return captureResult{}, errors.New("capture target is empty")
	}
	if o.Replay != "" && o.Replay != "off" {
		return captureResult{}, fmt.Errorf("replay mode %q is not implemented", o.Replay)
	}
	if o.Replay == "" {
		o.Replay = "off"
	}
	if o.Deep || o.Exhaustive || len(o.Collect) > 0 || len(o.Exclude) > 0 || o.FollowChildren {
		return captureResult{}, errors.New("requested collector policy is not implemented")
	}
	if o.RequireComplete {
		return captureResult{}, errors.New("capture cannot require complete data before a collector is configured")
	}
	if o.DrainTimeout <= 0 {
		o.DrainTimeout = 2 * time.Second
	}
	session, dir, err := newCaptureSession()
	if err != nil {
		return captureResult{}, err
	}
	outcome := captureOutcomeNoCUDA
	if o.Inject != "" {
		outcome = captureOutcomeInjection
	}
	result := captureResult{Protocol: captureProtocolVersion, Session: session, State: captureStatePlanned, Outcome: outcome, ScratchDir: dir}
	cmd := exec.Command(o.command[0], o.command[1:]...)
	cmd.Dir, cmd.Env = o.Dir, captureEnvironment(o, session, dir)
	cmd.Stdin, cmd.Stdout, cmd.Stderr = o.Stdin, o.Stdout, o.Stderr
	if cmd.Stdin == nil {
		cmd.Stdin = os.Stdin
	}
	if cmd.Stdout == nil {
		cmd.Stdout = os.Stdout
	}
	if cmd.Stderr == nil {
		cmd.Stderr = os.Stderr
	}
	if err := configureCaptureProcess(cmd); err != nil {
		return result, err
	}
	if err := cmd.Start(); err != nil {
		return result, fmt.Errorf("start capture target: %w", err)
	}
	result.State, result.TargetPID = captureStateSpawned, cmd.Process.Pid
	done := make(chan error, 1)
	go func() { done <- cmd.Wait() }()
	runctx := ctx
	cancel := func() {}
	if o.Timeout > 0 {
		runctx, cancel = context.WithTimeout(ctx, o.Timeout)
	}
	defer cancel()
	var waitErr error
	select {
	case waitErr = <-done:
	case <-runctx.Done():
		result.Cancelled = true
		signalCaptureProcess(cmd.Process, syscall.SIGTERM)
		timer := time.NewTimer(o.DrainTimeout)
		select {
		case waitErr = <-done:
			if !timer.Stop() {
				<-timer.C
			}
		case <-timer.C:
			killCaptureProcess(cmd.Process)
			waitErr = <-done
		}
	}
	result.State = captureStateFinalized
	result = observeCaptureSession(result, dir, session, observeOptions{injectionRequested: o.Inject != "", recoverUnfinalized: o.RecoverUnfinalized})
	if waitErr == nil {
		result.TargetExitCode = 0
		if result.Cancelled {
			return result, runctx.Err()
		}
		return result, nil
	}
	var ee *exec.ExitError
	if errors.As(waitErr, &ee) {
		result.TargetExitCode = -1
		if status, ok := ee.Sys().(syscall.WaitStatus); ok {
			if status.Signaled() {
				result.TargetSignal = status.Signal().String()
			} else {
				result.TargetExitCode = status.ExitStatus()
			}
		}
		if result.Cancelled {
			return result, runctx.Err()
		}
		return result, nil
	}
	return result, fmt.Errorf("wait for capture target: %w", waitErr)
}
