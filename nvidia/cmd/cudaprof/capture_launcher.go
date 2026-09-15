package main

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"errors"
	"flag"
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
	// runComplete means every observed generation qualified and no records
	// were dropped.
	runComplete = "complete"
	// runPartial means the target finalized but at least one generation did
	// not qualify or records were lost. Totals are a floor, never a run total.
	runPartial = "partial"
	// runFailed means the target generation itself never finalized, so no
	// generation in the run can be qualified from this capture.
	runFailed = "failed"
	// runUnknown means no transport was established, so the capture carries no
	// evidence either way. It is not a claim that work was lost.
	runUnknown = "unknown"
)

// captureLoss is the breakdown of records the producer never wrote, as the
// agent's finalized record already reports it. A single dropped_records total
// says a capture is truncated but not by which mechanism, and the mechanisms
// do not share a remedy: CUPTI discarding records before the agent sees them
// is not the agent's pending ring overflowing. Neither is addressed by
// --max-records, which bounds how many journal frames the reader decodes and
// cannot recover a record that was never written.
type captureLoss struct {
	Native          uint64 `json:"native_dropped,omitempty"`
	Queue           uint64 `json:"queue_dropped,omitempty"`
	QueueOversize   uint64 `json:"queue_oversize_dropped,omitempty"`
	QueueContention uint64 `json:"queue_contention_dropped,omitempty"`
	QueueCapacity   uint64 `json:"queue_capacity_dropped,omitempty"`
	Offered         uint64 `json:"offered_records,omitempty"`
	Retained        uint64 `json:"retained_records,omitempty"`
	// BackpressureMaxNanos and DrainWriteMaxNanos are the longest admission
	// wait and the longest batch write the producer saw. A capacity drop is an
	// admission that waited out ADMIT_WAIT_NANOS, and the wait ends only when
	// a batch write returns, so these two are what say whether the loss was
	// journal write latency. The agent records them for that purpose; without
	// them a capacity drop is a count with no cause.
	BackpressureMaxNanos uint64 `json:"backpressure_max_nanos,omitempty"`
	DrainWriteMaxNanos   uint64 `json:"drain_write_max_nanos,omitempty"`
}

// admitWaitNanos mirrors ADMIT_WAIT_NANOS in cupti_agent.c: how long admission
// waits for a freed slot before charging the record to queue_capacity_dropped.
const admitWaitNanos = 100 * 1000 * 1000

// writeLatencyNote reports whether the producer's own timings say the loss was
// journal write latency, which is the question a capacity drop count raises and
// cannot answer.
func (l captureLoss) writeLatencyNote() string {
	if l.BackpressureMaxNanos == 0 && l.DrainWriteMaxNanos == 0 {
		return ""
	}
	note := fmt.Sprintf(" (longest admission wait %s, longest batch write %s",
		time.Duration(l.BackpressureMaxNanos), time.Duration(l.DrainWriteMaxNanos))
	if l.BackpressureMaxNanos >= admitWaitNanos {
		note += fmt.Sprintf(", at the %s admission deadline: the journal writes, not the record count, are what lost these", time.Duration(admitWaitNanos))
	}
	return note + ")"
}

// orNil drops an all-zero breakdown so a capture that lost nothing, and one
// from a producer that reports no breakdown, publish no field rather than a
// row of zeros that reads like a measurement.
func (l captureLoss) orNil() *captureLoss {
	if l == (captureLoss{}) {
		return nil
	}
	c := l
	return &c
}

// dominant names the mechanism that lost the most records, and what to do
// about it. It is the sentence a caller can act on; the counts alone are not.
// The capability block decides the remedy rather than decorating it: the
// lever below only exists for a run that collected API attribution, and
// offering it to a run that already declined it would be wrong advice.
func (l captureLoss) dominant(capability json.RawMessage) string {
	switch {
	case l == captureLoss{}:
		return ""
	case l.QueueCapacity >= l.Native && l.QueueCapacity >= l.Queue:
		return fmt.Sprintf("%d lost to the agent's pending ring filling up: the producer outran the drain%s%s", l.QueueCapacity, l.writeLatencyNote(), volumeRemedy(capability))
	case l.Native >= l.Queue:
		return fmt.Sprintf("%d dropped by CUPTI before the agent saw them: the device outran the activity buffers%s", l.Native, volumeRemedy(capability))
	default:
		return fmt.Sprintf("%d refused by the agent queue (%d oversize, %d contention, %d capacity)", l.Queue, l.QueueOversize, l.QueueContention, l.QueueCapacity)
	}
}

// volumeRemedy names the one lever that reduces journal volume by an order of
// magnitude. Runtime and driver API attribution dominates the record stream --
// the agent's note at cupti_agent.c:880 puts it at 92% of the measured
// journals, and ds4 captures put it at 87% -- and a kernel timeline does not
// need it. Nothing outside the C source says so, which leaves a caller staring
// at a drop count with no reachable action.
//
// It stops short of promising fewer drops, because that does not follow and
// has been measured not to hold. A capacity drop is admission finding the ring
// full, and the wait ends only when the drain publishes a freed tail after its
// batch write returns, so the deadline at cupti_agent.c:50 is really a bound on
// journal write latency: "a write slower than this deadline is what a capacity
// drop is made of". Total volume does not set that latency. Six measured runs
// agree -- cutting volume eightfold left retention anywhere from 1% to 77%,
// a spread wider than the difference between the configurations.
func volumeRemedy(capability json.RawMessage) string {
	if !apiRecordsEnabled(capability) {
		return ", and this run already collected no API attribution, so the remaining levers are a shorter workload or fewer captured generations"
	}
	return ", so re-run with CUDAPROF_API_ACTIVITY=0 for a journal roughly an order of magnitude smaller (API attribution is ~87-92% of records and a kernel timeline does not need it); that buys smaller inputs, not fewer drops, which on this path track journal write latency rather than record count"
}

// apiRecordsEnabled reports whether the agent collected runtime and driver API
// records. A producer that reports no capability block is treated as having
// collected them, which is the agent's default when the variable is unset.
func apiRecordsEnabled(capability json.RawMessage) bool {
	if len(capability) == 0 {
		return true
	}
	var c struct {
		APIRecords *bool `json:"api_records"`
	}
	if err := json.Unmarshal(capability, &c); err != nil || c.APIRecords == nil {
		return true
	}
	return *c.APIRecords
}

// retainedFraction is the share of offered records that reached the journal.
// It is the number that says how truncated a capture is; a raw drop count
// beside an unknown total does not.
func (l captureLoss) retainedFraction() (float64, bool) {
	if l.Offered == 0 {
		return 0, false
	}
	return float64(l.Retained) / float64(l.Offered) * 100, true
}

// captureOptions is the policy for one capture invocation. command contains
// the target argv after the -- separator and is never interpreted by a shell.
type captureOptions struct {
	Trace, Deep, Exhaustive bool
	Collect, Exclude        []string
	Replay                  string
	FollowChildren          bool
	RequireComplete         bool
	NoInject                bool
	JSON                    bool
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
	MaxJournalRecords     int64
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
	// RecordLoss breaks down what the producer never wrote, when the agent
	// reported it. It is nil when nothing was lost or when the capture came
	// from a producer that does not publish the breakdown.
	RecordLoss *captureLoss `json:"record_loss,omitempty"`
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

// captureActivity represents one observed GPU kernel execution or memory copy.
type captureActivity = cudacapture.ActivityRecord

// ActivityRecord is exported for callers inspecting captured activities.
type ActivityRecord = cudacapture.ActivityRecord

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
		case "-h", "-help", "--help":
			return o, flag.ErrHelp
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
		case "--max-records", "-max-records":
			v, next, err := captureValue(args, i, value, hasValue)
			if err != nil {
				return o, err
			}
			i = next
			n, err := strconv.ParseInt(v, 10, 64)
			if err != nil || n <= 0 {
				return o, fmt.Errorf("invalid %s %q", name, v)
			}
			o.MaxJournalRecords = n
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
		case "--no-inject":
			o.NoInject = true
		case "--json":
			o.JSON = true
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
	if o.Inject == "" && !o.NoInject {
		if env := os.Getenv("CUDAPROF_INJECT"); env != "" {
			o.Inject = env
		} else if env := os.Getenv("CUDAPROF_AGENT"); env != "" {
			o.Inject = env
		} else if exe, err := os.Executable(); err == nil {
			adj := filepath.Join(filepath.Dir(exe), "cupti_agent.so")
			if fi, err := os.Stat(adj); err == nil && !fi.IsDir() {
				o.Inject = adj
			}
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
	if o.Inject == "" && !o.NoInject {
		result.CaptureError = "no injection library configured (specify --inject /path/to/agent.so, CUDAPROF_INJECT, or pass --no-inject to run uninstrumented)"
		result.RunCompleteness = runUnknown
		return result, nil
	}
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
	result = observeCaptureSession(result, dir, session, observeOptions{injectionRequested: o.Inject != "", noInject: o.NoInject, recoverUnfinalized: o.RecoverUnfinalized, maxJournalRecords: o.MaxJournalRecords})
	if o.RequireComplete && result.RunCompleteness != runComplete {
		result.CaptureSuccess = false
		if result.CaptureError == "" {
			if result.DroppedRecords > 0 {
				result.CaptureError = fmt.Sprintf("dropped %d records", result.DroppedRecords)
			} else if result.UnqualifiedGenerations > 0 {
				result.CaptureError = fmt.Sprintf("%d unqualified worker generation(s)", result.UnqualifiedGenerations)
			} else {
				result.CaptureError = fmt.Sprintf("run completeness is %s", result.RunCompleteness)
			}
		}
	}
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
