package main

import (
	"context"
	"crypto/rand"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"runtime"
	"slices"
	"strconv"
	"strings"
	"syscall"
	"time"

	"github.com/tmc/lib/nvidia/x/cudacapture"
	"github.com/tmc/lib/nvidia/x/cudaprof"
	"github.com/tmc/lib/nvidia/x/cudaprof/isolation"
	"github.com/tmc/lib/nvidia/x/internal/capturefiles"
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
	// captureOutcomeNotFinalized reports a target whose journal decoded
	// completely and whose producer never wrote a finalized record. It is
	// split out of transport-failed because the transport is exactly what did
	// work: the handshake completed and records arrived. Reporting it as a
	// transport failure sent at least one reader hunting the channel for a
	// fault that was never there, with transport_ready:false agreeing.
	captureOutcomeNotFinalized = "not-finalized"
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
// retained records is not a fault in a generation: a finalized generation that
// ran no GPU work qualifies, and nothing here infers lost work from a count.
// A run that retained no records at all is not complete either, though: it is
// empty, and "complete" on zero records would certify a capture that observed
// nothing.
const (
	// runComplete means every observed generation qualified, no records
	// were dropped, and at least one activity record was retained.
	runComplete = "complete"
	// runEmpty means every observed generation qualified and nothing was
	// dropped, but the run retained no activity records. It is not a claim
	// that work was lost, nor that the run did no device work.
	runEmpty = "empty"
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
	byKind          *captureKindLosses
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

// maxCrashTrace mirrors CRASH_RING in cupti_agent.c: the most API callbacks
// the agent keeps for a crash trace.
const maxCrashTrace = 4096

// maxHostSampleHz mirrors HOST_SAMPLE_MAX_HZ in cupti_agent.c.
const maxHostSampleHz = 1000

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
// at a drop count with no reachable action. The lever is named as the flag
// that pulls it rather than as CUDAPROF_API_ACTIVITY=0, the variable the flag
// exports: advice that asks a caller to edit the target's environment by hand
// is advice about the implementation.
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
	return ", so re-run with --api-activity=false for a journal roughly an order of magnitude smaller (API attribution is ~87-92% of records and a kernel timeline does not need it); that buys smaller inputs, not fewer drops, which on this path track journal write latency rather than record count"
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
	CapturePlanFile        string
	CapturePlanCooperative bool

	// Iterations, when set, runs the iterations gate on the written
	// capture, and the capture fails when the gate does.
	Iterations *iterationsOptions

	Trace     bool
	Stacks    bool
	stacksSet bool
	Replay    string
	Exclude   []string
	// FollowChildren keeps the generations of processes the target spawned in
	// this capture's totals and projection. It does not decide whether a child
	// is instrumented: the launcher sets only LD_PRELOAD (DYLD_INSERT_LIBRARIES
	// on darwin), which every exec'd process inherits, so a child loads the
	// agent and writes its own generation whatever this says. False leaves
	// those records out of the totals and the pprof while the generations stay
	// listed in worker_generations and their raw journals stay attached, so
	// nothing is hidden by not following them.
	FollowChildren bool
	// APIActivity collects runtime and driver API attribution alongside device
	// activity. It is on by default because the agent's default is on; setting
	// it false exports CUDAPROF_API_ACTIVITY=0, which is the one lever that
	// cuts journal volume by an order of magnitude. apiActivitySet records
	// whether the caller said anything, so an untouched flag leaves the
	// target's own environment alone, as the collection policy below does.
	APIActivity    bool
	apiActivitySet bool
	// HostSampleHz samples thread run states and CPU tick counters from the
	// native agent. It does not collect host stack traces. hostSampleSet lets
	// zero explicitly disable sampling inherited from the caller's environment.
	HostSampleHz  int
	hostSampleSet bool
	// Memsets collects memset activity. It is off by default: with the kind
	// enabled, cuGraphLaunch has faulted inside libcuda for a CUDA-graph MLX
	// workload on GB10. The launcher always exports CUDAPROF_MEMSETS, so a
	// value inherited from the caller's environment cannot turn it on.
	Memsets       bool
	UnifiedMemory bool
	MemoryPolicy  bool
	// PCSampling turns on the agent's program-counter sampler, which is what
	// turns a capture into per-line counts. It is off by default because it is
	// not free: measured on GB10 it costs 25-27 microseconds of host drain per
	// launch, which a kernel timeline should not pay for work it never reads.
	PCSampling bool
	// PCSamplingPeriod is the agent's sampling period exponent, 5 to 31. Zero
	// leaves the agent's own default.
	PCSamplingPeriod int
	// CrashTrace is how many of the target's last CUDA API callbacks the
	// agent writes to standard error if the target dies of a fault, exported
	// as CUDAPROF_CRASH_TRACE. Zero leaves the agent's default, off.
	CrashTrace int
	// GEMMs records each cuBLASLt matmul the target makes -- shape, types,
	// beta and algorithm -- and the kernels it launched. Off by default:
	// the library formats a log line per call.
	GEMMs            bool
	RequireComplete  bool
	IsolationReceipt string
	ObserveIsolation bool
	RequireIsolation bool
	NoInject         bool
	JSON             bool
	// LaunchManifest enables the bounded, metadata-only launch manifest in the
	// capture status. It never enables parameter decoding by itself.
	LaunchManifest bool
	// CaptureFileBytes is the per-file budget for artifacts attached to the
	// written capture. Zero means captureFileBytes. The capture records the
	// budget it was written under, and cudaprof's readers admit it from
	// there; a reader that does not trust the capture keeps the cudacapture
	// defaults.
	CaptureFileBytes int64
	// CaptureTotalBytes is the total budget for raw attachments plus generated
	// content. Zero means captureTotalBytes. It is recorded as
	// CaptureFileBytes is.
	CaptureTotalBytes int64
	// AllowPartial accepts a capture that omitted an attachment or holds a
	// journal over MaxJournalRecords. Without it such a capture is still
	// written, but the command exits nonzero naming the flags to rerun with.
	AllowPartial bool
	// RecoverUnfinalized admits records from readable generations that carry no
	// finalized record, as an explicitly labelled partial. It never makes a
	// capture successful and never affects a finalized generation.
	RecoverUnfinalized bool
	// CaptureRange, Finalize and FlushInterval are the agent's collection
	// policy. They are launcher options only in the sense that the launcher is
	// what puts them in the target's environment; every one of them is read and
	// enforced inside the injected agent, and a value this launcher accepts is
	// still inert against an agent that does not know it. The names are the
	// contract: CUDAPROF_CAPTURE_RANGE, CUDAPROF_FINALIZE and
	// CUDAPROF_FLUSH_INTERVAL_MS.
	CaptureRange string
	Finalize     string
	// GraphTrace is how the agent traces CUDA graph replays, exported as
	// CUDAPROF_GRAPH_TRACE. Empty leaves the agent's default, node mode.
	GraphTrace        graphTraceMode
	FlushInterval     time.Duration
	MaxJournalRecords int64
	// StampEnv names environment variables whose values are recorded in the
	// capture, by exact name or by a trailing-* prefix. Nothing is disclosed
	// unless it is named here: an environment carries credentials, and a
	// profile is an artifact people attach to bug reports. The digest below
	// discriminates arms without this.
	StampEnv              []string
	Timeout, DrainTimeout time.Duration
	Output                string
	Inject                string
	// InjectMode is how the agent is loaded: injectModePreload, the
	// default when empty, or injectModeCUDA.
	InjectMode     string
	Dir            string
	Env            []string
	Stdin          io.Reader
	Stdout, Stderr io.Writer
	// Log receives cudaprof's own warnings, which are not the target's
	// output. Nil means os.Stderr.
	Log     io.Writer
	command []string

	// openGPU opens the GPUs whose clocks are read at capture-range edges.
	// Nil reads them through NVML.
	openGPU gpuOpener

	// ReplaceEmptyOutput records that Output exists and holds no activity
	// records, so this run may take the name. The removal is deliberately not
	// done at parse time: parsing arguments must not delete a file, and the
	// command does it once, before the target runs.
	ReplaceEmptyOutput bool
}

// The values --capture-range and --finalize accept. They are the agent's
// vocabulary, not this command's: each is compared byte for byte against the
// corresponding variable in cupti_agent.c, so a spelling this launcher invents
// would turn the feature off while reading as if it turned it on.
//
// The two are independent requests even though both watch the profiler API.
// Gating asks what to collect; finalizing on a stop asks when to publish it. A
// target that calls only cuProfilerStop, to hand the agent the flush its exit
// path will never run, wants the second and not the first, so neither value
// implies the other and neither is refused without the other.
// The values --inject-mode accepts. Preload puts the agent in LD_PRELOAD, so
// every process the target starts loads it; cuda-injection names it in
// CUDA_INJECTION64_PATH, which the driver loads during cuInit, so only a
// process that initializes CUDA does. The agent is the same library either
// way.
const (
	injectModePreload = "preload"
	injectModeCUDA    = "cuda-injection"
)

const (
	captureRangeNone        = "none"
	captureRangeProfilerAPI = "cudaProfilerApi"
	finalizeExit            = "exit"
	finalizeProfilerStop    = "profiler-stop"
	finalizeExitGroup       = "exit-group"
)

// captureResult records target termination beside capture status. A capture
// can fail while its target succeeds, but not the other way round: a complete
// capture of a target that died or exited nonzero is reported as
// captureOutcomeTargetFailed, with capture_success false, because its records
// stop where the target did.
type captureResult struct {
	StacksRequested  bool              `json:"stacks_requested,omitempty"`
	Isolation        *isolation.Result `json:"gpu_isolation,omitempty"`
	IsolationBinding isolation.Binding `json:"isolation_binding"`
	isolationRaw     []byte
	planBudget       *cudaprof.CaptureBudget

	CapturePlanManifest json.RawMessage `json:"capture_plan_manifest,omitempty"`
	CapturePlanError    string          `json:"capture_plan_error,omitempty"`

	Protocol           int    `json:"protocol"`
	Session            string `json:"session"`
	State              string `json:"state"`
	Outcome            string `json:"outcome"`
	CaptureSuccess     bool   `json:"capture_success"`
	TargetPID          int    `json:"target_pid"`
	TargetExitCode     int    `json:"target_exit_code"`
	TargetSignal       string `json:"target_signal,omitempty"`
	Cancelled          bool   `json:"cancelled,omitempty"`
	TransportReady     bool   `json:"transport_ready"`
	BootstrapSeen      bool   `json:"bootstrap_seen"`
	CollectorFinalized bool   `json:"collector_finalized"`
	ActivityRecords    uint64 `json:"activity_records"`
	ExternalRecords    uint64 `json:"external_records,omitempty"`
	// HostSampleRecords counts the producer's per-thread host run-state
	// records. It is not device work and is deliberately separate from
	// ActivityRecords: it is what says whether the host was running, blocked
	// on I/O, or asleep during a device gap.
	HostSampleRecords uint64          `json:"host_sample_records,omitempty"`
	APIRecords        []captureAPI    `json:"api_records,omitempty"`
	WorkerGenerations []captureWorker `json:"worker_generations,omitempty"`
	// RecordLoss breaks down what the producer never wrote, when the agent
	// reported it. It is nil when nothing was lost or when the capture came
	// from a producer that does not publish the breakdown.
	RecordLoss *captureLoss `json:"record_loss,omitempty"`
	// RunCompleteness reports whole-run qualification independently of
	// CaptureSuccess, and UnqualifiedGenerations counts the generations that
	// cost it. Per-generation detail stays in WorkerGenerations.
	RunCompleteness        string `json:"run_completeness"`
	UnqualifiedGenerations uint64 `json:"unqualified_generations,omitempty"`
	// ChildrenFollowed says whether the totals above include the generations
	// of processes the target spawned. It carries no omitempty: a reader that
	// cannot see this field cannot tell a whole-run total from a target-only
	// one, and the dangerous value is the false one.
	ChildrenFollowed bool `json:"children_followed"`
	// InjectMode records how the agent was loaded when one was: "preload"
	// or "cuda-injection". Under cuda-injection a process that never
	// initialized CUDA writes no journal, so a target with no generation of
	// its own is expected there and is not a transport failure.
	InjectMode string `json:"inject_mode,omitempty"`
	// ExcludedPatterns, ExcludedRecords and ExcludedNanos report what --exclude
	// removed from the projection. The totals above stay journal totals, so
	// activity_samples == activity_records - excluded_records.
	ExcludedPatterns []string `json:"excluded_patterns,omitempty"`
	ExcludedRecords  uint64   `json:"excluded_records,omitempty"`
	ExcludedNanos    uint64   `json:"excluded_nanos,omitempty"`
	// OmittedFiles records artifacts left out of the written capture because
	// they exceeded the per-file budget. Their exact size and digest are kept
	// so an omission is always visible and the original stays identifiable.
	OmittedFiles []captureOmittedFile `json:"omitted_files,omitempty"`
	// Limits records the bounds the capture was written under and the sizes
	// that decide what a reader needs to decode it.
	Limits *captureLimits `json:"limits,omitempty"`
	// Modules records every device module image the agent saw the driver
	// load, including the ones it declined to store. It is the census the
	// embedded cubins are drawn from: a module missing from the capture is
	// either absent here, meaning the target never loaded it, or present with
	// Stored false, meaning the capture chose not to keep it.
	Modules []captureModule `json:"modules,omitempty"`
	// PCSamples and PCSampling report the agent's program-counter sampler:
	// one entry per distinct sampled PC and one summary of what the sampler
	// itself saw. They are absent unless the capture asked for sampling.
	PCSamples  []capturePCSample  `json:"pc_samples,omitempty"`
	PCSampling *capturePCSampling `json:"pc_sampling,omitempty"`
	// RecoveredGenerations and RecoveredActivityRecords count only what came
	// from unfinalized generations, so a reader can subtract the recovered
	// contribution from the totals above. They are zero unless recovery was
	// explicitly requested and at least one generation was recovered.
	RecoveredGenerations     uint64               `json:"recovered_generations,omitempty"`
	RecoveredActivityRecords uint64               `json:"recovered_activity_records,omitempty"`
	RecoveredDroppedRecords  uint64               `json:"recovered_dropped_records,omitempty"`
	DroppedRecords           uint64               `json:"dropped_records"`
	ActivityNanos            uint64               `json:"activity_nanos"`
	CopyBytes                uint64               `json:"copy_bytes"`
	Activities               []captureActivity    `json:"activities,omitempty"`
	LaunchManifest           json.RawMessage      `json:"launch_manifest,omitempty"`
	PhaseReceipt             *capturePhaseReceipt `json:"phase_receipt,omitempty"`
	// WallClock places the capture's CLOCK_MONOTONIC timestamps, the range
	// edges among them, in wall time. It is nil where the controller has no
	// CLOCK_MONOTONIC.
	WallClock    *captureWallClock  `json:"wall_clock,omitempty"`
	Capability   json.RawMessage    `json:"capability,omitempty"`
	Target       *targetEnvironment `json:"target_environment,omitempty"`
	CaptureError string             `json:"capture_error,omitempty"`
	ScratchDir   string             `json:"scratch_dir,omitempty"`
}

// captureActivity represents one observed GPU kernel execution or memory copy.
type captureActivity = cudacapture.ActivityRecord

// ActivityRecord is exported for callers inspecting captured activities.
type ActivityRecord = cudacapture.ActivityRecord

// captureAPI retains driver/runtime launch context separately from GPU
// activity totals. Its timestamps describe the host API call, not GPU work.
type captureAPI = cudacapture.APIRecord

// captureArtifactBudget bounds the artifact a capture writes. It is a producer
// policy, applied before anything is embedded, not a limit discovered at the
// end: an attachment that does not fit is omitted with its exact size and
// digest so the capture is still written. A zero field means the capture
// default, captureFileBytes or captureTotalBytes.
type captureArtifactBudget struct {
	FileBytes  int64
	TotalBytes int64
}

// effective substitutes the package default for every unset field.
func (b captureArtifactBudget) effective() captureArtifactBudget {
	if b.FileBytes <= 0 {
		b.FileBytes = captureFileBytes
	}
	if b.TotalBytes <= 0 {
		b.TotalBytes = captureTotalBytes
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
	o := captureOptions{Trace: true, APIActivity: true, FollowChildren: true, Replay: "off", Timeout: 0, DrainTimeout: 2 * time.Second}
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
		case "--stacks":
			v, err := captureBool(name, value, hasValue)
			if err != nil {
				return o, err
			}
			o.Stacks, o.stacksSet = v, true
		case "--follow-children", "-follow-children":
			v, err := captureBool(name, value, hasValue)
			if err != nil {
				return o, err
			}
			o.FollowChildren = v
		case "--api-activity", "-api-activity":
			v, err := captureBool(name, value, hasValue)
			if err != nil {
				return o, err
			}
			o.APIActivity, o.apiActivitySet = v, true
		case "--memsets", "-memsets":
			v, err := captureBool(name, value, hasValue)
			if err != nil {
				return o, err
			}
			o.Memsets = v
		case "--memory-policy", "-memory-policy":
			v, err := captureBool(name, value, hasValue)
			if err != nil {
				return o, err
			}
			o.MemoryPolicy = v
		case "--unified-memory", "-unified-memory":
			v, err := captureBool(name, value, hasValue)
			if err != nil {
				return o, err
			}
			o.UnifiedMemory = v
		case "--pcsampling", "-pcsampling":
			v, err := captureBool(name, value, hasValue)
			if err != nil {
				return o, err
			}
			o.PCSampling = v
		case "--host-sample-hz", "-host-sample-hz":
			v, next, err := captureValue(args, i, value, hasValue)
			if err != nil {
				return o, err
			}
			i = next
			n, err := strconv.Atoi(v)
			if err != nil || n < 0 || n > maxHostSampleHz {
				return o, fmt.Errorf("invalid %s %q (0 to %d)", name, v, maxHostSampleHz)
			}
			o.HostSampleHz, o.hostSampleSet = n, true
		case "--gemms", "-gemms":
			v, err := captureBool(name, value, hasValue)
			if err != nil {
				return o, err
			}
			o.GEMMs = v
		case "--pcsampling-period":
			v, next, err := captureValue(args, i, value, hasValue)
			if err != nil {
				return o, err
			}
			i = next
			n, err := strconv.Atoi(v)
			if err != nil || n < 5 || n > 31 {
				return o, fmt.Errorf("invalid %s %q", name, v)
			}
			o.PCSamplingPeriod, o.PCSampling = n, true
		case "--crash-trace":
			v, next, err := captureValue(args, i, value, hasValue)
			if err != nil {
				return o, err
			}
			i = next
			n, err := strconv.Atoi(v)
			if err != nil || n < 1 || n > maxCrashTrace {
				return o, fmt.Errorf("invalid %s %q (1 to %d)", name, v, maxCrashTrace)
			}
			o.CrashTrace = n
		case "--observe-isolation":
			o.ObserveIsolation = true
		case "--require-isolation":
			o.RequireIsolation = true
		case "--isolation-receipt":
			v, next, err := captureValue(args, i, value, hasValue)
			if err != nil {
				return o, err
			}
			i = next
			o.IsolationReceipt = v
		case "--require-complete":
			o.RequireComplete = true
		case "--allow-partial":
			o.AllowPartial = true
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
		case "--capture-range", "-capture-range":
			v, next, err := captureValue(args, i, value, hasValue)
			if err != nil {
				return o, err
			}
			i = next
			if v != captureRangeNone && v != captureRangeProfilerAPI {
				return o, fmt.Errorf("invalid capture range %q (%s or %s)", v, captureRangeNone, captureRangeProfilerAPI)
			}
			o.CaptureRange = v
		case "--graph-trace", "-graph-trace":
			v, next, err := captureValue(args, i, value, hasValue)
			if err != nil {
				return o, err
			}
			i = next
			m, err := parseGraphTraceMode(v)
			if err != nil {
				return o, err
			}
			o.GraphTrace = m
		case "--finalize", "-finalize":
			v, next, err := captureValue(args, i, value, hasValue)
			if err != nil {
				return o, err
			}
			i = next
			if v != finalizeExit && v != finalizeProfilerStop && v != finalizeExitGroup {
				return o, fmt.Errorf("invalid finalize trigger %q (%s, %s or %s)", v, finalizeExit, finalizeProfilerStop, finalizeExitGroup)
			}
			o.Finalize = v
		case "--flush-interval", "-flush-interval":
			v, next, err := captureValue(args, i, value, hasValue)
			if err != nil {
				return o, err
			}
			i = next
			d, err := time.ParseDuration(v)
			if err != nil || d <= 0 {
				return o, fmt.Errorf("invalid %s %q", name, v)
			}
			// The agent takes whole milliseconds, so a sub-millisecond interval
			// would round to zero and read as "no periodic flush" -- the exact
			// state the caller was asking to leave.
			if d < time.Millisecond {
				return o, fmt.Errorf("flush interval %q is below the agent's 1ms resolution", v)
			}
			o.FlushInterval = d
		case "--exclude", "-exclude":
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
		case "--stamp-env":
			v, next, err := captureValue(args, i, value, hasValue)
			if err != nil {
				return o, err
			}
			i = next
			for _, name := range strings.Split(v, ",") {
				if name = strings.TrimSpace(name); name != "" {
					o.StampEnv = append(o.StampEnv, name)
				}
			}
		case "--inject-mode", "-inject-mode":
			v, next, err := captureValue(args, i, value, hasValue)
			if err != nil {
				return o, err
			}
			i = next
			if v != injectModePreload && v != injectModeCUDA {
				return o, fmt.Errorf("invalid inject mode %q (%s or %s)", v, injectModePreload, injectModeCUDA)
			}
			o.InjectMode = v
		case "--no-inject":
			o.NoInject = true
		case "--json":
			o.JSON = true
		case "--capture-plan-cooperative":
			o.CapturePlanCooperative = true
		case "--capture-plan":
			v, next, err := captureValue(args, i, value, hasValue)
			if err != nil {
				return o, err
			}
			i = next
			if v == "" {
				return o, fmt.Errorf("capture plan path is empty")
			}
			o.CapturePlanFile = v
		case "--launch-manifest":
			o.LaunchManifest = true
		case "--iteration-marker", "--skip-iterations", "--expect-markers":
			v, next, err := captureValue(args, i, value, hasValue)
			if err != nil {
				return o, err
			}
			i = next
			if o.Iterations == nil {
				o.Iterations = &iterationsOptions{skip: 1}
			}
			switch name {
			case "--iteration-marker":
				if _, err := regexp.Compile(v); err != nil {
					return o, fmt.Errorf("invalid %s: %w", name, err)
				}
				o.Iterations.marker = v
			case "--skip-iterations":
				o.Iterations.skip, err = strconv.Atoi(v)
				if err != nil || o.Iterations.skip < 0 {
					return o, fmt.Errorf("invalid %s %q", name, v)
				}
			case "--expect-markers":
				o.Iterations.expect, err = strconv.Atoi(v)
				if err != nil || o.Iterations.expect < 1 {
					return o, fmt.Errorf("invalid %s %q", name, v)
				}
			}
		default:
			return o, unknownCaptureOption(name)
		}
		i++
	}
	if i == 0 || i >= len(args) {
		return o, errors.New("capture requires -- followed by a target command")
	}
	if o.CapturePlanCooperative && o.CapturePlanFile == "" {
		return o, errors.New("--capture-plan-cooperative requires --capture-plan")
	}
	if o.Iterations != nil {
		switch {
		case o.Iterations.marker == "":
			return o, errors.New("--skip-iterations and --expect-markers need --iteration-marker")
		case o.Output == "":
			return o, errors.New("--iteration-marker needs -o: the gate reads the written capture")
		}
	}
	o.command = append([]string(nil), args[i:]...)
	if o.Output != "" {
		if _, err := os.Stat(o.Output); err == nil {
			records, holds := outputHoldsWork(o.Output)
			if holds {
				return o, fmt.Errorf("capture output already exists and holds %s: %s\n"+
					"\tOnly a capture that observed nothing is replaced automatically. Move this one aside or choose another -o path.", records, o.Output)
			}
			o.ReplaceEmptyOutput = true
		} else if !errors.Is(err, os.ErrNotExist) {
			return o, fmt.Errorf("check capture output: %w", err)
		}
	}
	if o.Inject == "" && !o.NoInject {
		if env := os.Getenv("CUDAPROF_INJECT"); env != "" {
			o.Inject = env
		} else if env := os.Getenv("CUDAPROF_AGENT"); env != "" {
			o.Inject = env
		} else {
			var dir string
			if exe, err := os.Executable(); err == nil {
				dir = filepath.Dir(exe)
			}
			o.Inject = findAgent(dir, os.Getenv)
		}
	}
	if o.Inject != "" {
		if fi, err := os.Stat(o.Inject); err != nil {
			return o, fmt.Errorf("check injection library: %w", err)
		} else if fi.IsDir() {
			return o, fmt.Errorf("injection library is a directory: %s", o.Inject)
		}
	}
	if o.ObserveIsolation && o.IsolationReceipt != "" {
		return o, fmt.Errorf("--observe-isolation and --isolation-receipt select different observers; use one")
	}
	return o, nil
}

// captureRetired maps an option this command once accepted to what replaced
// it. Nothing here is parsed: an option is either implemented or unknown, and
// an accepted-then-refused flag is what this table exists to prevent a return
// to. It is a diagnosis, not a compatibility shim -- the replacement is not
// guessable from the name, and "unknown capture option" alone sends a caller
// hunting for a spelling when what they want is a different knob or a
// different tool.
var captureRetired = map[string]string{
	"--deep":       "runtime and driver API attribution is collected by default, so there is no deeper mode to ask for; --api-activity=false turns it off",
	"--exhaustive": "runtime and driver API attribution is collected by default, so there is no deeper mode to ask for; --api-activity=false turns it off",
	"--collect":    "capture records a CUPTI activity timeline, not hardware counters; counter collection needs the CUPTI Profiling API with replay, which github.com/tmc/lib/nvidia/x/cuptirange does in-process for a Go program (Plan.Capture), and cuptimetrics lists the metrics a chip offers",
}

func unknownCaptureOption(name string) error {
	if note, ok := captureRetired[name]; ok {
		return fmt.Errorf("unknown capture option %q: %s", name, note)
	}
	return fmt.Errorf("unknown capture option %q", name)
}

// captureBool parses a boolean option. Like the flag package, a boolean takes
// a value only in the -x=v form: a separated value cannot be told apart from
// the target command, and an option whose meaning depends on what follows it
// is worse than one spelling fewer.
func captureBool(name, value string, hasValue bool) (bool, error) {
	if !hasValue {
		return true, nil
	}
	v, err := strconv.ParseBool(value)
	if err != nil {
		return false, fmt.Errorf("invalid %s %q", name, value)
	}
	return v, nil
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

// targetEnvironment records what the target was run as.
//
// Two arms of the same program can differ only in an environment variable,
// produce different output, and pass every gate the capture applies -- and
// nothing in the capture said which arm it profiled. That cost a reader a day.
//
// Command and Dir are recorded in full: they are already visible to anyone who
// can read the capture's process list. The environment is not, so only its
// DIGEST is recorded by default. A digest answers the question that was
// actually being asked -- are these two captures the same arm -- without
// disclosing a single value, and it discriminates on variables nobody thought
// to name in advance, which is the case that bites. Values are recorded only
// for variables the operator named with --stamp-env.
type targetEnvironment struct {
	Command []string `json:"command,omitempty"`
	Dir     string   `json:"dir,omitempty"`
	// EnvDigest is over every NAME=VALUE the target received, sorted, so it is
	// stable across the map ordering the environment does not have.
	EnvDigest string `json:"env_digest,omitempty"`
	EnvCount  int    `json:"env_count,omitempty"`
	// Env holds the variables named by --stamp-env. A named variable that was
	// not set is recorded as absent rather than empty: "" is a value a program
	// can branch on differently than unset.
	Env        map[string]string `json:"env,omitempty"`
	EnvUnset   []string          `json:"env_unset,omitempty"`
	StampedFor []string          `json:"env_stamped_for,omitempty"`

	// Executable is the program the command ran, read through a leading
	// env(1), and Agent the injection library it was given. Both are
	// identified before the target starts.
	Executable *executableIdentity `json:"executable,omitempty"`
	Agent      *executableIdentity `json:"agent,omitempty"`
}

// stampTargetEnvironment records the environment the target actually received,
// which is the one captureEnvironment returned, not the parent's.
func stampTargetEnvironment(command []string, workDir string, env, stampFor []string) *targetEnvironment {
	t := &targetEnvironment{
		Command:  append([]string(nil), command...),
		Dir:      workDir,
		EnvCount: len(env),
	}
	sorted := slices.Sorted(slices.Values(env))
	h := sha256.New()
	for _, item := range sorted {
		h.Write([]byte(item))
		h.Write([]byte{0})
	}
	t.EnvDigest = hex.EncodeToString(h.Sum(nil))
	if len(stampFor) == 0 {
		return t
	}
	t.StampedFor = append([]string(nil), stampFor...)
	for _, want := range stampFor {
		prefix, isPrefix := strings.CutSuffix(want, "*")
		matched := false
		for _, item := range env {
			name, value, ok := strings.Cut(item, "=")
			if !ok {
				continue
			}
			if (isPrefix && strings.HasPrefix(name, prefix)) || (!isPrefix && name == want) {
				if t.Env == nil {
					t.Env = make(map[string]string)
				}
				t.Env[name] = value
				matched = true
			}
		}
		if !matched {
			t.EnvUnset = append(t.EnvUnset, want)
		}
	}
	return t
}

func captureEnvironment(o captureOptions, session, dir string) []string {
	env := append([]string(nil), o.Env...)
	if env == nil {
		env = os.Environ()
	}
	env = setCaptureEnv(env, "CUDAPROF_PROTOCOL", strconv.Itoa(captureProtocolVersion))
	env = setCaptureEnv(env, "CUDAPROF_SESSION", session)
	env = setCaptureEnv(env, "CUDAPROF_TRANSPORT", dir)
	if o.Stacks {
		env = setCaptureEnv(env, "CUDAPROF_STACK_DIR", filepath.Join(dir, "stacks"))
		env = setCaptureEnv(env, "CUDAPROF_GO_STACKS", "1")
		env = setCaptureEnv(env, "CUDAPROF_PYTHON_STACKS", "1")
		pythonPath := filepath.Join(dir, "python")
		if inherited := lookupEnv(env, "PYTHONPATH"); inherited != "" {
			pythonPath += string(os.PathListSeparator) + inherited
		}
		env = setCaptureEnv(env, "PYTHONPATH", pythonPath)
	} else if o.stacksSet {
		env = setCaptureEnv(env, "CUDAPROF_GO_STACKS", "0")
		env = setCaptureEnv(env, "CUDAPROF_PYTHON_STACKS", "0")
	}
	env = setCaptureEnv(env, cudaprof.ExternalStacksEnv, filepath.Join(dir, cudacapture.ExternalStacksName))
	if !hasCaptureEnv(env, "CUDAPROF_CORRELATE") {
		env = setCaptureEnv(env, "CUDAPROF_CORRELATE", "1")
	}
	// The collection policy is exported only when the caller asked for it, so
	// an unset flag leaves whatever the caller's own environment said and the
	// agent keeps its own default. "none" is exported like any other value: it
	// is how a caller turns off gating that the surrounding environment
	// enabled, and silently dropping it would leave the run gated.
	if o.CaptureRange != "" {
		env = setCaptureEnv(env, "CUDAPROF_CAPTURE_RANGE", o.CaptureRange)
	}
	if o.Finalize != "" {
		env = setCaptureEnv(env, "CUDAPROF_FINALIZE", o.Finalize)
	}
	if o.GraphTrace != "" {
		env = setCaptureEnv(env, "CUDAPROF_GRAPH_TRACE", string(o.GraphTrace))
	}
	// Module images are requested by this process because this process is the
	// one that can decode the record kind they arrive under. An agent old
	// enough to ignore the variable simply writes no modules, and the capture
	// is what it would have been.
	env = setCaptureEnv(env, "CUDAPROF_MODULES", "1")
	// The agent keeps a module image only up to its own per-module limit,
	// 16 MB by default, well under the file budget this process then applies
	// to it: sys1's 40.8 MB and 34.9 MB modules were dropped although they
	// fit. Unless the caller set the agent's limit, it is the file budget.
	if lookupEnv(env, "CUDAPROF_MODULE_FILE_BYTES") == "" {
		b := captureArtifactBudget{FileBytes: o.CaptureFileBytes}.effective()
		env = setCaptureEnv(env, "CUDAPROF_MODULE_FILE_BYTES", strconv.FormatInt(b.FileBytes, 10))
	}
	// PC sampling is exported only when asked for, and asking for it is what
	// makes the agent write the record kind this process has just learned to
	// decode. An agent that predates the kind ignores the variable.
	if o.PCSampling {
		env = setCaptureEnv(env, "CUDAPROF_PCSAMPLING", "1")
		if o.PCSamplingPeriod > 0 {
			env = setCaptureEnv(env, "CUDAPROF_PCSAMPLING_PERIOD", strconv.Itoa(o.PCSamplingPeriod))
		}
	}
	if o.hostSampleSet {
		env = setCaptureEnv(env, "CUDAPROF_HOST_SAMPLE", strconv.Itoa(o.HostSampleHz))
	}
	if o.CrashTrace > 0 {
		env = setCaptureEnv(env, "CUDAPROF_CRASH_TRACE", strconv.Itoa(o.CrashTrace))
	}
	// cuBLASLt calls, like PC samples, arrive under a kind only this process
	// knows, so they are asked for only when wanted.
	if o.GEMMs {
		env = setCaptureEnv(env, "CUDAPROF_GEMMS", "1")
	}
	if o.FlushInterval > 0 {
		env = setCaptureEnv(env, "CUDAPROF_FLUSH_INTERVAL_MS", strconv.FormatInt(o.FlushInterval.Milliseconds(), 10))
	}
	// The agent reads this variable as "off only when it is exactly 0", so an
	// explicit --api-activity exports 1 rather than relying on absence: the
	// caller's own environment may already hold a 0 that the flag is being
	// used to override.
	if o.apiActivitySet {
		value := "1"
		if !o.APIActivity {
			value = "0"
		}
		env = setCaptureEnv(env, "CUDAPROF_API_ACTIVITY", value)
	}
	// Memsets are opt-in, and exported either way so an inherited value
	// cannot decide.
	memsets := "0"
	if o.Memsets {
		memsets = "1"
	}
	env = setCaptureEnv(env, "CUDAPROF_MEMSETS", memsets)
	unifiedMemory := "0"
	if o.UnifiedMemory {
		unifiedMemory = "1"
	}
	env = setCaptureEnv(env, "CUDAPROF_UNIFIED_MEMORY", unifiedMemory)
	memoryPolicy := "0"
	if o.MemoryPolicy {
		memoryPolicy = "1"
	}
	env = setCaptureEnv(env, "CUDAPROF_MEMORY_POLICY", memoryPolicy)
	if o.Inject != "" && o.InjectMode == injectModeCUDA {
		// The driver dlopens this path from whatever directory the target
		// runs in, so a relative one would name a different file or none.
		path := o.Inject
		if abs, err := filepath.Abs(path); err == nil {
			path = abs
		}
		env = setCaptureEnv(env, "CUDA_INJECTION64_PATH", path)
	} else if o.Inject != "" {
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
	env = setNVTXInjection(env)
	return env
}

func hasCaptureEnv(env []string, name string) bool {
	for _, item := range env {
		if key, _, ok := strings.Cut(item, "="); ok && key == name {
			return true
		}
	}
	return false
}

// setNVTXInjection points NVTX_INJECTION64_PATH at a CUPTI library so that an
// in-process NVTX provider in the target actually emits.
//
// LD_PRELOAD above gets CUPTI collecting, and the producer then reports
// nvtx_markers:true -- but that bit only says cuptiActivityEnable(
// CUPTI_ACTIVITY_KIND_MARKER) succeeded, which is a claim about the collector,
// not about anything emitting. A Go target using nvidia/x/nvtx returns from
// initInjection with enabled still false when this variable is unset
// (nvtx/nvtx.go:259), and says nothing about it, so the capture collects zero
// markers while reporting that markers are on. That combination cost a full
// capture cycle on 2026-09-18 before the cause was found.
//
// An existing value is never overridden: the caller has then already chosen an
// injection library, and NVIDIA's own tooling sets this variable too.
//
// Only a path that stats is exported. Handing the target a name the loader
// cannot resolve reaches the same silent return through a different door, and
// would read as a landed fix. Setting the variable is still only a
// producer-side claim in exactly the way nvtx_markers is: the honest check is
// that a capture collected markers, not that this function found a file.
func setNVTXInjection(env []string) []string {
	const key = "NVTX_INJECTION64_PATH"
	if runtime.GOOS != "linux" {
		return env
	}
	if lookupEnv(env, key) != "" {
		return env
	}
	if path := findCUPTILibrary(); path != "" {
		env = setCaptureEnv(env, key, path)
	}
	return env
}

// findCUPTILibrary returns a CUPTI library that exists on disk, or "".
//
// This repeats nvidia/cupti's toolkit search rather than calling it, because
// that package resolves the path by dlopening the library into the calling
// process, and the launcher must not load CUPTI to profile something else. It
// also covers targets/<arch>-linux/lib for the running architecture, which the
// generated resolver in nvidia/cupti/init.gen.go does not: on aarch64 hosts
// libcupti lives in targets/sbsa-linux/lib, and that resolver finds it only
// when ldconfig already has the soname.
func findCUPTILibrary() string {
	if p := os.Getenv("CUDAPROF_NVTX_INJECTION"); p != "" {
		return p
	}
	arch := "x86_64-linux"
	if runtime.GOARCH == "arm64" {
		arch = "sbsa-linux"
	}
	var roots []string
	for _, dir := range []string{os.Getenv("CUDA_HOME"), os.Getenv("CUDA_PATH"), "/usr/local/cuda"} {
		if dir != "" {
			roots = append(roots, dir)
		}
	}
	subs := []string{"extras/CUPTI/lib64", "extras/CUPTI/lib", filepath.Join("targets", arch, "lib")}
	names := []string{"libcupti.so.13", "libcupti.so.12", "libcupti.so"}
	for _, root := range roots {
		for _, sub := range subs {
			for _, name := range names {
				p := filepath.Join(root, sub, name)
				if fi, err := os.Stat(p); err == nil && !fi.IsDir() {
					return p
				}
			}
		}
	}
	return ""
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
	result := captureResult{Protocol: captureProtocolVersion, Session: session, State: captureStatePlanned, Outcome: outcome, ChildrenFollowed: o.FollowChildren, ScratchDir: dir, StacksRequested: o.Stacks}
	if o.Stacks {
		if err := prepareStackTransport(dir); err != nil {
			return result, err
		}
	}
	if o.Inject == "" && !o.NoInject {
		result.CaptureError = "no injection library configured (specify --inject /path/to/agent.so, CUDAPROF_INJECT, or pass --no-inject to run uninstrumented)"
		result.RunCompleteness = runUnknown
		return result, nil
	}
	if o.Inject != "" {
		if o.InjectMode == "" {
			o.InjectMode = injectModePreload
		}
		result.InjectMode = o.InjectMode
	}
	// The driver loads one injection library. Replacing one the caller's
	// environment already names would silently unload another tool.
	if o.Inject != "" && o.InjectMode == injectModeCUDA {
		base := o.Env
		if base == nil {
			base = os.Environ()
		}
		if v := lookupEnv(base, "CUDA_INJECTION64_PATH"); v != "" {
			return result, fmt.Errorf("--inject-mode=%s: CUDA_INJECTION64_PATH is already set to %q", injectModeCUDA, v)
		}
	}
	cmd := exec.Command(o.command[0], o.command[1:]...)
	cmd.Dir, cmd.Env = o.Dir, captureEnvironment(o, session, dir)
	result.Target = stampTargetEnvironment(o.command, o.Dir, cmd.Env, o.StampEnv)
	if cmd.Err != nil {
		result.Target.Executable = &executableIdentity{Path: o.command[0], Error: cmd.Err.Error()}
	} else {
		result.Target.Executable = identifyTarget(cmd.Path, o.command, cmd.Env, o.Dir)
	}
	if o.Inject != "" {
		result.Target.Agent = identifyExecutable(o.Inject)
	}
	log := o.Log
	if log == nil {
		log = os.Stderr
	}
	goExit := o.Inject != "" && goExitRisk(result.Target.Executable, lookupEnv(cmd.Env, "CUDAPROF_FINALIZE"))
	if goExit {
		fmt.Fprintf(log, "cudaprof: warning: %s\n", goExitWarning(result.Target.Executable))
	}
	if o.Inject != "" && o.Memsets {
		fmt.Fprintf(log, "cudaprof: warning: %s\n", memsetsWarning)
	}
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
	// A gated target writes capture-range edges, and each is given the GPU's
	// clocks as the controller reads them. The GPU is opened before the
	// target starts and followed only while it runs.
	var clocks *edgeClocks
	if lookupEnv(cmd.Env, "CUDAPROF_CAPTURE_RANGE") == captureRangeProfilerAPI {
		clocks = openEdgeClocks(o.openGPU)
	}
	var isolationObserver *isolationWatch
	if o.ObserveIsolation {
		isolationObserver = startIsolationWatch()
	}
	result.WallClock = readWallClock()
	captureAdmission, captureClockKnown := monotonicNanos()
	if err := cmd.Start(); err != nil {
		if clocks != nil {
			clocks.finish()
		}
		if isolationObserver != nil {
			isolationObserver.finish(&result)
		}
		return result, fmt.Errorf("start capture target: %w", err)
	}
	result.State, result.TargetPID = captureStateSpawned, cmd.Process.Pid
	result.IsolationBinding = isolation.Binding{Session: session, Process: captureProcessIdentity(cmd.Process.Pid), Scope: "capture_process_lifetime"}
	if captureClockKnown {
		result.IsolationBinding.Window = isolation.Window{Domain: "CLOCK_MONOTONIC", Start: int64(captureAdmission)}
	}
	if clocks != nil {
		clocks.watch(dir, cmd.Process.Pid)
	}
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
	if end, ok := monotonicNanos(); ok && captureClockKnown {
		result.IsolationBinding.Window.End = int64(end)
	}
	result.State = captureStateFinalized
	var edgeSamples map[edgeKey]*gpuEdgeSample
	if clocks != nil {
		edgeSamples = clocks.finish()
	}
	observe := observeOptions{injectionRequested: o.Inject != "", cudaInjection: o.Inject != "" && o.InjectMode == injectModeCUDA, noInject: o.NoInject, targetOnly: !o.FollowChildren, recoverUnfinalized: o.RecoverUnfinalized, maxJournalRecords: o.MaxJournalRecords}
	if goExit {
		observe.goTarget = result.Target.Executable
	}
	result = observeCaptureSession(result, dir, session, observe)
	if isolationObserver != nil {
		isolationObserver.finish(&result)
	}
	if clocks != nil {
		attachEdgeClocks(result.PhaseReceipt, edgeSamples)
	}
	result = excludeKernels(result, o.Exclude)
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
			return result, context.Cause(runctx)
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
		result = markTargetFailed(result)
		if result.Cancelled {
			return result, context.Cause(runctx)
		}
		return result, nil
	}
	return result, fmt.Errorf("wait for capture target: %w", waitErr)
}

// excludeKernels drops the kernels named by --exclude from this capture's
// projection: the pprof, the activity samples and the aggregates computed from
// them.
//
// It is a projection filter and not a collection one, because by the time this
// runs the target has exited and the agent has written everything it saw.
// Nothing here can keep a kernel out of the capture: the raw journals stay
// attached, and journal-based analysis (cudaprof gaps) still sees every
// excluded kernel, which is right -- a timeline with a kernel cut out of it
// would report idle gaps that never happened.
//
// So the totals stay journal totals. ActivityRecords, ActivityNanos, CopyBytes
// and the loss breakdown keep describing what the producer wrote, and what the
// exclusion removed is reported beside them as ExcludedRecords and
// ExcludedNanos. The identity a reader can check is
// activity_samples == activity_records - excluded_records. Subtracting from
// the totals instead would publish numbers that disagree with the journal in
// the same file, and with the producer's own retained count, with nothing in
// the capture to say which convention produced them.
//
// A pattern matches when it appears anywhere in the kernel name. Mangled names
// are long and an exact match is unusable by hand. Only kernels are matched: a
// memcpy has no kernel name, so no --exclude can remove one.
func excludeKernels(result captureResult, patterns []string) captureResult {
	if len(patterns) == 0 {
		return result
	}
	result.ExcludedPatterns = append([]string(nil), patterns...)
	// A fresh slice rather than a filter in place: captureResult is passed by
	// value, so reusing the storage would rewrite the activities of the result
	// the caller still holds.
	kept := make([]captureActivity, 0, len(result.Activities))
	for _, a := range result.Activities {
		if a.Subtype == "kernel" && matchesAnyPattern(a.Name, patterns) {
			result.ExcludedRecords++
			if a.Valid && a.End >= a.Start {
				result.ExcludedNanos += a.End - a.Start
			}
			continue
		}
		kept = append(kept, a)
	}
	result.Activities = kept
	return result
}

func matchesAnyPattern(name string, patterns []string) bool {
	for _, p := range patterns {
		if strings.Contains(name, p) {
			return true
		}
	}
	return false
}

// outputHoldsWork reports whether the capture already at path holds a
// measurement, and describes what it holds. The description is returned even
// when it holds nothing so the caller never has to re-derive it.
//
// Refusing to overwrite exists to protect a capture someone paid a run under
// GPU lock for. A capture that observed nothing cost that run too, but there
// is no measurement in it to lose, and it is precisely the artifact a failed
// run leaves behind -- so the refusal lands on the retry, under the name the
// retry wants. The operator then renames the output, and two captures of one
// workload that differ by name stop being comparable by the tooling that pairs
// them, or deletes the file by hand, which is the operation the refusal was
// trying to make deliberate.
//
// The predicate is the file's own record count and nothing else: not a --force
// flag, not the name, not whether the previous run reported failure. A retry
// path keyed on the request rather than on the file can take a good capture
// whose run merely looks similar, and an outcome token can say "failed" about
// a run that still retained records. A file whose count cannot be read holds
// work by default -- an artifact cudaprof cannot parse is not one it may
// delete.
func outputHoldsWork(path string) (string, bool) {
	cap, err := readCapturePathLimits(path, cudacapture.Limits{})
	if err != nil || cap == nil {
		return "a capture cudaprof cannot read, so its record count is unknown", true
	}
	status := capturefiles.Status(cap)
	if status == nil {
		return "no capture status, so its record count is unknown", true
	}
	records, ok := status["activity_records"].(float64)
	if !ok {
		return "a capture status with no activity_records field, so its record count is unknown", true
	}
	if records == 0 {
		return "no activity records", false
	}
	return fmt.Sprintf("%.0f activity records", records), true
}
