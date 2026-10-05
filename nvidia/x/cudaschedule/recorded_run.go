package cudaschedule

import (
	"context"
	"fmt"
	"path/filepath"
	"time"
)

// RecordedRunOptions specifies a prebuilt, source-reviewed worker and helper.
// The zero value is invalid. Limits and executable pins are explicit.
// The worker must call RunRecordedHelper with worker-ready before CUDA or probes.
type RecordedRunOptions struct {
	PristinePTX      string        // Pristine source for baseline and fresh health execution.
	Executable       string        // Absolute prebuilt worker path.
	ExecutableSHA256 string        // Worker executable SHA-256.
	Args             []string      // Worker arguments; the supplied PTX path is appended.
	WorkDir          string        // Absolute worker working directory.
	Helper           string        // Absolute cudarecorded helper path.
	HelperSHA256     string        // Helper executable SHA-256.
	ArtifactDir      string        // Absolute parent for retained per-attempt artifacts.
	LockScript       string        // Absolute source-reviewed gpu-lock script path.
	LockDirectory    string        // Authoritative shared lock directory.
	Label            string        // Caller label recorded by the real lock.
	QueueTimeout     time.Duration // Whole-second queue budget.
	AdmissionTimeout time.Duration // Ownership and worker-ready ACK budget.
	Timeout          time.Duration // Candidate runtime after CUDA permit.
	PristineTimeout  time.Duration // Pristine runtime after CUDA permit.
	CleanupTimeout   time.Duration // Retirement and reaping budget.
	LeaseLimit       time.Duration // Whole-second real lock limit.
	LeaseExpected    time.Duration // Whole-second real lock estimate.
	OutputLimit      int           // Maximum retained bytes per output stream.
	OracleAttached   bool          // Require PID-bound instrumentation liveness.
	RequireOutput    bool          // Require a worker-bound output-contract receipt.
	CompareOutput    bool          // Execute a fresh pristine baseline before the candidate.
}

// RecordedCustody retains actual processes involved in one attempt.
// Closed requires checked retirement of holder and supervisor groups.
// Worker status comes from its direct supervisor, never from guardian success.
type RecordedCustody struct {
	// Phase records the furthest observed source transition: keeper-ready,
	// holder-started, supervisor-observed, worker-observed, worker-permitted.
	Phase string `json:"phase"`
	// SupervisorCreation is not-started before H exists, unknown after H starts,
	// and observed only after the actual S peer identity has been retained.
	SupervisorCreation string `json:"supervisor_creation"`

	Controller    RecordedProcess `json:"controller"`
	Keeper        RecordedProcess `json:"keeper"`
	Holder        RecordedProcess `json:"holder"`
	Supervisor    RecordedProcess `json:"supervisor"`
	Worker        RecordedProcess `json:"worker"`
	Closed        bool            `json:"closed"`
	WorkerReaped  bool            `json:"worker_reaped"`
	StdoutDrained bool            `json:"stdout_drained"`
	StderrDrained bool            `json:"stderr_drained"`
	Directory     string          `json:"directory"`
}

// RunRecorded executes supplied PTX, including an unchanged pristine candidate.
// It does not perturb, redraw or confirm. Its controller must be CUDA-free
// and outside an inherited lease. Artifacts are retained.
// Configuration and parent cancellation return errors. Execution and ownership
// failures are retained in HarnessErr. A timeout receives fresh pristine health
// only after complete retirement; a previous baseline is not reused as health.
func RunRecorded(ctx context.Context, seed uint64, recordedPTX string, opts RecordedRunOptions) (RunRecord, error) {
	if ctx == nil {
		return RunRecord{}, fmt.Errorf("recorded run: nil context")
	}
	if err := ctx.Err(); err != nil {
		return RunRecord{}, err
	}
	if err := recordedOutsideLease(); err != nil {
		return RunRecord{}, err
	}
	for _, source := range []string{opts.PristinePTX, recordedPTX} {
		if len(source) == 0 || len(source) > 4<<20 {
			return RunRecord{}, fmt.Errorf("recorded run: PTX size out of range")
		}
		if _, err := Analyze(source); err != nil {
			return RunRecord{}, fmt.Errorf("recorded run: PTX: %w", err)
		}
	}
	for _, path := range []string{opts.Executable, opts.WorkDir, opts.Helper, opts.ArtifactDir, opts.LockScript, opts.LockDirectory} {
		if !filepath.IsAbs(path) {
			return RunRecord{}, fmt.Errorf("recorded run: paths must be absolute")
		}
	}
	if !validPTXDigest(opts.ExecutableSHA256) || !validPTXDigest(opts.HelperSHA256) {
		return RunRecord{}, fmt.Errorf("recorded run: executable pins required")
	}
	for _, runtime := range []time.Duration{opts.Timeout, opts.PristineTimeout} {
		if _, err := (recordedLimits{Queue: opts.QueueTimeout, Admission: opts.AdmissionTimeout, Runtime: runtime, Cleanup: opts.CleanupTimeout, Lease: opts.LeaseLimit}).validate(); err != nil {
			return RunRecord{}, err
		}
	}
	var baseline *RunRecord
	if opts.CompareOutput {
		rec, err := runRecordedAttempt(ctx, seed, opts.PristinePTX, opts, opts.PristineTimeout, true)
		baseline = &rec
		if err != nil {
			return RunRecord{Baseline: baseline, HarnessErr: "baseline interrupted"}, err
		}
		if Classify(rec) != VerdictPass {
			return RunRecord{Baseline: baseline, ExpectedPTXDigest: digestPTX(recordedPTX), HarnessErr: "baseline did not establish pass"}, nil
		}
	}
	rec, err := runRecordedAttempt(ctx, seed, recordedPTX, opts, opts.Timeout, false)
	rec.Baseline = baseline
	rec.CompareOutput = opts.CompareOutput
	if err != nil {
		return rec, err
	}
	if rec.TimedOut && rec.HarnessErr == "" && rec.Custody != nil && rec.Custody.Closed {
		health, err := runRecordedAttempt(ctx, seed, opts.PristinePTX, opts, opts.PristineTimeout, true)
		rec.Pristine = &health
		if err != nil {
			return rec, err
		}
	}
	return rec, nil
}

// RunRecordedHelper runs the controlled protocol.
// Keeper and supervisor arguments are mode, absolute config path and SHA-256.
// The worker calls it with []string{"worker-ready"} before initializing CUDA.
// Successful transport does not imply a successful worker outcome.
// A caller must not initialize CUDA after helper refusal.
func RunRecordedHelper(ctx context.Context, args []string) error {
	if ctx == nil {
		return fmt.Errorf("recorded helper: nil context")
	}
	if len(args) == 1 && args[0] == "worker-ready" {
		return runRecordedWorkerReady(ctx)
	}
	if len(args) != 3 || (args[0] != "keeper" && args[0] != "supervisor") || !filepath.IsAbs(args[1]) || !validPTXDigest(args[2]) {
		return fmt.Errorf("recorded helper: expected keeper or supervisor, config path and digest")
	}
	return runRecordedHelperMode(ctx, args[0], args[1], args[2])
}
