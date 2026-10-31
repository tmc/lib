package autotune

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"strings"
	"sync"
	"time"
)

// TuneJobSpec describes an isolated tuning and PGO cache qualification job
// for one shape bucket.
type TuneJobSpec struct {
	Arch          string `json:"arch"`
	Device        string `json:"device"`
	DeviceSMs     int    `json:"device_sms"`
	DriverVersion string `json:"driver_version"`
	CUDAVersion   string `json:"cuda_version"`
	WeightDigest  string `json:"weight_digest"`
	PlanKey       string `json:"plan_key"`
	BucketM       int    `json:"bucket_m"`
	HiddenSize    int    `json:"hidden_size"`
	InterSize     int    `json:"inter_size"`
	Heads         int    `json:"heads"`
	KVHeads       int    `json:"kv_heads"`
	BeamWidth     int    `json:"beam_width"`
	CrossLayer    bool   `json:"cross_layer"`
	PTXASLoop     bool   `json:"ptxas_loop"`
	CacheDir      string `json:"cache_dir"`
}

// Validate checks that the required fields of s are populated and returns an
// error if any dimension or bucket is non-positive.
func (s TuneJobSpec) Validate() error {
	norm := s.Norm()
	if norm.Arch == "" {
		return errors.New("tune job spec: empty arch")
	}
	if norm.BucketM <= 0 {
		return fmt.Errorf("tune job spec: invalid bucket_m %d", s.BucketM)
	}
	if norm.HiddenSize <= 0 {
		return fmt.Errorf("tune job spec: invalid hidden_size %d", s.HiddenSize)
	}
	if norm.InterSize <= 0 {
		return fmt.Errorf("tune job spec: invalid inter_size %d", s.InterSize)
	}
	if norm.Heads <= 0 {
		return fmt.Errorf("tune job spec: invalid heads %d", s.Heads)
	}
	if norm.KVHeads <= 0 || norm.KVHeads > norm.Heads || norm.Heads%norm.KVHeads != 0 {
		return fmt.Errorf("tune job spec: invalid kv_heads %d for heads %d", s.KVHeads, norm.Heads)
	}
	return nil
}

// Norm returns a copy of s with default hardware, version, and model shape
// parameters populated when omitted.
func (s TuneJobSpec) Norm() TuneJobSpec {
	s.Arch = strings.TrimSpace(s.Arch)
	if s.Arch == "" {
		s.Arch = "sm_121a"
	}
	s.Device = strings.TrimSpace(s.Device)
	if s.Device == "" {
		s.Device = "GB10"
	}
	if s.DeviceSMs <= 0 {
		s.DeviceSMs = 48
	}
	s.DriverVersion = strings.TrimSpace(s.DriverVersion)
	if s.DriverVersion == "" {
		s.DriverVersion = "580.173.02"
	}
	s.CUDAVersion = strings.TrimSpace(s.CUDAVersion)
	if s.CUDAVersion == "" {
		s.CUDAVersion = "13.4"
	}
	if s.HiddenSize <= 0 {
		s.HiddenSize = 256
	}
	if s.InterSize <= 0 {
		s.InterSize = 512
	}
	if s.Heads <= 0 {
		s.Heads = 4
	}
	if s.KVHeads <= 0 {
		s.KVHeads = min(2, s.Heads)
	}
	if s.BeamWidth <= 0 {
		s.BeamWidth = 4
	}
	s.WeightDigest = strings.TrimSpace(s.WeightDigest)
	if s.WeightDigest == "" {
		s.WeightDigest = fmt.Sprintf("sha256:llama_bf16_h%d_i%d_q%d_kv%d", s.HiddenSize, s.InterSize, s.Heads, s.KVHeads)
	}
	s.PlanKey = strings.TrimSpace(s.PlanKey)
	if s.PlanKey == "" {
		mode := "single"
		if s.CrossLayer {
			mode = "cross"
		}
		s.PlanKey = fmt.Sprintf("egtune_%s_h%d_i%d_q%d_kv%d", mode, s.HiddenSize, s.InterSize, s.Heads, s.KVHeads)
	}
	s.CacheDir = strings.TrimSpace(s.CacheDir)
	return s
}

// PGOCacheKey returns the canonical [PGOCacheKey] for s.
func (s TuneJobSpec) PGOCacheKey() PGOCacheKey {
	norm := s.Norm()
	return PGOCacheKey{
		Arch:          norm.Arch,
		DriverVersion: norm.DriverVersion,
		CUDAVersion:   norm.CUDAVersion,
		WeightDigest:  norm.WeightDigest,
		PlanKey:       norm.PlanKey,
		ShapeBucket:   norm.BucketM,
	}
}

// TuneJobResult summarizes the outcome of a single isolated TuneJobSpec.
type TuneJobResult struct {
	Spec            TuneJobSpec       `json:"spec"`
	CacheKey        PGOCacheKey       `json:"cache_key"`
	CacheHit        bool              `json:"cache_hit"`
	CacheStored     bool              `json:"cache_stored"`
	Promoted        bool              `json:"promoted"`
	WinningSchedule string            `json:"winning_schedule"`
	KernelName      string            `json:"kernel_name"`
	PTXASIterations int               `json:"ptxas_iterations"`
	RegistersPerTh  int               `json:"registers_per_thread"`
	TotalSpillBytes int               `json:"total_spill_bytes"`
	Proof           PGOPromotionProof `json:"proof"`
	DurationMs      float64           `json:"duration_ms"`
}

// EpochDaemon coordinates single-tenant CUPTI PC-sampling epochs so at most
// one isolated profiling worker or epoch runs at a time and respects an
// optional [ThrottleGate].
type EpochDaemon struct {
	ThrottleGate ThrottleGate

	mu     sync.Mutex
	epochs uint64
	active int
}

// NewEpochDaemon returns a new [EpochDaemon] with the optional gate.
func NewEpochDaemon(gate ThrottleGate) *EpochDaemon {
	return &EpochDaemon{ThrottleGate: gate}
}

// Epochs returns the number of completed or started isolated epochs.
func (d *EpochDaemon) Epochs() uint64 {
	if d == nil {
		return 0
	}
	d.mu.Lock()
	defer d.mu.Unlock()
	return d.epochs
}

// Active returns the number of currently executing epochs (0 or 1).
func (d *EpochDaemon) Active() int {
	if d == nil {
		return 0
	}
	d.mu.Lock()
	defer d.mu.Unlock()
	return d.active
}

// RunEpoch executes fn under single-tenant epoch exclusivity. If d.ThrottleGate
// is non-nil and disallows execution, RunEpoch returns [ErrThrottled].
func (d *EpochDaemon) RunEpoch(ctx context.Context, fn func(ctx context.Context, epoch uint64) error) error {
	if ctx == nil {
		ctx = context.Background()
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	if fn == nil {
		return errors.New("epoch daemon: nil epoch function")
	}
	if d == nil {
		return fn(ctx, 1)
	}
	d.mu.Lock()
	defer d.mu.Unlock()
	if err := ctx.Err(); err != nil {
		return err
	}
	if d.ThrottleGate != nil && !d.ThrottleGate.Allow() {
		return ErrThrottled
	}
	d.epochs++
	epoch := d.epochs
	d.active++
	defer func() {
		d.active--
	}()
	return fn(ctx, epoch)
}

// SubprocessTuneRunner runs isolated [TuneJobSpec] jobs either in a child OS
// process via Command (passing `-worker-spec <json>` so CUPTI's process-wide
// hardware PC-sampling lock is confined to the child and released on exit) or
// via WorkerFunc in-process when Command is empty.
type SubprocessTuneRunner struct {
	// Command is the base command and optional arguments used to spawn an
	// isolated worker process (for example, []string{os.Executable()}).
	// The runner appends "-worker-spec", <json(spec)> to Command.
	Command []string

	// Env holds optional extra environment variables ("KEY=VALUE") appended to
	// os.Environ() when spawning a subprocess worker.
	Env []string

	// WorkerFunc is the fallback in-process worker invoked when Command is empty.
	WorkerFunc func(context.Context, TuneJobSpec) (TuneJobResult, error)

	// EpochDaemon enforces single-tenant GPU epoch exclusivity across jobs.
	// When nil, a per-runner mutex still serializes RunJob calls.
	EpochDaemon *EpochDaemon

	mu sync.Mutex
}

// RunJob executes a single [TuneJobSpec] within an exclusive epoch.
func (r *SubprocessTuneRunner) RunJob(ctx context.Context, spec TuneJobSpec) (TuneJobResult, error) {
	if r == nil {
		return TuneJobResult{}, errors.New("subprocess tune runner: nil runner")
	}
	if ctx == nil {
		ctx = context.Background()
	}
	if err := spec.Validate(); err != nil {
		return TuneJobResult{}, err
	}
	normSpec := spec.Norm()

	daemon := r.EpochDaemon
	var (
		res TuneJobResult
		err error
	)
	runFn := func(runCtx context.Context, _ uint64) error {
		t0 := time.Now()
		res, err = r.dispatch(runCtx, normSpec)
		if err != nil {
			return err
		}
		if res.DurationMs <= 0 {
			res.DurationMs = float64(time.Since(t0).Nanoseconds()) / 1e6
		}
		if res.CacheKey == (PGOCacheKey{}) {
			res.CacheKey = normSpec.PGOCacheKey()
		}
		if res.Spec.BucketM == 0 {
			res.Spec = normSpec
		}
		return nil
	}

	if daemon != nil {
		if runErr := daemon.RunEpoch(ctx, runFn); runErr != nil {
			return TuneJobResult{}, runErr
		}
		return res, nil
	}

	r.mu.Lock()
	defer r.mu.Unlock()
	if runErr := runFn(ctx, 1); runErr != nil {
		return TuneJobResult{}, runErr
	}
	return res, nil
}

// RunBuckets runs isolated tuning epochs sequentially across each shape bucket
// M in buckets, enforcing single-tenant epoch exclusivity via [EpochDaemon] so
// two CUPTI workers never collide on the GPU.
func (r *SubprocessTuneRunner) RunBuckets(ctx context.Context, baseSpec TuneJobSpec, buckets []int) ([]TuneJobResult, error) {
	if r == nil {
		return nil, errors.New("subprocess tune runner: nil runner")
	}
	if len(buckets) == 0 {
		return nil, errors.New("subprocess tune runner: no buckets specified")
	}
	for _, m := range buckets {
		if m <= 0 {
			return nil, fmt.Errorf("subprocess tune runner: invalid bucket M=%d", m)
		}
	}

	results := make([]TuneJobResult, 0, len(buckets))
	for _, m := range buckets {
		jobSpec := baseSpec
		jobSpec.BucketM = m
		res, err := r.RunJob(ctx, jobSpec)
		if err != nil {
			return results, fmt.Errorf("bucket M=%d: %w", m, err)
		}
		results = append(results, res)
	}
	return results, nil
}

func (r *SubprocessTuneRunner) dispatch(ctx context.Context, spec TuneJobSpec) (TuneJobResult, error) {
	if len(r.Command) > 0 {
		return r.runSubprocess(ctx, spec)
	}
	if r.WorkerFunc != nil {
		return r.WorkerFunc(ctx, spec)
	}
	return TuneJobResult{}, errors.New("subprocess tune runner: neither Command nor WorkerFunc configured")
}

func (r *SubprocessTuneRunner) runSubprocess(ctx context.Context, spec TuneJobSpec) (TuneJobResult, error) {
	exe := strings.TrimSpace(r.Command[0])
	if exe == "" {
		return TuneJobResult{}, errors.New("subprocess tune runner: empty command executable")
	}
	specBytes, err := json.Marshal(spec)
	if err != nil {
		return TuneJobResult{}, fmt.Errorf("subprocess tune runner: marshal spec: %w", err)
	}
	args := make([]string, 0, len(r.Command)-1+2)
	args = append(args, r.Command[1:]...)
	args = append(args, "-worker-spec", string(specBytes))

	cmd := exec.CommandContext(ctx, exe, args...)
	if len(r.Env) > 0 {
		cmd.Env = append(os.Environ(), r.Env...)
	}
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	if err := cmd.Run(); err != nil {
		errMsg := strings.TrimSpace(stderr.String())
		if errMsg == "" {
			errMsg = strings.TrimSpace(stdout.String())
		}
		if errMsg != "" {
			return TuneJobResult{}, fmt.Errorf("subprocess worker failed: %w: %s", err, errMsg)
		}
		return TuneJobResult{}, fmt.Errorf("subprocess worker failed: %w", err)
	}

	outBytes := bytes.TrimSpace(stdout.Bytes())
	if len(outBytes) == 0 {
		return TuneJobResult{}, errors.New("subprocess worker produced empty stdout")
	}
	var res TuneJobResult
	if err := json.Unmarshal(outBytes, &res); err != nil {
		// If test binary or wrapper emitted extra lines, try the last JSON object line.
		lines := bytes.Split(outBytes, []byte("\n"))
		parsed := false
		for i := len(lines) - 1; i >= 0; i-- {
			line := bytes.TrimSpace(lines[i])
			if len(line) > 0 && line[0] == '{' {
				if uErr := json.Unmarshal(line, &res); uErr == nil {
					parsed = true
					break
				}
			}
		}
		if !parsed {
			return TuneJobResult{}, fmt.Errorf("subprocess worker unmarshal output: %w (output: %s)", err, string(outBytes))
		}
	}
	return res, nil
}
