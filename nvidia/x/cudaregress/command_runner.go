package cudaregress

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/tmc/lib/nvidia/x/cudalock"
	"github.com/tmc/lib/nvidia/x/cudatrial"
)

const (
	commandSpecVersion  = 1
	maxCommandSpecBytes = 1 << 20
	maxCommandArgs      = 256
	maxCommandEnv       = 128
	maxCommandArtifacts = 256
	maxCommandString    = 16 << 10
)

// CommandArtifact binds a named plan input or library to its on-disk bytes.
type CommandArtifact struct {
	Name   string `json:"name"`
	Path   string `json:"path"`
	SHA256 string `json:"sha256"`
}

// CommandArm declares one direct executable invocation. Args are passed
// literally to the executable; no shell or environment expansion is used.
type CommandArm struct {
	Path   string   `json:"path"`
	SHA256 string   `json:"sha256"`
	Args   []string `json:"args"`
}

// ExecutionSpec fully declares the executable, input closure, and child
// environment for a replay. Its digest is SHA-256 of its canonical JSON with
// SpecDigest empty.
type ExecutionSpec struct {
	Version          int               `json:"version"`
	PlanID           string            `json:"plan_id"`
	PlanDigest       string            `json:"plan_digest"`
	SpecDigest       string            `json:"spec_digest"`
	Mode             string            `json:"mode"`
	WorkingDirectory string            `json:"working_directory"`
	Environment      map[string]string `json:"environment"`
	TimeoutNanos     int64             `json:"timeout_nanos"`
	MaxOutputBytes   int               `json:"max_output_bytes"`
	Baseline         CommandArm        `json:"baseline"`
	Candidate        CommandArm        `json:"candidate"`
	Inputs           []CommandArtifact `json:"inputs"`
	Libraries        []CommandArtifact `json:"libraries"`
}

// CommandTiming is the child's explicit measurement declaration.
type CommandTiming struct {
	Kind   cudatrial.TimingKind `json:"kind"`
	Status string               `json:"status"`
	Unit   string               `json:"unit"`
	Count  int64                `json:"count"`
}

// CommandResult is the exact result envelope accepted from a child process.
type CommandResult struct {
	Version      int           `json:"version"`
	OutputValues []float64     `json:"output_values"`
	OutputSHA256 string        `json:"output_sha256"`
	Timing       CommandTiming `json:"timing"`
}

// CommandAttempt preserves process evidence for each invocation requested by
// Replay, including failures that Replay cannot retain in its core receipt.
type CommandAttempt struct {
	Arm          string         `json:"arm"`
	Round        int            `json:"round"`
	Started      time.Time      `json:"started"`
	Finished     time.Time      `json:"finished"`
	ExitCode     int            `json:"exit_code"`
	StdoutBytes  int            `json:"stdout_bytes"`
	StderrBytes  int            `json:"stderr_bytes"`
	StdoutSHA256 string         `json:"stdout_sha256"`
	StderrSHA256 string         `json:"stderr_sha256"`
	Partial      bool           `json:"partial"`
	Result       *CommandResult `json:"result,omitempty"`
	Error        string         `json:"error,omitempty"`
}

// CommandReplayRunner executes one declared arm directly and retains bounded
// output and attempt evidence. Construct it with NewCommandReplayRunner.
type CommandReplayRunner struct {
	plan      *ExperimentPlan
	spec      ExecutionSpec
	mu        sync.Mutex
	attempts  []CommandAttempt
	lockOwner *cudalock.Owner
	lockDir   string
}

// NewCommandReplayRunner validates the plan, spec, artifact closure, and
// selected mode before any child process can start.
func NewCommandReplayRunner(plan *ExperimentPlan, spec ExecutionSpec) (*CommandReplayRunner, error) {
	if plan == nil {
		return nil, errors.New("cudaregress: nil plan")
	}
	if err := ValidatePlan(plan); err != nil {
		return nil, err
	}
	if err := ValidateExecutionSpec(plan, &spec); err != nil {
		return nil, err
	}
	if plan.DeviceConfig.TargetClockMHz != 0 || plan.DeviceConfig.PowerLimitWatts != 0 || plan.DeviceConfig.MIGMode != "" {
		return nil, errors.New("cudaregress: requested clock, power, or MIG setting is unsupported")
	}
	planJSON, err := json.Marshal(plan)
	if err != nil {
		return nil, fmt.Errorf("cudaregress: copy replay plan: %w", err)
	}
	var ownedPlan ExperimentPlan
	if err := json.Unmarshal(planJSON, &ownedPlan); err != nil {
		return nil, fmt.Errorf("cudaregress: copy replay plan: %w", err)
	}
	specJSON, err := json.Marshal(spec)
	if err != nil {
		return nil, fmt.Errorf("cudaregress: copy execution spec: %w", err)
	}
	var ownedSpec ExecutionSpec
	if err := json.Unmarshal(specJSON, &ownedSpec); err != nil {
		return nil, fmt.Errorf("cudaregress: copy execution spec: %w", err)
	}
	runner := &CommandReplayRunner{plan: &ownedPlan, spec: ownedSpec}
	if spec.Mode == "gpu" || plan.DeviceConfig.RequireHardwareLock {
		dir, err := canonicalLockDir()
		if err != nil {
			return nil, err
		}
		lookup := func(key string) string {
			if key == "GPU_LOCKDIR" {
				return dir
			}
			return os.Getenv(key)
		}
		owner, err := cudalock.Inherited(lookup)
		if err != nil {
			return nil, fmt.Errorf("cudaregress: GPU execution refused: %w", err)
		}
		if spec.Environment["GPU_LOCK_HELD"] != strconv.Itoa(owner.PID) {
			return nil, errors.New("cudaregress: execution spec must carry the inherited GPU lock owner")
		}
		if filepath.Clean(spec.Environment["GPU_LOCKDIR"]) != filepath.Clean(dir) {
			return nil, errors.New("cudaregress: execution spec must carry the canonical GPU lock directory")
		}
		runner.lockOwner = &owner
		runner.lockDir = dir
	}
	return runner, nil
}

func canonicalLockDir() (string, error) {
	home, err := os.UserHomeDir()
	if err != nil || home == "" {
		return "", errors.New("cudaregress: GPU execution refused: host home directory unavailable")
	}
	if !filepath.IsAbs(home) {
		return "", errors.New("cudaregress: GPU execution refused: host home directory is not absolute")
	}
	return filepath.Join(home, "ml-explore", ".gpu-lock", "lock"), nil
}

// ValidateExecutionSpec verifies versioned identity and exact artifact closure.
func ValidateExecutionSpec(plan *ExperimentPlan, spec *ExecutionSpec) error {
	if plan == nil || spec == nil {
		return errors.New("cudaregress: nil plan or execution spec")
	}
	if spec.Version != commandSpecVersion || spec.PlanID != plan.PlanID || spec.PlanDigest != plan.PlanDigest {
		return errors.New("cudaregress: execution spec version or plan identity mismatch")
	}
	if spec.Mode != "host_fixture" && spec.Mode != "gpu" {
		return errors.New("cudaregress: invalid execution mode")
	}
	if !validSHA256(spec.SpecDigest) {
		return errors.New("cudaregress: invalid execution spec digest")
	}
	copySpec := *spec
	digest, err := ComputeExecutionSpecDigest(copySpec)
	if err != nil {
		return fmt.Errorf("cudaregress: compute execution spec digest: %w", err)
	}
	if digest != spec.SpecDigest {
		return errors.New("cudaregress: execution spec digest mismatch")
	}
	if !filepath.IsAbs(spec.WorkingDirectory) || len(spec.WorkingDirectory) > maxCommandString {
		return errors.New("cudaregress: working directory must be an absolute bounded path")
	}
	info, err := os.Stat(spec.WorkingDirectory)
	if err != nil || !info.IsDir() {
		return errors.New("cudaregress: working directory unavailable")
	}
	if spec.TimeoutNanos <= 0 || spec.TimeoutNanos > int64(24*time.Hour) || spec.MaxOutputBytes <= 0 || spec.MaxOutputBytes > 16<<20 {
		return errors.New("cudaregress: invalid execution bounds")
	}
	if spec.Environment == nil || spec.Inputs == nil || spec.Libraries == nil || spec.Baseline.Args == nil || spec.Candidate.Args == nil {
		return errors.New("cudaregress: execution spec must declare environment, artifact arrays, and argv arrays")
	}
	if len(spec.Environment) > maxCommandEnv || len(spec.Inputs) > maxCommandArtifacts || len(spec.Libraries) > maxCommandArtifacts || len(spec.Baseline.Args) > maxCommandArgs || len(spec.Candidate.Args) > maxCommandArgs {
		return errors.New("cudaregress: execution spec collection limit exceeded")
	}
	for _, artifact := range append(append([]CommandArtifact(nil), spec.Inputs...), spec.Libraries...) {
		if artifact.Name == "" || len(artifact.Name) > maxCommandString || !filepath.IsAbs(artifact.Path) || len(artifact.Path) > maxCommandString || strings.ContainsRune(artifact.Path, 0) || !validSHA256(artifact.SHA256) {
			return errors.New("cudaregress: invalid input or library binding")
		}
	}
	for k, v := range spec.Environment {
		if k == "" || strings.ContainsAny(k, "=\x00") || strings.ContainsRune(v, 0) || len(k)+len(v) > maxCommandString {
			return errors.New("cudaregress: invalid execution environment entry")
		}
	}
	if err := validateCommandArm(plan, spec.Baseline); err != nil {
		return fmt.Errorf("baseline: %w", err)
	}
	if err := validateCommandArm(plan, spec.Candidate); err != nil {
		return fmt.Errorf("candidate: %w", err)
	}
	wantInputs := make([]artifactBinding, 0, len(plan.Inputs))
	for _, a := range plan.Inputs {
		wantInputs = append(wantInputs, artifactBinding{a.Name, a.Path, a.Digest})
	}
	gotInputs := make([]artifactBinding, 0, len(spec.Inputs))
	for _, a := range spec.Inputs {
		gotInputs = append(gotInputs, artifactBinding{a.Name, a.Path, a.SHA256})
	}
	if err := validateBindings(wantInputs, gotInputs); err != nil {
		return fmt.Errorf("inputs: %w", err)
	}
	libs := make([]artifactBinding, 0, len(plan.LibraryClosure))
	for _, l := range plan.LibraryClosure {
		libs = append(libs, artifactBinding{l.Name, l.Path, l.Digest})
	}
	gotLibs := make([]artifactBinding, 0, len(spec.Libraries))
	for _, l := range spec.Libraries {
		gotLibs = append(gotLibs, artifactBinding{l.Name, l.Path, l.SHA256})
	}
	if err := validateBindings(libs, gotLibs); err != nil {
		return fmt.Errorf("libraries: %w", err)
	}
	return nil
}

// ComputeExecutionSpecDigest returns the canonical spec digest. SpecDigest is
// ignored while hashing, so callers can set it to the returned value.
func ComputeExecutionSpecDigest(spec ExecutionSpec) (string, error) {
	spec.SpecDigest = ""
	b, err := json.Marshal(spec)
	if err != nil {
		return "", err
	}
	h := sha256.Sum256(b)
	return hex.EncodeToString(h[:]), nil
}

// OutputDigest hashes the canonical JSON encoding of a finite, nonempty output vector.
func OutputDigest(values []float64) (string, error) {
	if len(values) == 0 {
		return "", errors.New("cudaregress: empty output vector")
	}
	for _, v := range values {
		if math.IsNaN(v) || math.IsInf(v, 0) {
			return "", errors.New("cudaregress: non-finite output value")
		}
	}
	b, err := json.Marshal(values)
	if err != nil {
		return "", err
	}
	h := sha256.Sum256(b)
	return hex.EncodeToString(h[:]), nil
}

func validateCommandArm(plan *ExperimentPlan, arm CommandArm) error {
	if !filepath.IsAbs(arm.Path) || len(arm.Path) > maxCommandString || arm.Args == nil || len(arm.Args) > maxCommandArgs || !validSHA256(arm.SHA256) {
		return errors.New("invalid executable path, digest, or argument count")
	}
	found := false
	for _, b := range plan.BinaryArtifacts {
		if filepath.Clean(b.Path) == filepath.Clean(arm.Path) && b.Digest == arm.SHA256 {
			found = true
			break
		}
	}
	if !found {
		return errors.New("executable is not bound to a plan binary artifact")
	}
	for _, arg := range arm.Args {
		if strings.ContainsRune(arg, 0) || len(arg) > maxCommandString {
			return errors.New("invalid argument")
		}
	}
	d, _, err := ComputeFileDigest(arm.Path)
	if err != nil || d != arm.SHA256 {
		return errors.New("executable digest mismatch")
	}
	return nil
}

type artifactBinding struct{ name, path, digest string }

func validateBindings(want, got []artifactBinding) error {
	if len(want) != len(got) {
		return errors.New("artifact binding count mismatch")
	}
	key := func(x artifactBinding) string {
		return fmt.Sprintf("%s\x00%s\x00%s", x.name, filepath.Clean(x.path), x.digest)
	}
	w := make(map[string]bool, len(want))
	for _, x := range want {
		k := key(x)
		if w[k] {
			return errors.New("duplicate plan artifact")
		}
		w[k] = true
	}
	for _, x := range got {
		k := key(x)
		if !w[k] {
			return errors.New("missing, extra, or changed artifact binding")
		}
		delete(w, k)
	}
	if len(w) != 0 {
		return errors.New("missing artifact binding")
	}
	return nil
}

func validSHA256(s string) bool {
	b, err := hex.DecodeString(s)
	return err == nil && len(b) == sha256.Size && s == strings.ToLower(s)
}

// Attempts returns a copy of all child invocations attempted so far.
func (r *CommandReplayRunner) Attempts() []CommandAttempt {
	r.mu.Lock()
	defer r.mu.Unlock()
	attempts := append([]CommandAttempt(nil), r.attempts...)
	for i := range attempts {
		attempts[i].Result = cloneCommandResult(attempts[i].Result)
	}
	return attempts
}

func cloneCommandResult(result *CommandResult) *CommandResult {
	if result == nil {
		return nil
	}
	copyResult := *result
	copyResult.OutputValues = append([]float64(nil), result.OutputValues...)
	return &copyResult
}

// RunArm executes one declared arm with the spec's bounded timeout/output.
func (r *CommandReplayRunner) RunArm(ctx context.Context, armName string, round int) (cudatrial.Receipt, []float64, error) {
	if ctx == nil {
		return cudatrial.Receipt{}, nil, errors.New("cudaregress: nil command context")
	}
	arm := r.spec.Baseline
	if armName == "candidate" {
		arm = r.spec.Candidate
	} else if armName != "baseline" {
		return cudatrial.Receipt{}, nil, errors.New("cudaregress: unknown command arm")
	}
	if err := ValidatePlan(r.plan); err != nil {
		return cudatrial.Receipt{}, nil, err
	}
	if err := validateCommandArm(r.plan, arm); err != nil {
		return cudatrial.Receipt{}, nil, err
	}
	if err := r.verifyLock(); err != nil {
		return cudatrial.Receipt{}, nil, err
	}
	cmdCtx, cancel := context.WithTimeout(ctx, time.Duration(r.spec.TimeoutNanos))
	defer cancel()
	budget := &captureBudget{limit: r.spec.MaxOutputBytes, cancel: cancel}
	stdout := &boundedCapture{limit: r.spec.MaxOutputBytes, budget: budget}
	stderr := &boundedCapture{limit: r.spec.MaxOutputBytes, budget: budget}
	cmd := exec.CommandContext(cmdCtx, arm.Path, arm.Args...)
	cmd.Dir = r.spec.WorkingDirectory
	cmd.Env = sortedEnvironment(r.spec.Environment)
	cmd.Stdout, cmd.Stderr = stdout, stderr
	cmd.WaitDelay = time.Second
	started := time.Now()
	err := cmd.Run()
	finished := time.Now()
	if lockErr := r.verifyLock(); lockErr != nil {
		if err == nil {
			err = lockErr
		} else {
			err = errors.Join(err, lockErr)
		}
	}
	attempt := CommandAttempt{Arm: armName, Round: round, Started: started, Finished: finished, ExitCode: -1, StdoutBytes: stdout.total, StderrBytes: stderr.total, StdoutSHA256: stdout.digest(), StderrSHA256: stderr.digest(), Partial: err != nil}
	if cmd.ProcessState != nil {
		attempt.ExitCode = cmd.ProcessState.ExitCode()
	}
	if stdout.overflow || stderr.overflow {
		err = errors.New("child output limit exceeded")
	}
	if cmdCtx.Err() != nil {
		err = cmdCtx.Err()
	}
	var result CommandResult
	if err == nil {
		if parseErr := decodeCommandResult(stdout.buf.Bytes(), &result); parseErr != nil {
			err = parseErr
		}
	}
	if err == nil {
		err = validateCommandResult(r.plan, result)
	}
	var receipt cudatrial.Receipt
	if err == nil {
		attempt.Result = cloneCommandResult(&result)
		elapsed := finished.Sub(started)
		receipt = cudatrial.Receipt{Name: armName, Round: round, Complete: true, Digest: result.OutputSHA256, Started: started, Finished: finished, WallElapsed: elapsed, WallNs: elapsed.Nanoseconds(), Provenance: map[string]string{"plan_digest": r.plan.PlanDigest, "spec_digest": r.spec.SpecDigest, "mode": r.spec.Mode, "stdout_sha256": attempt.StdoutSHA256, "stderr_sha256": attempt.StderrSHA256}}
		if result.Timing.Status != "measured" {
			err = errors.New("child did not measure the requested timing channel")
		} else {
			duration := time.Duration(result.Timing.Count)
			if err = receipt.RecordTiming(result.Timing.Kind, duration); err == nil {
				receipt.TimingKind = result.Timing.Kind
			}
		}
	}
	if err != nil {
		attempt.Error = err.Error()
		attempt.Partial = true
		receipt = cudatrial.Receipt{Name: armName, Round: round, Complete: false, Err: err, Started: started, Finished: finished, Provenance: map[string]string{"plan_digest": r.plan.PlanDigest, "spec_digest": r.spec.SpecDigest, "mode": r.spec.Mode, "stdout_sha256": attempt.StdoutSHA256, "stderr_sha256": attempt.StderrSHA256}}
	}
	r.mu.Lock()
	r.attempts = append(r.attempts, attempt)
	r.mu.Unlock()
	if err != nil {
		return receipt, nil, err
	}
	return receipt, append([]float64(nil), result.OutputValues...), nil
}

func (r *CommandReplayRunner) verifyLock() error {
	if r.lockOwner == nil {
		return nil
	}
	dir, err := canonicalLockDir()
	if err != nil {
		return err
	}
	lookup := func(key string) string {
		if key == "GPU_LOCKDIR" {
			return dir
		}
		return os.Getenv(key)
	}
	owner, err := cudalock.Inherited(lookup)
	if err != nil {
		return fmt.Errorf("cudaregress: inherited GPU lock lost during replay: %w", err)
	}
	want := r.lockOwner
	if filepath.Clean(dir) != filepath.Clean(r.lockDir) || owner.PID != want.PID || owner.Host != want.Host || owner.Epoch != want.Epoch {
		return errors.New("cudaregress: inherited GPU lock owner changed during replay")
	}
	return nil
}

type captureBudget struct {
	mu       sync.Mutex
	limit    int
	total    int
	cancel   context.CancelFunc
	overflow bool
}

func (b *captureBudget) consume(n int) bool {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.total += n
	if b.total > b.limit {
		b.overflow = true
		b.cancel()
		return false
	}
	return true
}

type boundedCapture struct {
	limit    int
	buf      bytes.Buffer
	hash     hashWriter
	bytes    int
	total    int
	overflow bool
	budget   *captureBudget
}
type hashWriter interface {
	Write([]byte) (int, error)
	Sum([]byte) []byte
}

func (w *boundedCapture) Write(p []byte) (int, error) {
	if w.hash == nil {
		w.hash = sha256.New()
	}
	_, _ = w.hash.Write(p)
	w.total += len(p)
	if !w.budget.consume(len(p)) {
		w.overflow = true
		return len(p), errors.New("child combined output limit exceeded")
	}
	room := w.limit - w.bytes
	if room < 0 {
		room = 0
	}
	keep := len(p)
	if keep > room {
		keep = room
		w.overflow = true
	}
	if keep > 0 {
		_, _ = w.buf.Write(p[:keep])
		w.bytes += keep
	}
	if w.overflow {
		return len(p), errors.New("child output limit exceeded")
	}
	return len(p), nil
}
func (w *boundedCapture) digest() string {
	if w.hash == nil {
		return hex.EncodeToString(sha256.New().Sum(nil))
	}
	return hex.EncodeToString(w.hash.Sum(nil))
}

func sortedEnvironment(env map[string]string) []string {
	keys := make([]string, 0, len(env))
	for k := range env {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	out := make([]string, 0, len(keys))
	for _, k := range keys {
		out = append(out, k+"="+env[k])
	}
	return out
}

func validateCommandResult(plan *ExperimentPlan, result CommandResult) error {
	if result.Version != 1 || len(result.OutputValues) == 0 || !validSHA256(result.OutputSHA256) {
		return errors.New("invalid child result identity or output")
	}
	for _, v := range result.OutputValues {
		if math.IsNaN(v) || math.IsInf(v, 0) {
			return errors.New("child output contains non-finite value")
		}
	}
	digest, err := OutputDigest(result.OutputValues)
	if err != nil {
		return err
	}
	if digest != result.OutputSHA256 {
		return errors.New("child output digest mismatch")
	}
	t := result.Timing
	if t.Kind != plan.TimingScope.Channel || t.Unit != "ns" || t.Count < 0 || (t.Status != "measured" && t.Status != "unmeasured") {
		return errors.New("child timing does not match requested channel")
	}
	return nil
}

func decodeCommandResult(b []byte, out *CommandResult) error {
	if len(b) > maxCommandSpecBytes {
		return errors.New("child result byte limit exceeded")
	}
	if err := validateCommandResultNumbers(b); err != nil {
		return err
	}
	if err := checkStrictJSON(b, out); err != nil {
		return fmt.Errorf("decode child result: %w", err)
	}
	return nil
}

func validateCommandResultNumbers(b []byte) error {
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(b, &fields); err != nil {
		return err
	}
	var values []json.RawMessage
	rawValues, hasValues := fields["output_values"]
	if !hasValues || string(rawValues) == "null" {
		return errors.New("child output_values is required")
	}
	if err := json.Unmarshal(rawValues, &values); err != nil {
		return errors.New("child output_values must be an array of numbers")
	}
	for _, raw := range values {
		var value any
		d := json.NewDecoder(bytes.NewReader(raw))
		d.UseNumber()
		if err := d.Decode(&value); err != nil {
			return err
		}
		if _, ok := value.(json.Number); !ok {
			return errors.New("child output_values must contain numbers")
		}
	}
	var timing map[string]json.RawMessage
	rawTiming, hasTiming := fields["timing"]
	if !hasTiming || string(rawTiming) == "null" {
		return errors.New("child timing is required")
	}
	if err := json.Unmarshal(rawTiming, &timing); err != nil {
		return errors.New("child timing must be an object")
	}
	rawCount, ok := timing["count"]
	if !ok || string(rawCount) == "null" {
		return errors.New("child timing count is required")
	}
	var count json.Number
	d := json.NewDecoder(bytes.NewReader(rawCount))
	d.UseNumber()
	if err := d.Decode(&count); err != nil {
		return errors.New("child timing count must be an integer")
	}
	if _, err := count.Int64(); err != nil {
		return errors.New("child timing count must be an integer")
	}
	return nil
}

// ReadExecutionSpec reads one bounded, exact-schema JSON spec.
func ReadExecutionSpec(r io.Reader) (ExecutionSpec, error) {
	var spec ExecutionSpec
	b, err := io.ReadAll(io.LimitReader(r, maxCommandSpecBytes+1))
	if err != nil {
		return spec, err
	}
	if len(b) > maxCommandSpecBytes {
		return spec, errors.New("cudaregress: execution spec byte limit exceeded")
	}
	if err := checkExecutionSpecBounds(b); err != nil {
		return ExecutionSpec{}, fmt.Errorf("cudaregress: execution spec bounds: %w", err)
	}
	if err := checkStrictJSON(b, &spec); err != nil {
		return ExecutionSpec{}, fmt.Errorf("cudaregress: decode execution spec: %w", err)
	}
	return spec, nil
}

func checkExecutionSpecBounds(b []byte) error {
	var value any
	d := json.NewDecoder(bytes.NewReader(b))
	d.UseNumber()
	if err := d.Decode(&value); err != nil {
		return err
	}
	var check func(any, string) error
	check = func(v any, path string) error {
		switch x := v.(type) {
		case string:
			if len(x) > maxCommandString {
				return fmt.Errorf("string too long at %s", path)
			}
		case []any:
			limit := 0
			switch path {
			case "baseline.args", "candidate.args":
				limit = maxCommandArgs
			case "inputs", "libraries":
				limit = maxCommandArtifacts
			}
			if limit > 0 && len(x) > limit {
				return fmt.Errorf("collection too large at %s", path)
			}
			for i, item := range x {
				if err := check(item, fmt.Sprintf("%s[%d]", path, i)); err != nil {
					return err
				}
			}
		case map[string]any:
			if path == "environment" && len(x) > maxCommandEnv {
				return errors.New("environment entry limit exceeded")
			}
			for k, item := range x {
				if len(k) > maxCommandString {
					return fmt.Errorf("key too long at %s", path)
				}
				child := k
				if path != "" {
					child = path + "." + k
				}
				if path == "environment" {
					if value, ok := item.(string); ok && len(k)+len(value) > maxCommandString {
						return fmt.Errorf("environment entry too long at %s", child)
					}
				}
				if err := check(item, child); err != nil {
					return err
				}
			}
		}
		return nil
	}
	return check(value, "")
}

// checkStrictJSON rejects duplicate/case-alias keys, unknown fields and trailing data.
func checkStrictJSON(b []byte, out any) error {
	d := json.NewDecoder(bytes.NewReader(b))
	d.UseNumber()
	if err := scanJSONValue(d, reflect.TypeOf(out).Elem(), 0); err != nil {
		return err
	}
	if _, err := d.Token(); err != io.EOF {
		return errors.New("trailing JSON data")
	}
	d = json.NewDecoder(bytes.NewReader(b))
	d.DisallowUnknownFields()
	if err := d.Decode(out); err != nil {
		return err
	}
	if d.Decode(new(any)) != io.EOF {
		return errors.New("trailing JSON data")
	}
	return nil
}

func scanJSONValue(d *json.Decoder, typ reflect.Type, depth int) error {
	if depth > 64 {
		return errors.New("JSON nesting limit exceeded")
	}
	for typ.Kind() == reflect.Pointer {
		typ = typ.Elem()
	}
	jsonUnmarshaler := reflect.TypeOf((*json.Unmarshaler)(nil)).Elem()
	if reflect.PointerTo(typ).Implements(jsonUnmarshaler) || typ.Implements(jsonUnmarshaler) {
		var raw json.RawMessage
		if err := d.Decode(&raw); err != nil {
			return err
		}
		if err := scanJSONStructure(raw, depth); err != nil {
			return err
		}
		value := reflect.New(typ).Interface()
		return json.Unmarshal(raw, value)
	}
	tok, err := d.Token()
	if err != nil {
		return err
	}
	delim, ok := tok.(json.Delim)
	if !ok {
		return nil
	}
	switch delim {
	case '{':
		fields := map[string]reflect.Type{}
		if typ.Kind() == reflect.Struct {
			for i := 0; i < typ.NumField(); i++ {
				f := typ.Field(i)
				if f.PkgPath != "" {
					continue
				}
				name := strings.Split(f.Tag.Get("json"), ",")[0]
				if name == "" {
					name = f.Name
				}
				if name != "-" {
					fields[name] = f.Type
				}
			}
		}
		seen := map[string]bool{}
		for d.More() {
			k, err := d.Token()
			if err != nil {
				return err
			}
			key, ok := k.(string)
			if !ok {
				return errors.New("object key is not a string")
			}
			fold := strings.ToLower(key)
			if seen[fold] {
				return errors.New("duplicate or case-alias JSON key")
			}
			seen[fold] = true
			ft, ok := fields[key]
			if !ok {
				if typ.Kind() == reflect.Map {
					ft = typ.Elem()
				} else {
					return fmt.Errorf("unknown or case-alias JSON field %q", key)
				}
			}
			if err := scanJSONValue(d, ft, depth+1); err != nil {
				return err
			}
		}
		_, err = d.Token()
		return err
	case '[':
		et := reflect.TypeOf((*any)(nil)).Elem()
		if typ.Kind() == reflect.Slice || typ.Kind() == reflect.Array {
			et = typ.Elem()
		}
		for d.More() {
			if err := scanJSONValue(d, et, depth+1); err != nil {
				return err
			}
		}
		_, err = d.Token()
		return err
	default:
		return errors.New("unexpected JSON delimiter")
	}
}

func scanJSONStructure(b []byte, depth int) error {
	d := json.NewDecoder(bytes.NewReader(b))
	var walk func(int) error
	walk = func(level int) error {
		if level > 64 {
			return errors.New("JSON nesting limit exceeded")
		}
		tok, err := d.Token()
		if err != nil {
			return err
		}
		delim, ok := tok.(json.Delim)
		if !ok {
			return nil
		}
		switch delim {
		case '{':
			seen := make(map[string]bool)
			for d.More() {
				token, err := d.Token()
				if err != nil {
					return err
				}
				key, ok := token.(string)
				if !ok {
					return errors.New("object key is not a string")
				}
				fold := strings.ToLower(key)
				if seen[fold] {
					return errors.New("duplicate or case-alias JSON key")
				}
				seen[fold] = true
				if err := walk(level + 1); err != nil {
					return err
				}
			}
			_, err = d.Token()
			return err
		case '[':
			for d.More() {
				if err := walk(level + 1); err != nil {
					return err
				}
			}
			_, err = d.Token()
			return err
		default:
			return errors.New("unexpected JSON delimiter")
		}
	}
	if err := walk(depth); err != nil {
		return err
	}
	if _, err := d.Token(); err != io.EOF {
		return errors.New("trailing JSON data")
	}
	return nil
}
