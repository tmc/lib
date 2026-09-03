// Package inferbench validates matched inference benchmark receipts.
package inferbench

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"sort"
	"strings"
	"time"
)

const SchemaVersion = 3

// Campaign contains every sample in one matched benchmark campaign.
type Campaign struct {
	SchemaVersion int               `json:"schema_version"`
	RandomSeed    uint64            `json:"random_seed"`
	Runs          []Run             `json:"runs"`
	Order         []ScheduledSample `json:"order"`
}

// Run identifies one implementation and its measured samples.
type Run struct {
	Arm      string   `json:"arm"`
	Identity Identity `json:"identity"`
	Workload Workload `json:"workload"`
	Evidence Evidence `json:"evidence"`
	Samples  []Sample `json:"samples"`
}

// Identity pins the executable, model, tokenizer, backend, precision, and device.
type Identity struct {
	BinarySHA256    string `json:"binary_sha256"`
	ModelSHA256     string `json:"model_sha256"`
	TokenizerSHA256 string `json:"tokenizer_sha256"`
	ConfigSHA256    string `json:"config_sha256"`
	SourceSHA256    string `json:"source_sha256"`
	Backend         string `json:"backend"`
	Precision       string `json:"precision"`
	DeviceID        string `json:"device_id"`
}

// Workload describes the work that every arm must perform identically.
type Workload struct {
	PromptTokenIDs  [][]int32 `json:"prompt_token_ids"`
	ContextTokens   int       `json:"context_tokens"`
	GeneratedTokens int       `json:"generated_tokens"`
	Concurrency     int       `json:"concurrency"`
	WarmupRuns      int       `json:"warmup_runs"`
	Sampler         string    `json:"sampler"`
	CacheState      string    `json:"cache_state"`
}

// Evidence records conditions needed to interpret timing as GPU execution.
type Evidence struct {
	HardwareVerified     bool          `json:"hardware_verified"`
	DeviceResidentKV     bool          `json:"device_resident_kv"`
	CPUFallback          bool          `json:"cpu_fallback"`
	TimingBoundary       string        `json:"timing_boundary"`
	DeviceSynchronized   bool          `json:"device_synchronized"`
	IncludesQueueing     bool          `json:"includes_queueing"`
	RunnerLease          string        `json:"runner_lease"`
	DeviceEvidenceSHA256 string        `json:"device_evidence_sha256"`
	NullControlSHA256    string        `json:"null_control_sha256"`
	NullControlSamples   int           `json:"null_control_samples"`
	NullControlP95       time.Duration `json:"null_control_p95_ns"`
	NullControlLimit     time.Duration `json:"null_control_limit_ns"`
	PersistentRuntime    bool          `json:"persistent_runtime"`
	CompilationExcluded  bool          `json:"compilation_excluded"`
	WarmupRunsCompleted  int           `json:"warmup_runs_completed"`
}

// Sample contains one completed, synchronized serving observation.
type Sample struct {
	ID                string            `json:"id"`
	StartedAt         time.Time         `json:"started_at"`
	EndedAt           time.Time         `json:"ended_at"`
	TTFT              time.Duration     `json:"ttft_ns"`
	TPOT              time.Duration     `json:"tpot_ns"`
	InterToken        []time.Duration   `json:"inter_token_ns"`
	TokensPerSecond   float64           `json:"tokens_per_second"`
	Dispatches        uint64            `json:"dispatches"`
	DispatchMeasured  bool              `json:"dispatch_measured"`
	CacheHits         uint64            `json:"cache_hits"`
	CacheMisses       uint64            `json:"cache_misses"`
	TierReadBytes     map[string]uint64 `json:"tier_read_bytes"`
	TierWriteBytes    map[string]uint64 `json:"tier_write_bytes"`
	TierBytesMeasured bool              `json:"tier_bytes_measured"`
	GeneratedTokenIDs [][]int32         `json:"generated_token_ids"`
	OutputSHA256      string            `json:"output_sha256"`
	Requests          []RequestTiming   `json:"requests"`
}

// RequestTiming preserves the submission and token-arrival boundary for one
// request within a concurrent sample.
type RequestTiming struct {
	StartedAt time.Time   `json:"started_at"`
	TokenAt   []time.Time `json:"token_at"`
	EndedAt   time.Time   `json:"ended_at"`
}

// ScheduledSample names a sample in its required execution order.
type ScheduledSample struct {
	Arm string `json:"arm"`
	ID  string `json:"id"`
}

// Validate rejects receipts that cannot support a matched GPU performance claim.
func (c *Campaign) Validate() error {
	if c.SchemaVersion != SchemaVersion {
		return fmt.Errorf("schema version %d", c.SchemaVersion)
	}
	if c.RandomSeed == 0 {
		return errors.New("zero random seed")
	}
	if len(c.Runs) < 2 {
		return errors.New("fewer than two arms")
	}
	seenArms := make(map[string]bool)
	var workload Workload
	var identity Identity
	var outputs map[string]string
	var runnerLease string
	var nullControlSHA256 string
	for i := range c.Runs {
		r := &c.Runs[i]
		if r.Arm == "" || seenArms[r.Arm] {
			return fmt.Errorf("invalid arm %q", r.Arm)
		}
		seenArms[r.Arm] = true
		if i == 0 {
			workload = r.Workload
			identity = r.Identity
			runnerLease = r.Evidence.RunnerLease
			nullControlSHA256 = r.Evidence.NullControlSHA256
		} else if !reflect.DeepEqual(workload, r.Workload) {
			return fmt.Errorf("arm %q: unequal workload", r.Arm)
		}
		if i > 0 && (r.Identity.ModelSHA256 != identity.ModelSHA256 || r.Identity.TokenizerSHA256 != identity.TokenizerSHA256 || r.Identity.ConfigSHA256 != identity.ConfigSHA256 || r.Identity.Precision != identity.Precision) {
			return fmt.Errorf("arm %q: unequal model, tokenizer, config, or precision", r.Arm)
		}
		if i > 0 && r.Evidence.RunnerLease != runnerLease {
			return fmt.Errorf("arm %q: unequal runner lease", r.Arm)
		}
		if i > 0 && r.Evidence.NullControlSHA256 != nullControlSHA256 {
			return fmt.Errorf("arm %q: unequal null control", r.Arm)
		}
		if err := validateRun(r); err != nil {
			return fmt.Errorf("arm %q: %w", r.Arm, err)
		}
		if i == 0 {
			outputs = sampleOutputs(r.Samples)
		} else if !reflect.DeepEqual(outputs, sampleOutputs(r.Samples)) {
			return fmt.Errorf("arm %q: output hashes differ", r.Arm)
		}
	}
	return c.validateOrder()
}

func validateRun(r *Run) error {
	ids := []string{r.Identity.BinarySHA256, r.Identity.ModelSHA256, r.Identity.TokenizerSHA256, r.Identity.ConfigSHA256, r.Identity.SourceSHA256}
	for _, sum := range ids {
		if !validSHA256(sum) {
			return errors.New("invalid identity hash")
		}
	}
	if r.Identity.Backend != "metal" && r.Identity.Backend != "cuda" {
		return fmt.Errorf("unsupported backend %q", r.Identity.Backend)
	}
	if r.Identity.Precision == "" || r.Identity.DeviceID == "" {
		return errors.New("incomplete identity")
	}
	w := r.Workload
	if len(w.PromptTokenIDs) == 0 || w.ContextTokens <= 0 || w.GeneratedTokens < 2 || w.Concurrency <= 0 || w.WarmupRuns <= 0 || w.Sampler == "" || w.CacheState != "cold" {
		return errors.New("incomplete workload")
	}
	for _, p := range w.PromptTokenIDs {
		if len(p) != w.ContextTokens {
			return errors.New("prompt token count differs from context tokens")
		}
	}
	e := r.Evidence
	if !e.HardwareVerified || !e.DeviceResidentKV || e.CPUFallback {
		return errors.New("hardware path not proven")
	}
	if e.TimingBoundary != "submit-to-final-device-synchronize" || !e.DeviceSynchronized || !e.IncludesQueueing || e.RunnerLease == "" || !validSHA256(e.DeviceEvidenceSHA256) {
		return errors.New("unsafe timing boundary")
	}
	if !e.PersistentRuntime || !e.CompilationExcluded || e.WarmupRunsCompleted != w.WarmupRuns {
		return errors.New("compile or warmup boundary not proven")
	}
	if !validSHA256(e.NullControlSHA256) || e.NullControlSamples < 20 || e.NullControlP95 <= 0 || e.NullControlLimit <= 0 || e.NullControlP95 > e.NullControlLimit {
		return errors.New("host qualification not proven")
	}
	if len(r.Samples) < 20 {
		return fmt.Errorf("only %d samples", len(r.Samples))
	}
	seen := make(map[string]bool)
	for i := range r.Samples {
		s := &r.Samples[i]
		if s.ID == "" || seen[s.ID] {
			return errors.New("invalid sample id")
		}
		seen[s.ID] = true
		if s.StartedAt.IsZero() || !s.EndedAt.After(s.StartedAt) {
			return fmt.Errorf("sample %q: invalid interval", s.ID)
		}
		if s.TTFT <= 0 || s.TPOT <= 0 || len(s.InterToken) != w.GeneratedTokens-1 || s.TokensPerSecond <= 0 || math.IsNaN(s.TokensPerSecond) || math.IsInf(s.TokensPerSecond, 0) || s.Dispatches == 0 || !s.DispatchMeasured {
			return fmt.Errorf("sample %q: incomplete timing", s.ID)
		}
		for _, d := range s.InterToken {
			if d <= 0 {
				return fmt.Errorf("sample %q: invalid inter-token latency", s.ID)
			}
		}
		if !s.TierBytesMeasured || len(s.TierReadBytes) == 0 || len(s.TierWriteBytes) == 0 || len(s.GeneratedTokenIDs) != len(w.PromptTokenIDs) || !validSHA256(s.OutputSHA256) {
			return fmt.Errorf("sample %q: incomplete telemetry", s.ID)
		}
		if len(s.Requests) != len(w.PromptTokenIDs) {
			return fmt.Errorf("sample %q: request timing count differs", s.ID)
		}
		for _, request := range s.Requests {
			if request.StartedAt.Before(s.StartedAt) || request.EndedAt.After(s.EndedAt) || request.EndedAt.Before(request.StartedAt) || len(request.TokenAt) != w.GeneratedTokens {
				return fmt.Errorf("sample %q: invalid request timing", s.ID)
			}
			last := request.StartedAt
			for _, at := range request.TokenAt {
				if at.Before(last) || at.After(request.EndedAt) {
					return fmt.Errorf("sample %q: unordered token timing", s.ID)
				}
				last = at
			}
		}
		for _, tokens := range s.GeneratedTokenIDs {
			if len(tokens) != w.GeneratedTokens {
				return fmt.Errorf("sample %q: generated token count differs", s.ID)
			}
		}
		sum, err := HashJSON(s.GeneratedTokenIDs)
		if err != nil || sum != s.OutputSHA256 {
			return fmt.Errorf("sample %q: incorrect output hash", s.ID)
		}
	}
	return nil
}

func sampleOutputs(samples []Sample) map[string]string {
	out := make(map[string]string, len(samples))
	for _, s := range samples {
		out[s.ID] = s.OutputSHA256
	}
	return out
}

func (c *Campaign) validateOrder() error {
	want := make(map[ScheduledSample]bool)
	intervals := make(map[ScheduledSample]Sample)
	for _, r := range c.Runs {
		for _, s := range r.Samples {
			k := ScheduledSample{r.Arm, s.ID}
			want[k] = true
			intervals[k] = s
		}
	}
	if len(c.Order) != len(want) {
		return errors.New("schedule does not cover every sample")
	}
	got := make(map[ScheduledSample]bool)
	var last time.Time
	for _, k := range c.Order {
		s, ok := intervals[k]
		if !ok || got[k] {
			return errors.New("schedule contains missing or duplicate sample")
		}
		got[k] = true
		if !last.IsZero() && s.StartedAt.Before(last) {
			return errors.New("sample runners overlap or violate schedule")
		}
		last = s.EndedAt
	}
	wantOrder := Schedule(c.Runs, c.RandomSeed)
	if !reflect.DeepEqual(c.Order, wantOrder) {
		return errors.New("schedule is not the seeded randomized order")
	}
	return nil
}

// Schedule returns a deterministic, balanced randomized order for the runs.
func Schedule(runs []Run, seed uint64) []ScheduledSample {
	var out []ScheduledSample
	for _, r := range runs {
		for _, s := range r.Samples {
			out = append(out, ScheduledSample{r.Arm, s.ID})
		}
	}
	x := seed
	for i := len(out) - 1; i > 0; i-- {
		x ^= x << 13
		x ^= x >> 7
		x ^= x << 17
		j := int(x % uint64(i+1))
		out[i], out[j] = out[j], out[i]
	}
	return out
}

// HashJSON returns the SHA-256 of the canonical encoding/json representation of v.
func HashJSON(v any) (string, error) {
	b, err := json.Marshal(v)
	if err != nil {
		return "", err
	}
	sum := sha256.Sum256(b)
	return hex.EncodeToString(sum[:]), nil
}

// HashFile returns the SHA-256 of a regular file.
func HashFile(path string) (string, error) {
	f, err := os.Open(path)
	if err != nil {
		return "", fmt.Errorf("hash file: %w", err)
	}
	defer f.Close()
	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		return "", fmt.Errorf("hash file: %w", err)
	}
	return hex.EncodeToString(h.Sum(nil)), nil
}

// HashPath hashes a regular file, or a directory as sorted relative names,
// modes, sizes, and file contents. Symlinks and non-regular files are rejected.
func HashPath(path string) (string, error) {
	info, err := os.Lstat(path)
	if err != nil {
		return "", fmt.Errorf("hash path: %w", err)
	}
	if info.Mode().IsRegular() {
		return HashFile(path)
	}
	if info.Mode()&os.ModeSymlink != 0 {
		target, err := os.Stat(path)
		if err != nil || !target.Mode().IsRegular() {
			return "", errors.New("hash path: symlink does not resolve to a regular file")
		}
		return HashFile(path)
	}
	if !info.IsDir() {
		return "", errors.New("hash path: not a regular file or directory")
	}
	var names []string
	err = filepath.WalkDir(path, func(name string, entry os.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if name == path {
			return nil
		}
		if entry.Type()&os.ModeSymlink != 0 {
			info, err := os.Stat(name)
			if err != nil || !info.Mode().IsRegular() {
				return fmt.Errorf("hash path: symlink %s does not resolve to a regular file", name)
			}
		}
		if entry.Type().IsRegular() || entry.Type()&os.ModeSymlink != 0 {
			rel, err := filepath.Rel(path, name)
			if err != nil {
				return err
			}
			names = append(names, filepath.ToSlash(rel))
		}
		return nil
	})
	if err != nil {
		return "", fmt.Errorf("hash path: %w", err)
	}
	sort.Strings(names)
	h := sha256.New()
	for _, name := range names {
		full := filepath.Join(path, filepath.FromSlash(name))
		info, err := os.Stat(full)
		if err != nil {
			return "", fmt.Errorf("hash path: %w", err)
		}
		fmt.Fprintf(h, "%s\x00%d\x00%s\x00", name, info.Size(), strings.TrimSpace(info.Mode().String()))
		f, err := os.Open(full)
		if err != nil {
			return "", fmt.Errorf("hash path: %w", err)
		}
		_, copyErr := io.Copy(h, f)
		closeErr := f.Close()
		if copyErr != nil {
			return "", fmt.Errorf("hash path: %w", copyErr)
		}
		if closeErr != nil {
			return "", fmt.Errorf("hash path: %w", closeErr)
		}
	}
	return hex.EncodeToString(h.Sum(nil)), nil
}

// WriteJSONAtomic writes v only after its complete JSON encoding is durable in
// a sibling temporary file. It never exposes a partial benchmark receipt.
func WriteJSONAtomic(path string, v any) error {
	dir := filepath.Dir(path)
	f, err := os.CreateTemp(dir, ".inferbench-*")
	if err != nil {
		return fmt.Errorf("create receipt: %w", err)
	}
	tmp := f.Name()
	ok := false
	defer func() {
		f.Close()
		if !ok {
			os.Remove(tmp)
		}
	}()
	enc := json.NewEncoder(f)
	enc.SetIndent("", "  ")
	if err := enc.Encode(v); err != nil {
		return fmt.Errorf("encode receipt: %w", err)
	}
	if err := f.Sync(); err != nil {
		return fmt.Errorf("sync receipt: %w", err)
	}
	if err := f.Close(); err != nil {
		return fmt.Errorf("close receipt: %w", err)
	}
	if err := os.Rename(tmp, path); err != nil {
		return fmt.Errorf("publish receipt: %w", err)
	}
	ok = true
	return nil
}

func validSHA256(s string) bool {
	b, err := hex.DecodeString(s)
	return err == nil && len(b) == sha256.Size
}

// Lease prevents two local benchmark runners from sharing a host receipt path.
type Lease struct{ path string }

// AcquireLease atomically acquires path and fails when another runner owns it.
func AcquireLease(path string) (*Lease, error) {
	f, err := os.OpenFile(path, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0600)
	if err != nil {
		return nil, fmt.Errorf("acquire runner lease: %w", err)
	}
	if _, err := fmt.Fprintf(f, "%d\n", os.Getpid()); err != nil {
		f.Close()
		os.Remove(path)
		return nil, fmt.Errorf("write runner lease: %w", err)
	}
	if err := f.Close(); err != nil {
		os.Remove(path)
		return nil, fmt.Errorf("close runner lease: %w", err)
	}
	return &Lease{path: path}, nil
}

// Close releases the runner lease.
func (l *Lease) Close() error {
	if l == nil || l.path == "" {
		return nil
	}
	path := l.path
	l.path = ""
	return os.Remove(path)
}

// SortRuns sorts runs by arm for stable serialization before execution.
func SortRuns(runs []Run) { sort.Slice(runs, func(i, j int) bool { return runs[i].Arm < runs[j].Arm }) }
