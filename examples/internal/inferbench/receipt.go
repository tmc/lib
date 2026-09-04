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

const SchemaVersion = 7

const (
	PersistentEnginePolicy = "one-compiled-engine-warmed-per-arm"
	ColdIntervalPolicy     = "recreate-serving-prefix-runtime-and-clear-host-prefix-cache-before-each-interval"
	ThroughputOutputMode   = "generated-tokens"
	ObservedOutputMode     = "full-logits-and-populated-kv-v1"
)

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
	OutputMode      string    `json:"output_mode"`
	FullLogits      int       `json:"full_logits_elements,omitempty"`
	KVLayers        int       `json:"kv_layers,omitempty"`
	KVPerToken      int       `json:"kv_elements_per_token,omitempty"`
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
	RuntimePolicy        string        `json:"runtime_policy"`
	AllocationPolicy     string        `json:"allocation_policy"`
	OutputContract       string        `json:"output_contract"`
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
	PhysicalWaves     []PhysicalWave    `json:"physical_waves"`
	GeneratedTokenIDs [][]int32         `json:"generated_token_ids"`
	OutputSHA256      string            `json:"output_sha256"`
	ObservedOutputs   []ObservedOutput  `json:"observed_outputs,omitempty"`
	ObservedSHA256    string            `json:"observed_sha256,omitempty"`
	Requests          []RequestTiming   `json:"requests"`
}

// ObservedOutput identifies the full final logits and populated all-layer KV
// state retained for one request before its serving row is released.
type ObservedOutput struct {
	Request             int               `json:"request"`
	FullLogitsSHA256    string            `json:"full_logits_sha256"`
	FullLogitsElements  int               `json:"full_logits_elements"`
	PopulatedKVSHA256   string            `json:"populated_kv_sha256"`
	PopulatedKVElements int               `json:"populated_kv_elements"`
	PopulatedTokens     int               `json:"populated_tokens"`
	Layers              []ObservedKVLayer `json:"layers"`
}

// ObservedKVLayer identifies one populated K/V layer in an observed output.
type ObservedKVLayer struct {
	Layer        int    `json:"layer"`
	KeySHA256    string `json:"key_sha256"`
	ValueSHA256  string `json:"value_sha256"`
	ElementsEach int    `json:"elements_each"`
}

// PhysicalWave attributes one completed device wave to all participating
// requests. Dispatches are wave totals and must not be repeated per request.
type PhysicalWave struct {
	Sequence       uint64            `json:"sequence"`
	Kind           string            `json:"kind"`
	Requests       []string          `json:"requests"`
	Dispatches     uint64            `json:"dispatches"`
	TierReadBytes  map[string]uint64 `json:"tier_read_bytes"`
	TierWriteBytes map[string]uint64 `json:"tier_write_bytes"`
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
	if err := validateRunHeader(r); err != nil {
		return err
	}
	w := r.Workload
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
		if err := validateSample(s, w); err != nil {
			return err
		}
	}
	return nil
}

func validateRunHeader(r *Run) error {
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
	if len(w.PromptTokenIDs) == 0 || len(w.PromptTokenIDs) != w.Concurrency || w.ContextTokens <= 0 || w.GeneratedTokens < 2 || w.Concurrency <= 0 || w.WarmupRuns <= 0 || w.Sampler != "greedy-argmax" || w.CacheState != "cold" || w.OutputMode != ThroughputOutputMode && w.OutputMode != ObservedOutputMode {
		return errors.New("incomplete workload")
	}
	for _, p := range w.PromptTokenIDs {
		if len(p) != w.ContextTokens {
			return errors.New("prompt token count differs from context tokens")
		}
	}
	if w.OutputMode == ThroughputOutputMode && (w.FullLogits != 0 || w.KVLayers != 0 || w.KVPerToken != 0) {
		return errors.New("observed shape contaminates throughput mode")
	}
	if w.OutputMode == ObservedOutputMode && (w.FullLogits <= 0 || w.KVLayers <= 0 || w.KVPerToken <= 0) {
		return errors.New("incomplete observed output shape")
	}
	if w.OutputMode == ObservedOutputMode {
		if _, _, _, ok := observedShape(w); !ok {
			return errors.New("invalid observed output shape")
		}
	}
	e := r.Evidence
	if !e.HardwareVerified || !e.DeviceResidentKV || e.CPUFallback {
		return errors.New("hardware path not proven")
	}
	wantBoundary := "submit-to-final-device-synchronize"
	if w.OutputMode == ObservedOutputMode {
		wantBoundary = "submit-to-observed-output-digest-and-final-device-synchronize"
	}
	if e.TimingBoundary != wantBoundary || !e.DeviceSynchronized || !e.IncludesQueueing || e.RunnerLease == "" || !validSHA256(e.DeviceEvidenceSHA256) {
		return errors.New("unsafe timing boundary")
	}
	if !e.PersistentRuntime || !e.CompilationExcluded || e.WarmupRunsCompleted != w.WarmupRuns {
		return errors.New("compile or warmup boundary not proven")
	}
	if e.RuntimePolicy != PersistentEnginePolicy || e.AllocationPolicy != ColdIntervalPolicy {
		return errors.New("runtime or allocation policy not proven")
	}
	if e.OutputContract != w.OutputMode {
		return errors.New("output contract not proven")
	}
	if !validSHA256(e.NullControlSHA256) || e.NullControlSamples < 20 || e.NullControlP95 <= 0 || e.NullControlLimit <= 0 || e.NullControlP95 > e.NullControlLimit {
		return errors.New("host qualification not proven")
	}
	return nil
}

func validateSample(s *Sample, w Workload) error {
	if s.StartedAt.IsZero() || !s.EndedAt.After(s.StartedAt) {
		return fmt.Errorf("sample %q: invalid interval", s.ID)
	}
	elapsed := s.EndedAt.Sub(s.StartedAt)
	if s.TTFT <= 0 || s.TPOT <= 0 || len(s.InterToken) != w.GeneratedTokens-1 || s.TokensPerSecond <= 0 || math.IsNaN(s.TokensPerSecond) || math.IsInf(s.TokensPerSecond, 0) || s.Dispatches == 0 || !s.DispatchMeasured {
		return fmt.Errorf("sample %q: incomplete timing", s.ID)
	}
	wantTokensPerSecond := float64(len(w.PromptTokenIDs)*w.GeneratedTokens) / elapsed.Seconds()
	if s.TTFT > elapsed || s.TPOT > elapsed || math.Abs(s.TokensPerSecond-wantTokensPerSecond) > wantTokensPerSecond*1e-9 {
		return fmt.Errorf("sample %q: timing exceeds or differs from completion interval", s.ID)
	}
	for _, d := range s.InterToken {
		if d <= 0 || d > elapsed {
			return fmt.Errorf("sample %q: invalid inter-token latency", s.ID)
		}
	}
	if !s.TierBytesMeasured || len(s.TierReadBytes) == 0 || len(s.TierWriteBytes) == 0 || len(s.GeneratedTokenIDs) != len(w.PromptTokenIDs) || !validSHA256(s.OutputSHA256) {
		return fmt.Errorf("sample %q: incomplete telemetry", s.ID)
	}
	if err := validatePhysicalWaves(s, len(w.PromptTokenIDs)); err != nil {
		return fmt.Errorf("sample %q: %w", s.ID, err)
	}
	if err := validateObservedOutputs(s, w); err != nil {
		return fmt.Errorf("sample %q: %w", s.ID, err)
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
	ttft, tpot, interToken := requestMetrics(s.Requests, w.GeneratedTokens)
	if s.TTFT != ttft || s.TPOT != tpot || !reflect.DeepEqual(s.InterToken, interToken) {
		return fmt.Errorf("sample %q: summary timing differs from request timing", s.ID)
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
	return nil
}

func requestMetrics(requests []RequestTiming, generatedTokens int) (ttft, tpot time.Duration, interToken []time.Duration) {
	interToken = make([]time.Duration, generatedTokens-1)
	for _, request := range requests {
		ttft += request.TokenAt[0].Sub(request.StartedAt)
		for i := range interToken {
			d := request.TokenAt[i+1].Sub(request.TokenAt[i])
			tpot += d
			interToken[i] += d
		}
	}
	ttft /= time.Duration(len(requests))
	tpot /= time.Duration(len(requests) * (generatedTokens - 1))
	for i := range interToken {
		interToken[i] /= time.Duration(len(requests))
	}
	return ttft, tpot, interToken
}

func validateObservedOutputs(s *Sample, w Workload) error {
	if w.OutputMode == ThroughputOutputMode {
		if len(s.ObservedOutputs) != 0 || s.ObservedSHA256 != "" {
			return errors.New("observed output contaminates throughput mode")
		}
		return nil
	}
	populatedTokens, populatedElements, layerElements, ok := observedShape(w)
	if !ok || len(s.ObservedOutputs) != w.Concurrency || !validSHA256(s.ObservedSHA256) {
		return errors.New("incomplete observed output")
	}
	seen := make(map[int]bool)
	for _, output := range s.ObservedOutputs {
		if output.Request < 0 || output.Request >= w.Concurrency || seen[output.Request] ||
			!validSHA256(output.FullLogitsSHA256) || output.FullLogitsElements != w.FullLogits ||
			!validSHA256(output.PopulatedKVSHA256) || output.PopulatedKVElements != populatedElements ||
			output.PopulatedTokens != populatedTokens || len(output.Layers) != w.KVLayers {
			return errors.New("invalid observed output")
		}
		for layer, got := range output.Layers {
			if got.Layer != layer || !validSHA256(got.KeySHA256) || !validSHA256(got.ValueSHA256) || got.ElementsEach != layerElements {
				return errors.New("invalid observed KV layer")
			}
		}
		kvSum, err := HashJSON(output.Layers)
		if err != nil || kvSum != output.PopulatedKVSHA256 {
			return errors.New("incorrect observed KV hash")
		}
		seen[output.Request] = true
	}
	sum, err := HashJSON(s.ObservedOutputs)
	if err != nil || sum != s.ObservedSHA256 {
		return errors.New("incorrect observed output hash")
	}
	return nil
}

func observedShape(w Workload) (tokens, elements, layerElements int, ok bool) {
	if w.ContextTokens <= 0 || w.GeneratedTokens < 2 || w.KVLayers <= 0 || w.KVPerToken <= 0 ||
		w.ContextTokens > math.MaxInt-(w.GeneratedTokens-1) {
		return 0, 0, 0, false
	}
	tokens = w.ContextTokens + w.GeneratedTokens - 1
	if tokens > math.MaxInt/w.KVPerToken {
		return 0, 0, 0, false
	}
	elements = tokens * w.KVPerToken
	if w.KVLayers > math.MaxInt/2 {
		return 0, 0, 0, false
	}
	parts := 2 * w.KVLayers
	if elements%parts != 0 {
		return 0, 0, 0, false
	}
	return tokens, elements, elements / parts, true
}

func validatePhysicalWaves(s *Sample, requests int) error {
	if len(s.PhysicalWaves) == 0 {
		return errors.New("missing physical waves")
	}
	read := make(map[string]uint64)
	write := make(map[string]uint64)
	var dispatches, last uint64
	seenMembers := make(map[string]bool)
	seenRequests := make(map[string]bool)
	wantRequests := make(map[string]bool, requests)
	for i := 0; i < requests; i++ {
		wantRequests[fmt.Sprintf("%s-%d", s.ID, i)] = true
	}
	for _, wave := range s.PhysicalWaves {
		if wave.Sequence <= last || wave.Kind == "" || len(wave.Requests) == 0 || wave.Dispatches == 0 && sumByteMap(wave.TierReadBytes)+sumByteMap(wave.TierWriteBytes) == 0 {
			return errors.New("invalid physical wave")
		}
		last = wave.Sequence
		dispatches += wave.Dispatches
		waveMembers := make(map[string]bool)
		for _, request := range wave.Requests {
			logical, _, ok := strings.Cut(request, "@")
			if request == "" || !ok || !wantRequests[logical] || waveMembers[request] {
				return errors.New("invalid physical wave member")
			}
			waveMembers[request] = true
			seenMembers[request] = true
			seenRequests[logical] = true
		}
		for k, v := range wave.TierReadBytes {
			read[k] += v
		}
		for k, v := range wave.TierWriteBytes {
			write[k] += v
		}
	}
	if dispatches != s.Dispatches || !reflect.DeepEqual(read, nonzeroByteMap(s.TierReadBytes)) || !reflect.DeepEqual(write, nonzeroByteMap(s.TierWriteBytes)) {
		return errors.New("physical wave totals differ")
	}
	if len(seenMembers) < requests || !reflect.DeepEqual(seenRequests, wantRequests) {
		return errors.New("physical waves omit requests")
	}
	return nil
}

func sumByteMap(m map[string]uint64) uint64 {
	var n uint64
	for _, v := range m {
		n += v
	}
	return n
}

func nonzeroByteMap(m map[string]uint64) map[string]uint64 {
	out := make(map[string]uint64)
	for k, v := range m {
		if v != 0 {
			out[k] = v
		}
	}
	return out
}

func sampleOutputs(samples []Sample) map[string]string {
	out := make(map[string]string, len(samples))
	for _, s := range samples {
		out[s.ID] = s.OutputSHA256 + ":" + s.ObservedSHA256
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
	d, err := os.Open(dir)
	if err != nil {
		return fmt.Errorf("open receipt directory: %w", err)
	}
	if err := d.Sync(); err != nil {
		d.Close()
		return fmt.Errorf("sync receipt directory: %w", err)
	}
	if err := d.Close(); err != nil {
		return fmt.Errorf("close receipt directory: %w", err)
	}
	ok = true
	return nil
}

func validSHA256(s string) bool {
	b, err := hex.DecodeString(s)
	return err == nil && len(b) == sha256.Size
}

// Lease prevents two local benchmark runners from sharing a host receipt path.
type Lease struct {
	path string
	file *os.File
}

// Path returns the lock file identifying the lease.
func (l *Lease) Path() string {
	if l == nil {
		return ""
	}
	return l.path
}

// AcquireLease atomically acquires path and fails when another runner owns it.
func AcquireLease(path string) (*Lease, error) {
	f, err := os.OpenFile(path, os.O_RDWR|os.O_CREATE, 0600)
	if err != nil {
		return nil, fmt.Errorf("acquire runner lease: %w", err)
	}
	if err := lockFile(f); err != nil {
		f.Close()
		return nil, fmt.Errorf("acquire runner lease: %w", err)
	}
	if err := f.Truncate(0); err != nil {
		unlockFile(f)
		f.Close()
		return nil, fmt.Errorf("truncate runner lease: %w", err)
	}
	if _, err := f.Seek(0, 0); err != nil {
		unlockFile(f)
		f.Close()
		return nil, fmt.Errorf("seek runner lease: %w", err)
	}
	if _, err := fmt.Fprintf(f, "%d\n", os.Getpid()); err != nil {
		unlockFile(f)
		f.Close()
		return nil, fmt.Errorf("write runner lease: %w", err)
	}
	if err := f.Sync(); err != nil {
		unlockFile(f)
		f.Close()
		return nil, fmt.Errorf("sync runner lease: %w", err)
	}
	return &Lease{path: path, file: f}, nil
}

// AcquireDeviceLease prevents concurrent collectors from timing the same
// physical device, even when their output and receipt paths differ.
func AcquireDeviceLease(dir, backend, device string) (*Lease, error) {
	if dir == "" || backend == "" || device == "" {
		return nil, errors.New("acquire device lease: incomplete identity")
	}
	if err := os.MkdirAll(dir, 0o700); err != nil {
		return nil, fmt.Errorf("acquire device lease: %w", err)
	}
	sum := sha256.Sum256([]byte(backend + "\x00" + device))
	return AcquireLease(filepath.Join(dir, hex.EncodeToString(sum[:])+".lock"))
}

// Close releases the runner lease.
func (l *Lease) Close() error {
	if l == nil || l.file == nil {
		return nil
	}
	f := l.file
	l.file = nil
	if err := unlockFile(f); err != nil {
		f.Close()
		return err
	}
	return f.Close()
}

// SortRuns sorts runs by arm for stable serialization before execution.
func SortRuns(runs []Run) { sort.Slice(runs, func(i, j int) bool { return runs[i].Arm < runs[j].Arm }) }
