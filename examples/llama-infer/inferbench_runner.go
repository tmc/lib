package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sync"
	"time"

	"github.com/tmc/lib/examples/internal/continuousbatch"
	"github.com/tmc/lib/examples/internal/inferbench"
)

type inferbenchSpec struct {
	Arm                     string              `json:"arm"`
	Arms                    []string            `json:"arms,omitempty"`
	RandomSeed              uint64              `json:"random_seed,omitempty"`
	RawDir                  string              `json:"raw_dir,omitempty"`
	Output                  string              `json:"output"`
	Lease                   string              `json:"lease"`
	SourcePath              string              `json:"source_path"`
	TokenizerPath           string              `json:"tokenizer_path"`
	ConfigPath              string              `json:"config_path"`
	DeviceEvidencePath      string              `json:"device_evidence_path"`
	ExpectedBinarySHA256    string              `json:"expected_binary_sha256"`
	ExpectedModelSHA256     string              `json:"expected_model_sha256"`
	ExpectedTokenizerSHA256 string              `json:"expected_tokenizer_sha256"`
	ExpectedConfigSHA256    string              `json:"expected_config_sha256"`
	ExpectedSourceSHA256    string              `json:"expected_source_sha256"`
	SampleIDs               []string            `json:"sample_ids"`
	Workload                inferbench.Workload `json:"workload"`
}

type deviceEvidence struct {
	Backend            string        `json:"backend"`
	DeviceID           string        `json:"device_id"`
	Hostname           string        `json:"hostname"`
	CapturedAt         time.Time     `json:"captured_at"`
	Command            []string      `json:"command"`
	Output             string        `json:"output"`
	HardwareVerified   bool          `json:"hardware_verified"`
	DeviceResidentKV   bool          `json:"device_resident_kv"`
	CPUFallback        bool          `json:"cpu_fallback"`
	NullControlPath    string        `json:"null_control_path"`
	NullControlSamples int           `json:"null_control_samples"`
	NullControlP95     time.Duration `json:"null_control_p95_ns"`
	NullControlLimit   time.Duration `json:"null_control_limit_ns"`
}

type armReceipt struct {
	SchemaVersion int            `json:"schema_version"`
	State         string         `json:"state"`
	Run           inferbench.Run `json:"run"`
}

func runInferbench(ctx context.Context, cfg Config, specPath string) error {
	return runInferbenchWithEngine(ctx, cfg, specPath, getBenchmarkEngine)
}

func runInferbenchWithEngine(ctx context.Context, cfg Config, specPath string, newEngine func(Config) (*ModelEngine, error)) error {
	var spec inferbenchSpec
	if err := decodeStrictJSON(specPath, &spec); err != nil {
		return err
	}
	if err := validateBenchmarkArms(spec); err != nil {
		return err
	}
	wantTokenizer, err := filepath.EvalSymlinks(filepath.Join(cfg.ModelPath, "tokenizer.json"))
	if err != nil {
		return fmt.Errorf("inferbench: resolve model tokenizer: %w", err)
	}
	gotTokenizer, err := filepath.EvalSymlinks(spec.TokenizerPath)
	if err != nil || gotTokenizer != wantTokenizer {
		return errors.New("inferbench: tokenizer path is not the model tokenizer")
	}
	wantConfig, err := filepath.EvalSymlinks(filepath.Join(cfg.ModelPath, "config.json"))
	if err != nil {
		return fmt.Errorf("inferbench: resolve model config: %w", err)
	}
	gotConfig, err := filepath.EvalSymlinks(spec.ConfigPath)
	if err != nil || gotConfig != wantConfig {
		return errors.New("inferbench: config path is not the model config")
	}
	if spec.Output == "" || spec.Lease == "" || len(spec.SampleIDs) == 0 {
		return errors.New("inferbench: incomplete output, lease, or sample ids")
	}
	if spec.Workload.Sampler != "greedy-argmax" {
		return fmt.Errorf("inferbench: unsupported sampler %q", spec.Workload.Sampler)
	}
	if spec.Workload.CacheState != "cold" {
		return fmt.Errorf("inferbench: unsupported cache state %q", spec.Workload.CacheState)
	}
	if len(spec.Workload.PromptTokenIDs) != spec.Workload.Concurrency {
		return errors.New("inferbench: prompt count differs from concurrency")
	}
	if spec.Workload.GeneratedTokens < 2 || spec.Workload.WarmupRuns < 1 || spec.Workload.ContextTokens < 1 {
		return errors.New("inferbench: incomplete workload")
	}
	for _, prompt := range spec.Workload.PromptTokenIDs {
		if len(prompt) != spec.Workload.ContextTokens {
			return errors.New("inferbench: prompt token count differs from context tokens")
		}
	}
	if spec.Workload.OutputMode != inferbench.ThroughputOutputMode && spec.Workload.OutputMode != inferbench.ObservedOutputMode {
		return fmt.Errorf("inferbench: unsupported output mode %q", spec.Workload.OutputMode)
	}
	wantLogits, wantLayers := 0, 0
	wantKVPerToken := 0
	if spec.Workload.OutputMode == inferbench.ObservedOutputMode {
		wantLogits = cfg.VocabSize
		wantLayers = cfg.NumLayers
		wantKVPerToken = 2 * cfg.NumLayers * cfg.NumKVHeads * cfg.HeadDim
	}
	if spec.Workload.FullLogits != wantLogits || spec.Workload.KVLayers != wantLayers || spec.Workload.KVPerToken != wantKVPerToken {
		return errors.New("inferbench: output shape differs from model configuration")
	}
	lease, err := inferbench.AcquireLease(spec.Lease)
	if err != nil {
		return err
	}
	defer lease.Close()

	identity, evidence, err := benchmarkIdentity(cfg, spec)
	if err != nil {
		return err
	}
	cacheDir, err := os.UserCacheDir()
	if err != nil {
		return fmt.Errorf("inferbench: user cache directory: %w", err)
	}
	deviceLease, err := inferbench.AcquireDeviceLease(filepath.Join(cacheDir, "tmc-lib", "inferbench"), identity.Backend, identity.DeviceID)
	if err != nil {
		return err
	}
	defer deviceLease.Close()
	evidence.RunnerLease = deviceLease.Path()
	eng, err := newEngine(cfg)
	if err != nil {
		return fmt.Errorf("inferbench: compile engine: %w", err)
	}
	deviceID, err := benchmarkDeviceID(eng)
	if err != nil {
		return fmt.Errorf("inferbench: query runtime device: %w", err)
	}
	if deviceID != identity.DeviceID {
		return fmt.Errorf("inferbench: qualified device %q differs from runtime device %q", identity.DeviceID, deviceID)
	}
	if err := benchmarkPhysicalTelemetry(); err != nil {
		return fmt.Errorf("inferbench: physical telemetry unavailable: %w", err)
	}
	identity.Precision = "f32"
	if eng.IsF16() {
		identity.Precision = "f16"
	}
	adapter, err := newScheduledEngineAdapter(eng, cfg, spec.Workload.Concurrency)
	if err != nil {
		return fmt.Errorf("inferbench: serving adapter: %w", err)
	}
	defer adapter.Close()
	adapter.enablePhysicalLedger()
	if spec.Workload.OutputMode == inferbench.ObservedOutputMode {
		adapter.observedOutputs = newObservedOutputCollector()
	}
	scheduler, err := continuousbatch.New(adapter, continuousbatch.Config{MaxActive: spec.Workload.Concurrency, MaxQueue: spec.Workload.Concurrency, MaxBatch: spec.Workload.Concurrency, BatchWait: time.Millisecond, TokenBuffer: spec.Workload.GeneratedTokens + 1, PrefillWorkers: spec.Workload.Concurrency})
	if err != nil {
		return err
	}
	defer scheduler.Close()
	if len(spec.Arms) != 0 {
		return runInferbenchCampaign(ctx, cfg, spec, identity, evidence, eng, adapter, scheduler)
	}
	for i := 0; i < spec.Workload.WarmupRuns; i++ {
		if _, err := benchmarkSample(ctx, scheduler, adapter, eng, spec.Workload, fmt.Sprintf("warmup-%d", i)); err != nil {
			return err
		}
	}
	evidence.WarmupRunsCompleted = spec.Workload.WarmupRuns
	run := inferbench.Run{Arm: spec.Arm, Identity: identity, Workload: spec.Workload, Evidence: evidence}
	for _, id := range spec.SampleIDs {
		resetBenchmarkCaches(eng)
		if err := adapter.resetPrefixes(); err != nil {
			return fmt.Errorf("inferbench: reset serving prefixes: %w", err)
		}
		sample, err := benchmarkSample(ctx, scheduler, adapter, eng, spec.Workload, id)
		if err != nil {
			return err
		}
		run.Samples = append(run.Samples, sample)
	}
	run.Evidence.DeviceSynchronized = true
	return inferbench.WriteJSONAtomic(spec.Output, armReceipt{SchemaVersion: inferbench.SchemaVersion, State: "collected", Run: run})
}

func validateBenchmarkArms(spec inferbenchSpec) error {
	if len(spec.Arms) == 0 {
		if !benchmarkArmSupportsOutput(spec.Arm, spec.Workload.OutputMode) {
			return fmt.Errorf("inferbench: unsupported arm %q", spec.Arm)
		}
		return nil
	}
	if spec.Arm != "" || spec.RandomSeed == 0 || spec.RawDir == "" || len(spec.Arms) < 2 {
		return errors.New("inferbench: incomplete multi-arm campaign")
	}
	if len(spec.SampleIDs) < 20 {
		return fmt.Errorf("inferbench: only %d planned samples per arm", len(spec.SampleIDs))
	}
	seen := make(map[string]bool)
	for _, arm := range spec.Arms {
		if seen[arm] {
			return fmt.Errorf("inferbench: duplicate arm %q", arm)
		}
		seen[arm] = true
		if !benchmarkArmSupportsOutput(arm, spec.Workload.OutputMode) {
			return fmt.Errorf("inferbench: arm %q cannot perform output mode %q", arm, spec.Workload.OutputMode)
		}
		if err := configureBenchmarkArm(nil, arm); err != nil {
			return err
		}
	}
	return nil
}

func benchmarkArmSupportsOutput(arm, mode string) bool {
	switch mode {
	case inferbench.ThroughputOutputMode:
		return arm == "llama-infer" || arm == "llama-infer-unfused-projections"
	case inferbench.ObservedOutputMode:
		return benchmarkObservedOutputSupported() && (arm == "llama-infer-observed-output" || arm == "llama-infer-unfused-projections-observed-output")
	default:
		return false
	}
}

func runInferbenchCampaign(ctx context.Context, cfg Config, spec inferbenchSpec, identity inferbench.Identity, evidence inferbench.Evidence, eng *ModelEngine, adapter *scheduledEngineAdapter, scheduler *continuousbatch.Scheduler) error {
	if err := benchmarkPeakMemorySupport(eng); err != nil {
		return fmt.Errorf("inferbench: peak memory evidence: %w", err)
	}
	if err := os.Mkdir(spec.RawDir, 0o700); err != nil {
		return fmt.Errorf("inferbench: create raw receipt directory: %w", err)
	}
	evidence.DeviceSynchronized = true
	evidence.MemoryMetric = inferbench.PeakDeviceMemoryMetric
	evidence.WarmupRunsCompleted = spec.Workload.WarmupRuns
	plan := inferbench.Campaign{SchemaVersion: inferbench.SchemaVersion, RandomSeed: spec.RandomSeed}
	for _, arm := range spec.Arms {
		run := inferbench.Run{Arm: arm, Identity: identity, Workload: spec.Workload, Evidence: evidence}
		for _, id := range spec.SampleIDs {
			run.Samples = append(run.Samples, inferbench.Sample{ID: id})
		}
		plan.Runs = append(plan.Runs, run)
	}
	plan.Order = inferbench.Schedule(plan.Runs, plan.RandomSeed)
	if err := inferbench.WriteJSONAtomic(filepath.Join(spec.RawDir, "plan.json"), plan); err != nil {
		return fmt.Errorf("inferbench: write campaign plan: %w", err)
	}
	for _, arm := range spec.Arms {
		if err := configureBenchmarkArm(eng, arm); err != nil {
			return err
		}
		for i := 0; i < spec.Workload.WarmupRuns; i++ {
			if _, err := benchmarkSample(ctx, scheduler, adapter, eng, spec.Workload, fmt.Sprintf("warmup-%s-%d", arm, i)); err != nil {
				return err
			}
		}
	}
	interval := 0
	campaign, err := inferbench.Collect(plan, func(k inferbench.ScheduledSample) (inferbench.Sample, error) {
		if err := configureBenchmarkArm(eng, k.Arm); err != nil {
			return inferbench.Sample{}, err
		}
		resetBenchmarkCaches(eng)
		if err := adapter.resetPrefixes(); err != nil {
			return inferbench.Sample{}, fmt.Errorf("inferbench: reset serving prefixes: %w", err)
		}
		return benchmarkSample(ctx, scheduler, adapter, eng, spec.Workload, k.ID)
	}, func(raw inferbench.IntervalReceipt) error {
		path := filepath.Join(spec.RawDir, fmt.Sprintf("%06d.json", interval))
		interval++
		return inferbench.WriteJSONAtomic(path, raw)
	})
	if err != nil {
		return err
	}
	// Rehash outside every timed interval. Besides catching persistent input
	// drift, this avoids making model size part of the interleaved schedule.
	if err := verifyBenchmarkInputs(cfg, spec, identity, evidence); err != nil {
		return err
	}
	if err := campaign.BindRawSet(spec.RawDir); err != nil {
		return err
	}
	return inferbench.WriteJSONAtomic(spec.Output, campaign)
}

func verifyBenchmarkInputs(cfg Config, spec inferbenchSpec, identity inferbench.Identity, evidence inferbench.Evidence) error {
	exe, err := os.Executable()
	if err != nil {
		return fmt.Errorf("inferbench: recheck binary identity: %w", err)
	}
	paths := []struct {
		name string
		path string
		want string
		dir  bool
	}{
		{"binary", exe, identity.BinarySHA256, true},
		{"model", cfg.ModelPath, identity.ModelSHA256, true},
		{"tokenizer", spec.TokenizerPath, identity.TokenizerSHA256, true},
		{"config", spec.ConfigPath, identity.ConfigSHA256, true},
		{"source", spec.SourcePath, identity.SourceSHA256, true},
		{"device evidence", spec.DeviceEvidencePath, evidence.DeviceEvidenceSHA256, false},
	}
	for _, item := range paths {
		var (
			got string
			err error
		)
		if item.dir {
			got, err = inferbench.HashPath(item.path)
		} else {
			got, err = inferbench.HashFile(item.path)
		}
		if err != nil {
			return fmt.Errorf("inferbench: recheck %s identity: %w", item.name, err)
		}
		if got != item.want {
			return fmt.Errorf("inferbench: %s identity changed during campaign", item.name)
		}
	}
	var proof deviceEvidence
	if err := decodeStrictJSON(spec.DeviceEvidencePath, &proof); err != nil {
		return err
	}
	nullHash, err := inferbench.HashFile(proof.NullControlPath)
	if err != nil {
		return fmt.Errorf("inferbench: recheck null control identity: %w", err)
	}
	if nullHash != evidence.NullControlSHA256 {
		return errors.New("inferbench: null control identity changed during campaign")
	}
	return nil
}

func benchmarkIdentity(cfg Config, spec inferbenchSpec) (inferbench.Identity, inferbench.Evidence, error) {
	exe, err := os.Executable()
	if err != nil {
		return inferbench.Identity{}, inferbench.Evidence{}, fmt.Errorf("inferbench: executable: %w", err)
	}
	computed := make(map[string]string)
	paths := []struct{ name, path, want string }{
		{"binary", exe, spec.ExpectedBinarySHA256},
		{"model", cfg.ModelPath, spec.ExpectedModelSHA256},
		{"tokenizer", spec.TokenizerPath, spec.ExpectedTokenizerSHA256},
		{"config", spec.ConfigPath, spec.ExpectedConfigSHA256},
		{"source", spec.SourcePath, spec.ExpectedSourceSHA256},
	}
	for _, item := range paths {
		name, path := item.name, item.path
		if path == "" {
			return inferbench.Identity{}, inferbench.Evidence{}, fmt.Errorf("inferbench: missing %s path", name)
		}
		computed[name], err = inferbench.HashPath(path)
		if err != nil {
			return inferbench.Identity{}, inferbench.Evidence{}, fmt.Errorf("inferbench: %s identity: %w", name, err)
		}
		if item.want == "" || computed[name] != item.want {
			return inferbench.Identity{}, inferbench.Evidence{}, fmt.Errorf("inferbench: stale or unpinned %s identity: got %s want %s", name, computed[name], item.want)
		}
	}
	var proof deviceEvidence
	if err := decodeStrictJSON(spec.DeviceEvidencePath, &proof); err != nil {
		return inferbench.Identity{}, inferbench.Evidence{}, err
	}
	hostname, hostErr := os.Hostname()
	age := time.Since(proof.CapturedAt)
	if proof.Backend != benchmarkBackend() || proof.DeviceID == "" || hostErr != nil || proof.Hostname != hostname || proof.CapturedAt.IsZero() || age < -time.Minute || age > 15*time.Minute || len(proof.Command) == 0 || proof.Output == "" || !proof.HardwareVerified || !proof.DeviceResidentKV || proof.CPUFallback || proof.NullControlSamples < 20 || proof.NullControlP95 <= 0 || proof.NullControlLimit <= 0 || proof.NullControlP95 > proof.NullControlLimit {
		return inferbench.Identity{}, inferbench.Evidence{}, errors.New("inferbench: hardware qualification refused")
	}
	nullHash, err := inferbench.HashFile(proof.NullControlPath)
	if err != nil {
		return inferbench.Identity{}, inferbench.Evidence{}, fmt.Errorf("inferbench: null control: %w", err)
	}
	proofHash, err := inferbench.HashFile(spec.DeviceEvidencePath)
	if err != nil {
		return inferbench.Identity{}, inferbench.Evidence{}, err
	}
	identity := inferbench.Identity{BinarySHA256: computed["binary"], ModelSHA256: computed["model"], TokenizerSHA256: computed["tokenizer"], ConfigSHA256: computed["config"], SourceSHA256: computed["source"], Backend: proof.Backend, DeviceID: proof.DeviceID}
	timingBoundary := "submit-to-final-device-synchronize"
	if spec.Workload.OutputMode == inferbench.ObservedOutputMode {
		timingBoundary = "submit-to-observed-output-digest-and-final-device-synchronize"
	}
	evidence := inferbench.Evidence{HardwareVerified: true, DeviceResidentKV: true, TimingBoundary: timingBoundary, IncludesQueueing: true, RunnerLease: spec.Lease, DeviceEvidenceSHA256: proofHash, NullControlSHA256: nullHash, NullControlSamples: proof.NullControlSamples, NullControlP95: proof.NullControlP95, NullControlLimit: proof.NullControlLimit, PersistentRuntime: true, CompilationExcluded: true, RuntimePolicy: inferbench.PersistentEnginePolicy, AllocationPolicy: inferbench.ColdIntervalPolicy, OutputContract: spec.Workload.OutputMode}
	return identity, evidence, nil
}

func benchmarkSample(ctx context.Context, scheduler *continuousbatch.Scheduler, adapter *scheduledEngineAdapter, eng *ModelEngine, workload inferbench.Workload, id string) (inferbench.Sample, error) {
	if id == "" {
		return inferbench.Sample{}, errors.New("inferbench: empty sample id")
	}
	if err := benchmarkMemoryReset(ctx, eng); err != nil {
		return inferbench.Sample{}, fmt.Errorf("inferbench: reset memory interval: %w", err)
	}
	physicalBefore := adapter.physical.Snapshot()
	cacheBefore := eng.prefixCacheMetrics()
	started := time.Now()
	responses := make([]*continuousbatch.Response, len(workload.PromptTokenIDs))
	requestIDs := make([]string, len(responses))
	timings := make([]inferbench.RequestTiming, len(responses))
	for i, prompt := range workload.PromptTokenIDs {
		timings[i].StartedAt = time.Now()
		requestIDs[i] = fmt.Sprintf("%s-%d", id, i)
		response, err := scheduler.Submit(ctx, continuousbatch.Request{ID: requestIDs[i], PromptTokenIDs: prompt, MaxTokens: workload.GeneratedTokens, Sampler: SamplerConfig{Method: "argmax"}})
		if err != nil {
			return inferbench.Sample{}, fmt.Errorf("inferbench: submit request %d: %w", i, err)
		}
		responses[i] = response
	}
	generated := make([][]int32, len(responses))
	errs := make(chan error, len(responses))
	var wg sync.WaitGroup
	for i, response := range responses {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for token := range response.Tokens {
				generated[i] = append(generated[i], token.ID)
				timings[i].TokenAt = append(timings[i].TokenAt, time.Now())
			}
			if err := <-response.Done; err != nil {
				errs <- fmt.Errorf("inferbench: request %d: %w", i, err)
				return
			}
			timings[i].EndedAt = time.Now()
			if len(generated[i]) != workload.GeneratedTokens {
				errs <- fmt.Errorf("inferbench: request %d generated %d tokens, want %d", i, len(generated[i]), workload.GeneratedTokens)
			}
		}()
	}
	wg.Wait()
	close(errs)
	for err := range errs {
		return inferbench.Sample{}, err
	}
	if err := benchmarkSynchronize(ctx, eng); err != nil {
		return inferbench.Sample{}, fmt.Errorf("inferbench: final device synchronize: %w", err)
	}
	peakMemory, err := benchmarkPeakMemory(eng)
	if err != nil {
		return inferbench.Sample{}, fmt.Errorf("inferbench: snapshot memory interval: %w", err)
	}
	var observed []inferbench.ObservedOutput
	var observedHash string
	if workload.OutputMode == inferbench.ObservedOutputMode {
		if adapter.observedOutputs == nil {
			return inferbench.Sample{}, errors.New("inferbench: observed output collector unavailable")
		}
		var err error
		observed, err = adapter.observedOutputs.outputs(requestIDs)
		if err != nil {
			return inferbench.Sample{}, err
		}
		observedHash, err = inferbench.HashJSON(observed)
		if err != nil {
			return inferbench.Sample{}, err
		}
	}
	ended := time.Now()
	var ttft, tpot time.Duration
	var inter []time.Duration
	for _, request := range timings {
		ttft += request.TokenAt[0].Sub(request.StartedAt)
		for i := 1; i < len(request.TokenAt); i++ {
			d := request.TokenAt[i].Sub(request.TokenAt[i-1])
			tpot += d
			if len(timings) == 1 {
				inter = append(inter, d)
			}
		}
	}
	ttft /= time.Duration(len(timings))
	tokenGaps := len(timings) * (workload.GeneratedTokens - 1)
	if tokenGaps > 0 {
		tpot /= time.Duration(tokenGaps)
	}
	if len(timings) > 1 {
		inter = make([]time.Duration, workload.GeneratedTokens-1)
		for step := range inter {
			for _, request := range timings {
				inter[step] += request.TokenAt[step+1].Sub(request.TokenAt[step])
			}
			inter[step] /= time.Duration(len(timings))
		}
	}
	hash, err := inferbench.HashJSON(generated)
	if err != nil {
		return inferbench.Sample{}, err
	}
	duration := ended.Sub(started)
	cache := eng.prefixCacheMetrics()
	waves, err := adapter.physical.Since(physicalBefore)
	if err != nil {
		return inferbench.Sample{}, err
	}
	physicalWaves := make([]inferbench.PhysicalWave, len(waves))
	tierRead := map[string]uint64{"metal_row_cache": 0}
	tierWrite := map[string]uint64{"metal_prefix_cache": 0}
	var dispatches uint64
	for i, wave := range waves {
		physicalWaves[i] = inferbench.PhysicalWave{Sequence: wave.Sequence, Kind: wave.Kind, Requests: wave.Requests, Dispatches: wave.Dispatches, TierReadBytes: wave.TierReadBytes, TierWriteBytes: wave.TierWriteBytes}
		dispatches += wave.Dispatches
		for k, v := range wave.TierReadBytes {
			tierRead[k] += v
		}
		for k, v := range wave.TierWriteBytes {
			tierWrite[k] += v
		}
	}
	if dispatches == 0 {
		return inferbench.Sample{}, errors.New("inferbench: physical ledger contains no dispatches")
	}
	return inferbench.Sample{ID: id, StartedAt: started, EndedAt: ended, TTFT: ttft, TPOT: tpot, InterToken: inter, TokensPerSecond: float64(len(timings)*workload.GeneratedTokens) / duration.Seconds(), Dispatches: dispatches, DispatchMeasured: true, CacheHits: uint64(cache.Hits - cacheBefore.Hits), CacheMisses: uint64(cache.Misses - cacheBefore.Misses), TierReadBytes: tierRead, TierWriteBytes: tierWrite, TierBytesMeasured: true, PhysicalWaves: physicalWaves, GeneratedTokenIDs: generated, OutputSHA256: hash, ObservedOutputs: observed, ObservedSHA256: observedHash, Requests: timings, PeakMemoryBytes: peakMemory}, nil
}

func decodeStrictJSON(path string, dst any) error {
	f, err := os.Open(path)
	if err != nil {
		return fmt.Errorf("inferbench: open %s: %w", path, err)
	}
	defer f.Close()
	dec := json.NewDecoder(f)
	dec.DisallowUnknownFields()
	if err := dec.Decode(dst); err != nil {
		return fmt.Errorf("inferbench: decode %s: %w", path, err)
	}
	if err := dec.Decode(&struct{}{}); err != io.EOF {
		return fmt.Errorf("inferbench: decode %s: trailing data", path)
	}
	return nil
}
