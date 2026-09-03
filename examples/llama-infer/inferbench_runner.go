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
	Output                  string              `json:"output"`
	Lease                   string              `json:"lease"`
	SourcePath              string              `json:"source_path"`
	TokenizerPath           string              `json:"tokenizer_path"`
	DeviceEvidencePath      string              `json:"device_evidence_path"`
	ExpectedBinarySHA256    string              `json:"expected_binary_sha256"`
	ExpectedModelSHA256     string              `json:"expected_model_sha256"`
	ExpectedTokenizerSHA256 string              `json:"expected_tokenizer_sha256"`
	ExpectedSourceSHA256    string              `json:"expected_source_sha256"`
	SampleIDs               []string            `json:"sample_ids"`
	Workload                inferbench.Workload `json:"workload"`
}

type deviceEvidence struct {
	Backend          string    `json:"backend"`
	DeviceID         string    `json:"device_id"`
	Hostname         string    `json:"hostname"`
	CapturedAt       time.Time `json:"captured_at"`
	Command          []string  `json:"command"`
	Output           string    `json:"output"`
	HardwareVerified bool      `json:"hardware_verified"`
	DeviceResidentKV bool      `json:"device_resident_kv"`
	CPUFallback      bool      `json:"cpu_fallback"`
}

type armReceipt struct {
	SchemaVersion int            `json:"schema_version"`
	State         string         `json:"state"`
	Run           inferbench.Run `json:"run"`
}

func runInferbench(ctx context.Context, cfg Config, specPath string) error {
	var spec inferbenchSpec
	if err := decodeStrictJSON(specPath, &spec); err != nil {
		return err
	}
	if spec.Arm != "llama-infer" {
		return fmt.Errorf("inferbench: unsupported arm %q", spec.Arm)
	}
	wantTokenizer, err := filepath.EvalSymlinks(filepath.Join(cfg.ModelPath, "tokenizer.json"))
	if err != nil {
		return fmt.Errorf("inferbench: resolve model tokenizer: %w", err)
	}
	gotTokenizer, err := filepath.EvalSymlinks(spec.TokenizerPath)
	if err != nil || gotTokenizer != wantTokenizer {
		return errors.New("inferbench: tokenizer path is not the model tokenizer")
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
	lease, err := inferbench.AcquireLease(spec.Lease)
	if err != nil {
		return err
	}
	defer lease.Close()

	identity, evidence, err := benchmarkIdentity(cfg, spec)
	if err != nil {
		return err
	}
	eng, err := getBenchmarkEngine(cfg)
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
	identity.Precision = "f32"
	if eng.IsF16() {
		identity.Precision = "f16"
	}
	adapter, err := newScheduledEngineAdapter(eng, cfg, spec.Workload.Concurrency)
	if err != nil {
		return fmt.Errorf("inferbench: serving adapter: %w", err)
	}
	defer adapter.Close()
	scheduler, err := continuousbatch.New(adapter, continuousbatch.Config{MaxActive: spec.Workload.Concurrency, MaxQueue: spec.Workload.Concurrency, MaxBatch: spec.Workload.Concurrency, BatchWait: time.Millisecond, TokenBuffer: spec.Workload.GeneratedTokens + 1, PrefillWorkers: spec.Workload.Concurrency})
	if err != nil {
		return err
	}
	defer scheduler.Close()
	for i := 0; i < spec.Workload.WarmupRuns; i++ {
		if _, err := benchmarkSample(ctx, scheduler, adapter, eng, spec.Workload, fmt.Sprintf("warmup-%d", i)); err != nil {
			return err
		}
	}
	run := inferbench.Run{Arm: spec.Arm, Identity: identity, Workload: spec.Workload, Evidence: evidence}
	for _, id := range spec.SampleIDs {
		resetBenchmarkCaches(eng)
		sample, err := benchmarkSample(ctx, scheduler, adapter, eng, spec.Workload, id)
		if err != nil {
			return err
		}
		run.Samples = append(run.Samples, sample)
	}
	return inferbench.WriteJSONAtomic(spec.Output, armReceipt{SchemaVersion: inferbench.SchemaVersion, State: "collected", Run: run})
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
	if proof.Backend != benchmarkBackend() || proof.DeviceID == "" || hostErr != nil || proof.Hostname != hostname || proof.CapturedAt.IsZero() || age < -time.Minute || age > 15*time.Minute || len(proof.Command) == 0 || proof.Output == "" || !proof.HardwareVerified || !proof.DeviceResidentKV || proof.CPUFallback {
		return inferbench.Identity{}, inferbench.Evidence{}, errors.New("inferbench: hardware qualification refused")
	}
	proofHash, err := inferbench.HashFile(spec.DeviceEvidencePath)
	if err != nil {
		return inferbench.Identity{}, inferbench.Evidence{}, err
	}
	identity := inferbench.Identity{BinarySHA256: computed["binary"], ModelSHA256: computed["model"], TokenizerSHA256: computed["tokenizer"], SourceSHA256: computed["source"], Backend: proof.Backend, DeviceID: proof.DeviceID}
	evidence := inferbench.Evidence{HardwareVerified: true, DeviceResidentKV: true, TimingBoundary: "submit-to-final-device-synchronize", DeviceSynchronized: true, IncludesQueueing: true, RunnerLease: spec.Lease, DeviceEvidenceSHA256: proofHash}
	return identity, evidence, nil
}

func benchmarkSample(ctx context.Context, scheduler *continuousbatch.Scheduler, adapter *scheduledEngineAdapter, eng *ModelEngine, workload inferbench.Workload, id string) (inferbench.Sample, error) {
	if id == "" {
		return inferbench.Sample{}, errors.New("inferbench: empty sample id")
	}
	dispatchesBefore := adapter.Dispatches()
	cacheBefore := eng.prefixCacheMetrics()
	started := time.Now()
	responses := make([]*continuousbatch.Response, len(workload.PromptTokenIDs))
	timings := make([]inferbench.RequestTiming, len(responses))
	for i, prompt := range workload.PromptTokenIDs {
		timings[i].StartedAt = time.Now()
		response, err := scheduler.Submit(ctx, continuousbatch.Request{ID: fmt.Sprintf("%s-%d", id, i), PromptTokenIDs: prompt, MaxTokens: workload.GeneratedTokens, Sampler: SamplerConfig{Method: "argmax"}})
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
	ended := time.Now()
	dispatches := adapter.Dispatches() - dispatchesBefore
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
	return inferbench.Sample{ID: id, StartedAt: started, EndedAt: ended, TTFT: ttft, TPOT: tpot, InterToken: inter, TokensPerSecond: float64(len(timings)*workload.GeneratedTokens) / duration.Seconds(), Dispatches: dispatches, CacheHits: uint64(cache.Hits - cacheBefore.Hits), CacheMisses: uint64(cache.Misses - cacheBefore.Misses), TierReadBytes: map[string]uint64{"kv_host_to_device": 0}, TierWriteBytes: map[string]uint64{"kv_device_to_host": 0}, GeneratedTokenIDs: generated, OutputSHA256: hash, Requests: timings}, nil
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
