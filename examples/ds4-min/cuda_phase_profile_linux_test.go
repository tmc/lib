//go:build linux

package main

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/x/cuda"
)

type cudaPhaseEvent struct {
	name  string
	event *cuda.Event
}

type cudaPhaseProfile struct {
	runtime *cudaRuntime
	events  []cudaPhaseEvent
}

type cudaPhaseResult struct {
	Tokens          int              `json:"tokens"`
	HostElapsedNS   int64            `json:"host_elapsed_ns"`
	StreamElapsedNS map[string]int64 `json:"stream_elapsed_ns"`
	Counts          map[string]int   `json:"counts"`
	LogitsSHA256    string           `json:"logits_sha256"`
	ReferenceSHA256 string           `json:"reference_logits_sha256"`
	Scope           string           `json:"scope"`
}

func (p *cudaPhaseProfile) mark(name string) error {
	event, err := p.runtime.device.NewEvent()
	if err != nil {
		return fmt.Errorf("cuda phase profile: create event: %w", err)
	}
	if err := event.Record(p.runtime.device.DefaultStream()); err != nil {
		event.Destroy()
		return fmt.Errorf("cuda phase profile: record event: %w", err)
	}
	p.events = append(p.events, cudaPhaseEvent{name: name, event: event})
	return nil
}

func (p *cudaPhaseProfile) finish() (map[string]int64, map[string]int, error) {
	if err := p.mark("profile-end"); err != nil {
		return nil, nil, err
	}
	defer func() {
		for _, event := range p.events {
			event.event.Destroy()
		}
	}()
	if err := p.events[len(p.events)-1].event.Synchronize(context.Background()); err != nil {
		return nil, nil, fmt.Errorf("cuda phase profile: synchronize: %w", err)
	}
	durations := make(map[string]int64)
	counts := make(map[string]int)
	for i := 0; i+1 < len(p.events); i++ {
		d, err := p.events[i+1].event.ElapsedSince(p.events[i].event)
		if err != nil {
			return nil, nil, fmt.Errorf("cuda phase profile: elapsed %s: %w", p.events[i].name, err)
		}
		durations[p.events[i].name] += d.Nanoseconds()
		counts[p.events[i].name]++
	}
	return durations, counts, nil
}

func validateCUDAPhaseCounts(tokens int, counts map[string]int) error {
	want := map[string]int{
		"dense-front":       cudaLayers,
		"attention":         tokens * cudaLayers,
		"attention-output":  tokens * cudaLayers,
		"ffn-pre":           tokens * cudaLayers,
		"shared-ffn":        tokens * cudaLayers,
		"expert-ffn":        tokens * cudaLayers,
		"ffn-post":          tokens * cudaLayers,
		"qkv-normalization": tokens * cudaLayers,
		"output":            1,
	}
	if len(counts) != len(want) {
		return fmt.Errorf("cuda phase profile: got %d phase names, want %d", len(counts), len(want))
	}
	for name, n := range want {
		if counts[name] != n {
			return fmt.Errorf("cuda phase profile: %s count %d, want %d", name, counts[name], n)
		}
	}
	return nil
}

func validateCUDAPhaseDurations(durations map[string]int64, counts map[string]int) error {
	if len(durations) != len(counts) {
		return fmt.Errorf("cuda phase profile: got %d duration names, want %d", len(durations), len(counts))
	}
	var total int64
	for name, count := range counts {
		duration, ok := durations[name]
		if !ok {
			return fmt.Errorf("cuda phase profile: missing %s duration", name)
		}
		if duration < 0 {
			return fmt.Errorf("cuda phase profile: negative %s duration %d", name, duration)
		}
		if count < 1 {
			return fmt.Errorf("cuda phase profile: non-positive %s count %d", name, count)
		}
		total += duration
	}
	if total <= 0 {
		return fmt.Errorf("cuda phase profile: non-positive aggregate duration %d", total)
	}
	return nil
}

func TestCUDAPhaseProfile(t *testing.T) {
	if os.Getenv("DS4_CUDA_PHASE_PROFILE") == "" {
		t.Skip("DS4_CUDA_PHASE_PROFILE is not set")
	}
	if os.Getenv("DS4_CUDA_PHASE_PROFILE") != "1" {
		t.Fatal("DS4_CUDA_PHASE_PROFILE must be 1")
	}
	required := func(name string) string {
		t.Helper()
		value := os.Getenv(name)
		if value == "" {
			t.Fatalf("%s is required", name)
		}
		return value
	}
	model := required("DS4_MODEL")
	tokensPath := required("DS4_PREFILL_TOKENS")
	resultPath := required("DS4_CUDA_PHASE_PROFILE_OUT")
	input, err := os.ReadFile(tokensPath)
	if err != nil {
		t.Fatal(err)
	}
	tokens, err := parsePrefillTokens(input)
	if err != nil {
		t.Fatal(err)
	}
	const profileTokens = 8
	tokens = tokens[:profileTokens]
	backend, closer, err := openSessionBackend(config{model: model, backend: "cuda", context: 1024, log: io.Discard})
	if err != nil {
		t.Fatal(err)
	}
	closed := false
	defer func() {
		if !closed {
			closer.Close()
		}
	}()
	runtime, ok := backend.(*cudaRuntime)
	if !ok {
		t.Fatal("CUDA phase profile opened a non-CUDA runtime")
	}
	if err := runtime.EvalBatch(tokens, 0); err != nil {
		t.Fatal(err)
	}
	reference := append([]float32(nil), runtime.Logits()...)
	referenceHash, err := oracleWorkloadHash(reference)
	if err != nil {
		t.Fatal(err)
	}
	if err := runtime.Reset(); err != nil {
		t.Fatal(err)
	}
	profile := &cudaPhaseProfile{runtime: runtime}
	runtime.phaseMark = profile.mark
	start := time.Now()
	if err := runtime.EvalBatch(tokens, 0); err != nil {
		t.Fatal(err)
	}
	hostElapsed := time.Since(start)
	durations, counts, err := profile.finish()
	if err != nil {
		t.Fatal(err)
	}
	runtime.phaseMark = nil
	if err := validateCUDAPhaseCounts(profileTokens, counts); err != nil {
		t.Fatal(err)
	}
	if err := validateCUDAPhaseDurations(durations, counts); err != nil {
		t.Fatal(err)
	}
	if err := exactFloat32Bits(runtime.Logits(), reference); err != nil {
		t.Fatalf("instrumented logits differ from reference: %v", err)
	}
	logitsHash, err := oracleWorkloadHash(runtime.Logits())
	if err != nil {
		t.Fatal(err)
	}
	if err := closer.Close(); err != nil {
		t.Fatal(err)
	}
	closed = true
	result := cudaPhaseResult{
		Tokens:          profileTokens,
		HostElapsedNS:   hostElapsed.Nanoseconds(),
		StreamElapsedNS: durations,
		Counts:          counts,
		LogitsSHA256:    logitsHash,
		ReferenceSHA256: referenceHash,
		Scope:           "warmed CUDA positions 0..7 only; default-stream elapsed intervals include device idle time; they are phase attribution, not kernel busy time or throughput",
	}
	data, err := json.MarshalIndent(result, "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	f, err := os.OpenFile(resultPath, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0600)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := f.Write(data); err != nil {
		f.Close()
		t.Fatal(err)
	}
	if err := f.Close(); err != nil {
		t.Fatal(err)
	}
}

func TestValidateCUDAPhaseCounts(t *testing.T) {
	const tokens = 8
	want := map[string]int{
		"dense-front": cudaLayers,
	}
	for _, name := range []string{"qkv-normalization", "attention", "attention-output", "ffn-pre", "shared-ffn", "expert-ffn", "ffn-post"} {
		want[name] = tokens * cudaLayers
	}
	want["output"] = 1
	if err := validateCUDAPhaseCounts(tokens, want); err != nil {
		t.Fatal(err)
	}
	for name := range want {
		bad := make(map[string]int, len(want))
		for key, value := range want {
			bad[key] = value
		}
		delete(bad, name)
		if err := validateCUDAPhaseCounts(tokens, bad); err == nil {
			t.Fatalf("missing %s passed", name)
		}
	}
	badCount := make(map[string]int, len(want))
	for key, value := range want {
		badCount[key] = value
	}
	badCount["attention"]--
	if err := validateCUDAPhaseCounts(tokens, badCount); err == nil {
		t.Fatal("wrong attention count passed")
	}
	replaced := make(map[string]int, len(want))
	for key, value := range want {
		replaced[key] = value
	}
	delete(replaced, "attention")
	replaced["fabricated"] = tokens * cudaLayers
	if err := validateCUDAPhaseCounts(tokens, replaced); err == nil {
		t.Fatal("replacement phase name passed")
	}
}

func TestValidateCUDAPhaseDurations(t *testing.T) {
	counts := map[string]int{"a": 1, "b": 2}
	for _, tt := range []struct {
		name      string
		durations map[string]int64
		ok        bool
	}{
		{name: "positive", durations: map[string]int64{"a": 1, "b": 2}, ok: true},
		{name: "negative", durations: map[string]int64{"a": -1, "b": 2}},
		{name: "zero aggregate", durations: map[string]int64{"a": 0, "b": 0}},
		{name: "missing", durations: map[string]int64{"a": 1}},
		{name: "replacement", durations: map[string]int64{"a": 1, "c": 2}},
	} {
		t.Run(tt.name, func(t *testing.T) {
			err := validateCUDAPhaseDurations(tt.durations, counts)
			if (err == nil) != tt.ok {
				t.Fatalf("error = %v, want success %t", err, tt.ok)
			}
		})
	}
}
