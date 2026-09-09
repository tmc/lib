package metalprof

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"time"

	"github.com/google/pprof/profile"
	"github.com/tmc/lib/apple/x/metal/metalcapture"
)

// ReplayProfilerOutput matches the structured JSON exported by gputrace profiler.
type ReplayProfilerOutput struct {
	MetalDeviceName *string `json:"metal_device_name"`
	Dispatches      []struct {
		Index         *int     `json:"index"`
		PipelineID    *int     `json:"pipeline_id"`
		FunctionIndex *int     `json:"function_index"`
		FunctionName  *string  `json:"function_name"`
		DurationUs    *int64   `json:"duration_us"`
		SIMDShare     *float64 `json:"simd_share"`
	} `json:"dispatches"`
	Pipelines []struct {
		PipelineID       *int    `json:"pipeline_id"`
		FunctionName     *string `json:"function_name"`
		InstructionCount *int64  `json:"instruction_count"`
		SpilledBytes     *int64  `json:"spilled_bytes"`
	} `json:"pipelines"`
	TotalDispatchTimeUs *int64 `json:"total_dispatch_time_us"`
}

// ReplayEnrichmentResult holds the outputs of a successful replay enrichment, including
// the separate replay pprof profile, typed observations, and the raw profiler export JSON.
type ReplayEnrichmentResult struct {
	Profile      *profile.Profile
	Observations []metalcapture.Observation
	RawExport    []byte
}

// EnrichTraceReplay headlessly replays inTracePath, validates fresh output, runs
// gputrace profiler to extract structured per-dispatch and pipeline execution costs,
// and produces a separate replay pprof profile alongside typed Observations and raw export JSON.
func EnrichTraceReplay(ctx context.Context, inTracePath string, timeout time.Duration) (*ReplayEnrichmentResult, error) {
	if timeout <= 0 {
		timeout = 30 * time.Second
	}
	ctx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()

	// Use ~/tmp persistent working root per guidelines
	home, err := os.UserHomeDir()
	if err != nil {
		return nil, fmt.Errorf("metalprof: get home dir: %w", err)
	}
	parent := filepath.Join(home, "tmp")
	if err := os.MkdirAll(parent, 0755); err != nil {
		return nil, fmt.Errorf("metalprof: create tmp dir: %w", err)
	}

	tmpDir, err := os.MkdirTemp(parent, "metalprof-replay-")
	if err != nil {
		return nil, fmt.Errorf("metalprof: create replay dir: %w", err)
	}
	var retainedDir bool
	defer func() {
		if !retainedDir {
			_ = os.RemoveAll(tmpDir)
		}
	}()

	profiledDir := filepath.Join(tmpDir, "profiled")

	// 1. Headless Replay with explicit wall-time bound and unresolved-ownership retention on timeout
	if replayErr := ReplayTraceContext(ctx, inTracePath, profiledDir); replayErr != nil {
		if ctx.Err() != nil {
			// Replay timed out or was cancelled. Retain scratch to prevent corrupting
			// an active writer and honestly signal unresolved process ownership.
			retainedDir = true
			return nil, fmt.Errorf("metalprof: replay timed out (%v), retained scratch at %s: %w", timeout, tmpDir, replayErr)
		}
		return nil, fmt.Errorf("metalprof: replay failed: %w", replayErr)
	}

	// 2. Validate fresh expected output files
	rawDir := filepath.Join(profiledDir, filepath.Base(inTracePath)+".gpuprofiler_raw")
	if _, err := os.Stat(rawDir); err != nil {
		matches, mErr := filepath.Glob(filepath.Join(profiledDir, "*.gpuprofiler_raw"))
		if mErr == nil && len(matches) > 0 {
			rawDir = matches[0]
		} else {
			return nil, fmt.Errorf("metalprof: replay produced no fresh .gpuprofiler_raw: %w", err)
		}
	}

	streamPath := filepath.Join(rawDir, "streamData")
	if _, err := os.Stat(streamPath); err != nil {
		return nil, fmt.Errorf("metalprof: streamData missing in replay output: %w", err)
	}

	// 3. Subprocess public export via gputrace profiler --json
	cmd := exec.CommandContext(ctx, "gputrace", "profiler", "--json", rawDir)
	out, err := cmd.CombinedOutput()
	if err != nil {
		return nil, fmt.Errorf("metalprof: gputrace profiler failed: %w (output: %s)", err, string(out))
	}

	prof, obs, err := ParseReplayProfilerOutput(out)
	if err != nil {
		return nil, err
	}

	return &ReplayEnrichmentResult{
		Profile:      prof,
		Observations: obs,
		RawExport:    out,
	}, nil
}

// ParseReplayProfilerOutput decodes gputrace profiler --json output, validates schemas,
// reconciles dispatch totals, builds a separate replay pprof profile with source-only frames,
// and produces typed observations.
func ParseReplayProfilerOutput(out []byte) (*profile.Profile, []metalcapture.Observation, error) {
	if len(bytes.TrimSpace(out)) == 0 {
		return nil, nil, fmt.Errorf("metalprof: empty profiler export")
	}

	var stats ReplayProfilerOutput
	if err := json.Unmarshal(out, &stats); err != nil {
		return nil, nil, fmt.Errorf("metalprof: unmarshal gputrace profiler json: %w", err)
	}

	if len(stats.Dispatches) == 0 && len(stats.Pipelines) == 0 {
		return nil, nil, fmt.Errorf("metalprof: profiler output contains no dispatches or pipelines")
	}

	// Validate and reconcile dispatch totals
	var sumDispatchUs int64
	for i, d := range stats.Dispatches {
		if d.DurationUs == nil {
			return nil, nil, fmt.Errorf("metalprof: dispatch %d missing duration_us", i)
		}
		if *d.DurationUs < 0 {
			return nil, nil, fmt.Errorf("metalprof: dispatch %d has negative duration_us: %d", i, *d.DurationUs)
		}
		if *d.DurationUs > math.MaxInt64/1000 {
			return nil, nil, fmt.Errorf("metalprof: dispatch %d duration overflows nanoseconds: %d us", i, *d.DurationUs)
		}
		sumDispatchUs += *d.DurationUs
	}

	if stats.TotalDispatchTimeUs != nil {
		if *stats.TotalDispatchTimeUs != sumDispatchUs {
			return nil, nil, fmt.Errorf("metalprof: dispatch sum mismatch: sum=%d us, total_dispatch_time_us=%d us",
				sumDispatchUs, *stats.TotalDispatchTimeUs)
		}
	}

	// Construct separate Replay Profile with source-only frames (no synthetic line 1)
	replayProf := &profile.Profile{
		SampleType: []*profile.ValueType{
			{Type: "samples", Unit: "count"},
			{Type: "gpu_time", Unit: "nanoseconds"},
		},
		PeriodType: &profile.ValueType{
			Type: "gpu_time", Unit: "nanoseconds",
		},
		Comments: []string{
			"Replay profile generated from headless MTLReplayer and gputrace profiler streamData.",
			"Note: replay durations are derived from command-buffer cumulative dispatch offsets rather than direct hardware counter sampling.",
			"Note: replay dispatches include all captured execution phases (including warmup).",
		},
	}

	var observations []metalcapture.Observation
	funcByName := make(map[string]*profile.Function)
	locByName := make(map[string]*profile.Location)

	getLoc := func(name string) *profile.Location {
		if loc, ok := locByName[name]; ok {
			return loc
		}
		fn, ok := funcByName[name]
		if !ok {
			fn = &profile.Function{
				ID:         uint64(len(replayProf.Function) + 1),
				Name:       name,
				SystemName: name,
			}
			replayProf.Function = append(replayProf.Function, fn)
			funcByName[name] = fn
		}
		loc := &profile.Location{
			ID: uint64(len(replayProf.Location) + 1),
			Line: []profile.Line{
				{Function: fn}, // Source-only anchor: no synthetic line number fabricated
			},
		}
		replayProf.Location = append(replayProf.Location, loc)
		locByName[name] = loc
		return loc
	}

	deviceName := ""
	if stats.MetalDeviceName != nil {
		deviceName = *stats.MetalDeviceName
	}

	for i, d := range stats.Dispatches {
		name := "unknown_kernel"
		if d.FunctionName != nil && *d.FunctionName != "" {
			name = *d.FunctionName
		} else if d.FunctionIndex != nil {
			name = fmt.Sprintf("function_%d", *d.FunctionIndex)
		}

		nanos := *d.DurationUs * 1000
		loc := getLoc(name)

		sampleLabels := map[string][]string{
			"kernel": {name},
			"origin": {"replay"},
			"phase":  {"replay_trace"},
		}
		if d.PipelineID != nil {
			sampleLabels["pipeline_id"] = []string{fmt.Sprintf("%d", *d.PipelineID)}
		}
		if deviceName != "" {
			sampleLabels["device"] = []string{deviceName}
		}

		replayProf.Sample = append(replayProf.Sample, &profile.Sample{
			Location: []*profile.Location{loc},
			Value:    []int64{1, nanos},
			Label:    sampleLabels,
		})

		dispatchIdx := i
		if d.Index != nil {
			dispatchIdx = *d.Index
		}

		obsID := fmt.Sprintf("replay_dispatch_%d_inferred_duration", dispatchIdx)
		observations = append(observations, metalcapture.Observation{
			ID:          obsID,
			Name:        "Replay Inferred Dispatch Duration (Cumulative Offset Delta)",
			Unit:        "nanoseconds",
			Kind:        metalcapture.MetricKindDuration,
			Aggregation: metalcapture.AggregationSum,
			Scope:       metalcapture.ScopeDispatch,
			Origin:      metalcapture.OriginReplay,
			Status:      metalcapture.StatusCollected,
			Device:      deviceName,
			Pipeline:    name,
			PipelineID:  d.PipelineID,
			DispatchID:  &dispatchIdx,
			Value:       metalcapture.NewIntObservationValue(nanos),
		})

		if d.SIMDShare != nil {
			observations = append(observations, metalcapture.Observation{
				ID:          fmt.Sprintf("replay_dispatch_%d_simd_share", dispatchIdx),
				Name:        "Replay SIMD Execution Share",
				Unit:        "ratio",
				Kind:        metalcapture.MetricKindRatio,
				Aggregation: metalcapture.AggregationMean,
				Scope:       metalcapture.ScopeDispatch,
				Origin:      metalcapture.OriginReplay,
				Status:      metalcapture.StatusCollected,
				Device:      deviceName,
				Pipeline:    name,
				PipelineID:  d.PipelineID,
				DispatchID:  &dispatchIdx,
				Value:       metalcapture.NewFloatObservationValue(*d.SIMDShare),
				NonAdditive: true,
			})
		}
	}

	for _, p := range stats.Pipelines {
		pipeName := "unknown_pipeline"
		if p.FunctionName != nil && *p.FunctionName != "" {
			pipeName = *p.FunctionName
		}
		pipeIDStr := ""
		if p.PipelineID != nil {
			pipeIDStr = fmt.Sprintf("_%d", *p.PipelineID)
		}
		if p.InstructionCount != nil {
			observations = append(observations, metalcapture.Observation{
				ID:          fmt.Sprintf("compiled_pipeline%s_%s_instructions", pipeIDStr, pipeName),
				Name:        "Compiled Pipeline Instruction Count",
				Unit:        "instructions",
				Kind:        metalcapture.MetricKindCount,
				Aggregation: metalcapture.AggregationMax,
				Scope:       metalcapture.ScopePipeline,
				Origin:      metalcapture.OriginReplay,
				Status:      metalcapture.StatusCollected,
				Device:      deviceName,
				Pipeline:    pipeName,
				PipelineID:  p.PipelineID,
				Value:       metalcapture.NewIntObservationValue(*p.InstructionCount),
				NonAdditive: true,
			})
		}
		if p.SpilledBytes != nil {
			observations = append(observations, metalcapture.Observation{
				ID:          fmt.Sprintf("compiled_pipeline%s_%s_spilled_bytes", pipeIDStr, pipeName),
				Name:        "Compiled Pipeline Spilled Memory",
				Unit:        "bytes",
				Kind:        metalcapture.MetricKindBytes,
				Aggregation: metalcapture.AggregationMax,
				Scope:       metalcapture.ScopePipeline,
				Origin:      metalcapture.OriginReplay,
				Status:      metalcapture.StatusCollected,
				Device:      deviceName,
				Pipeline:    pipeName,
				PipelineID:  p.PipelineID,
				Value:       metalcapture.NewIntObservationValue(*p.SpilledBytes),
				NonAdditive: true,
			})
		}
	}

	return replayProf, observations, nil
}
