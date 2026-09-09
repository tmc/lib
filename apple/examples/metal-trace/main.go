// Command metal-trace executes real Metal GPU compute work, measures GPU intervals,
// verifies results against a CPU reference, and packages output into a single portable capture.pprof.
package main

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"runtime/trace"
	"time"

	"github.com/google/pprof/profile"
	"github.com/tmc/apple/objectivec"
	"github.com/tmc/lib/apple/x/metal"
	"github.com/tmc/lib/apple/x/metal/metalcapture"
	"github.com/tmc/lib/apple/x/metal/metalprof"
)

func sha256Sum(b []byte) [32]byte {
	return sha256.Sum256(b)
}

type config struct {
	elements int
	rounds   int
	passes   int
	timing   string
	outDir   string
	capture  bool
	enrich   bool
}

func main() {
	log.SetFlags(0)
	if err := run(); err != nil {
		log.Fatalf("metal-trace: %v", err)
	}
}

func run() error {
	var cfg config
	flag.IntVar(&cfg.elements, "elements", 1048576, "number of float32 elements")
	flag.IntVar(&cfg.rounds, "rounds", 512, "number of dependent FMA rounds")
	flag.IntVar(&cfg.passes, "passes", 12, "number of measured passes")
	flag.StringVar(&cfg.timing, "timing", "auto", "timing mode: auto, counter, command-buffer")
	flag.StringVar(&cfg.outDir, "out", "", "output directory (defaults to ~/tmp/metal-trace-<timestamp>)")
	flag.BoolVar(&cfg.capture, "capture", false, "record programmatic .gputrace capture")
	flag.BoolVar(&cfg.enrich, "enrich", false, "enrich capture with deep gputrace replay profile and observations")
	flag.Parse()

	if cfg.elements <= 0 || cfg.rounds <= 0 || cfg.passes <= 0 {
		return fmt.Errorf("elements, rounds, and passes must be positive integers")
	}

	if cfg.outDir == "" {
		home, err := os.UserHomeDir()
		if err != nil {
			return err
		}
		parent := filepath.Join(home, "tmp")
		if err := os.MkdirAll(parent, 0755); err != nil {
			return err
		}
		cfg.outDir = filepath.Join(parent, fmt.Sprintf("metal-trace-%d", time.Now().UnixNano()))
	}

	if err := os.MkdirAll(cfg.outDir, 0755); err != nil {
		return fmt.Errorf("create output directory: %w", err)
	}

	tracePath := filepath.Join(cfg.outDir, "go.trace")
	traceFile, err := os.Create(tracePath)
	if err != nil {
		return fmt.Errorf("create trace file: %w", err)
	}
	defer traceFile.Close()

	if err := trace.Start(traceFile); err != nil {
		return fmt.Errorf("start trace: %w", err)
	}
	defer trace.Stop()

	ctx, task := trace.NewTask(context.Background(), "metal-trace: workload execution")
	defer task.End()

	dev, err := metal.NewDevice()
	if err != nil {
		return fmt.Errorf("initialize metal device: %w", err)
	}
	defer dev.Close()

	shaderBytes, err := os.ReadFile(filepath.Join("kernels.metal"))
	if err != nil {
		// Fallback to local embedded constant or current directory
		shaderBytes = []byte(defaultShaderSource)
	}

	lib, err := dev.CompileLibrary(string(shaderBytes))
	if err != nil {
		return fmt.Errorf("compile MSL library: %w", err)
	}
	defer objectivec.ObjectFromID(lib.GetID()).Release()

	builder := metalprof.NewBuilder()
	code, _ := metalcapture.LoadCode(&metalcapture.Capture{
		Files: map[string][]byte{"kernels.metal": shaderBytes},
		Code:  metalcapture.CodeFiles{Source: "kernels.metal"},
	})

	var captureSession *metalprof.CaptureSession
	if cfg.capture {
		gputraceDir := filepath.Join(cfg.outDir, "run.gputrace")
		captureSession, err = metalprof.StartCapture(dev, gputraceDir)
		if err != nil {
			log.Printf("warning: programmatic capture disabled or failed: %v", err)
		}
	}

	workloadRes, err := runWorkload(ctx, dev, lib, builder, code, cfg)
	if err != nil {
		return fmt.Errorf("workload execution: %w", err)
	}

	if captureSession != nil {
		_ = captureSession.Stop()
	}

	trace.Stop()
	traceFile.Close()

	// Encode pprof profile
	var profBuf bytes.Buffer
	if err := builder.Encode(&profBuf); err != nil {
		return fmt.Errorf("encode profile: %w", err)
	}

	p, err := profile.ParseData(profBuf.Bytes())
	if err != nil {
		return fmt.Errorf("parse generated profile: %w", err)
	}

	// Read trace bytes with explicit error checking
	traceData, err := os.ReadFile(tracePath)
	if err != nil {
		return fmt.Errorf("read generated trace file: %w", err)
	}

	// Separately compile AIR from source for offline inspection.
	// Note: this is an offline compilation with xcrun metal; it is recorded with
	// toolchain details and an explicit lack of exact-pipeline identity with the
	// runtime-compiled MTLLibrary pipeline.
	var airBytes []byte
	var airMeta map[string]any
	airTmp := filepath.Join(cfg.outDir, "kernels.air")
	srcTmp := filepath.Join(cfg.outDir, "kernels.metal")
	if err := os.WriteFile(srcTmp, shaderBytes, 0644); err != nil {
		return fmt.Errorf("write source file for AIR compilation: %w", err)
	}
	cmd := exec.Command("xcrun", "-sdk", "macosx", "metal", "-c", srcTmp, "-o", airTmp)
	if out, err := cmd.CombinedOutput(); err == nil {
		data, readErr := os.ReadFile(airTmp)
		if readErr != nil {
			return fmt.Errorf("read compiled AIR file: %w", readErr)
		}
		airBytes = data
		_ = os.Remove(airTmp)
		airMeta = map[string]any{
			"tool":                    "xcrun -sdk macosx metal -c",
			"exact_pipeline_identity": false,
			"source_bytes_sha256":     fmt.Sprintf("%x", sha256Sum(shaderBytes)),
			"air_bytes_sha256":        fmt.Sprintf("%x", sha256Sum(airBytes)),
		}
	} else {
		log.Printf("note: metal compile to AIR skipped: %v (output: %s)", err, string(out))
	}

	// Assemble Capture
	files := map[string][]byte{
		"kernels.metal": shaderBytes,
		"go.trace":      traceData,
	}
	codeFiles := metalcapture.CodeFiles{
		Source: "kernels.metal",
	}
	if len(airBytes) > 0 {
		files["kernels.air"] = airBytes
		codeFiles.AIR = "kernels.air"
	}

	meta := map[string]any{
		"device":                 dev.Name(),
		"elements":               cfg.elements,
		"fma_rounds":             cfg.rounds,
		"passes":                 cfg.passes,
		"timing_requested":       workloadRes.RequestedTiming,
		"timing_actual":          workloadRes.ActualTiming,
		"timing_scope":           workloadRes.TimingScope,
		"timing_fallback_reason": workloadRes.FallbackReason,
		"verified_results":       workloadRes.Verified,
		"max_absolute_error":     workloadRes.MaxError,
		"intervals_nanos":        workloadRes.Intervals,
		"air_compilation":        airMeta,
	}
	metaBytes, err := json.Marshal(meta)
	if err != nil {
		return fmt.Errorf("marshal capture metadata: %w", err)
	}

	var liveObs []metalcapture.Observation
	for kName, intervals := range workloadRes.Intervals {
		for passIdx, ns := range intervals {
			liveObs = append(liveObs, metalcapture.Observation{
				ID:          fmt.Sprintf("live_pass_%d_%s_duration", passIdx, kName),
				Name:        fmt.Sprintf("Live %s Pass Duration", kName),
				Unit:        "nanoseconds",
				Kind:        metalcapture.MetricKindDuration,
				Aggregation: metalcapture.AggregationSum,
				Scope:       metalcapture.Scope(workloadRes.TimingScope),
				Origin:      metalcapture.OriginLive,
				Status:      metalcapture.StatusCollected,
				Device:      dev.Name(),
				Pipeline:    kName,
				RunID:       fmt.Sprintf("pass_%d", passIdx),
				Value:       metalcapture.NewIntObservationValue(ns),
			})
		}
	}
	liveObs = append(liveObs, metalcapture.Observation{
		ID:          "live_max_absolute_error",
		Name:        "Live Output Max Absolute Error vs CPU Reference",
		Unit:        "error",
		Kind:        metalcapture.MetricKindCount,
		Aggregation: metalcapture.AggregationMax,
		Scope:       metalcapture.ScopePipeline,
		Origin:      metalcapture.OriginLive,
		Status:      metalcapture.StatusCollected,
		Device:      dev.Name(),
		Value:       metalcapture.NewFloatObservationValue(workloadRes.MaxError),
		NonAdditive: true,
	})

	cap := &metalcapture.Capture{
		Profile:      p,
		Files:        files,
		Code:         codeFiles,
		Metadata:     metaBytes,
		Observations: liveObs,
	}

	if cfg.capture && cfg.enrich && captureSession != nil {
		enrichRes, err := metalprof.EnrichTraceReplay(ctx, filepath.Join(cfg.outDir, "run.gputrace"), 60*time.Second)
		if err != nil {
			log.Printf("warning: replay enrichment failed: %v", err)
		} else {
			var replayBuf bytes.Buffer
			if err := enrichRes.Profile.Write(&replayBuf); err == nil {
				cap.Files["replay.pprof"] = replayBuf.Bytes()
			}
			if len(enrichRes.RawExport) > 0 {
				cap.Files["replay_export.json"] = enrichRes.RawExport
			}
			cap.Observations = append(cap.Observations, enrichRes.Observations...)
			fmt.Printf("Enriched Replay:   %d observations, replay.pprof & replay_export.json attached\n", len(enrichRes.Observations))
		}
	}

	capturePprofPath := filepath.Join(cfg.outDir, "capture.pprof")
	capFile, err := os.OpenFile(capturePprofPath, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, 0644)
	if err != nil {
		return fmt.Errorf("create capture.pprof: %w", err)
	}
	defer capFile.Close()

	if err := metalcapture.Write(capFile, cap); err != nil {
		return fmt.Errorf("write capture.pprof: %w", err)
	}

	fmt.Println("=== Metal Profiling Execution Complete ===")
	fmt.Printf("Device:           %s\n", dev.Name())
	fmt.Printf("Elements:         %d\n", cfg.elements)
	fmt.Printf("FMA Rounds:       %d\n", cfg.rounds)
	fmt.Printf("Passes:           %d\n", cfg.passes)
	fmt.Printf("Verified Outputs: %d (warmup excluded)\n", workloadRes.Verified)
	fmt.Printf("Max Error:        %g\n", workloadRes.MaxError)
	fmt.Printf("Timing Mode:      %s (actual: %s, scope: %s)\n", workloadRes.RequestedTiming, workloadRes.ActualTiming, workloadRes.TimingScope)
	if workloadRes.FallbackReason != "" {
		fmt.Printf("Fallback Reason:  %s\n", workloadRes.FallbackReason)
	}
	fmt.Printf("Saved Capture:    %s\n", capturePprofPath)

	return nil
}

const defaultShaderSource = `
#include <metal_stdlib>
using namespace metal;

kernel void vector_add(device const float *a [[buffer(0)]],
                       device const float *b [[buffer(1)]],
                       device float *c [[buffer(2)]],
                       constant uint &n [[buffer(3)]],
                       uint id [[thread_position_in_grid]]) {
    if (id < n) {
        c[id] = a[id] + b[id];
    }
}

kernel void compute(device float *c [[buffer(0)]],
                    constant uint &n [[buffer(1)]],
                    constant uint &rounds [[buffer(2)]],
                    uint id [[thread_position_in_grid]]) {
    if (id < n) {
        float v = c[id];
        for (uint r = 0; r < rounds; r++) {
            v = fma(v, 1.000001f, 0.0001f);
        }
        c[id] = v;
    }
}
`
