package metalprof

import (
	"context"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"time"

	"github.com/google/pprof/profile"
	"github.com/tmc/lib/apple/x/metal/metalcapture"
)

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
