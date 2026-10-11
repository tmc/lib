package nvtelemetry

import (
	"fmt"
	"io"
)

// Replay reads a recorded stream and computes window and energy summaries across all declared phases.
func Replay(r io.Reader, maxBytes int64, opts WindowOptions) (*Manifest, map[string][]WindowSummary, map[string][]EnergySummary, error) {
	manifest, phases, samples, err := ReadStream(r, maxBytes)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("read stream: %w", err)
	}

	windowSummaries := make(map[string][]WindowSummary)
	energySummaries := make(map[string][]EnergySummary)

	// If explicit phases were marked, aggregate each phase
	for _, p := range phases {
		ws, es := AggregateWindow(p.Name, p.StartMonotonicNanos, p.EndMonotonicNanos, samples, opts)
		windowSummaries[p.Name] = ws
		energySummaries[p.Name] = es
	}

	// Always compute a full-run window from start to end
	if manifest.EndTimeMonotonic > manifest.StartTimeMonotonic {
		ws, es := AggregateWindow("full_run", manifest.StartTimeMonotonic, manifest.EndTimeMonotonic, samples, opts)
		windowSummaries["full_run"] = ws
		energySummaries["full_run"] = es
	}

	return manifest, windowSummaries, energySummaries, nil
}
