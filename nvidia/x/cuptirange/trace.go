package cuptirange

import (
	"errors"
	"fmt"
	"math"
	"strings"

	"github.com/tmc/lib/nvidia/x/perfetto"
)

// Metric is a named evaluated value. Name should be the exact CUPTI metric
// name, including its aggregation suffix; no unit conversion is performed.
type Metric struct {
	Name  string
	Value float64
}

// WriteMetrics writes a metric snapshot to consecutive Perfetto counter tracks
// starting at firstTrack. The caller owns those track IDs and supplies timestamp
// in its trace clock domain. A range metric has no implied sampling timestamp;
// do not treat this display position as a measured kernel start or end time.
// All values are validated before writing. Writer errors may leave a partial
// trace, which the caller must discard or otherwise mark incomplete.
func WriteMetrics(w *perfetto.Writer, firstTrack, timestamp uint64, metrics []Metric) error {
	if w == nil || len(metrics) == 0 || firstTrack == 0 || uint64(len(metrics)-1) > ^uint64(0)-firstTrack {
		return errors.New("cuptirange: invalid metric export arguments")
	}
	seen := make(map[string]bool)
	for _, metric := range metrics {
		if metric.Name == "" || strings.ContainsRune(metric.Name, 0) || seen[metric.Name] || math.IsNaN(metric.Value) || math.IsInf(metric.Value, 0) {
			return fmt.Errorf("cuptirange: invalid exported metric %q", metric.Name)
		}
		seen[metric.Name] = true
	}
	for i, metric := range metrics {
		track := firstTrack + uint64(i)
		if err := w.WriteCounterDescriptor(track, metric.Name, "", 0); err != nil {
			return fmt.Errorf("cuptirange: metric descriptor: %w", err)
		}
		if err := w.WriteDoubleCounterValue(track, timestamp, metric.Value); err != nil {
			return fmt.Errorf("cuptirange: metric value: %w", err)
		}
	}
	return nil
}
