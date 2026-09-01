package metalprof

import (
	"context"
	"fmt"
	"runtime/pprof"
	"sort"
	"strings"
)

// Do calls f with a copy of the parent context into which the supplied
// labels have been merged, and annotates current goroutine CPU profiling.
func Do(ctx context.Context, labels pprof.LabelSet, f func(context.Context)) {
	pprof.Do(ctx, labels, f)
}

// LabelsFromContext extracts active pprof label key-value pairs from the context.
func LabelsFromContext(ctx context.Context) map[string]string {
	result := make(map[string]string)
	pprof.ForLabels(ctx, func(key, value string) bool {
		result[key] = value
		return true
	})
	return result
}

// FormatLabels formats a label map as a sorted, colon-delimited string.
func FormatLabels(labels map[string]string) string {
	if len(labels) == 0 {
		return ""
	}
	keys := make([]string, 0, len(labels))
	for k := range labels {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	var parts []string
	for _, k := range keys {
		parts = append(parts, fmt.Sprintf("%s=%s", k, labels[k]))
	}
	return strings.Join(parts, ",")
}
