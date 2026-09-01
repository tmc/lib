package metalprof

import (
	"context"
	"fmt"
	"runtime/pprof"
	"sort"
	"strings"

	apple "github.com/tmc/apple/metal"
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

// PushDebugGroup attaches an active pprof context label string as a Metal debug scope.
func PushDebugGroup(enc apple.MTLCommandEncoder, groupName string) {
	if enc.GetID() != 0 && groupName != "" {
		enc.PushDebugGroup(groupName)
	}
}

// PopDebugGroup pops the current Metal debug scope.
func PopDebugGroup(enc apple.MTLCommandEncoder) {
	if enc.GetID() != 0 {
		enc.PopDebugGroup()
	}
}

// InsertDebugSignpost inserts an instant debug signpost marker into the Metal command stream.
func InsertDebugSignpost(enc apple.MTLCommandEncoder, signpost string) {
	if enc.GetID() != 0 && signpost != "" {
		enc.InsertDebugSignpost(signpost)
	}
}
