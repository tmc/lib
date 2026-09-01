package cudaprof

import (
	"context"
	"runtime/pprof"
	"strings"
	"sync/atomic"

	"github.com/tmc/lib/nvidia/x/nvtx"
)

var nextCorrID uint64

// Do calls f with a context containing labels. GPU activities, CUPTI PC samples,
// and CUDA driver launches initiated inside f automatically inherit these labels
// in both pprof profiles and Perfetto timeline traces.
func Do(ctx context.Context, labels pprof.LabelSet, f func(ctx context.Context)) {
	pprof.Do(ctx, labels, func(labeledCtx context.Context) {
		corrID := atomic.AddUint64(&nextCorrID, 1)
		pushCorrelation(labeledCtx, corrID)
		defer popCorrelation(corrID)

		f(labeledCtx)
	})
}

// Labels returns a pprof.LabelSet for the given key/value pairs.
func Labels(args ...string) pprof.LabelSet {
	return pprof.Labels(args...)
}

// Label returns a copy of ctx in which the pprof label with the given key is set to value.
func Label(ctx context.Context, key, value string) context.Context {
	return pprof.WithLabels(ctx, pprof.Labels(key, value))
}

func pushCorrelation(ctx context.Context, id uint64) {
	if nvtx.Enabled() {
		var parts []string
		pprof.ForLabels(ctx, func(key, value string) bool {
			parts = append(parts, key+"="+value)
			return true
		})
		msg := "cudaprof.Do"
		if len(parts) > 0 {
			msg += " (" + strings.Join(parts, ", ") + ")"
		}
		nvtx.RangePush(msg)
	}
}

func popCorrelation(id uint64) {
	if nvtx.Enabled() {
		nvtx.RangePop()
	}
}

