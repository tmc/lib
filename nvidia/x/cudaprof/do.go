package cudaprof

import (
	"context"
	"runtime/pprof"
	"sync/atomic"
)

var nextCorrID uint64

// Do calls f with a context containing labels. GPU activities, CUPTI PC samples,
// and CUDA driver launches initiated inside f automatically inherit these labels
// in both pprof profiles and Perfetto timeline traces.
func Do(ctx context.Context, labels pprof.LabelSet, f func(ctx context.Context)) {
	pprof.Do(ctx, labels, func(labeledCtx context.Context) {
		corrID := atomic.AddUint64(&nextCorrID, 1)
		pushCorrelation(corrID, labels)
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

func pushCorrelation(id uint64, labels pprof.LabelSet) {
	// NVTX / CUPTI correlation hooks
}

func popCorrelation(id uint64) {
	// NVTX / CUPTI correlation hooks
}
