package cudaprof

import (
	"context"
	"os"
	"runtime"
	"runtime/pprof"
	"strings"
	"sync/atomic"

	"github.com/tmc/lib/nvidia/cupti"
	"github.com/tmc/lib/nvidia/x/nvtx"
)

var nextCorrID uint64

// Correlation dispositions. A reader must never have to infer from an absent
// external-correlation record whether correlation was asked for, was refused,
// or simply found nothing to correlate.
const (
	// CorrelationOff means Do pushes no external correlation ID. This is the
	// default: the first push loads libcupti into the process, which a
	// labelling helper must not do uninvited.
	CorrelationOff = "off"
	// CorrelationPushing means every region so far pushed and popped its ID.
	CorrelationPushing = "pushing"
	// CorrelationUnavailable means correlation was enabled but libcupti or
	// the external-correlation symbols are not callable in this process.
	CorrelationUnavailable = "unavailable"
	// CorrelationPushFailed means CUPTI rejected a push.
	CorrelationPushFailed = "push_failed"
	// CorrelationPopMismatch means a pop returned an ID other than the one
	// this region pushed, so the per-thread stack was not the one entered.
	CorrelationPopMismatch = "pop_mismatch"
)

// correlationKind is the namespace Do pushes into, or Invalid when correlation
// is off. It is read on every region, so it is atomic rather than mutex-guarded.
var correlationKind atomic.Int32

// correlationDisposition holds one of the Correlation* constants. It only ever
// moves away from CorrelationPushing: a failure is sticky, because a reader
// asking after the fact needs to know a push failed, not that the last one
// happened to succeed.
var correlationDisposition atomic.Value

// Correlate makes Do push id onto the calling thread's CUPTI external
// correlation stack for kind, so that every CUPTI activity recorded inside the
// region is preceded by an ActivityExternalCorrelation record naming it. That
// is what lets [LabelExternal] label a kernel with the Go region it was
// launched from.
//
// It is opt-in because the first push loads libcupti into the process. Passing
// [cupti.ExternalCorrelationKindInvalid] turns correlation back off.
//
// Custom0 is the conventional namespace for a first-party framework, and is
// what CUDAPROF_CORRELATE selects. A target that also pushes another framework's
// scopes -- PyTorch uses Custom0 and Custom1 -- publishes two namespaces, and a
// consumer joins only the lowest-numbered one; the capture reports the rest as
// unjoined rather than as work nothing was pushed for.
//
// Correlate is safe to call from any goroutine, but a region already running
// keeps the kind it entered with.
func Correlate(kind cupti.ExternalCorrelationKind) {
	correlationKind.Store(int32(kind))
	if kind == cupti.ExternalCorrelationKindInvalid {
		correlationDisposition.Store(CorrelationOff)
		return
	}
	correlationDisposition.Store(CorrelationPushing)
}

// CorrelationDisposition reports whether Do is pushing external correlation
// IDs, and if not, why. It returns one of the Correlation* constants.
func CorrelationDisposition() string {
	d, _ := correlationDisposition.Load().(string)
	if d == "" {
		return CorrelationOff
	}
	return d
}

// failCorrelation records a sticky failure. The kind is left in place so a
// caller can see what was attempted.
func failCorrelation(d string) { correlationDisposition.Store(d) }

func init() {
	correlationDisposition.Store(CorrelationOff)
	// CUDAPROF_CORRELATE lets a capture enable correlation in a target whose
	// source does not call Correlate, which is the whole injected-agent path:
	// the agent is preloaded by the launcher, not linked by the program.
	if v := os.Getenv("CUDAPROF_CORRELATE"); v != "" && v != "0" {
		Correlate(cupti.ExternalCorrelationKindCustom0)
	}
}

// Do calls f with a context containing labels, brackets the call with an NVTX
// range when NVTX is available, and pushes a CUPTI external correlation ID for
// the region when [Correlate] has enabled it. Recorded GPU observations inherit
// labels only when the caller passes the labeled context to the recording method.
//
// Both brackets are per-thread: NVTX range nesting and the CUPTI external
// correlation stack belong to the calling OS thread, and CUPTI attributes an ID
// only to activities launched from the thread holding it. A goroutine may move
// between OS threads at any preemption point, which would pop a stack it never
// pushed and leave launches from the new thread uncorrelated. So Do pins the
// goroutine with [runtime.LockOSThread] for the duration of f whenever it has
// pushed something thread-local, in the same way [x/cuda.Context.Run] does. A
// Do with neither bracket active does not pin and costs nothing beyond
// [pprof.Do].
func Do(ctx context.Context, labels pprof.LabelSet, f func(ctx context.Context)) {
	pprof.Do(ctx, labels, func(labeledCtx context.Context) {
		corrID := atomic.AddUint64(&nextCorrID, 1)
		entered := pushCorrelation(labeledCtx, corrID)
		if entered.pinned {
			defer runtime.UnlockOSThread()
		}
		defer popCorrelation(corrID, entered)

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

// regionBrackets records what a region actually entered, so its exit unwinds
// exactly that and no more. It travels on the stack from push to pop rather
// than through package state, because a nested Do may enter a different set
// than its parent and two goroutines hold different regions concurrently.
type regionBrackets struct{ pinned, nvtx, cupti bool }

// pushCorrelation enters the region's thread-local brackets and reports which
// it entered. It pins before pushing anything, so no bracket can be entered on
// one thread and left on another.
func pushCorrelation(ctx context.Context, id uint64) (entered regionBrackets) {
	wantNVTX := nvtx.Enabled()
	kind := cupti.ExternalCorrelationKind(correlationKind.Load())
	wantCUPTI := kind != cupti.ExternalCorrelationKindInvalid
	if !wantNVTX && !wantCUPTI {
		return entered
	}
	runtime.LockOSThread()
	entered.pinned = true
	if wantNVTX {
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
		entered.nvtx = true
	}
	if wantCUPTI {
		// A push that fails is recorded rather than retried: the symbol is
		// either callable in this process or it is not, and a region that
		// silently skipped its push would be indistinguishable from one whose
		// kernels the framework never scoped.
		if err := cupti.ActivityPushExternalCorrelationId(kind, id); err != nil {
			if _, ok := err.(*cupti.SymbolError); ok || !cupti.Loaded() {
				failCorrelation(CorrelationUnavailable)
			} else {
				failCorrelation(CorrelationPushFailed)
			}
		} else {
			entered.cupti = true
		}
	}
	return entered
}

// popCorrelation leaves exactly the brackets pushCorrelation entered, in
// reverse order. The popped correlation ID is checked against the one this
// region pushed: CUPTI pops the calling thread's stack, so a mismatch means the
// region left on a thread other than the one it entered, and every activity
// recorded since is attributed to the wrong scope.
func popCorrelation(id uint64, entered regionBrackets) {
	if entered.cupti {
		kind := cupti.ExternalCorrelationKind(correlationKind.Load())
		got, err := cupti.ActivityPopExternalCorrelationId(kind)
		switch {
		case err != nil:
			failCorrelation(CorrelationPushFailed)
		case got != id:
			failCorrelation(CorrelationPopMismatch)
		}
	}
	if entered.nvtx {
		nvtx.RangePop()
	}
}
