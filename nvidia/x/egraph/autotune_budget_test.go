package egraph

import (
	"context"
	"runtime"
	"sync/atomic"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/x/cudatest"
)

// TestMeasureExpressionOutlivesBudget checks that the search deadline stops
// the block-size sweep between candidates but does not cancel a compile and
// benchmark already in progress. A deadline far shorter than one NVRTC
// compile must still yield exactly one measured candidate.
func TestMeasureExpressionOutlivesBudget(t *testing.T) {
	dev := cudatest.Require(t)
	defer dev.Close()
	expr := &Expr{Op: "fused_rmsnorm_gemv", Children: []*Expr{{Op: "x"}, {Op: "w"}}}
	const budget = 5 * time.Millisecond
	ctx := &compileWatchContext{Context: context.Background()}
	deadline := time.Now().Add(budget)
	cand, latency, err := NewTuner(dev).measureExpression(ctx, expr, deadline)
	if err != nil {
		t.Fatalf("measureExpression: %v", err)
	}
	if n := ctx.compiles.Load(); n != 1 {
		t.Fatalf("compiled %d candidates after the budget expired during the first, want 1", n)
	}
	if time.Now().Before(deadline) {
		t.Fatalf("measurement finished within the %v budget; the test needs a compile that outlives it", budget)
	}
	if latency <= 0 || cand.CUDASource == "" || cand.BlockDim != autotuneBlockSizes[0] || cand.GridDim != benchOutDim {
		t.Fatalf("candidate=%+v latency=%v, want first block size %d measured", cand, latency, autotuneBlockSizes[0])
	}
}

// TestMeasureExpressionHonorsCancel checks that the caller's context, unlike
// the budget, reaches the candidate's NVRTC compile, so cancelling it stops
// the candidate in flight rather than only at the next block size.
func TestMeasureExpressionHonorsCancel(t *testing.T) {
	dev := cudatest.Require(t)
	defer dev.Close()
	expr := &Expr{Op: "fused_rmsnorm_gemv", Children: []*Expr{{Op: "x"}, {Op: "w"}}}
	tuner := NewTuner(dev)

	ctx := &compileWatchContext{Context: context.Background()}
	if _, _, err := tuner.measureExpression(ctx, expr, time.Now().Add(5*time.Millisecond)); err != nil {
		t.Fatalf("measureExpression: %v", err)
	}
	if ctx.compiles.Load() == 0 {
		t.Fatal("caller's context never reached the nvrtcx compile")
	}

	cancelCtx, cancel := context.WithTimeout(context.Background(), 5*time.Millisecond)
	defer cancel()
	if _, _, err := tuner.measureExpression(cancelCtx, expr, time.Now().Add(time.Hour)); err != context.DeadlineExceeded {
		t.Fatalf("measureExpression err = %v, want %v", err, context.DeadlineExceeded)
	}
}

// compileWatchContext counts nvrtcx.Compile calls made with it. Compile
// checks its context once on entry, before any cache lookup.
type compileWatchContext struct {
	context.Context
	compiles atomic.Int32
}

func (c *compileWatchContext) Err() error {
	pcs := make([]uintptr, 1)
	if runtime.Callers(2, pcs) == 1 {
		f, _ := runtime.CallersFrames(pcs).Next()
		if f.Function == "github.com/tmc/lib/nvidia/x/nvrtcx.Compile" {
			c.compiles.Add(1)
		}
	}
	return c.Context.Err()
}
