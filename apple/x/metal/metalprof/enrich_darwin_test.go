package metalprof_test

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/tmc/lib/apple/x/metal/metalprof"
)

func TestEnrichTraceReplayTimeoutScratchRetention(t *testing.T) {
	// Create a real dummy trace directory so ReplayTraceContext begins execution
	dir := t.TempDir()
	dummyTrace := filepath.Join(dir, "dummy.gputrace")
	if err := os.MkdirAll(dummyTrace, 0755); err != nil {
		t.Fatal(err)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 1*time.Millisecond)
	defer cancel()

	time.Sleep(2 * time.Millisecond) // Ensure context is already done
	_, err := metalprof.EnrichTraceReplay(ctx, dummyTrace, 1*time.Millisecond)
	if err == nil {
		t.Fatal("expected error on expired context")
	}

	// Verify errors.Is for both the unresolved ownership sentinel and context deadline
	if !errors.Is(err, metalprof.ErrReplayUnresolvedOwnership) {
		t.Errorf("error %v does not wrap ErrReplayUnresolvedOwnership", err)
	}
	if !errors.Is(err, context.DeadlineExceeded) && !errors.Is(err, context.Canceled) {
		t.Errorf("error %v does not wrap context deadline or cancellation", err)
	}
}
