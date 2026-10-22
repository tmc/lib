package cudacapture_test

import (
	"errors"
	"sync"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudacapture"
)

func TestSyncModeString(t *testing.T) {
	tests := []struct {
		mode cudacapture.SyncMode
		want string
	}{
		{cudacapture.SyncAllow, "allow"},
		{cudacapture.SyncWarn, "warn"},
		{cudacapture.SyncReject, "reject"},
		{cudacapture.SyncMode(99), "SyncMode(99)"},
	}
	for _, tc := range tests {
		if got := tc.mode.String(); got != tc.want {
			t.Fatalf("SyncMode(%d).String() = %q, want %q", int(tc.mode), got, tc.want)
		}
	}
}

func TestSyncArbiterNestingAndModes(t *testing.T) {
	var warned []cudacapture.SyncViolation
	arb := cudacapture.NewSyncArbiter(cudacapture.SyncAllow, func(v cudacapture.SyncViolation) {
		warned = append(warned, v)
	})

	// Baseline SyncAllow permits CheckSync without recording violations.
	if err := arb.CheckSync("cuStreamSynchronize"); err != nil {
		t.Fatalf("baseline CheckSync: %v", err)
	}
	if got := arb.Violations(); len(got) != 0 {
		t.Fatalf("baseline Violations() = %v, want empty", got)
	}

	// Outer Forbid(SyncReject) rejects syncs.
	restoreOuter := arb.Forbid(cudacapture.SyncReject)
	err := arb.CheckSync("cuMemcpyDtoH")
	if !errors.Is(err, cudacapture.ErrForbiddenSync) {
		t.Fatalf("CheckSync inside Forbid(SyncReject) = %v, want ErrForbiddenSync", err)
	}

	// Inner Allow() overrides outer Forbid(SyncReject), analogous to allow_cuda_sync()
	// inside PingPongTimer.elapsed_time().
	restoreAllow := arb.Allow()
	if err := arb.CheckSync("cuEventSynchronize"); err != nil {
		t.Fatalf("CheckSync inside inner Allow() = %v, want nil", err)
	}

	// Further inner Forbid(SyncWarn) records violation and invokes onWarn without returning an error.
	restoreInnerWarn := arb.Forbid(cudacapture.SyncWarn)
	if err := arb.CheckSync(".item()"); err != nil {
		t.Fatalf("CheckSync inside inner Forbid(SyncWarn) = %v, want nil", err)
	}
	if len(warned) != 1 || warned[0].Op != ".item()" || warned[0].Mode != cudacapture.SyncWarn || warned[0].Depth != 2 || warned[0].AllowedDepth != 1 {
		t.Fatalf("warned = %+v, want 1 warning at Depth=2 AllowedDepth=1", warned)
	}

	// Exiting inner Forbid(SyncWarn) returns to inner Allow().
	restoreInnerWarn()
	restoreInnerWarn() // idempotent second call must be a no-op
	if err := arb.CheckSync("cuEventElapsedTime"); err != nil {
		t.Fatalf("CheckSync after exiting inner warn = %v, want nil", err)
	}

	// Exiting inner Allow() returns to outer Forbid(SyncReject).
	restoreAllow()
	if err := arb.CheckSync("cuCtxSynchronize"); !errors.Is(err, cudacapture.ErrForbiddenSync) {
		t.Fatalf("CheckSync after exiting Allow() = %v, want ErrForbiddenSync", err)
	}

	// Exiting outer Forbid(SyncReject) restores baseline SyncAllow.
	restoreOuter()
	if err := arb.CheckSync("cuCtxSynchronize"); err != nil {
		t.Fatalf("CheckSync after exiting outer Forbid = %v, want nil", err)
	}

	violations := arb.Violations()
	if len(violations) != 3 {
		t.Fatalf("len(Violations()) = %d, want 3: %+v", len(violations), violations)
	}
	if violations[0].Op != "cuMemcpyDtoH" || violations[0].Mode != cudacapture.SyncReject || violations[0].Depth != 1 || violations[0].AllowedDepth != 0 {
		t.Fatalf("violations[0] = %+v", violations[0])
	}
	if violations[1].Op != ".item()" || violations[1].Mode != cudacapture.SyncWarn || violations[1].Depth != 2 || violations[1].AllowedDepth != 1 {
		t.Fatalf("violations[1] = %+v", violations[1])
	}
	if violations[2].Op != "cuCtxSynchronize" || violations[2].Mode != cudacapture.SyncReject || violations[2].Depth != 1 || violations[2].AllowedDepth != 0 {
		t.Fatalf("violations[2] = %+v", violations[2])
	}

	arb.ResetViolations()
	if got := arb.Violations(); got != nil {
		t.Fatalf("Violations() after ResetViolations() = %v, want nil", got)
	}
}

func TestSyncArbiterInterleavedConcurrency(t *testing.T) {
	// Matches test_interleaved_exit_no_leak from DeepInfer's test_cuda_sync_guards.py:
	// Goroutine A enters Forbid -> Goroutine B enters Allow -> Goroutine A exits -> Goroutine B exits.
	// Neither inside B's Allow nor after both exit may SyncReject leak.
	arb := cudacapture.NewSyncArbiter(cudacapture.SyncAllow, nil)

	aEntered := make(chan struct{})
	bEntered := make(chan struct{})
	aExited := make(chan struct{})

	var wg sync.WaitGroup
	wg.Add(2)

	go func() {
		defer wg.Done()
		restoreA := arb.Forbid(cudacapture.SyncReject)
		close(aEntered)
		<-bEntered
		restoreA()
		close(aExited)
	}()

	var bErrBeforeAExit, bErrAfterAExit error
	go func() {
		defer wg.Done()
		<-aEntered
		restoreB := arb.Allow()
		bErrBeforeAExit = arb.CheckSync("cuEventSynchronize")
		close(bEntered)
		<-aExited
		bErrAfterAExit = arb.CheckSync("cuEventSynchronize")
		restoreB()
	}()

	wg.Wait()

	if bErrBeforeAExit != nil {
		t.Fatalf("bErrBeforeAExit = %v, want nil", bErrBeforeAExit)
	}
	if bErrAfterAExit != nil {
		t.Fatalf("bErrAfterAExit = %v, want nil", bErrAfterAExit)
	}
	if err := arb.CheckSync("cuCtxSynchronize"); err != nil {
		t.Fatalf("post-exit CheckSync leaked forbidden state: %v", err)
	}

	// Concurrent stress test across multiple goroutines entering/exiting Forbid and Allow.
	var stressWG sync.WaitGroup
	for i := range 8 {
		stressWG.Add(1)
		go func(id int) {
			defer stressWG.Done()
			for j := range 200 {
				if (id+j)%2 == 0 {
					r1 := arb.Forbid(cudacapture.SyncWarn)
					_ = arb.CheckSync("cuStreamSynchronize")
					r2 := arb.Allow()
					_ = arb.CheckSync("cuEventSynchronize")
					r2()
					r1()
				} else {
					r := arb.Allow()
					_ = arb.CheckSync("cuMemcpyDtoH")
					r()
				}
			}
		}(i)
	}
	stressWG.Wait()
	arb.ResetViolations()

	if err := arb.CheckSync("cuCtxSynchronize"); err != nil {
		t.Fatalf("after concurrent stress test, baseline CheckSync = %v, want nil", err)
	}
}
