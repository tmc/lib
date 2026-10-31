package main

import (
	"fmt"
	"strings"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cudacapture"
	"github.com/tmc/lib/nvidia/x/cudagraph"
)

// parseSyncGuardMode parses the -sync-guard CLI flag into a cudacapture.SyncMode.
// An empty string or "allow" maps to SyncAllow; "warn" maps to SyncWarn;
// "reject" maps to SyncReject.
func parseSyncGuardMode(s string) (cudacapture.SyncMode, error) {
	switch strings.ToLower(strings.TrimSpace(s)) {
	case "", "allow":
		return cudacapture.SyncAllow, nil
	case "warn":
		return cudacapture.SyncWarn, nil
	case "reject":
		return cudacapture.SyncReject, nil
	default:
		return cudacapture.SyncAllow, fmt.Errorf("unknown sync-guard mode %q (want allow, warn, or reject)", s)
	}
}

// newDecodeSyncArbiter creates a cudacapture.SyncArbiter configured with a
// forbidden-sync scope when mode is SyncWarn or SyncReject. When mode is
// SyncAllow, it returns nil so no sync checks are enforced.
func newDecodeSyncArbiter(mode cudacapture.SyncMode, onWarn func(cudacapture.SyncViolation)) *cudacapture.SyncArbiter {
	if mode == cudacapture.SyncAllow {
		return nil
	}
	return cudacapture.NewSyncArbiter(mode, onWarn)
}

// SetSyncArbiter attaches an optional host-synchronization guard arbiter to dg.
// When non-nil, synchronous Enqueue (stream == nil) and Step calls consult
// a.CheckSync before blocking the host on device completion.
func (dg *DecodeGraph) SetSyncArbiter(a *cudacapture.SyncArbiter) {
	if dg == nil {
		return
	}
	dg.syncArbiter = a
}

// AttachPingPongTimer binds a double-buffered cudagraph.PingPongTimer and its
// target EventRecordNodes to dg. Before each graph launch in Enqueue, dg calls
// timer.PrepareReplay(exec, nodes) to swap the active event bank and harvest
// the previous replay's elapsed durations without host-blocking on the first launch.
func (dg *DecodeGraph) AttachPingPongTimer(timer *cudagraph.PingPongTimer, exec driver.GraphExec, nodes []cudagraph.EventRecordNode) error {
	if dg == nil {
		return fmt.Errorf("attach ping-pong timer: nil DecodeGraph")
	}
	if timer == nil {
		return fmt.Errorf("attach ping-pong timer: nil PingPongTimer")
	}
	if exec == 0 {
		return fmt.Errorf("attach ping-pong timer: zero GraphExec handle")
	}
	if len(nodes) == 0 {
		return fmt.Errorf("attach ping-pong timer: empty event record nodes")
	}
	for i, n := range nodes {
		if n.Node() == 0 {
			return fmt.Errorf("attach ping-pong timer: zero event record node at index %d", i)
		}
	}
	dg.pingPong = timer
	dg.eventExec = exec
	dg.eventNodes = append([]cudagraph.EventRecordNode(nil), nodes...)
	return nil
}

// LastReplayDurationsMS returns the most recently harvested event-record
// replay durations in milliseconds from dg's attached PingPongTimer, or nil
// if no timer is attached or fewer than two replays have been prepared.
func (dg *DecodeGraph) LastReplayDurationsMS() []float64 {
	if dg == nil || dg.pingPong == nil {
		return nil
	}
	return dg.pingPong.LastDurationsMS()
}

// newDecodeBucketRunner creates a cudagraph.BucketRunner over DeepInfer's
// piecewise batch-size buckets up to maxBsz and registers dg for every
// configured bucket so padded batch sizes resolve to a ready graph handle.
func newDecodeBucketRunner(maxBsz int, dg *DecodeGraph) (*cudagraph.BucketRunner[*DecodeGraph], error) {
	if dg == nil {
		return nil, fmt.Errorf("decode bucket runner: nil DecodeGraph")
	}
	runner, err := cudagraph.NewBucketRunnerFromPolicy[*DecodeGraph](cudagraph.BucketPolicy{
		MaxBsz: maxBsz,
	})
	if err != nil {
		return nil, err
	}
	for _, b := range runner.WarmupOrder() {
		if err := runner.Register(b, dg); err != nil {
			return nil, err
		}
	}
	return runner, nil
}

// selectDecodeBucket resolves actualBsz to the smallest configured CUDA graph
// bucket >= actualBsz in runner and returns the bound DecodeGraph and padding.
func selectDecodeBucket(runner *cudagraph.BucketRunner[*DecodeGraph], actualBsz int) (*DecodeGraph, cudagraph.BucketPadding, error) {
	if runner == nil {
		return nil, cudagraph.BucketPadding{}, fmt.Errorf("select decode bucket: nil BucketRunner")
	}
	return runner.Select(actualBsz)
}
