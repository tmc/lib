// Package cudalock coordinates exclusive GPU access across processes and tests.
//
// It implements an atomic filesystem mutex protocol that is fully interoperable
// with the reference bash gpu-lock script and gpulease on DGX and multi-agent
// GPU hosts:
//
//	$HOME/ml-explore/.gpu-lock/lock
//
// Acquisition is performed by creating a staging directory with the owner
// record inside and atomically renaming it into place, ensuring the lock never
// exists without naming its holder. An owner record (pid, host, label, epoch,
// at) identifies the holder process.
//
// Unlike naive time-based lock acquisition, cudalock performs process liveness
// checks (signal 0) to distinguish active long-running jobs from genuinely dead
// processes, refuses to reclaim locks owned by foreign hosts, and queries device
// compute processes via nvidia-smi. Locks with missing, empty, or unparsable
// owner files are treated as abandoned and reclaimed only after exceeding a
// 15-second grace period measured from their modification time.
//
// Basic usage:
//
//	handle, err := cudalock.Acquire(ctx, cudalock.WithLabel("my-job"))
//	if err != nil {
//		return err
//	}
//	defer handle.Release()
//
// For testing integration, package cudatest automatically acquires cudalock
// in cudatest.Require to prevent test collision on shared GPUs.
//
// This package is outside the compatibility promise of
// github.com/tmc/lib/nvidia. Its API may change or be removed in patch releases.
package cudalock
