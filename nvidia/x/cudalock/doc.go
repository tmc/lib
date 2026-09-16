// Package cudalock coordinates exclusive GPU access across processes and tests.
//
// It implements an atomic filesystem mutex protocol that is fully interoperable
// with the reference bash gpu-lock script and gpulease on DGX and multi-agent
// GPU hosts:
//
//	$HOME/ml-explore/.gpu-lock/lock
//
// Acquisition is performed via atomic directory creation (os.Mkdir). An owner
// record (pid, host, label, epoch, at) is written inside the lock directory upon
// acquisition.
//
// Unlike naive time-based lock acquisition, cudalock performs process liveness
// checks (signal 0) to distinguish active long-running jobs from genuinely dead
// processes, refuses to reclaim locks owned by foreign hosts, and queries device
// compute processes via nvidia-smi.
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
package cudalock
