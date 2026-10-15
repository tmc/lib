// Package cudalock coordinates exclusive GPU access across cooperating processes.
//
// Protocol v2 uses a permanent sibling file, lock.guard, beside the owner
// directory (by default $HOME/ml-explore/.gpu-lock/lock). On Linux and Darwin,
// an OS advisory flock serializes owner inspection, stale recovery, publication
// and root retirement. Reentrant and inherited admission also verifies the
// exact owner snapshot under this guard. The guard closes before Acquire returns;
// GPU work holds the owner directory, not an advisory lock. Process death drops
// a held advisory lock without deleting or replacing its inode.
//
// Every client must use the same physical parent and guard inode. Never remove
// the guard, rename its parent, or replace that namespace while clients run.
// Symlinked lock leaves, aliased guard inodes and observed namespace replacement
// are refused. The parent may be reached through symlinks; it is resolved before
// selecting the lock and guard. This is a cooperative local-filesystem protocol,
// not protection against concurrent uncoordinated namespace modification.
//
// The repository reference/gpu-lock.sh implements v2 for Linux with util-linux
// flock. Older Go clients, installed scripts and gpulease are not qualified by
// owner-file compatibility. Mixed versions remain unsafe. Native admission must
// remain held until the actual participants and installed script are coordinated
// and independently checked. Other operating systems are refused.
//
// Default acquisition publishes a complete owner record by staging and rename.
// Dead local owners are recovered automatically; live and foreign-host owners
// are retained. Missing, empty or malformed owners have a 15-second grace
// period. WithNoReclaim uses the same guard but retains every contended lock.
// Reentrant and inherited references do not retire an ancestor's directory.
// Inherited verifies without waiting and can return ErrNotInherited on a busy
// guard or process state; that means unverified, not that no owner exists.
//
// Failed retirement restores only an intact original inode/owner snapshot into
// an absent canonical name. Otherwise the handle retains explicit cleanup
// custody and refuses new local admission. Retry cleans only its original
// retired inode and cannot remove a newer canonical owner.
// Acquisition rechecks cancellation after process-state waits and before
// admission. Cancellation after publication retires its exact root under the
// guard. If cleanup fails, the error preserves both causes and implements
// io.Closer for retry; this cleanup owner cannot authorize work even if its
// canonical owner was restored. Obtain the closer with errors.As.
// Status may query nvidia-smi; acquisition does not query the device.
//
// Basic usage:
//
//	handle, err := cudalock.Acquire(ctx, cudalock.WithLabel("my-job"))
//	if err != nil {
//		return err
//	}
//	defer handle.Release()
//
// This package is outside the compatibility promise of
// github.com/tmc/lib/nvidia. Its API may change or be removed in patch releases.
package cudalock
