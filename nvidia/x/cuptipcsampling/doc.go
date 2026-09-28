// Package cuptipcsampling provides bounded modern CUPTI PC sampling.
// StallReasons queries native reason IDs and names for an owned CUDA context on
// which sampling is already enabled; Enable turns it on and returns its own off.
// IdentifyCubin obtains a native code identifier and an exact artifact digest.
// Start owns an explicit sampling range with caller-supplied memory and read
// limits. Callers coordinate workload completion, preserve every Read batch,
// and call Close before destroying the context. Start may return both a session
// and an error when native cleanup ownership must be retained. Enable likewise
// returns a cleanup function with an error if native enable succeeded but its
// returned parameters were invalid. Both cleanup paths retry failed disable.
// PC sampling excludes cooperating Range, PM and SASS collectors on its device;
// callers still coordinate external profilers.
//
// RingBuffer (also aliased as MultiBuffer) provides zero-copy circular
// multi-buffering for PC sampling and CUPTI activity streams, enabling
// asynchronous non-blocking handoffs and concurrent background draining during
// high-rate profiling without stopping sampling.
//
// CapturePhase composes the PC sampling lifecycle into a bounded phase operation
// with explicit limits, workload completion and device synchronization, final
// draining, and cleanup, reporting loss, remaining samples, consumer failure,
// and cancellation separately.
// ProfileBridge exposes named-PC, dropped, and unattributed sample counts.
// Its stall coverage uses all three counts in the denominator, so CUPTI loss
// and empty kernel names lower coverage rather than disappearing from it.
//
// Collection is restricted to Linux amd64/arm64 and CUPTI 13.x layouts. A 13.x
// release newer than the newest qualified one uses that release's layout.
// This layout restriction is not native hardware qualification. Native reason
// counts are not legacy stall enums or inferred instruction metrics.
//
// This package is outside the compatibility promise of
// github.com/tmc/lib/nvidia. Its API may change or be removed in patch releases.
package cuptipcsampling
