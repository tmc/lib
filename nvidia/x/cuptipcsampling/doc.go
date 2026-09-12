// Package cuptipcsampling provides bounded modern CUPTI PC sampling.
// StallReasons queries native reason IDs and names for an owned CUDA context.
// IdentifyCubin obtains a native code identifier and an exact artifact digest.
// Start owns an explicit sampling range with caller-supplied memory and read
// limits. Callers coordinate workload completion, preserve every Read batch,
// and call Close before destroying the context. Start may return both a session
// and an error when native cleanup ownership must be retained.
//
// Collection is restricted to Linux amd64/arm64 and CUPTI 130001 layouts.
// This layout restriction is not native hardware qualification. Native reason
// counts are not legacy stall enums or inferred instruction metrics.
package cuptipcsampling
