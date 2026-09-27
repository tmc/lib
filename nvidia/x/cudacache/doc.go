// Package cudacache provides content-addressed local and remote caching
// for compiled CUDA kernel artifacts and search evaluation receipts.
//
// # Overview
//
// The package manages two distinct categories of cached data with strict
// separation between execution identities and exploration policies:
//
//  1. Content-Addressed Artifacts: Compiled kernel artifacts (PTX, CUBIN, TileIR)
//     are indexed strictly by binary and toolchain compatibility keys (see [CompilerCacheKey]).
//     Artifact storage enforces atomic publication, SHA-256 digest integrity,
//     size limits, and quarantine of corrupted payloads.
//
//  2. Search Evaluation Receipts: Performance evaluations and exploration traces
//     (see [SearchReceipt]) are indexed by [SearchPolicyID] and workload identity.
//     Changes to search algorithms, heuristics, beam widths, or random seeds
//     produce distinct receipt identities without invalidating identical underlying
//     binary artifacts.
//
// # Key Invalidation
//
// To guarantee that driver updates, toolchain upgrades, target shifts, or compilation
// options immediately invalidate stale cached binaries, [CompilerCacheKey] explicitly
// binds:
//   - Input content / source key
//   - NVRTC version (major.minor)
//   - Driver shared object name (SONAME) / path
//   - Target architecture (e.g. "sm_121a" vs "sm_100a")
//   - Exact compilation flags in deterministic order
//   - Output kind magic bytes
//
// # Remote Retrieval & Quarantine
//
// Remote sources are accessed through the [RemoteClient] transport interface.
// Remote downloads are subject to configurable size limits ([Config.MaxRemoteDownloadBytes]),
// SHA-256 digest validation, and trust policies. Corrupt or unverified artifacts
// are isolated in a quarantine directory and never returned to callers or executed.
//
// This package is outside the compatibility promise of
// github.com/tmc/lib/nvidia. Its API may change or be removed in patch releases.
package cudacache
