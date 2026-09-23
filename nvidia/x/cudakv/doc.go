// Package cudakv provides static and paged key-value caches for autoregressive
// transformer decoding, plus host-side residency and prefix bookkeeping.
//
// [New] allocates float32 K and V buffers for each layer on a CUDA device.
// [Cache.Layer] returns views into those owned buffers; [Cache.Advance] changes
// the token position but does not write tensor data. Finish device operations
// before calling [Cache.Close]. [NewPagedCache] provides a block pool shared by
// sequences; release sequence allocations with [PagedCache.FreeSequence] and
// release the pool with [PagedCache.Close].
//
// [RollingKVCache] tracks sink-token and rolling-window residency without
// allocating GPU storage. Use [RollingKVCache.Slot] to check residency as well
// as obtain the physical slot: an old position may have been overwritten.
// Callers coordinate tensor writes and device completion with position updates.
//
// [SSDOffloader] spills layer data through host memory into process-local files.
// Its files are scratch storage, not crash-durable checkpoints. Call
// [SSDOffloader.Close] to remove files owned by that instance.
//
// # Compaction and Defragmentation
//
// Continuous batching causes VRAM pool fragmentation as sequences allocate and
// free blocks with varying lifetimes. [Compactor] monitors pool fragmentation
// through [Fragmentation] (1.0 - LargestBlock / TotalFree) and [DefragPolicy].
// When fragmentation crosses a threshold (default 0.35) or when allocation would
// fail, [Compactor.Compact] or [PagedCache.Defrag] computes an optimal compaction
// plan moving scattered active pages into contiguous low-index blocks. Relocations
// execute via stream-ordered copies or can be captured into a CUDA Graph using
// [Compactor.CaptureGraph]. Allocator virtual page tables and attached [RadixTree]
// prefix cache mappings are updated atomically so active generation streams
// transparently read from compacted locations without stalling or losing tokens.
package cudakv
