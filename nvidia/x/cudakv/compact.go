// SPDX-License-Identifier: Apache-2.0

package cudakv

import (
	"context"
	"errors"
	"fmt"

	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudagraph"
	"github.com/tmc/lib/nvidia/x/cudamem"
)

// DefaultDefragThreshold is the default fragmentation index (0.35)
// that triggers proactive VRAM defragmentation.
const DefaultDefragThreshold = 0.35

// FragmentationStats reports metrics about the allocator's free and allocated blocks.
type FragmentationStats struct {
	TotalBlocks        int
	AllocatedBlocks    int
	FreeBlocks         int
	LargestContiguous  int
	FragmentationIndex float64
}

// VirtualPageTable represents a sequence's virtual-to-physical block table.
type VirtualPageTable struct {
	SeqID  int64
	Blocks []int
}

// Fragmentation calculates the memory fragmentation ratio:
//
//	Fragmentation = 1.0 - (LargestContiguousFree / TotalFree)
//
// A value of 0.0 indicates zero fragmentation (all free memory is in one contiguous block).
// A value approaching 1.0 indicates severe fragmentation (free memory is split into tiny disjoint islands).
func Fragmentation(totalFree, largestFree int) float64 {
	if totalFree <= 0 || largestFree >= totalFree {
		return 0.0
	}
	return 1.0 - (float64(largestFree) / float64(totalFree))
}

// TrackerFragmentation computes the fragmentation index for a cudamem.Tracker-managed
// memory pool given the total pool capacity in bytes and the largest contiguous free bytes.
func TrackerFragmentation(tracker *cudamem.Tracker, totalCapacityBytes, largestFreeBytes uint64) float64 {
	if tracker == nil || totalCapacityBytes == 0 {
		return 0.0
	}
	current, _ := tracker.Snapshot()
	if current >= totalCapacityBytes {
		return 0.0
	}
	totalFree := totalCapacityBytes - current
	if totalFree == 0 || largestFreeBytes >= totalFree {
		return 0.0
	}
	return 1.0 - (float64(largestFreeBytes) / float64(totalFree))
}

// DefragPolicy specifies conditions that trigger proactive KV pool defragmentation.
type DefragPolicy struct {
	// Threshold is the fragmentation index threshold (0.0 to 1.0) above which
	// compaction is triggered. Defaults to DefaultDefragThreshold (0.35).
	Threshold float64

	// MaxBlocksPerPass limits the maximum number of blocks moved in a single
	// compaction pass. 0 means unlimited (all scattered blocks are moved).
	MaxBlocksPerPass int
}

// DefaultDefragPolicy returns a DefragPolicy initialized with DefaultDefragThreshold (0.35).
func DefaultDefragPolicy() DefragPolicy {
	return DefragPolicy{
		Threshold: DefaultDefragThreshold,
	}
}

// ShouldCompact reports whether compaction should be executed based on the current
// fragmentation statistics and required contiguous blocks.
func (p DefragPolicy) ShouldCompact(stats FragmentationStats, neededBlocks int) bool {
	threshold := p.Threshold
	if threshold <= 0 {
		threshold = DefaultDefragThreshold
	}
	if stats.FragmentationIndex > threshold {
		return true
	}
	// Trigger if an allocation would fail solely due to fragmentation:
	// there are enough total free blocks, but no single contiguous run is large enough.
	if neededBlocks > 0 && stats.FreeBlocks >= neededBlocks && stats.LargestContiguous < neededBlocks {
		return true
	}
	return false
}

// BlockMove describes a single physical block relocation from SrcBlock to DstBlock.
type BlockMove struct {
	SrcBlock int // original physical slot (scattered, high index)
	DstBlock int // target physical slot (contiguous, low index)
}

// CompactionPlan details the relocations needed to eliminate or reduce pool fragmentation.
type CompactionPlan struct {
	Moves              []BlockMove
	Remappings         map[int]int // SrcBlock -> DstBlock
	InitialFragIndex   float64
	EstimatedFragIndex float64
	ActiveBlocks       int
	TotalBlocks        int
}

// Empty reports whether the plan has no block moves.
func (p CompactionPlan) Empty() bool {
	return len(p.Moves) == 0
}

// BlockCopier copies physical block contents between pool slots.
type BlockCopier interface {
	CopyBlock(srcSlot, dstSlot int) error
}

// Compactor orchestrates proactive VRAM defragmentation and paged KV pool compaction.
type Compactor struct {
	policy DefragPolicy
	tree   *RadixTree
	copier BlockCopier
}

// NewCompactor creates a compactor with the given defragmentation policy.
func NewCompactor(policy DefragPolicy) *Compactor {
	if policy.Threshold <= 0 {
		policy.Threshold = DefaultDefragThreshold
	}
	return &Compactor{
		policy: policy,
	}
}

// WithRadixTree attaches a RadixTree prefix cache whose block mappings
// will be updated atomically during compaction.
func (c *Compactor) WithRadixTree(tree *RadixTree) *Compactor {
	c.tree = tree
	return c
}

// WithCopier attaches a custom BlockCopier for physical data relocation.
func (c *Compactor) WithCopier(copier BlockCopier) *Compactor {
	c.copier = copier
	return c
}

// Policy returns the compactor policy.
func (c *Compactor) Policy() DefragPolicy {
	return c.policy
}

// ShouldCompact checks whether pc currently requires compaction under the configured policy.
func (c *Compactor) ShouldCompact(pc *PagedCache, neededBlocks int) bool {
	stats := pc.FragmentationStats()
	return c.policy.ShouldCompact(stats, neededBlocks)
}

// Plan computes the compaction plan for pc without applying it.
func (c *Compactor) Plan(pc *PagedCache) CompactionPlan {
	pc.mu.Lock()
	defer pc.mu.Unlock()
	return pc.blocks.PlanCompaction(c.policy.MaxBlocksPerPass)
}

// Compact executes compaction synchronously on pc according to policy.
func (c *Compactor) Compact(pc *PagedCache) (CompactionPlan, error) {
	return c.CompactOnStream(pc, nil)
}

// CompactOnStream executes compaction on pc using the given stream for asynchronous copies.
// Virtual page tables and the attached RadixTree mappings are updated atomically.
func (c *Compactor) CompactOnStream(pc *PagedCache, stream *cuda.Stream) (CompactionPlan, error) {
	if pc == nil {
		return CompactionPlan{}, fmt.Errorf("%s: nil paged cache", pkgName)
	}

	pc.mu.Lock()
	defer pc.mu.Unlock()

	if c.tree != nil {
		c.tree.mu.Lock()
		defer c.tree.mu.Unlock()
	}

	plan := pc.blocks.PlanCompaction(c.policy.MaxBlocksPerPass)
	if plan.Empty() {
		return plan, nil
	}

	// Relocate physical block data
	if c.copier != nil {
		for _, m := range plan.Moves {
			if err := c.copier.CopyBlock(m.SrcBlock, m.DstBlock); err != nil {
				return plan, fmt.Errorf("%s: copy block %d->%d: %w", pkgName, m.SrcBlock, m.DstBlock, err)
			}
		}
	} else {
		for _, m := range plan.Moves {
			if err := pc.copyBlockAsync(m.SrcBlock, m.DstBlock, stream); err != nil {
				return plan, fmt.Errorf("%s: copy block %d->%d: %w", pkgName, m.SrcBlock, m.DstBlock, err)
			}
		}
	}

	// If using an asynchronous stream, synchronize before publishing new page tables
	if stream != nil {
		if err := stream.Synchronize(context.Background()); err != nil {
			return plan, fmt.Errorf("%s: synchronize compaction stream: %w", pkgName, err)
		}
	}

	// Update allocator virtual page tables and free list atomically
	pc.blocks.ApplyCompaction(plan)

	// Update RadixTree prefix cache block mappings atomically
	if c.tree != nil {
		c.tree.remapBlocksLocked(plan.Remappings)
	}

	return plan, nil
}

// CompactionGraph represents a captured CUDA execution graph of compaction memory copies.
type CompactionGraph struct {
	graph       *cudagraph.Graph
	exec        *cudagraph.Exec
	plan        CompactionPlan
	updates     []cudagraph.KernelNodeUpdate
	deviceNodes []cudagraph.DeviceNode
}

// CaptureGraph captures the compaction block copies for plan into a reusable CUDA Graph.
func (c *Compactor) CaptureGraph(pc *PagedCache, stream *cuda.Stream, plan CompactionPlan) (*CompactionGraph, error) {
	if pc == nil {
		return nil, fmt.Errorf("%s: nil cache for graph capture", pkgName)
	}
	if stream == nil {
		return nil, fmt.Errorf("%s: nil stream for graph capture", pkgName)
	}
	if plan.Empty() {
		return nil, fmt.Errorf("%s: cannot capture empty compaction plan", pkgName)
	}

	var captureErr error
	graph, err := cudagraph.Capture(stream, func() {
		pc.mu.Lock()
		defer pc.mu.Unlock()
		for _, m := range plan.Moves {
			if err := pc.copyBlockAsync(m.SrcBlock, m.DstBlock, stream); err != nil {
				captureErr = err
				return
			}
		}
	})
	if err != nil {
		return nil, fmt.Errorf("%s: capture compaction graph: %w", pkgName, err)
	}
	if captureErr != nil {
		_ = graph.Destroy()
		return nil, fmt.Errorf("%s: capture copy operation: %w", pkgName, captureErr)
	}

	exec, err := graph.Instantiate()
	if err != nil {
		_ = graph.Destroy()
		return nil, fmt.Errorf("%s: instantiate compaction graph: %w", pkgName, err)
	}

	return &CompactionGraph{
		graph: graph,
		exec:  exec,
		plan:  plan,
	}, nil
}

// Plan returns the compaction plan captured in this graph.
func (cg *CompactionGraph) Plan() CompactionPlan {
	if cg == nil {
		return CompactionPlan{}
	}
	return cg.plan
}

// Launch replays the captured compaction copies on stream.
func (cg *CompactionGraph) Launch(stream *cuda.Stream) error {
	if cg == nil || cg.exec == nil {
		return fmt.Errorf("%s: compaction graph not instantiated", pkgName)
	}
	if stream == nil {
		return fmt.Errorf("%s: nil stream for graph launch", pkgName)
	}
	return cg.exec.Launch(stream)
}

// Destroy releases the captured graph and instantiated execution resources.
func (cg *CompactionGraph) Destroy() error {
	if cg == nil {
		return nil
	}
	var errs []error
	if cg.exec != nil {
		if err := cg.exec.Destroy(); err != nil {
			errs = append(errs, err)
		}
		cg.exec = nil
	}
	if cg.graph != nil {
		if err := cg.graph.Destroy(); err != nil {
			errs = append(errs, err)
		}
		cg.graph = nil
	}
	if len(errs) > 0 {
		return errors.Join(errs...)
	}
	return nil
}

// AddDeviceUpdate registers a kernel node update for device-side update pointers.
func (cg *CompactionGraph) AddDeviceUpdate(update cudagraph.KernelNodeUpdate) {
	if cg != nil {
		cg.updates = append(cg.updates, update)
	}
}

// DeviceUpdates converts registered updates into C-ABI compliant DeviceKernelNodeUpdate structs.
func (cg *CompactionGraph) DeviceUpdates() []cudagraph.DeviceKernelNodeUpdate {
	if cg == nil || len(cg.updates) == 0 {
		return nil
	}
	return cudagraph.EncodeDeviceUpdates(cg.updates)
}

// DeviceNodes returns the list of device nodes associated with this compaction graph.
func (cg *CompactionGraph) DeviceNodes() []cudagraph.DeviceNode {
	if cg == nil {
		return nil
	}
	return append([]cudagraph.DeviceNode(nil), cg.deviceNodes...)
}

// ApplyToCache applies the compaction plan to pc and tree after graph execution completes.
func (cg *CompactionGraph) ApplyToCache(pc *PagedCache, stream *cuda.Stream, tree ...*RadixTree) error {
	if pc == nil {
		return fmt.Errorf("%s: nil paged cache", pkgName)
	}
	if stream != nil {
		if err := stream.Synchronize(context.Background()); err != nil {
			return fmt.Errorf("%s: synchronize stream after graph launch: %w", pkgName, err)
		}
	}
	pc.mu.Lock()
	defer pc.mu.Unlock()
	pc.blocks.ApplyCompaction(cg.plan)
	for _, t := range tree {
		if t != nil {
			t.RemapBlocks(cg.plan.Remappings)
		}
	}
	return nil
}
