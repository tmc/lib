package cuptipcsampling

import (
	"fmt"
	"math"
	"time"

	"github.com/tmc/lib/nvidia/cupti"
	"github.com/tmc/lib/nvidia/x/cudaprof"
)

// SnapshotConfig binds one drained batch to a caller-qualified code generation.
// The caller must verify Target.CodeDigest against the loaded module generation;
// matching a native CRC alone cannot establish that binding. Coverage is supplied
// by the collector owner, never inferred by summing overlapping reason counts.
type SnapshotConfig struct {
	RunID                        string
	ModuleGeneration             uint64
	Target                       cudaprof.ProfileTarget
	Timestamp                    time.Time
	Coverage                     float64
	ClockUncertaintyNs           int64
	ModuleCRC, RangeID           uint64
	FunctionIndex, CorrelationID uint32
	KernelName                   string
	CollectionMode               cupti.PCSamplingCollectionMode
	MaxPCs, MaxCounters          int // Positive limits, at most 65536 and 1048576.
}

// Snapshot exports a bounded batch without native calls or modifying the bridge.
// All PCs must match the declared native identity. A partially drained/full batch,
// unknown reason, invalid inventory or exhausted budget returns no snapshot.
// Native reason names and sample counts, including companions, remain separate.
func (b *ProfileBridge) Snapshot(batch Batch, cfg SnapshotConfig) (*cudaprof.ProfileSnapshot, error) {
	if b == nil || !batch.Valid || batch.CUPTIVersion == 0 || batch.CUPTIVersion != b.Inventory.CUPTIVersion {
		return nil, fmt.Errorf("cuptipcsampling: invalid snapshot batch or inventory version")
	}
	if cfg.MaxPCs <= 0 || cfg.MaxPCs > 65536 || cfg.MaxCounters <= 0 || cfg.MaxCounters > 1048576 || len(batch.PCs) > cfg.MaxPCs {
		return nil, fmt.Errorf("cuptipcsampling: snapshot budget invalid or exceeded")
	}
	if cfg.KernelName == "" || cfg.Timestamp.IsZero() || batch.RangeID != cfg.RangeID || batch.CollectionMode != cfg.CollectionMode || batch.RemainingPCs != 0 || batch.HardwareBufferFull != 0 {
		return nil, fmt.Errorf("cuptipcsampling: incomplete or mismatched snapshot batch")
	}
	if cfg.CollectionMode != cupti.PcSamplingCollectionModeKernelSerialized {
		return nil, fmt.Errorf("cuptipcsampling: snapshot requires kernel-serialized correlation")
	}
	// Validate metadata before allocating counter maps.
	provenance := &cudaprof.SamplingProvenance{RunID: cfg.RunID, ModuleGeneration: cfg.ModuleGeneration, ModuleCRC: cfg.ModuleCRC, KernelName: cfg.KernelName, FunctionIndex: cfg.FunctionIndex, CorrelationID: cfg.CorrelationID, RangeID: batch.RangeID, CUPTIVersion: batch.CUPTIVersion, CollectionMode: uint32(batch.CollectionMode), TotalSamples: batch.TotalSamples, NonUserSamples: batch.NonUserSamples}
	s, err := cudaprof.NewProfileSnapshot(cudaprof.ProfileSnapshot{CodeDigest: cfg.Target.CodeDigest, Region: cfg.Target.Region, Workload: cfg.Target.Workload, TokensHash: cfg.Target.TokensHash, Device: cfg.Target.Device, CollectorMode: cfg.Target.CollectorMode, Timestamp: cfg.Timestamp, Coverage: cfg.Coverage, ClockUncertaintyNs: cfg.ClockUncertaintyNs, Losses: batch.DroppedSamples, Sampling: provenance})
	if err != nil {
		return nil, err
	}
	if len(b.Inventory.Reasons) > 65536 {
		return nil, fmt.Errorf("cuptipcsampling: reason inventory exceeds snapshot limit")
	}
	names := make(map[string]bool, len(b.Inventory.Reasons))
	for i, r := range b.Inventory.Reasons {
		if r.Name == "" || len(r.Name) > 1024 || names[r.Name] || (i > 0 && b.Inventory.Reasons[i-1].Index >= r.Index) {
			return nil, fmt.Errorf("cuptipcsampling: invalid snapshot reason inventory")
		}
		names[r.Name] = true
	}
	s.Metrics = make(map[string]int64)
	s.MetricUnits = make(map[string]string)
	s.PCStalls = make(map[uint64]map[string]int64)
	used := 0
	for _, pc := range batch.PCs {
		if pc.CRC != cfg.ModuleCRC || pc.Name != cfg.KernelName || pc.FunctionIndex != cfg.FunctionIndex || pc.CorrelationID != cfg.CorrelationID {
			return nil, fmt.Errorf("cuptipcsampling: mixed snapshot code attribution")
		}
		if len(pc.Reasons) > cfg.MaxCounters-used {
			return nil, fmt.Errorf("cuptipcsampling: snapshot counter budget exceeded")
		}
		used += len(pc.Reasons)
		counts := s.PCStalls[pc.Offset]
		if counts == nil {
			counts = make(map[string]int64)
			s.PCStalls[pc.Offset] = counts
		}
		seen := make(map[uint32]bool, len(pc.Reasons))
		for _, reason := range pc.Reasons {
			name := b.Inventory.Lookup(reason.Index)
			if name == "" || seen[reason.Index] {
				return nil, fmt.Errorf("cuptipcsampling: unknown or duplicate snapshot reason %d", reason.Index)
			}
			seen[reason.Index] = true
			count := int64(reason.Samples)
			if s.Metrics[name] > math.MaxInt64-count || counts[name] > math.MaxInt64-count {
				return nil, fmt.Errorf("cuptipcsampling: snapshot counter overflow")
			}
			s.Metrics[name] += count
			counts[name] += count
			s.MetricUnits[name] = "samples"
		}
	}
	return s, nil
}
