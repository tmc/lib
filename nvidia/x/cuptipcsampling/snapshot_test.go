package cuptipcsampling_test

import (
	"fmt"
	"strings"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/cupti"
	"github.com/tmc/lib/nvidia/x/cudaprof"
	"github.com/tmc/lib/nvidia/x/cuptipcsampling"
)

func snapshotFixture() (*cuptipcsampling.ProfileBridge, cuptipcsampling.Batch, cuptipcsampling.SnapshotConfig) {
	b := &cuptipcsampling.ProfileBridge{Inventory: cuptipcsampling.ReasonInventory{CUPTIVersion: 130001, Reasons: []cuptipcsampling.StallReason{{Index: 1, Name: "long_scoreboard"}, {Index: 2, Name: "long_scoreboard_not_issued"}}}}
	batch := cuptipcsampling.Batch{Valid: true, CUPTIVersion: 130001, CollectionMode: cupti.PcSamplingCollectionModeKernelSerialized, RangeID: 7, DroppedSamples: 3, PCs: []cuptipcsampling.PC{{CRC: 123, Name: "kernel", FunctionIndex: 4, CorrelationID: 5, Offset: 16, Reasons: []cuptipcsampling.StallSamples{{Index: 1, Samples: 10}, {Index: 2, Samples: 8}}}}}
	cfg := cuptipcsampling.SnapshotConfig{Target: cudaprof.ProfileTarget{CodeDigest: strings.Repeat("a", 64), Region: "region", Workload: "workload", Device: "device", CollectorMode: "synthetic-serialized"}, Timestamp: time.Unix(123, 0), Coverage: 0.8, ClockUncertaintyNs: -1, ModuleCRC: 123, RangeID: 7, FunctionIndex: 4, CorrelationID: 5, KernelName: "kernel", CollectionMode: batch.CollectionMode, MaxPCs: 4, MaxCounters: 8}
	batch.TotalSamples, batch.NonUserSamples = 30, 2
	cfg.RunID, cfg.ModuleGeneration = "synthetic-run", 8
	return b, batch, cfg
}

func TestProfileBridgeSnapshot(t *testing.T) {
	b, batch, cfg := snapshotFixture()
	s, err := b.Snapshot(batch, cfg)
	if err != nil {
		t.Fatal(err)
	}
	g, err := s.Guidance(cfg.Target)
	if err != nil {
		t.Fatal(err)
	}
	if g.StallCounts["long_scoreboard"] != 10 || g.StallCounts["long_scoreboard_not_issued"] != 8 || g.Losses != 3 || g.Coverage != 0.8 || g.MetricUnits["long_scoreboard"] != "samples" {
		t.Fatalf("altered observations: %+v", g)
	}
	if g.Sampling == nil || g.Sampling.RunID != cfg.RunID || g.Sampling.ModuleGeneration != 8 || g.Sampling.TotalSamples != 30 || g.Sampling.NonUserSamples != 2 || g.Sampling.RangeID != 7 || g.Sampling.CorrelationID != 5 {
		t.Fatalf("missing batch provenance: %+v", g.Sampling)
	}
	batch.PCs[0].Reasons[0].Samples = 99
	if s.PCStalls[16]["long_scoreboard"] != 10 {
		t.Fatal("snapshot aliases batch")
	}
	if b.ReasonSamples != nil {
		t.Fatal("export mutated bridge recording")
	}
}

func TestProfileBridgeSnapshotWideCounts(t *testing.T) {
	b, batch, cfg := snapshotFixture()
	batch.PCs[0].Reasons = []cuptipcsampling.StallSamples{{Index: 1, Samples: ^uint32(0)}}
	batch.PCs = append(batch.PCs, batch.PCs[0])
	s, err := b.Snapshot(batch, cfg)
	if err != nil {
		t.Fatal(err)
	}
	want := 2 * int64(^uint32(0))
	if s.Metrics["long_scoreboard"] != want || s.PCStalls[16]["long_scoreboard"] != want {
		t.Fatal("count truncated")
	}
	cfg.MaxPCs = 1
	if s, err := b.Snapshot(batch, cfg); err == nil || s != nil {
		t.Fatal("PC bound ignored")
	}
}

func TestProfileBridgeSnapshotRefusals(t *testing.T) {
	for _, kind := range []string{"version", "range", "crc", "kernel", "function", "correlation", "unknown", "duplicate", "inventory", "pc_budget", "counter_budget", "remaining", "full", "continuous"} {
		t.Run(kind, func(t *testing.T) {
			b, batch, cfg := snapshotFixture()
			switch kind {
			case "version":
				batch.CUPTIVersion++
			case "range":
				batch.RangeID++
			case "crc":
				batch.PCs[0].CRC++
			case "kernel":
				batch.PCs[0].Name = "other"
			case "function":
				batch.PCs[0].FunctionIndex++
			case "correlation":
				batch.PCs[0].CorrelationID++
			case "unknown":
				batch.PCs[0].Reasons[0].Index = 99
			case "duplicate":
				batch.PCs[0].Reasons[1].Index = 1
			case "inventory":
				b.Inventory.Reasons[1].Name = b.Inventory.Reasons[0].Name
			case "pc_budget":
				cfg.MaxPCs = 0
			case "counter_budget":
				cfg.MaxCounters = 1
			case "remaining":
				batch.RemainingPCs = 1
			case "full":
				batch.HardwareBufferFull = 1
			case "continuous":
				batch.CollectionMode = cupti.PcSamplingCollectionModeContinuous
				cfg.CollectionMode = batch.CollectionMode
			}
			if s, err := b.Snapshot(batch, cfg); err == nil || s != nil {
				t.Fatalf("accepted %s: %+v %v", kind, s, err)
			}
		})
	}
}

func ExampleProfileBridge_Snapshot() {
	bridge, batch, cfg := snapshotFixture() // Entirely synthetic; no GPU collection.
	snapshot, err := bridge.Snapshot(batch, cfg)
	if err != nil {
		fmt.Println(err)
		return
	}
	guidance, err := snapshot.Guidance(cfg.Target)
	fmt.Println(guidance.StallCounts["long_scoreboard"], guidance.Losses, err)
	// Output: 10 3 <nil>
}
