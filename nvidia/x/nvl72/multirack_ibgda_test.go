package nvl72

import (
	"strings"
	"testing"
)

func TestMultiRackHierarchicalCollective(t *testing.T) {
	spec := MultiRackClusterSpec{
		NumRacks:             8,
		GPUsPerRack:          72,
		IntraRackBWGBps:      900.0,
		InterRackRDMA_BWGBps: 100.0,
		EnableIBGDA:          true,
		EnableSHARPv4:        true,
		StragglerDeadlineNs:  50_000,
	}

	plan, err := PlanMultiRackHierarchicalCollective(spec, 64<<20) // 64 MiB
	if err != nil {
		t.Fatalf("PlanMultiRackHierarchicalCollective failed: %v", err)
	}
	if plan.TotalGPUs != 576 {
		t.Fatalf("TotalGPUs = %d, want 576", plan.TotalGPUs)
	}
	if plan.SHARPTreeDepth != 3 {
		t.Fatalf("SHARPTreeDepth = %d, want 3", plan.SHARPTreeDepth)
	}
	if plan.SpeedupFactor <= 1.0 {
		t.Errorf("expected SpeedupFactor > 1.0 with IBGDA + SHARP v4, got %v", plan.SpeedupFactor)
	}
	if plan.SHARPv4TrafficSavingRatio <= 0 {
		t.Errorf("expected positive SHARPv4TrafficSavingRatio, got %v", plan.SHARPv4TrafficSavingRatio)
	}
	if plan.CPUProxyOverheadSavedUs <= 0 {
		t.Errorf("expected positive CPUProxyOverheadSavedUs, got %v", plan.CPUProxyOverheadSavedUs)
	}

	// Compare against non-IBGDA, non-SHARP baseline.
	baseSpec := spec
	baseSpec.EnableIBGDA = false
	baseSpec.EnableSHARPv4 = false
	basePlan, err := PlanMultiRackHierarchicalCollective(baseSpec, 64<<20)
	if err != nil {
		t.Fatalf("baseline PlanMultiRackHierarchicalCollective failed: %v", err)
	}
	if plan.TotalLatencyUs >= basePlan.TotalLatencyUs {
		t.Errorf("IBGDA+SHARPv4 latency (%v us) should be strictly less than baseline (%v us)",
			plan.TotalLatencyUs, basePlan.TotalLatencyUs)
	}

	// Test Rubin NVL144 16-rack scale.
	rubinSpec := MultiRackClusterSpec{
		NumRacks:             16,
		GPUsPerRack:          144,
		IntraRackBWGBps:      1800.0,
		InterRackRDMA_BWGBps: 200.0,
		EnableIBGDA:          true,
		EnableSHARPv4:        true,
		StragglerDeadlineNs:  25_000,
	}
	rubinPlan, err := PlanMultiRackHierarchicalCollective(rubinSpec, 128<<20)
	if err != nil {
		t.Fatalf("Rubin PlanMultiRackHierarchicalCollective failed: %v", err)
	}
	if rubinPlan.TotalGPUs != 2304 {
		t.Errorf("Rubin TotalGPUs = %d, want 2304", rubinPlan.TotalGPUs)
	}
}

func TestMultiRackIBGDADoorbellCUDA(t *testing.T) {
	spec := MultiRackClusterSpec{
		NumRacks:             4,
		GPUsPerRack:          72,
		IntraRackBWGBps:      900.0,
		InterRackRDMA_BWGBps: 100.0,
		EnableIBGDA:          true,
		EnableSHARPv4:        true,
		StragglerDeadlineNs:  40_000,
	}
	plan, err := PlanMultiRackHierarchicalCollective(spec, 16<<20)
	if err != nil {
		t.Fatal(err)
	}

	cudaSrc := plan.IBGDADoorbellCUDA()
	requiredSubstrings := []string{
		"eg_ibgda_wqe_ctrl_seg",
		"eg_ibgda_post_wqe_doorbell",
		"st.release.sys.global.u64",
		"ld.acquire.sys.global.u64",
		"eg_ready_wait_sys_deadline",
		"%globaltimer",
		"eg_multirack_straggler_resilient_moe_combine",
		"40000ull",
	}
	for _, sub := range requiredSubstrings {
		if !strings.Contains(cudaSrc, sub) {
			t.Errorf("IBGDADoorbellCUDA missing expected substring %q", sub)
		}
	}

	// Verify .sys memory coherence on emitted CUDA source.
	if err := AuditC2CCoherence(cudaSrc); err != nil {
		t.Errorf("AuditC2CCoherence failed on IBGDADoorbellCUDA: %v", err)
	}
}

func TestMultiRackElasticRankMigration(t *testing.T) {
	spec := MultiRackClusterSpec{
		NumRacks:             4,
		GPUsPerRack:          72,
		IntraRackBWGBps:      900.0,
		InterRackRDMA_BWGBps: 100.0,
		EnableIBGDA:          true,
		EnableSHARPv4:        true,
		StragglerDeadlineNs:  50_000,
	}

	failedRank := 145 // Rack 2, local rank 1
	receipt, err := PlanElasticRankMigration(spec, failedRank, 1024)
	if err != nil {
		t.Fatalf("PlanElasticRankMigration failed: %v", err)
	}
	if receipt.FailedRank != failedRank {
		t.Errorf("FailedRank = %d, want %d", receipt.FailedRank, failedRank)
	}
	if receipt.FailedRack != 2 {
		t.Errorf("FailedRack = %d, want 2", receipt.FailedRack)
	}
	if receipt.HealthyRanksRemaining != 287 {
		t.Errorf("HealthyRanksRemaining = %d, want 287", receipt.HealthyRanksRemaining)
	}
	if receipt.ReplacementRank == failedRank {
		t.Errorf("ReplacementRank %d must differ from failedRank", receipt.ReplacementRank)
	}
	if receipt.RoutingTable[failedRank] != receipt.ReplacementRank {
		t.Errorf("RoutingTable[%d] = %d, want %d", failedRank, receipt.RoutingTable[failedRank], receipt.ReplacementRank)
	}
	if len(receipt.ShardReRoutes) == 0 {
		t.Fatal("expected non-empty ShardReRoutes")
	}
	for _, sr := range receipt.ShardReRoutes {
		if sr.OldOwnerRank != failedRank {
			t.Errorf("shard %d OldOwnerRank = %d, want %d", sr.ShardID, sr.OldOwnerRank, failedRank)
		}
		if sr.NewOwnerRank == failedRank {
			t.Errorf("shard %d NewOwnerRank must not be failedRank %d", sr.ShardID, failedRank)
		}
		if !sr.SameRack {
			t.Errorf("shard %d expected intra-rack NVSwitch replica migration", sr.ShardID)
		}
	}
	if !receipt.EpochSwap.ZeroDowntimeVerified || receipt.DowntimeMs != 0.0 {
		t.Errorf("expected zero-downtime CUgraphExec epoch swap, got Verified=%v DowntimeMs=%v",
			receipt.EpochSwap.ZeroDowntimeVerified, receipt.DowntimeMs)
	}
	if receipt.EpochSwap.NewEpoch <= receipt.EpochSwap.OldEpoch {
		t.Errorf("expected NewEpoch (%d) > OldEpoch (%d)", receipt.EpochSwap.NewEpoch, receipt.EpochSwap.OldEpoch)
	}
}

func TestMultiRackValidationErrors(t *testing.T) {
	badSpecs := []MultiRackClusterSpec{
		{NumRacks: 0, GPUsPerRack: 72, IntraRackBWGBps: 900, InterRackRDMA_BWGBps: 100},
		{NumRacks: 4, GPUsPerRack: 0, IntraRackBWGBps: 900, InterRackRDMA_BWGBps: 100},
		{NumRacks: 4, GPUsPerRack: 72, IntraRackBWGBps: 0, InterRackRDMA_BWGBps: 100},
		{NumRacks: 4, GPUsPerRack: 72, IntraRackBWGBps: 900, InterRackRDMA_BWGBps: -5},
	}
	for i, s := range badSpecs {
		if _, err := PlanMultiRackHierarchicalCollective(s, 1024); err == nil {
			t.Errorf("case %d: expected error for invalid spec %+v", i, s)
		}
	}

	validSpec := MultiRackClusterSpec{
		NumRacks:             2,
		GPUsPerRack:          72,
		IntraRackBWGBps:      900,
		InterRackRDMA_BWGBps: 100,
	}
	if _, err := PlanMultiRackHierarchicalCollective(validSpec, 0); err == nil {
		t.Error("expected error for zero payloadBytes")
	}
	if _, err := PlanElasticRankMigration(validSpec, -1, 100); err == nil {
		t.Error("expected error for negative failedRank")
	}
	if _, err := PlanElasticRankMigration(validSpec, 144, 100); err == nil {
		t.Error("expected error for out-of-range failedRank")
	}
	if _, err := PlanElasticRankMigration(validSpec, 0, 0); err == nil {
		t.Error("expected error for zero activeDecodeStreams")
	}
}
