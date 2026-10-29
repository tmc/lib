package nvl72

import (
	"fmt"
	"math"
	"testing"
)

func TestLookupPodSpec(t *testing.T) {
	cases := []struct {
		gen        string
		wantCapGB  int
		wantNIC    string
		wantRDMA   float64
		wantTFLOPS float64
		wantErr    bool
	}{
		{
			gen:        "GB200",
			wantCapGB:  192,
			wantNIC:    "ConnectX-7 400G",
			wantRDMA:   50.0,
			wantTFLOPS: 2250.0,
		},
		{
			gen:        "gb300",
			wantCapGB:  288,
			wantNIC:    "ConnectX-8 800G PCIe Gen6",
			wantRDMA:   100.0,
			wantTFLOPS: 3375.0,
		},
		{
			gen:     "B100",
			wantErr: true,
		},
	}

	for _, tc := range cases {
		t.Run(tc.gen, func(t *testing.T) {
			spec, err := LookupPodSpec(tc.gen)
			if tc.wantErr {
				if err == nil {
					t.Fatalf("LookupPodSpec(%q) succeeded, want error", tc.gen)
				}
				return
			}
			if err != nil {
				t.Fatalf("LookupPodSpec(%q): %v", tc.gen, err)
			}
			if spec.GPUsPerRack != 72 || spec.TraysPerRack != 18 {
				t.Fatalf("rack topology = (%d GPUs, %d trays), want (72, 18)", spec.GPUsPerRack, spec.TraysPerRack)
			}
			if spec.HBM3eCapacityGB != tc.wantCapGB {
				t.Fatalf("HBM3eCapacityGB = %d, want %d", spec.HBM3eCapacityGB, tc.wantCapGB)
			}
			if spec.Tier1HBM3eGBps != 8000.0 || spec.Tier2NVLink5UniGBps != 900.0 || spec.Tier2NVLink5BidiGBps != 1800.0 || spec.Tier3GraceC2CGBps != 450.0 {
				t.Fatalf("unexpected 4-tier bandwidths: %+v", spec)
			}
			if spec.Tier4ScaleOutNIC != tc.wantNIC || spec.Tier4ScaleOutGBps != tc.wantRDMA {
				t.Fatalf("Tier4 = (%q, %v GB/s), want (%q, %v GB/s)",
					spec.Tier4ScaleOutNIC, spec.Tier4ScaleOutGBps, tc.wantNIC, tc.wantRDMA)
			}
			if spec.TensorTFLOPS != tc.wantTFLOPS {
				t.Fatalf("TensorTFLOPS = %v, want %v", spec.TensorTFLOPS, tc.wantTFLOPS)
			}
		})
	}
}

func TestEstimateMultiRackPodEP(t *testing.T) {
	cases := []struct {
		name         string
		numRacks     int
		tokensPerGPU int
		hiddenSize   int
		topK         int
		totalExperts int
		wantGPUs     int
	}{
		{
			name:         "1Rack_72GPUs_IntraRackOnly",
			numRacks:     1,
			tokensPerGPU: 64,
			hiddenSize:   4096,
			topK:         8,
			totalExperts: 288,
			wantGPUs:     72,
		},
		{
			name:         "2Racks_144GPUs_Pod",
			numRacks:     2,
			tokensPerGPU: 64,
			hiddenSize:   4096,
			topK:         8,
			totalExperts: 288,
			wantGPUs:     144,
		},
		{
			name:         "4Racks_288GPUs_Pod",
			numRacks:     4,
			tokensPerGPU: 64,
			hiddenSize:   7168,
			topK:         8,
			totalExperts: 288,
			wantGPUs:     288,
		},
		{
			name:         "8Racks_576GPUs_Pod",
			numRacks:     8,
			tokensPerGPU: 64,
			hiddenSize:   7168,
			topK:         8,
			totalExperts: 576,
			wantGPUs:     576,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			gb200, err := EstimateMultiRackPodEP(GenGB200, tc.numRacks, tc.tokensPerGPU, tc.hiddenSize, tc.topK, tc.totalExperts)
			if err != nil {
				t.Fatalf("EstimateMultiRackPodEP(GB200): %v", err)
			}
			gb300, err := EstimateMultiRackPodEP(GenGB300, tc.numRacks, tc.tokensPerGPU, tc.hiddenSize, tc.topK, tc.totalExperts)
			if err != nil {
				t.Fatalf("EstimateMultiRackPodEP(GB300): %v", err)
			}

			if gb200.TotalGPUs != tc.wantGPUs || gb300.TotalGPUs != tc.wantGPUs {
				t.Fatalf("TotalGPUs = (%d, %d), want %d", gb200.TotalGPUs, gb300.TotalGPUs, tc.wantGPUs)
			}
			if gb200.HBM3eCapacityPerGPUGB != 192 || gb300.HBM3eCapacityPerGPUGB != 288 {
				t.Fatalf("HBM3eCapacityPerGPUGB = (%d, %d), want (192, 288)",
					gb200.HBM3eCapacityPerGPUGB, gb300.HBM3eCapacityPerGPUGB)
			}
			if math.Abs(gb300.PodHBM3eCapacityTB-1.5*gb200.PodHBM3eCapacityTB) > 1e-6 {
				t.Fatalf("GB300 PodHBM3eCapacityTB (%v) != 1.5 * GB200 (%v)",
					gb300.PodHBM3eCapacityTB, gb200.PodHBM3eCapacityTB)
			}
			if gb200.Tier4RDMAUniGBps != 50.0 || gb300.Tier4RDMAUniGBps != 100.0 {
				t.Fatalf("Tier4RDMAUniGBps = (%v, %v), want (50, 100)",
					gb200.Tier4RDMAUniGBps, gb300.Tier4RDMAUniGBps)
			}

			if tc.numRacks == 1 {
				if gb200.HierarchicalInterRackDispatchBytes != 0 || gb300.HierarchicalInterRackDispatchBytes != 0 {
					t.Fatalf("expected 0 inter-rack RDMA bytes for 1 rack, got (%d, %d)",
						gb200.HierarchicalInterRackDispatchBytes, gb300.HierarchicalInterRackDispatchBytes)
				}
			} else {
				// Multi-rack pod checks:
				// 1. Hierarchical rack-deduped dispatch sends strictly fewer bytes over RDMA than flat routing when TopK > 1.
				if gb300.HierarchicalInterRackDispatchBytes >= gb300.FlatInterRackDispatchBytes {
					t.Fatalf("expected hierarchical rack-deduped RDMA bytes (%d) < flat RDMA bytes (%d)",
						gb300.HierarchicalInterRackDispatchBytes, gb300.FlatInterRackDispatchBytes)
				}
				if gb300.RackDedupSavingsRatio <= 0 {
					t.Fatalf("expected positive RackDedupSavingsRatio, got %v", gb300.RackDedupSavingsRatio)
				}
				if gb300.HierarchicalDispatchNs >= gb300.FlatDispatchNs {
					t.Fatalf("expected HierarchicalDispatchNs (%v) < FlatDispatchNs (%v)",
						gb300.HierarchicalDispatchNs, gb300.FlatDispatchNs)
				}

				// 2. ConnectX-8 (100 GB/s on GB300) achieves lower Stage 1 RDMA dispatch and Stage 2 RDMA combine latency than ConnectX-7 (50 GB/s on GB200).
				if gb300.Stage1InterRackDispatchNs >= gb200.Stage1InterRackDispatchNs {
					t.Fatalf("expected GB300 ConnectX-8 Stage1InterRackDispatchNs (%v) < GB200 ConnectX-7 (%v)",
						gb300.Stage1InterRackDispatchNs, gb200.Stage1InterRackDispatchNs)
				}
				if gb300.Stage2InterRackCombineNs >= gb200.Stage2InterRackCombineNs {
					t.Fatalf("expected GB300 ConnectX-8 Stage2InterRackCombineNs (%v) < GB200 ConnectX-7 (%v)",
						gb300.Stage2InterRackCombineNs, gb200.Stage2InterRackCombineNs)
				}
				if gb300.SingleBatchTotalNs >= gb200.SingleBatchTotalNs {
					t.Fatalf("expected GB300 SingleBatchTotalNs (%v) < GB200 (%v)",
						gb300.SingleBatchTotalNs, gb200.SingleBatchTotalNs)
				}
			}

			// 3. Two-Batch Overlap (TBO) strictly reduces 2-microbatch latency on both GB200 and GB300.
			for _, est := range []PodEPEstimate{gb200, gb300} {
				if est.TBOTwoBatchNs >= est.SequentialTwoBatchNs || est.TBOOverlapSavingsNs <= 0 || est.TBOSpeedup <= 1.0 {
					t.Fatalf("%s TBO invalid: seq=%v tbo=%v savings=%v speedup=%v",
						est.Generation, est.SequentialTwoBatchNs, est.TBOTwoBatchNs, est.TBOOverlapSavingsNs, est.TBOSpeedup)
				}
			}
		})
	}
}

func TestEstimateMultiRackPodEPErrors(t *testing.T) {
	badCases := []struct {
		name string
		gen  string
		r    int
		s    int
		h    int
		k    int
		e    int
	}{
		{"bad_gen", "GB100", 2, 64, 4096, 8, 288},
		{"zero_racks", GenGB300, 0, 64, 4096, 8, 288},
		{"zero_tokens", GenGB300, 2, 0, 4096, 8, 288},
		{"odd_hidden", GenGB300, 2, 64, 4095, 8, 288},
		{"zero_experts", GenGB300, 2, 64, 4096, 8, 0},
		{"topk_gt_experts", GenGB300, 2, 64, 4096, 16, 8},
	}
	for _, tc := range badCases {
		t.Run(tc.name, func(t *testing.T) {
			if _, err := EstimateMultiRackPodEP(tc.gen, tc.r, tc.s, tc.h, tc.k, tc.e); err == nil {
				t.Fatalf("EstimateMultiRackPodEP(%+v) succeeded, want error", tc)
			}
		})
	}
}

func ExampleEstimateMultiRackPodEP() {
	gb200, _ := EstimateMultiRackPodEP(GenGB200, 4, 64, 4096, 8, 288)
	gb300, _ := EstimateMultiRackPodEP(GenGB300, 4, 64, 4096, 8, 288)
	fmt.Println(
		gb300.TotalGPUs,
		gb200.HBM3eCapacityPerGPUGB,
		gb300.HBM3eCapacityPerGPUGB,
		gb200.Tier4RDMAUniGBps,
		gb300.Tier4RDMAUniGBps,
		gb300.HierarchicalInterRackDispatchBytes < gb300.FlatInterRackDispatchBytes,
		gb300.SingleBatchTotalNs < gb200.SingleBatchTotalNs,
	)
	// Output:
	// 288 192 288 50 100 true true
}
