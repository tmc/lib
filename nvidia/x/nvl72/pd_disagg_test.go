package nvl72

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudahostemu"
)

func TestNVL72KVPageTransfer(t *testing.T) {
	cases := []struct {
		name               string
		numRanks           int
		numPages           int
		totalPoolPages     int
		pageBytes          int
		useMLALatent       bool
		useGraceC2CStaging bool
	}{
		{
			name:               "R8_MLA_Latent_DirectNVLink5",
			numRanks:           8,
			numPages:           8,
			totalPoolPages:     16,
			pageBytes:          640, // 1 token * (512B FP8 KVLoRA + 64*2B BF16 RoPE)
			useMLALatent:       true,
			useGraceC2CStaging: false,
		},
		{
			name:               "R8_GQA_GraceC2CStaging",
			numRanks:           8,
			numPages:           8,
			totalPoolPages:     16,
			pageBytes:          512,
			useMLALatent:       false,
			useGraceC2CStaging: true,
		},
		{
			name:               "R72_FullNVL72_MLA_Latent_DirectNVLink5",
			numRanks:           72,
			numPages:           8,
			totalPoolPages:     72,
			pageBytes:          640,
			useMLALatent:       true,
			useGraceC2CStaging: false,
		},
		{
			name:               "R72_FullNVL72_GQA_GraceC2CStaging",
			numRanks:           72,
			numPages:           6,
			totalPoolPages:     72,
			pageBytes:          256,
			useMLALatent:       false,
			useGraceC2CStaging: true,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			k := NVL72KVPageTransfer{
				Name:               "nvl72_kv_xfer",
				NumPages:           tc.numPages,
				PageBytes:          tc.pageBytes,
				UseMLALatent:       tc.useMLALatent,
				UseGraceC2CStaging: tc.useGraceC2CStaging,
				NumRanks:           tc.numRanks,
			}

			src := k.Source()
			for _, want := range []string{
				"eg_fabric_peer_ptr",
				"st.global.v4.u32",
				"fence.AcqRel.sys",
				"fence.acq_rel.sys",
				"st.release.sys.global.u32",
				"ld.acquire.sys.global.u32",
			} {
				if !strings.Contains(src, want) {
					t.Fatalf("Source missing expected token %q", want)
				}
			}

			srcPages := make([]byte, tc.numPages*tc.pageBytes)
			for p := 0; p < tc.numPages; p++ {
				for b := 0; b < tc.pageBytes; b++ {
					srcPages[p*tc.pageBytes+b] = byte((p*37 + b*13 + 0x5a) & 0xff)
				}
			}

			srcPageIDs := make([]int32, tc.numPages)
			dstPageIDs := make([]int32, tc.numPages)
			pagesPerRank := tc.totalPoolPages / tc.numRanks
			for p := 0; p < tc.numPages; p++ {
				srcPageIDs[p] = int32(p)
				// Scatter pages from prefill rank 0 across distinct remote decode ranks (1..numRanks-1).
				targetRank := 1 + ((p * 9) % (tc.numRanks - 1))
				localPage := p % pagesPerRank
				dstPage := targetRank*pagesPerRank + localPage
				// Avoid duplicate destination pages within the same test case.
				for used := true; used; {
					used = false
					for prev := 0; prev < p; prev++ {
						if int(dstPageIDs[prev]) == dstPage {
							dstPage = (dstPage + pagesPerRank) % tc.totalPoolPages
							if dstPage < pagesPerRank {
								dstPage += pagesPerRank
							}
							used = true
							break
						}
					}
				}
				dstPageIDs[p] = int32(dstPage)
			}

			wantPool, wantReady := k.Reference(srcPages, dstPageIDs, tc.totalPoolPages, tc.pageBytes)

			dstKVPool := make([]byte, tc.totalPoolPages*tc.pageBytes)
			dstPageReady := make([]byte, tc.totalPoolPages*4)
			c2cStagingPool := make([]byte, tc.totalPoolPages*tc.pageBytes)

			grid := k.Grid()
			block := k.Block()
			launchGridX := grid[0]
			if tc.numRanks >= 72 {
				// Single cooperative CTA loops over all NUM_PAGES pages to keep 72-rank host thread creation fast.
				launchGridX = 1
			}

			err := cudahostemu.RunMultiRank(context.Background(), src, tc.numRanks, cudahostemu.Launch{
				Kernel: k.Name,
				Grid:   cudahostemu.Dim3{X: launchGridX, Y: grid[1], Z: grid[2]},
				Block:  cudahostemu.Dim3{X: block[0], Y: block[1], Z: block[2]},
			},
				cudahostemu.IntArg(tc.totalPoolPages),
				cudahostemu.BufferArg("src_pages", "uint8_t", srcPages),
				cudahostemu.BufferArg("src_page_ids", "int32_t", i32Bytes(srcPageIDs)),
				cudahostemu.BufferArg("dst_page_ids", "int32_t", i32Bytes(dstPageIDs)),
				cudahostemu.BufferArg("dst_kv_pool", "uint8_t", dstKVPool),
				cudahostemu.BufferArg("dst_page_ready", "uint32_t", dstPageReady),
				cudahostemu.BufferArg("c2c_staging_pool", "uint8_t", c2cStagingPool),
			)
			if errors.Is(err, cudahostemu.ErrNoCompiler) {
				t.Skip(err)
			}
			if err != nil {
				t.Fatalf("RunMultiRank(%d ranks): %v", tc.numRanks, err)
			}

			gotReady := bytesU32(dstPageReady)
			for i, w := range wantReady {
				if gotReady[i] != w {
					t.Fatalf("dst_page_ready[%d] = %d, want %d", i, gotReady[i], w)
				}
			}
			if !bytes.Equal(dstKVPool, wantPool) {
				for p := 0; p < tc.totalPoolPages; p++ {
					start := p * tc.pageBytes
					end := start + tc.pageBytes
					if !bytes.Equal(dstKVPool[start:end], wantPool[start:end]) {
						t.Fatalf("dst_kv_pool page %d mismatch: got %x, want %x",
							p, dstKVPool[start:start+16], wantPool[start:start+16])
					}
				}
			}
			if !bytes.Equal(c2cStagingPool, wantPool) {
				t.Fatalf("c2c_staging_pool (decode verification / Grace C2C buffer) mismatch against Reference")
			}
		})
	}
}

func TestPlanPDDisaggregation(t *testing.T) {
	sla := PDSLA{
		PromptTokens:      4096,
		DecodeBatchPerGPU: 64,
		MaxTTFTMs:         250.0,
		MaxTPOTMs:         25.0,
	}

	baseModel := PDModelSpec{
		NumLayers:            61,
		HiddenSize:           7168,
		NumQHeads:            128,
		NumKVHeads:           32,
		HeadDim:              128,
		CacheKind:            CacheGQA,
		KVLoRARank:           512,
		RoPEDim:              64,
		TotalExperts:         288,
		TopK:                 8,
		ActiveParamsBillions: 37.0,
		TotalParamsBillions:  671.0,
	}

	// 1. Compare GQA vs MLA on a 4-rack pod: MLA must reduce TotalKVTransferBytes
	// and KVTransferMs by > 5x compared to GQA.
	gqaModel := baseModel
	gqaModel.CacheKind = CacheGQA

	mlaModel := baseModel
	mlaModel.CacheKind = CacheMLA

	gqaPlan, err := PlanPDDisaggregation(GenGB300, 4, sla, gqaModel)
	if err != nil {
		t.Fatalf("PlanPDDisaggregation(GQA): %v", err)
	}
	mlaPlan, err := PlanPDDisaggregation(GenGB300, 4, sla, mlaModel)
	if err != nil {
		t.Fatalf("PlanPDDisaggregation(MLA): %v", err)
	}

	if mlaPlan.KVBytesPerToken != 640 {
		t.Fatalf("MLA KVBytesPerToken = %d, want 640 (512 FP8 + 64*2 BF16)", mlaPlan.KVBytesPerToken)
	}
	wantGQABytes := int64(2 * gqaModel.NumKVHeads * gqaModel.HeadDim * 2)
	if gqaPlan.KVBytesPerToken != wantGQABytes {
		t.Fatalf("GQA KVBytesPerToken = %d, want %d", gqaPlan.KVBytesPerToken, wantGQABytes)
	}

	bytesRatio := float64(gqaPlan.TotalKVTransferBytes) / float64(mlaPlan.TotalKVTransferBytes)
	if bytesRatio <= 5.0 {
		t.Fatalf("expected MLA to reduce TotalKVTransferBytes by >5x vs GQA, got %.2fx (%d vs %d)",
			bytesRatio, gqaPlan.TotalKVTransferBytes, mlaPlan.TotalKVTransferBytes)
	}
	msRatio := gqaPlan.KVTransferMs / mlaPlan.KVTransferMs
	if msRatio <= 5.0 {
		t.Fatalf("expected MLA to reduce KVTransferMs by >5x vs GQA, got %.2fx (%.3f ms vs %.3f ms)",
			msRatio, gqaPlan.KVTransferMs, mlaPlan.KVTransferMs)
	}

	// 2. Compare GB200 vs GB300 on a 4-rack (288 GPU) pod.
	gb200Plan, err := PlanPDDisaggregation(GenGB200, 4, sla, mlaModel)
	if err != nil {
		t.Fatalf("PlanPDDisaggregation(GB200, 4 racks): %v", err)
	}
	gb300Plan, err := PlanPDDisaggregation(GenGB300, 4, sla, mlaModel)
	if err != nil {
		t.Fatalf("PlanPDDisaggregation(GB300, 4 racks): %v", err)
	}

	if gb200Plan.Generation != "gb200" || gb300Plan.Generation != "gb300" {
		t.Fatalf("Generation = (%q, %q), want (\"gb200\", \"gb300\")", gb200Plan.Generation, gb300Plan.Generation)
	}
	if gb200Plan.TotalRacks != 4 || gb300Plan.TotalRacks != 4 {
		t.Fatalf("TotalRacks = (%d, %d), want (4, 4)", gb200Plan.TotalRacks, gb300Plan.TotalRacks)
	}
	if gb200Plan.PrefillGPUs+gb200Plan.DecodeGPUs != 288 || gb300Plan.PrefillGPUs+gb300Plan.DecodeGPUs != 288 {
		t.Fatalf("total GPUs = (%d, %d), want 288",
			gb200Plan.PrefillGPUs+gb200Plan.DecodeGPUs, gb300Plan.PrefillGPUs+gb300Plan.DecodeGPUs)
	}
	if gb200Plan.TransferTransport != TransportConnectX7RDMA {
		t.Fatalf("GB200 TransferTransport = %q, want %q", gb200Plan.TransferTransport, TransportConnectX7RDMA)
	}
	if gb300Plan.TransferTransport != TransportConnectX8RDMA {
		t.Fatalf("GB300 TransferTransport = %q, want %q", gb300Plan.TransferTransport, TransportConnectX8RDMA)
	}
	if !gb200Plan.MeetsSLA || !gb300Plan.MeetsSLA {
		t.Fatalf("expected both GB200 and GB300 4-rack plans to meet SLA, got (%v, %v)",
			gb200Plan.MeetsSLA, gb300Plan.MeetsSLA)
	}
	if gb300Plan.KVTransferMs >= gb200Plan.KVTransferMs {
		t.Fatalf("expected GB300 KVTransferMs (%.3f ms) < GB200 KVTransferMs (%.3f ms)",
			gb300Plan.KVTransferMs, gb200Plan.KVTransferMs)
	}
	if gb300Plan.EstimatedTTFTMs >= gb200Plan.EstimatedTTFTMs {
		t.Fatalf("expected GB300 EstimatedTTFTMs (%.3f ms) < GB200 EstimatedTTFTMs (%.3f ms)",
			gb300Plan.EstimatedTTFTMs, gb200Plan.EstimatedTTFTMs)
	}
	if gb300Plan.TokensPerSecCluster <= gb200Plan.TokensPerSecCluster {
		t.Fatalf("expected GB300 TokensPerSecCluster (%.1f) > GB200 TokensPerSecCluster (%.1f)",
			gb300Plan.TokensPerSecCluster, gb200Plan.TokensPerSecCluster)
	}

	// 3. Verify single-rack (TotalRacks == 1) intra-rack NVLink 5 split.
	singleRackPlan, err := PlanPDDisaggregation(GenGB300, 1, sla, mlaModel)
	if err != nil {
		t.Fatalf("PlanPDDisaggregation(GB300, 1 rack): %v", err)
	}
	if singleRackPlan.TransferTransport != TransportNVLink5IntraRack {
		t.Fatalf("1-rack TransferTransport = %q, want %q", singleRackPlan.TransferTransport, TransportNVLink5IntraRack)
	}
	if singleRackPlan.PrefillGPUs+singleRackPlan.DecodeGPUs != 72 {
		t.Fatalf("1-rack GPUs = %d + %d, want 72", singleRackPlan.PrefillGPUs, singleRackPlan.DecodeGPUs)
	}

	// 4. Verify impossible SLA returns Pareto-closest plan with MeetsSLA == false.
	tightSLA := PDSLA{
		PromptTokens:      32768,
		DecodeBatchPerGPU: 128,
		MaxTTFTMs:         0.01,
		MaxTPOTMs:         0.01,
	}
	tightPlan, err := PlanPDDisaggregation(GenGB200, 2, tightSLA, gqaModel)
	if err != nil {
		t.Fatalf("PlanPDDisaggregation(tightSLA): %v", err)
	}
	if tightPlan.MeetsSLA {
		t.Fatalf("expected MeetsSLA == false for tightSLA, got %+v", tightPlan)
	}
}

func ExamplePlanPDDisaggregation() {
	sla := PDSLA{
		PromptTokens:      4096,
		DecodeBatchPerGPU: 64,
		MaxTTFTMs:         250.0,
		MaxTPOTMs:         25.0,
	}
	model := PDModelSpec{
		NumLayers:            61,
		HiddenSize:           7168,
		NumQHeads:            128,
		NumKVHeads:           32,
		HeadDim:              128,
		CacheKind:            CacheMLA,
		KVLoRARank:           512,
		RoPEDim:              64,
		TotalExperts:         288,
		TopK:                 8,
		ActiveParamsBillions: 37.0,
		TotalParamsBillions:  671.0,
	}
	gb200, _ := PlanPDDisaggregation(GenGB200, 4, sla, model)
	gb300, _ := PlanPDDisaggregation(GenGB300, 4, sla, model)
	fmt.Println(
		gb300.Generation,
		gb300.PrefillGPUs+gb300.DecodeGPUs,
		gb300.KVBytesPerToken,
		gb300.TransferTransport,
		gb300.MeetsSLA,
		gb300.KVTransferMs < gb200.KVTransferMs,
		gb300.EstimatedTTFTMs < gb200.EstimatedTTFTMs,
		gb300.TokensPerSecCluster > gb200.TokensPerSecCluster,
	)
	// Output:
	// gb300 288 640 connectx8-800g-rdma true true true true
}
