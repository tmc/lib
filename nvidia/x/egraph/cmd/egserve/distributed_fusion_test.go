package main

import (
	"encoding/json"
	"math"
	"net/http"
	"net/http/httptest"
	"slices"
	"strings"
	"testing"
)

func TestDistributedFusionReportBuilder(t *testing.T) {
	srv, err := NewEGServer(ServerConfig{
		Arch:      "sm_121a",
		HiddenDim: 256,
		IntermDim: 512,
		Layers:    4,
	})
	if err != nil {
		t.Fatalf("NewEGServer: %v", err)
	}

	iters, err := srv.BuildKernelStreamIterations("simple", 1)
	if err != nil {
		t.Fatalf("BuildKernelStreamIterations: %v", err)
	}
	if len(iters) == 0 {
		t.Fatal("expected non-empty simple iterations")
	}

	champ := iters[len(iters)-1]
	rep, err := BuildDistributedFusionReport("sm_121a", champ)
	if err != nil {
		t.Fatalf("BuildDistributedFusionReport: %v", err)
	}

	if rep.Arch != "sm_121a" {
		t.Fatalf("Arch = %q, want sm_121a", rep.Arch)
	}
	if rep.SourceIteration != champ.Iteration {
		t.Fatalf("SourceIteration = %d, want %d", rep.SourceIteration, champ.Iteration)
	}
	if rep.SourcePhase != champ.Phase {
		t.Fatalf("SourcePhase = %q, want %q", rep.SourcePhase, champ.Phase)
	}
	if math.Abs(rep.SingleGPULatencyUs-7.96) > 1e-6 {
		t.Fatalf("SingleGPULatencyUs = %.2f, want 7.96", rep.SingleGPULatencyUs)
	}
	if !rep.SLAAttained {
		t.Fatal("expected SLAAttained = true")
	}
	if rep.DisaggPDPrefillTTFTMs <= 0 || rep.DisaggPDPrefillTTFTMs >= 200.0 {
		t.Fatalf("DisaggPDPrefillTTFTMs = %.2f, want in (0, 200)", rep.DisaggPDPrefillTTFTMs)
	}
	if rep.DisaggPDDecodeTPOTMs <= 0 || rep.DisaggPDDecodeTPOTMs >= 20.0 {
		t.Fatalf("DisaggPDDecodeTPOTMs = %.2f, want in (0, 20)", rep.DisaggPDDecodeTPOTMs)
	}

	wantTP := []int{1, 2, 4, 8, 72}
	if len(rep.Tiers) != len(wantTP) {
		t.Fatalf("len(Tiers) = %d, want %d", len(rep.Tiers), len(wantTP))
	}

	wantSpeedups := map[int]float64{
		1:  1.00,
		2:  1.42,
		4:  1.68,
		8:  1.94,
		72: 2.38,
	}

	for i, tier := range rep.Tiers {
		if tier.TensorParallelDegree != wantTP[i] {
			t.Fatalf("tier[%d].TensorParallelDegree = %d, want %d", i, tier.TensorParallelDegree, wantTP[i])
		}
		if strings.TrimSpace(tier.TopologyLabel) == "" {
			t.Fatalf("tier[%d] empty TopologyLabel", i)
		}
		if tier.InterconnectBWGBps <= 0 {
			t.Fatalf("tier[%d].InterconnectBWGBps = %.1f, want > 0", i, tier.InterconnectBWGBps)
		}
		wantSp := wantSpeedups[tier.TensorParallelDegree]
		if math.Abs(tier.FusionSpeedup-wantSp) > 0.02 {
			t.Fatalf("TP=%d FusionSpeedup = %.2f, want %.2f", tier.TensorParallelDegree, tier.FusionSpeedup, wantSp)
		}
		if tier.GraceC2CKVPrefetchUs <= 0 {
			t.Fatalf("TP=%d GraceC2CKVPrefetchUs = %.2f, want > 0", tier.TensorParallelDegree, tier.GraceC2CKVPrefetchUs)
		}
		if len(tier.CUDAMultimemEpilogue) == 0 {
			t.Fatalf("TP=%d empty CUDAMultimemEpilogue", tier.TensorParallelDegree)
		}
		if tier.TensorParallelDegree > 1 {
			if tier.UnfusedNCCLAllReduceUs <= 0 {
				t.Fatalf("TP=%d UnfusedNCCLAllReduceUs = %.2f, want > 0", tier.TensorParallelDegree, tier.UnfusedNCCLAllReduceUs)
			}
			if tier.FusedMultimemTotalUs >= tier.UnfusedTotalUs {
				t.Fatalf("TP=%d FusedMultimemTotalUs (%.2f) >= UnfusedTotalUs (%.2f)",
					tier.TensorParallelDegree, tier.FusedMultimemTotalUs, tier.UnfusedTotalUs)
			}
			for _, instr := range []string{
				"multimem.ld_reduce.relaxed.sys.global.add.v4.bf16x2",
				"multimem.st.relaxed.sys.global.v4.bf16x2",
				"fence.proxy.async.global",
			} {
				if !strings.Contains(tier.PTXMultimemSnippet, instr) {
					t.Fatalf("TP=%d PTXMultimemSnippet missing %q:\n%s", tier.TensorParallelDegree, instr, tier.PTXMultimemSnippet)
				}
			}
		}
	}
}

func TestDistributedFusionDefaultFallbackAndHTTPHandler(t *testing.T) {
	// Zero-value iteration falls back cleanly to 7.96 us single-GPU latency.
	fallbackRep, err := BuildDistributedFusionReport("", KernelStreamIteration{})
	if err != nil {
		t.Fatalf("BuildDistributedFusionReport(zero): %v", err)
	}
	if fallbackRep.Arch != "sm_121a" {
		t.Fatalf("fallback Arch = %q, want sm_121a", fallbackRep.Arch)
	}
	if math.Abs(fallbackRep.SingleGPULatencyUs-7.96) > 1e-6 {
		t.Fatalf("fallback SingleGPULatencyUs = %.2f, want 7.96", fallbackRep.SingleGPULatencyUs)
	}

	srv, err := NewEGServer(ServerConfig{
		Arch:      "sm_121a",
		HiddenDim: 256,
		IntermDim: 512,
		Layers:    4,
	})
	if err != nil {
		t.Fatalf("NewEGServer: %v", err)
	}

	for _, tc := range []struct {
		name           string
		method         string
		target         string
		wantStatus     int
		wantIter       int
		wantSingleUs   float64
		checkMultimem  bool
	}{
		{
			name:          "default_champion_iter23",
			method:        http.MethodGet,
			target:        "/v1/kernels/distributed-fusion",
			wantStatus:    http.StatusOK,
			wantIter:      23,
			wantSingleUs:  7.96,
			checkMultimem: true,
		},
		{
			name:          "explicit_iter0_baseline",
			method:        http.MethodGet,
			target:        "/v1/kernels/distributed-fusion?iter=0",
			wantStatus:    http.StatusOK,
			wantIter:      0,
			wantSingleUs:  48.20,
			checkMultimem: true,
		},
		{
			name:       "post_method_not_allowed",
			method:     http.MethodPost,
			target:     "/v1/kernels/distributed-fusion",
			wantStatus: http.StatusMethodNotAllowed,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			req := httptest.NewRequest(tc.method, tc.target, nil)
			rec := httptest.NewRecorder()
			srv.handleKernelsDistributedFusion(rec, req)

			if rec.Code != tc.wantStatus {
				t.Fatalf("status = %d, want %d (body=%s)", rec.Code, tc.wantStatus, rec.Body.String())
			}
			if tc.wantStatus != http.StatusOK {
				return
			}

			var rep DistributedFusionReport
			if err := json.NewDecoder(rec.Body).Decode(&rep); err != nil {
				t.Fatalf("decode JSON response: %v", err)
			}
			if rep.SourceIteration != tc.wantIter {
				t.Fatalf("SourceIteration = %d, want %d", rep.SourceIteration, tc.wantIter)
			}
			if math.Abs(rep.SingleGPULatencyUs-tc.wantSingleUs) > 1e-2 {
				t.Fatalf("SingleGPULatencyUs = %.2f, want %.2f", rep.SingleGPULatencyUs, tc.wantSingleUs)
			}
			var gotTPs []int
			for _, tier := range rep.Tiers {
				gotTPs = append(gotTPs, tier.TensorParallelDegree)
			}
			if !slices.Equal(gotTPs, []int{1, 2, 4, 8, 72}) {
				t.Fatalf("got TPs = %v, want [1 2 4 8 72]", gotTPs)
			}
			if tc.checkMultimem {
				nvl72Tier := rep.Tiers[len(rep.Tiers)-1]
				if !strings.Contains(nvl72Tier.TopologyLabel, "NVL72") {
					t.Fatalf("expected NVL72 in tier label, got %q", nvl72Tier.TopologyLabel)
				}
				if nvl72Tier.InterconnectBWGBps != 1800.0 {
					t.Fatalf("NVL72 InterconnectBWGBps = %.1f, want 1800.0", nvl72Tier.InterconnectBWGBps)
				}
			}
		})
	}
}
