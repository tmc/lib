package expertparallel

import (
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/egraph/hopper"
)

func TestTBOMegakernelRolesAndPartition(t *testing.T) {
	roleTests := []struct {
		role TBOWarpgroupRole
		want string
	}{
		{WarpgroupRoleCompute, "compute"},
		{WarpgroupRoleFabricDispatch, "fabric_dispatch"},
		{WarpgroupRoleFabricCombine, "fabric_combine"},
		{WarpgroupRoleC2CDoorbell, "c2c_doorbell"},
		{TBOWarpgroupRole(99), "TBOWarpgroupRole(99)"},
	}
	for _, tc := range roleTests {
		if got := tc.role.String(); got != tc.want {
			t.Errorf("TBOWarpgroupRole(%d).String() = %q, want %q", uint8(tc.role), got, tc.want)
		}
	}

	partTests := []struct {
		name                             string
		totalCTAs                        int
		computeUs, dispatchUs, combineUs float64
		enableC2C                        bool
		wantSum                          int
		wantMinEach                      int
	}{
		{
			name:      "ZeroCTAs",
			totalCTAs: 0,
			wantSum:   0,
		},
		{
			name:      "OneCTA",
			totalCTAs: 1,
			wantSum:   1,
		},
		{
			name:      "TwoCTAs",
			totalCTAs: 2,
			wantSum:   2,
		},
		{
			name:        "MinThreeCTAs",
			totalCTAs:   3,
			computeUs:   50,
			dispatchUs:  10,
			combineUs:   10,
			enableC2C:   true,
			wantSum:     3,
			wantMinEach: 1,
		},
		{
			name:        "GB200_148CTAs_NoC2C",
			totalCTAs:   148,
			computeUs:   80,
			dispatchUs:  25,
			combineUs:   20,
			enableC2C:   false,
			wantSum:     148,
			wantMinEach: 1,
		},
		{
			name:        "GB200_148CTAs_WithC2C",
			totalCTAs:   148,
			computeUs:   80,
			dispatchUs:  25,
			combineUs:   20,
			enableC2C:   true,
			wantSum:     148,
			wantMinEach: 1,
		},
		{
			name:        "Rubin_160CTAs_DefaultLatencies",
			totalCTAs:   160,
			computeUs:   0,
			dispatchUs:  0,
			combineUs:   0,
			enableC2C:   true,
			wantSum:     160,
			wantMinEach: 1,
		},
	}

	for _, tc := range partTests {
		t.Run(tc.name, func(t *testing.T) {
			comp, disp, comb := PartitionTBOCTAs(tc.totalCTAs, tc.computeUs, tc.dispatchUs, tc.combineUs, tc.enableC2C)
			if got := comp + disp + comb; got != tc.wantSum {
				t.Fatalf("PartitionTBOCTAs sum = %d (%d+%d+%d), want %d", got, comp, disp, comb, tc.wantSum)
			}
			if tc.wantMinEach > 0 {
				if comp < tc.wantMinEach || disp < tc.wantMinEach || comb < tc.wantMinEach {
					t.Fatalf("PartitionTBOCTAs = (%d, %d, %d), want each >= %d", comp, disp, comb, tc.wantMinEach)
				}
			}
		})
	}

	// Verify enabling C2C shifts 1 CTA from Compute to Dispatch when computeCTAs > 1.
	c0, d0, cb0 := PartitionTBOCTAs(148, 80, 25, 20, false)
	c1, d1, cb1 := PartitionTBOCTAs(148, 80, 25, 20, true)
	if c1 != c0-1 || d1 != d0+1 || cb1 != cb0 {
		t.Fatalf("C2C shift mismatch: noC2C=(%d,%d,%d) withC2C=(%d,%d,%d)", c0, d0, cb0, c1, d1, cb1)
	}
}

func TestTBOMegakernelBuildAndSource(t *testing.T) {
	cases := []struct {
		name      string
		spec      TBOMegakernelSpec
		wantC2C   int
		wantError bool
	}{
		{
			name: "GB200_EP4_Shared_C2CEnabled",
			spec: TBOMegakernelSpec{
				Name:               "gb200_tbo",
				Config:             testConfig(4, 16, 128, 64, 64),
				TotalCTAs:          148,
				ThreadsPerCTA:      256,
				Microbatches:       4,
				ComputeUs:          90,
				DispatchUs:         30,
				CombineUs:          25,
				EnableC2CDoorbell:  true,
				ColdTokenCrossover: 8,
			},
			wantC2C: 1,
		},
		{
			name: "GB300_EP8_NoShared_NoC2C",
			spec: TBOMegakernelSpec{
				Name:              "gb300_tbo",
				Config:            testConfig(8, 16, 128, 64, 0),
				TotalCTAs:         64,
				ThreadsPerCTA:     256,
				Microbatches:      2,
				ComputeUs:         60,
				DispatchUs:        20,
				CombineUs:         20,
				EnableC2CDoorbell: false,
			},
			wantC2C: 0,
		},
		{
			name: "Defaults_EP2_C2CEnabled",
			spec: TBOMegakernelSpec{
				Config:            testConfig(2, 8, 128, 64, 64),
				EnableC2CDoorbell: true,
			},
			wantC2C: 1,
		},
		{
			name: "BadName",
			spec: TBOMegakernelSpec{
				Name:   "bad name",
				Config: testConfig(2, 8, 128, 64, 0),
			},
			wantError: true,
		},
		{
			name: "TooFewCTAs",
			spec: TBOMegakernelSpec{
				Name:      "too_few_ctas",
				Config:    testConfig(2, 8, 128, 64, 0),
				TotalCTAs: 2,
			},
			wantError: true,
		},
		{
			name: "BadThreadsPerCTA",
			spec: TBOMegakernelSpec{
				Name:          "bad_threads",
				Config:        testConfig(2, 8, 128, 64, 0),
				ThreadsPerCTA: 100,
			},
			wantError: true,
		},
		{
			name: "TooFewMicrobatches",
			spec: TBOMegakernelSpec{
				Name:         "bad_mb",
				Config:       testConfig(2, 8, 128, 64, 0),
				Microbatches: 1,
			},
			wantError: true,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			plan, err := BuildTBOMegakernel(tc.spec)
			if tc.wantError {
				if err == nil {
					t.Fatalf("BuildTBOMegakernel succeeded, want error")
				}
				return
			}
			if err != nil {
				t.Fatalf("BuildTBOMegakernel: %v", err)
			}
			if plan.ComputeCTAs+plan.DispatchCTAs+plan.CombineCTAs != plan.TotalCTAs {
				t.Fatalf("CTA sum %d+%d+%d != TotalCTAs %d",
					plan.ComputeCTAs, plan.DispatchCTAs, plan.CombineCTAs, plan.TotalCTAs)
			}
			if plan.C2CDoorbellCTAs != tc.wantC2C {
				t.Fatalf("C2CDoorbellCTAs = %d, want %d", plan.C2CDoorbellCTAs, tc.wantC2C)
			}
			if plan.OverlappedLatencyUs >= plan.SequentialLatencyUs || plan.TBOSpeedup <= 1.0 {
				t.Fatalf("expected TBO speedup > 1.0: seq=%v us overlapped=%v us speedup=%v",
					plan.SequentialLatencyUs, plan.OverlappedLatencyUs, plan.TBOSpeedup)
			}
			if plan.WorkspaceBytes() <= 2*plan.BasePlan.WorkspaceBytes() {
				t.Fatalf("WorkspaceBytes = %d, want > %d", plan.WorkspaceBytes(), 2*plan.BasePlan.WorkspaceBytes())
			}

			// Verify CTA role assignments across the grid.
			if got := plan.RoleForCTA(0); got != WarpgroupRoleCompute {
				t.Fatalf("RoleForCTA(0) = %v, want %v", got, WarpgroupRoleCompute)
			}
			if got := plan.RoleForCTA(plan.ComputeCTAs); got != WarpgroupRoleFabricDispatch && got != WarpgroupRoleC2CDoorbell {
				t.Fatalf("RoleForCTA(ComputeCTAs) = %v, want dispatch/doorbell", got)
			}
			if tc.wantC2C > 0 {
				doorbellIdx := plan.ComputeCTAs + plan.DispatchCTAs - 1
				if got := plan.RoleForCTA(doorbellIdx); got != WarpgroupRoleC2CDoorbell {
					t.Fatalf("RoleForCTA(%d) = %v, want %v", doorbellIdx, got, WarpgroupRoleC2CDoorbell)
				}
			}
			if got := plan.RoleForCTA(plan.TotalCTAs - 1); got != WarpgroupRoleFabricCombine {
				t.Fatalf("RoleForCTA(last) = %v, want %v", got, WarpgroupRoleFabricCombine)
			}

			// Verify unified persistent CUDA C++ megakernel source substrings.
			for _, want := range []string{
				plan.Name + "_tbo_megakernel",
				"const int slot = step & 1;",
				"cta < COMPUTE_CTAS",
				"cta < COMPUTE_CTAS + DISPATCH_CTAS",
				"st.release.sys.global.u32",
				"ld.acquire.sys.global.u32",
				"st.release.sys.global.u64",
				"ld.acquire.sys.global.u64",
				"eg_tbo_c2c_doorbell_signal",
				"routed_tokens >= 1u && routed_tokens <= (uint32_t)COLD_TOKEN_CROSSOVER",
			} {
				if !strings.Contains(plan.Source, want) {
					t.Fatalf("TBOMegakernelPlan.Source missing %q", want)
				}
			}
		})
	}
}

func TestTBOMegakernelReferenceRankInvariance(t *testing.T) {
	cfg1 := testConfig(1, 8, 128, 64, 64)
	cfg4 := testConfig(4, 8, 128, 64, 64)

	p1, err := BuildTBOMegakernel(TBOMegakernelSpec{
		Name:              "tbo_ep1",
		Config:            cfg1,
		Microbatches:      2,
		EnableC2CDoorbell: true,
	})
	if err != nil {
		t.Fatalf("BuildTBOMegakernel EP=1: %v", err)
	}
	p4, err := BuildTBOMegakernel(TBOMegakernelSpec{
		Name:              "tbo_ep4",
		Config:            cfg4,
		Microbatches:      2,
		EnableC2CDoorbell: true,
	})
	if err != nil {
		t.Fatalf("BuildTBOMegakernel EP=4: %v", err)
	}

	batches := []map[string]hopper.Value{
		testEnv(cfg1, 101),
		testEnv(cfg1, 202),
	}
	out1, err := p1.Reference(batches)
	if err != nil {
		t.Fatalf("p1.Reference: %v", err)
	}
	out4, err := p4.Reference(batches)
	if err != nil {
		t.Fatalf("p4.Reference: %v", err)
	}
	for mb := range batches {
		if idx := firstDiff(out1[mb], out4[mb]); idx >= 0 {
			t.Fatalf("microbatch %d: EP=4 differs from EP=1 at index %d: %g vs %g",
				mb, idx, out4[mb][idx], out1[mb][idx])
		}
	}
}
