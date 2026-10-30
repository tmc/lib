package hopper

import (
	"context"
	"errors"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudahostemu"
)

func TestQualifyStep(t *testing.T) {
	ctx := context.Background()

	rmsnorm := NewRMSNormKernel("qualify_rmsnorm_f32", 64, 1e-5)
	skinny := SkinnyGEMM{
		Name:     "qualify_skinny_gemm",
		N:        64,
		K:        256,
		MaxM:     4,
		Weight:   WeightBF16,
		Out:      F32,
		Epilogue: Identity,
	}
	tileNVFP4 := TileGEMM{
		Name:     "qualify_tile_gemm_wgmma_nvfp4",
		N:        128,
		K:        64,
		BM:       128,
		BN:       128,
		Stages:   2,
		WGMMA:    true,
		Repacked: true,
		Weight:   WeightNVFP4,
		Out:      F32,
		Epilogue: Identity,
	}
	pagedAttn := PagedAttention{
		Name:   "qualify_paged_attention",
		Hq:     8,
		Hkv:    2,
		QLen:   1,
		Window: 0,
		Page:   16,
		Q:      E4M3,
		Out:    F32,
		Chunk:  64,
		Depth:  1,
		Stages: 2,
	}

	tests := []struct {
		name       string
		step       Step
		M, N, K    int
		eps        float32
		wantSASS   bool
		wantTileIR bool
	}{
		{
			name:       "RowKernel_RMSNorm",
			step:       Step{Row: &rmsnorm, Rows: 4},
			M:          4,
			N:          64,
			K:          64,
			eps:        1e-5,
			wantSASS:   false,
			wantTileIR: true,
		},
		{
			name:       "SkinnyGEMM",
			step:       Step{GEMM: &skinny, M: 4},
			M:          4,
			N:          64,
			K:          256,
			eps:        1e-5,
			wantSASS:   false,
			wantTileIR: true,
		},
		{
			name:       "TileGEMM_WGMMA_NVFP4",
			step:       Step{Tile: &tileNVFP4, M: 16},
			M:          16,
			N:          128,
			K:          64,
			eps:        1e-5,
			wantSASS:   true,
			wantTileIR: true,
		},
		{
			name:       "PagedAttention",
			step:       Step{Attention: &pagedAttn, Batch: 1, MaxKeys: 64, MaxPages: 4},
			M:          1,
			N:          64,
			K:          64,
			eps:        1e-5,
			wantSASS:   false,
			wantTileIR: false,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			receipt, err := QualifyStep(ctx, tc.step, tc.M, tc.N, tc.K, tc.eps)
			if errors.Is(err, cudahostemu.ErrNoCompiler) {
				t.Skip("no host C++ compiler; skipping qualification check")
			}
			if err != nil {
				t.Fatalf("QualifyStep(%s) failed: %v", tc.name, err)
			}
			if !receipt.Passed() {
				t.Fatalf("QualifyStep(%s): receipt.Passed() = false: %+v", tc.name, receipt)
			}
			if !receipt.CUDAValid {
				t.Errorf("QualifyStep(%s): CUDAValid = false", tc.name)
			}
			if !receipt.HazardsChecked {
				t.Errorf("QualifyStep(%s): HazardsChecked = false", tc.name)
			}
			if receipt.HazardCount != 0 || len(receipt.Hazards) != 0 {
				t.Errorf("QualifyStep(%s): HazardCount = %d, Hazards = %+v", tc.name, receipt.HazardCount, receipt.Hazards)
			}
			if receipt.SASSEquivVerified != tc.wantSASS {
				t.Errorf("QualifyStep(%s): SASSEquivVerified = %v, want %v", tc.name, receipt.SASSEquivVerified, tc.wantSASS)
			}
			if receipt.TileIRAgreement != tc.wantTileIR {
				t.Errorf("QualifyStep(%s): TileIRAgreement = %v, want %v", tc.name, receipt.TileIRAgreement, tc.wantTileIR)
			}
		})
	}
}

func TestQualifyPlan(t *testing.T) {
	ctx := context.Background()

	b := NewBuilder()
	x := b.Input("x", BF16, 4, 256)
	w := b.Input("w", BF16, 64, 256)
	nw := b.Input("norm_w", F32, 64)
	mm := b.MatMul(x, w)
	ss := b.ReduceSum(b.Map(MustLambda("(mul $0 $0)"), mm))
	normed := b.Map(MustLambda("(mul (mul $0 (rsqrt (add (div $1 64) 1e-5))) $2)"), mm, ss, nw)

	expr, err := Optimize(b.G, normed)
	if err != nil {
		t.Fatalf("Optimize failed: %v", err)
	}
	plan, err := Compile(expr, "qualify_plan", F32)
	if err != nil {
		t.Fatalf("Compile failed: %v", err)
	}
	if len(plan.Steps()) < 2 {
		t.Fatalf("expected multi-step plan, got %d steps", len(plan.Steps()))
	}

	receipts, err := QualifyPlan(ctx, plan, 4, 64, 256, 1e-5)
	if errors.Is(err, cudahostemu.ErrNoCompiler) {
		t.Skip("no host C++ compiler; skipping plan qualification check")
	}
	if err != nil {
		t.Fatalf("QualifyPlan failed: %v", err)
	}
	if len(receipts) != len(plan.Steps()) {
		t.Fatalf("got %d receipts, want %d", len(receipts), len(plan.Steps()))
	}
	for i, r := range receipts {
		if !r.Passed() {
			t.Errorf("step %d (%s): Passed() = false: %+v", i, r.Name, r)
		}
		if r.HazardCount != 0 {
			t.Errorf("step %d (%s): HazardCount = %d, want 0", i, r.Name, r.HazardCount)
		}
		if !r.TileIRAgreement {
			t.Errorf("step %d (%s): TileIRAgreement = false, want true", i, r.Name)
		}
	}
}

func TestQualifySM100Kernels(t *testing.T) {
	ctx := context.Background()

	receipt, err := QualifySM100Kernels(ctx)
	if err != nil {
		t.Fatalf("QualifySM100Kernels failed: %v", err)
	}
	if !receipt.Passed() {
		t.Fatalf("QualifySM100Kernels: receipt.Passed() = false: %+v", receipt)
	}
	if !receipt.CUDAValid {
		t.Errorf("QualifySM100Kernels: CUDAValid = false")
	}
	if !receipt.TMEMTraceVerified {
		t.Errorf("QualifySM100Kernels: TMEMTraceVerified = false")
	}
	if !receipt.SASSEquivVerified {
		t.Errorf("QualifySM100Kernels: SASSEquivVerified = false")
	}
	if receipt.TMEMPeakColumns != 160 {
		t.Errorf("QualifySM100Kernels: TMEMPeakColumns = %d, want 160", receipt.TMEMPeakColumns)
	}

	suiteReceipt, err := QualifySM100Suite(ctx)
	if err != nil {
		t.Fatalf("QualifySM100Suite failed: %v", err)
	}
	if !suiteReceipt.Passed() || suiteReceipt.TMEMPeakColumns != receipt.TMEMPeakColumns {
		t.Errorf("QualifySM100Suite receipt mismatch: %+v vs %+v", suiteReceipt, receipt)
	}
}
