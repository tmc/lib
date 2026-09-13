package tileir_test

import (
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/tileir"
)

const auditSample = `
cuda_tile.module @gemm_module {
  cuda_tile.entry @gemm_opt(%A: !cuda_tile.tile<ptr<f16>>, %B: !cuda_tile.tile<ptr<f16>>, %C: !cuda_tile.tile<ptr<f32>>) {
    %tok0 = cuda_tile.make_token : !cuda_tile.token
    %vA = cuda_tile.load_view_tko %A, %tok0 : (!cuda_tile.tile<ptr<f16>>, !cuda_tile.token) -> !cuda_tile.tile<64x64xf16>
    %vB = cuda_tile.load_view_tko %B, %tok0 : (!cuda_tile.tile<ptr<f16>>, !cuda_tile.token) -> !cuda_tile.tile<64x64xf16>
    %cst = cuda_tile.constant <f32: 0.0> : !cuda_tile.tile<64x64xf32>
    %acc = cuda_tile.mma_f %vA, %vB, %cst : (!cuda_tile.tile<64x64xf16>, !cuda_tile.tile<64x64xf16>, !cuda_tile.tile<64x64xf32>) -> !cuda_tile.tile<64x64xf32>
    %tok1 = cuda_tile.store_view_tko %acc, %C, %tok0 : (!cuda_tile.tile<64x64xf32>, !cuda_tile.tile<ptr<f32>>, !cuda_tile.token) -> !cuda_tile.token
    cuda_tile.return
  }

  cuda_tile.entry @gemm_scalar_fallback(%A: !cuda_tile.tile<ptr<f16>>, %B: !cuda_tile.tile<ptr<f16>>, %C: !cuda_tile.tile<ptr<f32>>) {
    %tok0 = cuda_tile.make_token : !cuda_tile.token
    %vA = cuda_tile.load_ptr_tko %A, %tok0 : (!cuda_tile.tile<ptr<f16>>, !cuda_tile.token) -> !cuda_tile.tile<64x64xf16>
    %prod = cuda_tile.mul_f %vA, %vA : !cuda_tile.tile<64x64xf16>
    %sum = cuda_tile.add_f %prod, %prod : !cuda_tile.tile<64x64xf16>
    %p = cuda_tile.ptr_to_int %A : !cuda_tile.tile<ptr<f16>> -> i64
    cuda_tile.return
  }
}
`

func TestOpcodeSummary(t *testing.T) {
	mod, err := tileir.ParseString(auditSample)
	if err != nil {
		t.Fatalf("ParseString: %v", err)
	}

	fnOpt := mod.FindFunction("gemm_opt")
	if fnOpt == nil {
		t.Fatal("gemm_opt not found")
	}

	sumOpt := fnOpt.Summary()
	if sumOpt.Total != 7 {
		t.Errorf("gemm_opt Total = %d, want 7", sumOpt.Total)
	}

	if got := sumOpt.Count("load_view_tko"); got != 2 {
		t.Errorf("Count(load_view_tko) = %d, want 2", got)
	}
	if got := sumOpt.Count("cuda_tile.load_view_tko"); got != 2 {
		t.Errorf("Count(cuda_tile.load_view_tko) = %d, want 2", got)
	}
	if got := sumOpt.Count("mma"); got != 1 {
		t.Errorf("Count(mma) = %d, want 1", got)
	}
	if got := sumOpt.Count("mma_f"); got != 1 {
		t.Errorf("Count(mma_f) = %d, want 1", got)
	}
	if got := sumOpt.Count("store_view_tko"); got != 1 {
		t.Errorf("Count(store_view_tko) = %d, want 1", got)
	}

	// Categories
	if got := sumOpt.CategoryCount(tileir.CatComputeMMA); got != 1 {
		t.Errorf("CategoryCount(CatComputeMMA) = %d, want 1", got)
	}
	if got := sumOpt.CategoryCount(tileir.CatTileMemory); got != 3 { // 2 load_view_tko + 1 store_view_tko
		t.Errorf("CategoryCount(CatTileMemory) = %d, want 3", got)
	}
	if got := sumOpt.CategoryCount(tileir.CatSyncToken); got != 1 { // make_token
		t.Errorf("CategoryCount(CatSyncToken) = %d, want 1", got)
	}
	if got := sumOpt.CategoryCount(tileir.CatControlFlow); got != 1 { // return
		t.Errorf("CategoryCount(CatControlFlow) = %d, want 1", got)
	}

	// Module-level summary
	modSum := mod.Summary()
	if modSum.Total != 13 { // 7 from gemm_opt + 6 from gemm_scalar_fallback
		t.Errorf("mod.Summary().Total = %d, want 13", modSum.Total)
	}
}

func TestAssertOpcodes(t *testing.T) {
	mod, err := tileir.ParseString(auditSample)
	if err != nil {
		t.Fatalf("ParseString: %v", err)
	}

	tests := []struct {
		name    string
		kernel  string
		require []string
		forbid  []string
		wantErr string
	}{
		{
			name:    "optimized kernel passes requirements and forbid fallbacks",
			kernel:  "gemm_opt",
			require: []string{"cuda_tile.mma_f"},
			forbid:  tileir.ForbidScalarMemoryFallbacks,
		},
		{
			name:    "optimized kernel passes TMA requirement and unfused forbid list",
			kernel:  "gemm_opt",
			require: tileir.RequireTMAOperations,
			forbid:  tileir.ForbidUnfusedFallbacks,
		},
		{
			name:    "optimized kernel missing required int mma",
			kernel:  "gemm_opt",
			require: []string{"cuda_tile.mma_i"},
			wantErr: `kernel "gemm_opt" missing required opcode "cuda_tile.mma_i"`,
		},
		{
			name:    "fallback kernel fails scalar memory forbid list",
			kernel:  "gemm_scalar_fallback",
			forbid:  tileir.ForbidScalarMemoryFallbacks,
			wantErr: `kernel "gemm_scalar_fallback" contains forbidden opcode "cuda_tile.load_ptr_tko" (count 1)`,
		},
		{
			name:    "fallback kernel fails unfused forbid list",
			kernel:  "gemm_scalar_fallback",
			forbid:  tileir.ForbidUnfusedFallbacks,
			wantErr: `kernel "gemm_scalar_fallback" contains forbidden opcode "cuda_tile.mul_f" (count 1)`,
		},
		{
			name:    "fallback kernel missing tensor core requirement",
			kernel:  "gemm_scalar_fallback",
			require: []string{"cuda_tile.mma_f"},
			wantErr: `kernel "gemm_scalar_fallback" missing required opcode "cuda_tile.mma_f"`,
		},
		{
			name:    "missing kernel returns not found",
			kernel:  "nosuchkernel",
			wantErr: `kernel "nosuchkernel" not found`,
		},
		{
			name:    "module audit without kernel name checks all kernels",
			kernel:  "",
			forbid:  []string{"load_ptr_tko"},
			wantErr: `module contains forbidden opcode "load_ptr_tko" (count 1)`,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := mod.AssertOpcodes(tt.kernel, tt.require, tt.forbid)
			switch {
			case tt.wantErr == "" && err != nil:
				t.Errorf("AssertOpcodes() unexpected error: %v", err)
			case tt.wantErr != "" && err == nil:
				t.Errorf("AssertOpcodes() = nil, want error containing %q", tt.wantErr)
			case tt.wantErr != "" && !strings.Contains(err.Error(), tt.wantErr):
				t.Errorf("AssertOpcodes() error = %v, want error containing %q", err, tt.wantErr)
			}
		})
	}
}

func TestAuditNilSafety(t *testing.T) {
	var fn *tileir.Function
	sum := fn.Summary()
	if sum.Total != 0 || sum.Count("mma") != 0 || sum.CategoryCount(tileir.CatComputeMMA) != 0 {
		t.Errorf("(*Function)(nil).Summary() = %+v, want zeros", sum)
	}

	var mod *tileir.Module
	modSum := mod.Summary()
	if modSum.Total != 0 {
		t.Errorf("(*Module)(nil).Summary() = %+v, want zeros", modSum)
	}

	if err := mod.AssertOpcodes("foo", nil, nil); err == nil {
		t.Error("(*Module)(nil).AssertOpcodes() = nil, want error")
	}

	var opSum *tileir.OpcodeSummary
	if got := opSum.Count("mma"); got != 0 {
		t.Errorf("(*OpcodeSummary)(nil).Count() = %d, want 0", got)
	}
	if got := opSum.CategoryCount(tileir.CatComputeMMA); got != 0 {
		t.Errorf("(*OpcodeSummary)(nil).CategoryCount() = %d, want 0", got)
	}
}
