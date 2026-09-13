package cudasass_test

import (
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudasass"
)

const testSASSFixture = `
--- Kernel: ds4_iq2_simd ---
/* 0010 */   @P0 LDG.E.CONSTANT.F32 R4, [R2.64] ;
/* 0020 */   LOP3.LUT R6, R4, R5, 0xca ;
/* 0030 */   IDP4A.U8.U8 R0, R6, R7, R0 ;
/* 0040 */   IDP4A.S8.S8 R1, R8, R9, R1 ;
/* 0050 */   EXIT ;

--- Kernel: ds4_iq2_scalar ---
/* 0100 */   LDG.E.F32 R4, [R2.64] ;
/* 0110 */   SHR.B64 R5, R4, 0x8 ;
/* 0120 */   IMAD.WIDE R0, R5, R6, R0 ;
/* 0130 */   IMAD.WIDE R1, R7, R8, R1 ;
/* 0140 */   EXIT ;

--- Kernel: ds4_iq2_scalar_unpack ---
/* 0150 */   SHF.R.WRAP.B32 R5, R4, 0x8, R5 ;
/* 0160 */   BFE.U32 R6, R4, 0x808 ;
/* 0170 */   IMAD R0, R5, R6, R0 ;
/* 0180 */   EXIT ;

--- Kernel: fma_unfused_fallback ---
/* 0200 */   FMUL R0, R1, R2 ;
/* 0210 */   FADD R3, R0, R4 ;
/* 0220 */   DMUL R6, R7, R8 ;
/* 0230 */   DADD R10, R6, R9 ;
/* 0240 */   EXIT ;

--- Kernel: shuffle_shared_fallback ---
/* 0300 */   STS.128 [R0], R4 ;
/* 0310 */   LDS.128 R8, [R0] ;
/* 0320 */   EXIT ;
`

func TestSummary(t *testing.T) {
	d, err := cudasass.ParseString(testSASSFixture)
	if err != nil {
		t.Fatalf("ParseString: %v", err)
	}

	tests := []struct {
		name       string
		kernel     string
		wantTotal  int
		wantPred   int
		wantCounts map[string]int
		wantErr    bool
	}{
		{
			name:      "simd_kernel",
			kernel:    "ds4_iq2_simd",
			wantTotal: 5,
			wantPred:  1,
			wantCounts: map[string]int{
				"IDP4A":              2,
				"IDP4A.U8.U8":        1,
				"IDP4A.S8.S8":        1,
				"LOP3":               1,
				"LDG.E.CONSTANT.F32": 1,
				"EXIT":               1,
				"IMAD":               0,
			},
		},
		{
			name:      "scalar_kernel",
			kernel:    "ds4_iq2_scalar",
			wantTotal: 5,
			wantPred:  0,
			wantCounts: map[string]int{
				"IMAD":      2,
				"IMAD.WIDE": 2,
				"SHR.B64":   1,
				"IDP4A":     0,
			},
		},
		{
			name:      "scalar_unpack_kernel",
			kernel:    "ds4_iq2_scalar_unpack",
			wantTotal: 4,
			wantPred:  0,
			wantCounts: map[string]int{
				"SHF":  1,
				"BFE":  1,
				"IMAD": 1,
				"EXIT": 1,
			},
		},
		{
			name:      "fma_unfused_kernel",
			kernel:    "fma_unfused_fallback",
			wantTotal: 5,
			wantPred:  0,
			wantCounts: map[string]int{
				"FMUL": 1,
				"FADD": 1,
				"DMUL": 1,
				"DADD": 1,
				"EXIT": 1,
			},
		},
		{
			name:      "shuffle_shared_kernel",
			kernel:    "shuffle_shared_fallback",
			wantTotal: 3,
			wantPred:  0,
			wantCounts: map[string]int{
				"STS":  1,
				"LDS":  1,
				"EXIT": 1,
			},
		},
		{
			name:    "missing_kernel",
			kernel:  "nonexistent",
			wantErr: true,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			sum, err := d.Summary(tc.kernel)
			if (err != nil) != tc.wantErr {
				t.Fatalf("Summary(%q) error = %v, wantErr %v", tc.kernel, err, tc.wantErr)
			}
			if tc.wantErr {
				return
			}
			if sum.Total != tc.wantTotal {
				t.Errorf("Total = %d, want %d", sum.Total, tc.wantTotal)
			}
			if sum.Predicated != tc.wantPred {
				t.Errorf("Predicated = %d, want %d", sum.Predicated, tc.wantPred)
			}
			for op, want := range tc.wantCounts {
				if got := sum.Count(op); got != want {
					t.Errorf("Count(%q) = %d, want %d", op, got, want)
				}
			}
		})
	}
}

func TestAssertOpcodes(t *testing.T) {
	d, err := cudasass.ParseString(testSASSFixture)
	if err != nil {
		t.Fatalf("ParseString: %v", err)
	}

	tests := []struct {
		name      string
		kernel    string
		require   []string
		forbid    []string
		wantErr   bool
		errSubstr string
	}{
		{
			name:    "simd_success",
			kernel:  "ds4_iq2_simd",
			require: []string{"IDP4A", "LOP3.LUT"},
			forbid:  []string{"IMAD", "SHR"},
			wantErr: false,
		},
		{
			name:      "simd_missing_required",
			kernel:    "ds4_iq2_simd",
			require:   []string{"IDP4A", "HMMA"},
			forbid:    nil,
			wantErr:   true,
			errSubstr: "missing required opcode \"HMMA\"",
		},
		{
			name:      "simd_forbidden_present",
			kernel:    "ds4_iq2_simd",
			require:   []string{"IDP4A"},
			forbid:    []string{"EXIT"},
			wantErr:   true,
			errSubstr: "contains forbidden opcode \"EXIT\" (count 1)",
		},
		{
			name:    "case_insensitive",
			kernel:  "ds4_iq2_simd",
			require: []string{"idp4a.u8.u8"},
			forbid:  []string{"imad"},
			wantErr: false,
		},
		{
			name:    "scalar_asserts_imad",
			kernel:  "ds4_iq2_scalar",
			require: []string{"IMAD.WIDE"},
			forbid:  []string{"IDP4A"},
			wantErr: false,
		},
		{
			name:      "scalar_fails_forbidden_imad",
			kernel:    "ds4_iq2_scalar",
			forbid:    []string{"IMAD"},
			wantErr:   true,
			errSubstr: `contains forbidden opcode "IMAD" (count 2)`,
		},
		{
			name:      "scalar_fails_forbidden_imad_wide",
			kernel:    "ds4_iq2_scalar",
			forbid:    []string{"IMAD.WIDE"},
			wantErr:   true,
			errSubstr: `contains forbidden opcode "IMAD.WIDE" (count 2)`,
		},
		{
			name:      "scalar_fails_forbidden_shr",
			kernel:    "ds4_iq2_scalar",
			forbid:    []string{"SHR"},
			wantErr:   true,
			errSubstr: `contains forbidden opcode "SHR" (count 1)`,
		},
		{
			name:      "scalar_fails_forbidden_shf",
			kernel:    "ds4_iq2_scalar_unpack",
			forbid:    []string{"SHF"},
			wantErr:   true,
			errSubstr: `contains forbidden opcode "SHF" (count 1)`,
		},
		{
			name:      "scalar_fails_forbidden_bfe",
			kernel:    "ds4_iq2_scalar_unpack",
			forbid:    []string{"BFE"},
			wantErr:   true,
			errSubstr: `contains forbidden opcode "BFE" (count 1)`,
		},
		{
			name:      "scalar_fails_forbid_simd_fallbacks_slice",
			kernel:    "ds4_iq2_scalar",
			forbid:    cudasass.ForbidSIMDFallbacks,
			wantErr:   true,
			errSubstr: `contains forbidden opcode "IMAD" (count 2)`,
		},
		{
			name:    "simd_passes_forbid_simd_fallbacks_slice",
			kernel:  "ds4_iq2_simd",
			require: []string{"IDP4A"},
			forbid:  cudasass.ForbidSIMDFallbacks,
			wantErr: false,
		},
		{
			name:      "fma_unfused_fails_forbidden_fmul",
			kernel:    "fma_unfused_fallback",
			forbid:    []string{"FMUL"},
			wantErr:   true,
			errSubstr: `contains forbidden opcode "FMUL" (count 1)`,
		},
		{
			name:      "fma_unfused_fails_forbidden_fadd",
			kernel:    "fma_unfused_fallback",
			forbid:    []string{"FADD"},
			wantErr:   true,
			errSubstr: `contains forbidden opcode "FADD" (count 1)`,
		},
		{
			name:      "fma_unfused_fails_forbidden_dmul",
			kernel:    "fma_unfused_fallback",
			forbid:    []string{"DMUL"},
			wantErr:   true,
			errSubstr: `contains forbidden opcode "DMUL" (count 1)`,
		},
		{
			name:      "fma_unfused_fails_forbidden_dadd",
			kernel:    "fma_unfused_fallback",
			forbid:    []string{"DADD"},
			wantErr:   true,
			errSubstr: `contains forbidden opcode "DADD" (count 1)`,
		},
		{
			name:      "fma_unfused_fails_forbid_fma_fallbacks_slice",
			kernel:    "fma_unfused_fallback",
			forbid:    cudasass.ForbidFMAFallbacks,
			wantErr:   true,
			errSubstr: `contains forbidden opcode "FMUL" (count 1)`,
		},
		{
			name:      "shuffle_shared_fails_forbidden_lds",
			kernel:    "shuffle_shared_fallback",
			forbid:    []string{"LDS"},
			wantErr:   true,
			errSubstr: `contains forbidden opcode "LDS" (count 1)`,
		},
		{
			name:      "shuffle_shared_fails_forbidden_sts",
			kernel:    "shuffle_shared_fallback",
			forbid:    []string{"STS"},
			wantErr:   true,
			errSubstr: `contains forbidden opcode "STS" (count 1)`,
		},
		{
			name:      "shuffle_shared_fails_forbid_warp_shuffle_fallbacks_slice",
			kernel:    "shuffle_shared_fallback",
			forbid:    cudasass.ForbidWarpShuffleFallbacks,
			wantErr:   true,
			errSubstr: `contains forbidden opcode "LDS" (count 1)`,
		},
		{
			name:      "kernel_not_found",
			kernel:    "unknown_kernel",
			require:   []string{"IDP4A"},
			wantErr:   true,
			errSubstr: "kernel \"unknown_kernel\" not found",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			err := d.AssertOpcodes(tc.kernel, tc.require, tc.forbid)
			if (err != nil) != tc.wantErr {
				t.Fatalf("AssertOpcodes() error = %v, wantErr %v", err, tc.wantErr)
			}
			if tc.wantErr && tc.errSubstr != "" {
				if !strings.Contains(err.Error(), tc.errSubstr) {
					t.Errorf("AssertOpcodes() error = %q, want substr %q", err.Error(), tc.errSubstr)
				}
			}
		})
	}
}
