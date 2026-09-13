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
