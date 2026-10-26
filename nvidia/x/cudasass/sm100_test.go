package cudasass

import (
	"strings"
	"testing"
)

func TestSM100ClassifyInst(t *testing.T) {
	tests := []struct {
		name string
		inst Instruction
		want SM100InstInfo
		ok   bool
	}{
		{
			name: "utmmma_f16_cta1_smem",
			inst: Instruction{
				Opcode:   "UTMMA.F16.CTA_GROUP_1",
				Operands: "[TMEM:0x0], [SMEM:0x100], [SMEM:0x200]",
			},
			want: SM100InstInfo{
				Kind:        SM100UTMMA,
				Mnemonic:    "UTMMA.F16.CTA_GROUP_1",
				CTAGroup:    1,
				BlockScaled: false,
				TMEMSourceA: false,
				DataType:    "F16",
			},
			ok: true,
		},
		{
			name: "utmmma_bf16_cta2_tmem_a",
			inst: Instruction{
				Opcode:   "UTMMA.BF16.CTA_GROUP_2",
				Operands: "[TMEM:0x0], [TMEM:0x80], [SMEM:0x200]",
			},
			want: SM100InstInfo{
				Kind:        SM100UTMMA,
				Mnemonic:    "UTMMA.BF16.CTA_GROUP_2",
				CTAGroup:    2,
				BlockScaled: false,
				TMEMSourceA: true,
				DataType:    "BF16",
			},
			ok: true,
		},
		{
			name: "utmmma_mxf4nvf4_2cta_block_scale",
			inst: Instruction{
				Opcode:   "UTMMA.MXF4NVF4.2CTA.BLOCK_SCALE",
				Operands: "[TMEM:0x0], [TMEM:0x40], [SMEM:0x100], [TMEM:0x180]",
			},
			want: SM100InstInfo{
				Kind:        SM100UTMMA,
				Mnemonic:    "UTMMA.MXF4NVF4.2CTA.BLOCK_SCALE",
				CTAGroup:    2,
				BlockScaled: true,
				TMEMSourceA: true,
				DataType:    "MXF4NVF4",
			},
			ok: true,
		},
		{
			name: "utmmma_mxf8f6f4_default_cta1",
			inst: Instruction{
				Opcode:   "UTMMA.MXF8F6F4",
				Operands: "[TMEM:0x0], UR4, UR6",
			},
			want: SM100InstInfo{
				Kind:        SM100UTMMA,
				Mnemonic:    "UTMMA.MXF8F6F4",
				CTAGroup:    1,
				BlockScaled: true,
				TMEMSourceA: false,
				DataType:    "MXF8F6F4",
			},
			ok: true,
		},
		{
			name: "utmmma_mxf4nvf4_scale_vec_4x_sm103a",
			inst: Instruction{
				Opcode:   "UTMMA.MXF4NVF4.CTA_GROUP_1.BLOCK_SCALE.SCALE_VEC_4X",
				Operands: "[TMEM:0x0], [SMEM:0x100], [SMEM:0x200], [TMEM:0x80]",
			},
			want: SM100InstInfo{
				Kind:        SM100UTMMA,
				Mnemonic:    "UTMMA.MXF4NVF4.CTA_GROUP_1.BLOCK_SCALE.SCALE_VEC_4X",
				CTAGroup:    1,
				BlockScaled: true,
				ScaleVec:    4,
				TMEMSourceA: false,
				DataType:    "MXF4NVF4",
			},
			ok: true,
		},
		{
			name: "utmmma_mxf4nvf4_scale_vec_2x",
			inst: Instruction{
				Opcode:   "UTMMA.MXF4NVF4.CTA_GROUP_1.BLOCK_SCALE.2X",
				Operands: "[TMEM:0x0], [SMEM:0x100], [SMEM:0x200], [TMEM:0x80]",
			},
			want: SM100InstInfo{
				Kind:        SM100UTMMA,
				Mnemonic:    "UTMMA.MXF4NVF4.CTA_GROUP_1.BLOCK_SCALE.2X",
				CTAGroup:    1,
				BlockScaled: true,
				ScaleVec:    2,
				TMEMSourceA: false,
				DataType:    "MXF4NVF4",
			},
			ok: true,
		},
		{
			name: "utmaldg_2d",
			inst: Instruction{
				Opcode:   "UTMALDG.2D.CTA_GROUP_2",
				Operands: "[SMEM:0x0], [UR8]",
			},
			want: SM100InstInfo{
				Kind:     SM100UTMALDG,
				Mnemonic: "UTMALDG.2D.CTA_GROUP_2",
				CTAGroup: 2,
			},
			ok: true,
		},
		{
			name: "utmacp_scales",
			inst: Instruction{
				Opcode:   "UTMACP.CTA_GROUP_1",
				Operands: "[TMEM:0x180], [SMEM:0x300]",
			},
			want: SM100InstInfo{
				Kind:     SM100UTMACP,
				Mnemonic: "UTMACP.CTA_GROUP_1",
				CTAGroup: 1,
			},
			ok: true,
		},
		{
			name: "utmacmdflush",
			inst: Instruction{
				Opcode:   "UTMACMDFLUSH.CTA_GROUP_2",
				Operands: "[SMEM:0x400]",
			},
			want: SM100InstInfo{
				Kind:     SM100UTMAFlush,
				Mnemonic: "UTMACMDFLUSH.CTA_GROUP_2",
				CTAGroup: 2,
			},
			ok: true,
		},
		{
			name: "ldtm_128",
			inst: Instruction{
				Opcode:   "LDTM.128",
				Operands: "R16, [TMEM:0x0]",
			},
			want: SM100InstInfo{
				Kind:     SM100LDTM,
				Mnemonic: "LDTM.128",
				TMEMCols: 128,
			},
			ok: true,
		},
		{
			name: "sttm_64",
			inst: Instruction{
				Opcode:   "STTM.X64",
				Operands: "[TMEM:0x80], R8",
			},
			want: SM100InstInfo{
				Kind:     SM100STTM,
				Mnemonic: "STTM.X64",
				TMEMCols: 64,
			},
			ok: true,
		},
		{
			name: "elect",
			inst: Instruction{
				Opcode:   "ELECT",
				Operands: "P0, URZ, 0xffffffff",
			},
			want: SM100InstInfo{
				Kind:     SM100ELECT,
				Mnemonic: "ELECT",
			},
			ok: true,
		},
		{
			name: "non_sm100_ffma",
			inst: Instruction{
				Opcode:   "FFMA",
				Operands: "R0, R1, R2, R3",
			},
			want: SM100InstInfo{},
			ok:   false,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got, ok := ClassifySM100Inst(tc.inst)
			if ok != tc.ok {
				t.Fatalf("ClassifySM100Inst(%q) ok = %v, want %v", tc.inst.Opcode, ok, tc.ok)
			}
			if got != tc.want {
				t.Errorf("ClassifySM100Inst(%q) = %+v, want %+v", tc.inst.Opcode, got, tc.want)
			}
		})
	}
}

func TestSM100SummarizeAndAuditFunction(t *testing.T) {
	validFn := Function{
		Name: "blackwell_mxf4_gemm",
		Instructions: []Instruction{
			{Address: 0x0000, Opcode: "ELECT", Operands: "UP0, URZ, 0xffffffff"},
			{Address: 0x0010, Predicate: "@UP0", Opcode: "UTMALDG.2D.CTA_GROUP_2", Operands: "[SMEM:0x0], [UR4]"},
			{Address: 0x0020, Predicate: "@UP0", Opcode: "UTMACP.CTA_GROUP_2", Operands: "[TMEM:0x100], [SMEM:0x400]"},
			{Address: 0x0030, Opcode: "STTM.32", Operands: "[TMEM:0x80], R4"},
			{Address: 0x0040, Predicate: "@UP0", Opcode: "UTMMA.MXF4NVF4.CTA_GROUP_2.BLOCK_SCALE", Operands: "[TMEM:0x0], [TMEM:0x80], [SMEM:0x0], [TMEM:0x100]"},
			{Address: 0x0050, Predicate: "@UP0", Opcode: "UTMACMDFLUSH.CTA_GROUP_2", Operands: "[SMEM:0x800]"},
			{Address: 0x0060, Opcode: "LDTM.128", Operands: "R16, [TMEM:0x0]"},
			{Address: 0x0070, Opcode: "EXIT"},
		},
	}

	sum := SummarizeSM100Function(validFn)
	if sum.Total != 7 {
		t.Errorf("Total = %d, want 7", sum.Total)
	}
	if sum.UTMMA != 1 || sum.UTMALDG != 1 || sum.UTMACP != 1 || sum.UTMACMDFLUSH != 1 || sum.LDTM != 1 || sum.STTM != 1 || sum.ELECT != 1 {
		t.Errorf("unexpected counts: %+v", sum)
	}
	if sum.CTAGroup2 != 1 || sum.CTAGroup1 != 0 {
		t.Errorf("CTA group counts = (1:%d, 2:%d), want (1:0, 2:1)", sum.CTAGroup1, sum.CTAGroup2)
	}
	if sum.BlockScaled != 1 || sum.TMEMSourceA != 1 {
		t.Errorf("BlockScaled=%d TMEMSourceA=%d, want 1, 1", sum.BlockScaled, sum.TMEMSourceA)
	}
	if findings := AuditSM100Function(validFn); len(findings) != 0 {
		t.Errorf("AuditSM100Function(validFn) unexpected findings: %v", findings)
	}

	t.Run("missing_utmacmdflush_and_ldtm", func(t *testing.T) {
		fn := Function{
			Name: "missing_commit",
			Instructions: []Instruction{
				{Address: 0x0000, Opcode: "ELECT", Operands: "P0, RZ, 0xffffffff"},
				{Address: 0x0010, Predicate: "@P0", Opcode: "UTMMA.F16.CTA_GROUP_1", Operands: "[TMEM:0x0], [SMEM:0x100], [SMEM:0x200]"},
			},
		}
		findings := AuditSM100Function(fn)
		joined := strings.Join(findings, "\n")
		if !strings.Contains(joined, "UTMACMDFLUSH") {
			t.Errorf("expected finding for missing UTMACMDFLUSH, got: %v", findings)
		}
		if !strings.Contains(joined, "LDTM") {
			t.Errorf("expected finding for missing LDTM, got: %v", findings)
		}
	})

	t.Run("missing_utmacp_for_block_scaled_utmmma", func(t *testing.T) {
		fn := Function{
			Name: "missing_utmacp",
			Instructions: []Instruction{
				{Address: 0x0000, Opcode: "ELECT", Operands: "UP0, URZ, 0xffffffff"},
				{Address: 0x0010, Predicate: "@UP0", Opcode: "UTMMA.MXF4NVF4.CTA_GROUP_1", Operands: "[TMEM:0x0], [SMEM:0x100], [SMEM:0x200]"},
				{Address: 0x0020, Predicate: "@UP0", Opcode: "UTMACMDFLUSH.CTA_GROUP_1", Operands: "[SMEM:0x300]"},
				{Address: 0x0030, Opcode: "LDTM.64", Operands: "R8, [TMEM:0x0]"},
			},
		}
		findings := AuditSM100Function(fn)
		joined := strings.Join(findings, "\n")
		if !strings.Contains(joined, "UTMACP") {
			t.Errorf("expected finding for missing UTMACP on block-scaled UTMMA, got: %v", findings)
		}
	})

	t.Run("missing_elect_and_leader_predicate", func(t *testing.T) {
		fn := Function{
			Name: "unpredicated_utmmma",
			Instructions: []Instruction{
				{Address: 0x0000, Opcode: "UTMMA.E4M3.CTA_GROUP_1", Operands: "[TMEM:0x0], [SMEM:0x100], [SMEM:0x200]"},
				{Address: 0x0010, Opcode: "UTMACMDFLUSH.CTA_GROUP_1", Operands: "[SMEM:0x300]"},
				{Address: 0x0020, Opcode: "LDTM.64", Operands: "R8, [TMEM:0x0]"},
			},
		}
		findings := AuditSM100Function(fn)
		joined := strings.Join(findings, "\n")
		if !strings.Contains(joined, "ELECT") {
			t.Errorf("expected finding for missing ELECT/leader predicate, got: %v", findings)
		}
	})
}
