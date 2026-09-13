package cudaptx_test

import (
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudaptx"
)

// TestOpcodeSummary audits the real nvcc fixtures. Every expected number below
// was read off the fixture text, not off the parser; see testdata/README.
func TestOpcodeSummary(t *testing.T) {
	tests := []struct {
		name       string
		file       string
		kernel     string
		total      int
		predicated int
		counts     map[string]int
	}{{
		// guarded lowers "if (i < n) out[i] = scale(in[i], k);" to one
		// global load, one inlined fma, one global store, and a single
		// guarded branch "@%p1 bra" (testdata/lexer.sm90.ptx:73).
		name:       "guarded",
		file:       "lexer.sm90.ptx",
		kernel:     "guarded",
		total:      19,
		predicated: 1,
		counts: map[string]int{
			"ld.param":  4,
			"mov":       3,
			"ld.global": 1,
			"st.global": 1,
			"fma":       1,
			"setp":      1,
			"bra":       1,
			"ret":       1,
			"call":      0,
		},
	}, {
		// calls invokes the __noinline__ notInlined, so the lowering keeps
		// a call.uni (testdata/lexer.sm90.ptx:223-234).
		name:   "calls",
		file:   "lexer.sm90.ptx",
		kernel: "calls",
		total:  11,
		counts: map[string]int{
			"call":     1,
			"call.uni": 1,
			"st.param": 1,
			"ld.param": 2,
		},
	}, {
		// vec4 loads and stores through vectors, so the f32 forms never
		// appear: the only global accesses are ld/st.global.v4.f32
		// (testdata/vector.sm90.ptx:45,54).
		name:       "vec4",
		file:       "vector.sm90.ptx",
		kernel:     "vec4",
		total:      21,
		predicated: 1,
		counts: map[string]int{
			"ld.global":        1,
			"ld.global.v4":     1,
			"ld.global.v4.f32": 1,
			"ld.global.f32":    0,
			"st.global.v4":     1,
			"st.global.f32":    0,
			"add.f32":          4,
		},
	}}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			m := parseFixture(t, tt.file)
			s, err := m.Summary(tt.kernel)
			if err != nil {
				t.Fatal(err)
			}
			if s.Total != tt.total {
				t.Errorf("Total = %d, want %d", s.Total, tt.total)
			}
			if s.Predicated != tt.predicated {
				t.Errorf("Predicated = %d, want %d", s.Predicated, tt.predicated)
			}
			for prefix, want := range tt.counts {
				if got := s.Count(prefix); got != want {
					t.Errorf("Count(%q) = %d, want %d", prefix, got, want)
				}
				// Matching is case-insensitive.
				if got := s.Count(strings.ToUpper(prefix)); got != want {
					t.Errorf("Count(%q) = %d, want %d", strings.ToUpper(prefix), got, want)
				}
			}
		})
	}
}

// TestAssertOpcodes checks the audit against the lowering the fixtures record:
// a kernel whose callee was inlined keeps no call, and one whose callee was not
// keeps one.
func TestAssertOpcodes(t *testing.T) {
	tests := []struct {
		name    string
		file    string
		kernel  string
		require []string
		forbid  []string
		wantErr string // substring, empty if the assertion should hold
	}{{
		name:    "InlinedCalleeLeavesNoCall",
		file:    "lexer.sm90.ptx",
		kernel:  "guarded",
		require: []string{"ld.global", "fma"},
		forbid:  []string{"call"},
	}, {
		name:    "NoinlineCalleeKeepsCall",
		file:    "lexer.sm90.ptx",
		kernel:  "calls",
		require: []string{"call"},
	}, {
		name:    "VectorizedAccessOnly",
		file:    "vector.sm90.ptx",
		kernel:  "vec4",
		require: []string{"ld.global.v4"},
		forbid:  []string{"ld.global.f32"},
	}, {
		name:    "MissingRequired",
		file:    "vector.sm90.ptx",
		kernel:  "vec4",
		require: []string{"ld.global.f32"},
		wantErr: `missing required opcode "ld.global.f32"`,
	}, {
		name:    "ForbiddenPresent",
		file:    "lexer.sm90.ptx",
		kernel:  "calls",
		forbid:  []string{"call"},
		wantErr: `contains forbidden opcode "call" (count 1)`,
	}, {
		name:    "ScalarLoadForbiddenOnScalarKernel",
		file:    "lexer.sm90.ptx",
		kernel:  "guarded",
		forbid:  []string{"ld.global.f32"},
		wantErr: `contains forbidden opcode "ld.global.f32" (count 1)`,
	}, {
		name:    "UnfusedAddForbiddenOnVectorKernel",
		file:    "vector.sm90.ptx",
		kernel:  "vec4",
		forbid:  []string{"add.f32"},
		wantErr: `contains forbidden opcode "add.f32" (count 4)`,
	}, {
		name:    "UnknownKernel",
		file:    "vector.sm90.ptx",
		kernel:  "nosuchkernel",
		wantErr: `kernel "nosuchkernel" not found`,
	}}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			m := parseFixture(t, tt.file)
			err := m.AssertOpcodes(tt.kernel, tt.require, tt.forbid)
			switch {
			case tt.wantErr == "" && err != nil:
				t.Errorf("AssertOpcodes() = %v, want nil", err)
			case tt.wantErr != "" && err == nil:
				t.Errorf("AssertOpcodes() = nil, want error containing %q", tt.wantErr)
			case tt.wantErr != "" && !strings.Contains(err.Error(), tt.wantErr):
				t.Errorf("AssertOpcodes() = %v, want error containing %q", err, tt.wantErr)
			}
		})
	}
}

func TestAssertOpcodesForbidFallbacks(t *testing.T) {
	const ptxText = `.version 8.0
.target sm_90
.address_size 64

.visible .entry simd_kernel() {
	dp4a.s32.s32 %r1, %r2, %r3, %r4;
	ret;
}

.visible .entry simd_scalar_fallback() {
	shr.u32        %r5, %r4, 8;
	bfe.u32        %r6, %r4, 8, 8;
	shf.l.wrap.b32 %r7, %r2, %r3, 16;
	mul.lo.s32     %r8, %r5, %r6;
	mul.wide.s32   %rd1, %r5, %r6;
	mad.lo.s32     %r9, %r5, %r6, %r8;
	mad.wide.s32   %rd2, %r5, %r6, %rd1;
	ret;
}

.visible .entry fma_unfused_fallback() {
	mul.f32 %f3, %f1, %f2;
	add.f32 %f4, %f3, %f0;
	ret;
}

.visible .entry shuffle_shared_fallback() {
	st.shared.u32 [%r1], %r2;
	ld.shared.u32 %r3, [%r1];
	ret;
}
`
	m, err := cudaptx.ParseString(ptxText)
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
		// SIMD scalar fallbacks:
		{
			name:    "simd_passes_forbidden_fallbacks_slice",
			kernel:  "simd_kernel",
			require: []string{"dp4a"},
			forbid:  cudaptx.ForbidSIMDFallbacks,
			wantErr: "",
		},
		{
			name:    "simd_fallback_fails_forbidden_mul_lo",
			kernel:  "simd_scalar_fallback",
			forbid:  []string{"mul.lo"},
			wantErr: `contains forbidden opcode "mul.lo" (count 1)`,
		},
		{
			name:    "simd_fallback_fails_forbidden_mul_wide",
			kernel:  "simd_scalar_fallback",
			forbid:  []string{"mul.wide"},
			wantErr: `contains forbidden opcode "mul.wide" (count 1)`,
		},
		{
			name:    "simd_fallback_fails_forbidden_mad_lo",
			kernel:  "simd_scalar_fallback",
			forbid:  []string{"mad.lo"},
			wantErr: `contains forbidden opcode "mad.lo" (count 1)`,
		},
		{
			name:    "simd_fallback_fails_forbidden_mad_wide",
			kernel:  "simd_scalar_fallback",
			forbid:  []string{"mad.wide"},
			wantErr: `contains forbidden opcode "mad.wide" (count 1)`,
		},
		{
			name:    "simd_fallback_fails_forbidden_shr",
			kernel:  "simd_scalar_fallback",
			forbid:  []string{"shr"},
			wantErr: `contains forbidden opcode "shr" (count 1)`,
		},
		{
			name:    "simd_fallback_fails_forbidden_bfe",
			kernel:  "simd_scalar_fallback",
			forbid:  []string{"bfe"},
			wantErr: `contains forbidden opcode "bfe" (count 1)`,
		},
		{
			name:    "simd_fallback_fails_forbidden_shf",
			kernel:  "simd_scalar_fallback",
			forbid:  []string{"shf"},
			wantErr: `contains forbidden opcode "shf" (count 1)`,
		},
		{
			name:    "simd_fallback_fails_forbidden_simd_fallbacks_slice",
			kernel:  "simd_scalar_fallback",
			forbid:  cudaptx.ForbidSIMDFallbacks,
			wantErr: `contains forbidden opcode "mul.lo" (count 1)`,
		},

		// FMA unfused arithmetic fallbacks:
		{
			name:    "fma_unfused_fails_forbidden_mul",
			kernel:  "fma_unfused_fallback",
			forbid:  []string{"mul"},
			wantErr: `contains forbidden opcode "mul" (count 1)`,
		},
		{
			name:    "fma_unfused_fails_forbidden_add",
			kernel:  "fma_unfused_fallback",
			forbid:  []string{"add"},
			wantErr: `contains forbidden opcode "add" (count 1)`,
		},
		{
			name:    "fma_unfused_fails_forbidden_fma_fallbacks_slice",
			kernel:  "fma_unfused_fallback",
			forbid:  cudaptx.ForbidFMAFallbacks,
			wantErr: `contains forbidden opcode "mul" (count 1)`,
		},

		// Warp shuffle shared memory fallbacks:
		{
			name:    "shuffle_shared_fails_forbidden_ld_shared",
			kernel:  "shuffle_shared_fallback",
			forbid:  []string{"ld.shared"},
			wantErr: `contains forbidden opcode "ld.shared" (count 1)`,
		},
		{
			name:    "shuffle_shared_fails_forbidden_st_shared",
			kernel:  "shuffle_shared_fallback",
			forbid:  []string{"st.shared"},
			wantErr: `contains forbidden opcode "st.shared" (count 1)`,
		},
		{
			name:    "shuffle_shared_fails_forbidden_warp_shuffle_fallbacks_slice",
			kernel:  "shuffle_shared_fallback",
			forbid:  cudaptx.ForbidWarpShuffleFallbacks,
			wantErr: `contains forbidden opcode "ld.shared" (count 1)`,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := m.AssertOpcodes(tt.kernel, tt.require, tt.forbid)
			switch {
			case tt.wantErr == "" && err != nil:
				t.Errorf("AssertOpcodes() = %v, want nil", err)
			case tt.wantErr != "" && err == nil:
				t.Errorf("AssertOpcodes() = nil, want error containing %q", tt.wantErr)
			case tt.wantErr != "" && !strings.Contains(err.Error(), tt.wantErr):
				t.Errorf("AssertOpcodes() = %v, want error containing %q", err, tt.wantErr)
			}
		})
	}
}

// TestSummaryZeroValues pins that the API does not panic on absent input, as
// cudasass.Summary does not (docs/ptx/04-syntax.md:39-40 has nothing to say
// about this; it is the package's own contract).
func TestSummaryZeroValues(t *testing.T) {
	var k *cudaptx.Kernel
	if s := k.Summary(); s == nil || s.Total != 0 || s.Count("ret") != 0 {
		t.Errorf("(*Kernel)(nil).Summary() = %+v, want empty summary", s)
	}
	var s *cudaptx.OpcodeSummary
	if got := s.Count("ret"); got != 0 {
		t.Errorf("(*OpcodeSummary)(nil).Count() = %d, want 0", got)
	}
	var m *cudaptx.Module
	if _, err := m.Summary("k"); err == nil {
		t.Error("(*Module)(nil).Summary() = nil error, want error")
	}
}

func TestIsInstructionKeyword(t *testing.T) {
	// Authority: docs/ptx/04-syntax.md:78-125 Table 2, plus 09-instruction-set.md Section 9.7.11 (fabric)
	if !cudaptx.IsInstructionKeyword("fma") {
		t.Errorf("expected IsInstructionKeyword('fma') = true")
	}
	if !cudaptx.IsInstructionKeyword("dp4a") {
		t.Errorf("expected IsInstructionKeyword('dp4a') = true")
	}
	if !cudaptx.IsInstructionKeyword("fabric") {
		t.Errorf("expected IsInstructionKeyword('fabric') = true")
	}
	if !cudaptx.IsInstructionKeyword("tcgen05") {
		t.Errorf("expected IsInstructionKeyword('tcgen05') = true")
	}
	if cudaptx.IsInstructionKeyword("not_a_real_opcode_123") {
		t.Errorf("expected IsInstructionKeyword('not_a_real_opcode_123') = false")
	}
}

func TestInstructionCategory(t *testing.T) {
	tests := []struct {
		mnemonic string
		want     cudaptx.Category
		isCompute bool
		isMemory  bool
		isMatrix  bool
	}{
		{"fma", cudaptx.CatFloatingPoint, true, false, false},
		{"add", cudaptx.CatInteger, true, false, false},
		{"ld", cudaptx.CatDataMovement, false, true, false},
		{"st", cudaptx.CatDataMovement, false, true, false},
		{"fabric", cudaptx.CatFabric, false, true, false},
		{"bra", cudaptx.CatControlFlow, false, false, false},
		{"bar", cudaptx.CatParallelSync, false, false, false},
		{"wmma", cudaptx.CatMatrixWMMA, true, false, true},
		{"wgmma", cudaptx.CatMatrixWGMMA, true, false, true},
		{"tcgen05", cudaptx.CatTensorCore5thGen, true, false, true},
	}

	for _, tt := range tests {
		cat := cudaptx.InstructionCategory(tt.mnemonic)
		if cat != tt.want {
			t.Errorf("InstructionCategory(%q) = %v (%s), want %v", tt.mnemonic, cat, cat, tt.want)
		}
		if got := cat.IsCompute(); got != tt.isCompute {
			t.Errorf("Category(%v).IsCompute() = %t, want %t", cat, got, tt.isCompute)
		}
		if got := cat.IsMemory(); got != tt.isMemory {
			t.Errorf("Category(%v).IsMemory() = %t, want %t", cat, got, tt.isMemory)
		}
		if got := cat.IsMatrix(); got != tt.isMatrix {
			t.Errorf("Category(%v).IsMatrix() = %t, want %t", cat, got, tt.isMatrix)
		}
	}
}

func TestCategoryCount(t *testing.T) {
	m := parseFixture(t, "lexer.sm90.ptx")
	sum, err := m.Summary("guarded")
	if err != nil {
		t.Fatalf("Summary(guarded): %v", err)
	}

	// guarded: 4 ld.param, 3 mov, 1 ld.global, 1 st.global, 1 fma, 1 setp, 1 bra, 1 ret (total 19 - some directives/registers)
	memCount := sum.CategoryCount(cudaptx.CatDataMovement)
	if memCount == 0 {
		t.Errorf("expected CatDataMovement > 0, got %d", memCount)
	}
	fpCount := sum.CategoryCount(cudaptx.CatFloatingPoint)
	if fpCount != 1 {
		t.Errorf("expected 1 CatFloatingPoint (fma), got %d", fpCount)
	}
	cfCount := sum.CategoryCount(cudaptx.CatControlFlow)
	if cfCount != 2 { // bra, ret
		t.Errorf("expected 2 CatControlFlow, got %d", cfCount)
	}
}

func TestAuditArchitecture(t *testing.T) {
	ptxSm90 := `.version 9.0
.target sm_90
.address_size 64
.visible .entry k_sm90() {
	fma.rn.f32 %f1, %f2, %f3, %f4;
	ret;
}`
	m90, err := cudaptx.ParseString(ptxSm90)
	if err != nil {
		t.Fatalf("ParseString: %v", err)
	}
	if err := m90.AuditArchitecture("k_sm90"); err != nil {
		t.Errorf("AuditArchitecture on sm_90 kernel = %v, want nil", err)
	}

	ptxSm100OnSm90 := `.version 9.0
.target sm_90
.address_size 64
.visible .entry k_blackwell() {
	tcgen05.alloc.cta_group::1.sync.aligned.b32 [%r1], 4;
	ret;
}`
	mBad, err := cudaptx.ParseString(ptxSm100OnSm90)
	if err != nil {
		t.Fatalf("ParseString: %v", err)
	}
	if err := mBad.AuditArchitecture("k_blackwell"); err == nil {
		t.Error("expected AuditArchitecture to report error for tcgen05 on sm_90, got nil")
	} else if !strings.Contains(err.Error(), "requires sm_100") {
		t.Errorf("expected error containing 'requires sm_100', got %v", err)
	}
}

func TestInstructionMinPTXAndTarget(t *testing.T) {
	if sm := cudaptx.InstructionMinTarget("tcgen05"); !strings.HasPrefix(sm, "sm_100") {
		t.Errorf("InstructionMinTarget(tcgen05) = %q, want sm_100*", sm)
	}
	if sm := cudaptx.InstructionMinTarget("wgmma"); !strings.HasPrefix(sm, "sm_90") {
		t.Errorf("InstructionMinTarget(wgmma) = %q, want sm_90*", sm)
	}
	if sm := cudaptx.InstructionMinTarget("wmma"); sm != "sm_70" {
		t.Errorf("InstructionMinTarget(wmma) = %q, want sm_70", sm)
	}
	if ptx := cudaptx.InstructionMinPTX("tcgen05"); ptx != "8.6" {
		t.Errorf("InstructionMinPTX(tcgen05) = %q, want 8.6", ptx)
	}
	if ptx := cudaptx.InstructionMinPTX("fabric"); ptx != "9.3" {
		t.Errorf("InstructionMinPTX(fabric) = %q, want 9.3", ptx)
	}
}
