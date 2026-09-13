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
	// Authority: docs/ptx/04-syntax.md:78-125 Table 2
	if !cudaptx.IsInstructionKeyword("fma") {
		t.Errorf("expected IsInstructionKeyword('fma') = true")
	}
	if !cudaptx.IsInstructionKeyword("dp4a") {
		t.Errorf("expected IsInstructionKeyword('dp4a') = true")
	}
	if cudaptx.IsInstructionKeyword("not_a_real_opcode_123") {
		t.Errorf("expected IsInstructionKeyword('not_a_real_opcode_123') = false")
	}
}
