package cudaptx_test

import (
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudaptx"
)

func TestLookupDoc(t *testing.T) {
	tests := []struct {
		query    string
		wantKind cudaptx.DocKind
		wantMin  string
	}{
		{"fma", cudaptx.DocKindInstruction, "1.4"},
		{"FMA", cudaptx.DocKindInstruction, "1.4"},
		{"tcgen05.alloc", cudaptx.DocKindInstruction, "8.6"},
		{"ld", cudaptx.DocKindInstruction, "1.0"},
		{"wgmma.mma_async", cudaptx.DocKindInstruction, "8.0"},
		{"%tid", cudaptx.DocKindRegister, "1.0"},
		{"%tid.x", cudaptx.DocKindRegister, "1.0"},
		{"tid", cudaptx.DocKindRegister, "1.0"},
		{"%laneid", cudaptx.DocKindRegister, "1.3"},
		{"%clock64", cudaptx.DocKindRegister, "2.0"},
		{".maxnreg", cudaptx.DocKindDirective, "1.3"},
		{"maxnreg", cudaptx.DocKindDirective, "1.3"},
		{".target", cudaptx.DocKindDirective, "1.0"},
		{".version", cudaptx.DocKindDirective, "1.0"},
		{".f16x2", cudaptx.DocKindType, ""},
		{"f16x2", cudaptx.DocKindType, ""},
		{".s8x4", cudaptx.DocKindType, ""},
		{".rn", cudaptx.DocKindModifier, ""},
		{".global", cudaptx.DocKindModifier, ""},
		{".cta_group::1", cudaptx.DocKindModifier, ""},
		{".ca", cudaptx.DocKindModifier, ""},
	}

	for _, tt := range tests {
		t.Run(tt.query, func(t *testing.T) {
			doc := cudaptx.LookupDoc(tt.query)
			if doc == nil {
				t.Fatalf("LookupDoc(%q) = nil, want doc", tt.query)
			}
			if doc.Kind != tt.wantKind {
				t.Errorf("doc.Kind = %q, want %q", doc.Kind, tt.wantKind)
			}
			if tt.wantMin != "" && doc.MinPTX != tt.wantMin {
				t.Errorf("doc.MinPTX = %q, want %q", doc.MinPTX, tt.wantMin)
			}
			if doc.Summary == "" {
				t.Errorf("doc.Summary is empty for %q", tt.query)
			}
		})
	}

	if got := cudaptx.LookupDoc("non_existent_xyz"); got != nil {
		t.Errorf("LookupDoc(non_existent_xyz) = %+v, want nil", got)
	}
}

func TestAllDocs(t *testing.T) {
	docs := cudaptx.AllDocs()
	if len(docs) < 400 {
		t.Fatalf("AllDocs() returned %d docs, want >= 400", len(docs))
	}
	for i := 1; i < len(docs); i++ {
		if docs[i].Name < docs[i-1].Name {
			t.Errorf("AllDocs() not sorted: %q < %q", docs[i].Name, docs[i-1].Name)
			break
		}
	}
}

func TestExplain(t *testing.T) {
	t.Run("GuardedFMA", func(t *testing.T) {
		h := cudaptx.Explain("@%p1 fma.rn.ftz.f32 %f1, %f2, %f3, %f4;")
		if h == nil {
			t.Fatal("Explain returned nil")
		}
		if h.Predicate != "@%p1" {
			t.Errorf("Predicate = %q, want @%%p1", h.Predicate)
		}
		if h.Mnemonic != "fma" {
			t.Errorf("Mnemonic = %q, want fma", h.Mnemonic)
		}
		if h.Doc == nil || h.Doc.Name != "fma" {
			t.Errorf("Doc = %+v, want fma", h.Doc)
		}
		if len(h.Modifiers) != 3 {
			t.Fatalf("len(Modifiers) = %d, want 3", len(h.Modifiers))
		}
		if h.Modifiers[0].Name != ".rn" || !strings.Contains(h.Modifiers[0].Meaning, "nearest even") {
			t.Errorf("Modifiers[0] = %+v", h.Modifiers[0])
		}
		if h.Modifiers[1].Name != ".ftz" {
			t.Errorf("Modifiers[1] = %+v", h.Modifiers[1])
		}
		if h.Modifiers[2].Name != ".f32" {
			t.Errorf("Modifiers[2] = %+v", h.Modifiers[2])
		}
		md := h.Markdown()
		if !strings.Contains(md, "### `fma`") || !strings.Contains(md, "Fused multiply-add") {
			t.Errorf("Markdown() missing expected content:\n%s", md)
		}
	})

	t.Run("CompoundTCGen05", func(t *testing.T) {
		h := cudaptx.Explain("tcgen05.alloc.cta_group::1.sync.aligned.b32 [%r1], 4;")
		if h == nil {
			t.Fatal("Explain returned nil")
		}
		if h.Mnemonic != "tcgen05.alloc" {
			t.Errorf("Mnemonic = %q, want tcgen05.alloc", h.Mnemonic)
		}
		if h.MinPTX != "8.6" {
			t.Errorf("MinPTX = %q, want 8.6", h.MinPTX)
		}
		if h.MinTarget != "sm_100a" {
			t.Errorf("MinTarget = %q, want sm_100a", h.MinTarget)
		}
		foundCTAGroup := false
		for _, m := range h.Modifiers {
			if m.Name == ".cta_group::1" {
				foundCTAGroup = true
				if !strings.Contains(m.Meaning, "CTA group size 1") {
					t.Errorf(".cta_group::1 meaning = %q", m.Meaning)
				}
			}
		}
		if !foundCTAGroup {
			t.Error("Modifiers missing .cta_group::1")
		}
	})

	t.Run("SpecialRegisterOperand", func(t *testing.T) {
		h := cudaptx.Explain("mov.u32 %r0, %tid.x;")
		if h == nil {
			t.Fatal("Explain returned nil")
		}
		if len(h.Operands) != 2 {
			t.Fatalf("len(Operands) = %d, want 2", len(h.Operands))
		}
		tidOp := h.Operands[1]
		if tidOp.Doc == nil || tidOp.Doc.Kind != cudaptx.DocKindRegister {
			t.Errorf("Operands[1].Doc = %+v, want register doc", tidOp.Doc)
		}
		if !strings.Contains(tidOp.Summary, "Thread identifier") {
			t.Errorf("Operands[1].Summary = %q", tidOp.Summary)
		}
	})

	t.Run("Directive", func(t *testing.T) {
		h := cudaptx.Explain(".maxnreg 128")
		if h == nil {
			t.Fatal("Explain returned nil")
		}
		if h.Mnemonic != ".maxnreg" {
			t.Errorf("Mnemonic = %q, want .maxnreg", h.Mnemonic)
		}
		if h.Doc == nil || h.Doc.Kind != cudaptx.DocKindDirective {
			t.Errorf("Doc = %+v, want directive doc", h.Doc)
		}
		md := h.Markdown()
		if !strings.Contains(md, "### `.maxnreg`") {
			t.Errorf("Markdown() = %s", md)
		}
	})
}

func BenchmarkLookupDoc(b *testing.B) {
	queries := []string{"fma", "%tid", ".maxnreg", ".f16x2", ".rn"}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = cudaptx.LookupDoc(queries[i%len(queries)])
	}
}

func BenchmarkExplain(b *testing.B) {
	stmts := []string{
		"@%p1 fma.rn.ftz.f32 %f1, %f2, %f3, %f4;",
		"tcgen05.alloc.cta_group::1.sync.aligned.b32 [%r1], 4;",
		"ld.global.ca.u32 %r0, [%r1];",
		"mov.u32 %r0, %tid.x;",
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = cudaptx.Explain(stmts[i%len(stmts)])
	}
}
