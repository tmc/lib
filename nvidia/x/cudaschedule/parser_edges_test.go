package cudaschedule_test

import (
	"fmt"
	"reflect"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudaptx"
	"github.com/tmc/lib/nvidia/x/cudaschedule"
)

type parserEdgeFixture struct {
	strings.Builder
	want []cudaschedule.Site
}

func (f *parserEdgeFixture) statement(callable, text string, kind cudaschedule.SiteKind) {
	start := f.Len()
	f.WriteString(text)
	f.want = append(f.want, cudaschedule.Site{Ordinal: len(f.want), Callable: callable,
		Kind: kind, Span: cudaschedule.Span{Start: start, End: f.Len()}, Text: text,
		Head: strings.TrimSuffix(strings.Fields(text)[0], ";")})
}

func TestAnalyze_StatementScannerEdges(t *testing.T) {
	for _, tc := range []struct {
		name, prefix, statement string
		kind                    cudaschedule.SiteKind
		repeat, secondCallable  bool
	}{
		{"block comment", "/* membar.gl; */ ", "membar.gl;", cudaschedule.SiteFence, false, false},
		{"multiline block comment", "/* ignored\nmembar.gl;\n*/ ", "membar.gl;", cudaschedule.SiteFence, false, false},
		{"line comment", "// membar.gl;\n", "membar.gl;", cudaschedule.SiteFence, false, false},
		{"comment after neutral statement", ".reg .b32 %r0;\nmov.u32 %r0, 1; // membar.gl;\n", "membar.gl;", cudaschedule.SiteFence, false, false},
		{"quoted pragma", ".pragma \"membar.gl\";\n", "membar.gl;", cudaschedule.SiteFence, false, false},
		{"comment semicolon", ".reg .b32 %r1;\n.reg .b64 %rd1;\n", "atom.global.add.u32 /* ; */ %r1, [%rd1], 1;", cudaschedule.SiteAtomic, false, false},
		{"same line repeated", "", "membar.gl;", cudaschedule.SiteFence, true, false},
		{"repeated callable", "", "membar.gl;", cudaschedule.SiteFence, true, true},
		{"same line valid label", "L:", "membar.gl;", cudaschedule.SiteFence, false, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var fixture parserEdgeFixture
			fixture.WriteString(".version 8.7\n.target sm_90\n.address_size 64\n.visible .entry first(){\n")
			fixture.WriteString(tc.prefix)
			fixture.statement("first", tc.statement, tc.kind)
			if tc.repeat {
				fixture.statement("first", tc.statement, tc.kind)
			}
			fixture.WriteString("\nret;\n}\n")
			if tc.secondCallable {
				fixture.WriteString(".visible .entry second(){\n")
				fixture.statement("second", tc.statement, tc.kind)
				fixture.statement("second", tc.statement, tc.kind)
				fixture.WriteString("\nret;\n}\n")
			}
			source := fixture.String()
			sites, err := cudaschedule.Analyze(source)
			if err != nil {
				t.Fatal(err)
			}
			if len(sites) != len(fixture.want) {
				t.Fatalf("sites = %+v, want %d distinct statements", sites, len(fixture.want))
			}
			for i, want := range fixture.want {
				got := sites[i]
				if got.Ordinal != want.Ordinal || got.Callable != want.Callable || got.Kind != want.Kind || got.Head != want.Head || got.Span != want.Span || got.Text != want.Text || got.Predicate != "" {
					t.Fatalf("site %d = %+v, want %+v", i, got, want)
				}
				if i > 0 && got.Span.Start < sites[i-1].Span.End {
					t.Fatalf("sites share or overlap spans: %+v", sites)
				}
			}
			baseline := parserEdgeInstructions(t, source)
			for seed := uint64(0); seed < 32; seed++ {
				rewrite, err := cudaschedule.Perturb(source, cudaschedule.Options{Seed: seed, Intensity: 2})
				if err != nil {
					t.Fatalf("seed %d: %v", seed, err)
				}
				if rewrite.Report.SitesInjected != len(fixture.want) || len(rewrite.Report.Decisions) != len(fixture.want) {
					t.Fatalf("seed %d: incomplete injection report %+v", seed, rewrite.Report)
				}
				for i, decision := range rewrite.Report.Decisions {
					if decision.Site.Span != fixture.want[i].Span || decision.Site.Callable != fixture.want[i].Callable {
						t.Fatalf("seed %d: decision %d has wrong statement identity: %+v", seed, i, decision)
					}
				}
				if got := parserEdgeInstructions(t, rewrite.PTX); !reflect.DeepEqual(got, baseline) {
					t.Fatalf("seed %d: original PTX instructions changed:\ngot %v\nwant %v\n%s", seed, got, baseline, rewrite.PTX)
				}
			}
		})
	}
}

// Retain callable identity and every instruction's operation, operands and
// predicate. Only the newly injected sleeps may differ after perturbation.
func parserEdgeInstructions(t *testing.T, source string) []string {
	t.Helper()
	module, err := cudaptx.ParseString(source)
	if err != nil {
		t.Fatal(err)
	}
	var instructions []string
	for _, kernel := range module.Kernels {
		for _, inst := range kernel.Instructions {
			if inst.Opcode == "nanosleep.u32" {
				continue
			}
			instructions = append(instructions, fmt.Sprintf("%s|%s|%s|%s", kernel.Name, inst.Opcode, inst.Operands, inst.Predicate))
		}
	}
	return instructions
}
