package cudaptx_test

import (
	"reflect"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudaptx"
)

func TestTargetDeclarationsPublicParsers(t *testing.T) {
	parsers := []struct {
		name  string
		parse func(string) (*cudaptx.Module, error)
	}{
		{"string", cudaptx.ParseString},
		{"bytes", func(s string) (*cudaptx.Module, error) { return cudaptx.ParseBytes([]byte(s)) }},
		{"reader", func(s string) (*cudaptx.Module, error) { return cudaptx.Parse(strings.NewReader(s)) }},
	}
	for _, tc := range []struct {
		name, text string
		want       []cudaptx.TargetDeclaration
	}{
		{"absent", "// .target sm_70\n.pragma \".target sm_80; false:\";\n", nil},
		{"comments", "// .target sm_70\n/* .target sm_80 */\n.version 8.0\n.target sm_90, debug // .target sm_70\n", []cudaptx.TargetDeclaration{{Line: 4, Target: "sm_90, debug"}}},
		{"inline-comment", ".target/* .target sm_70 */sm_90\n", []cudaptx.TargetDeclaration{{Line: 1, Target: "sm_90"}}},
		{"order", ".target sm_70\n.target sm_90a, debug\n", []cudaptx.TargetDeclaration{{Line: 1, Target: "sm_70"}, {Line: 2, Target: "sm_90a, debug"}}},
		{"same-twice", ".target sm_90\n.target sm_90\n", []cudaptx.TargetDeclaration{{Line: 1, Target: "sm_90"}, {Line: 2, Target: "sm_90"}}},
		{"empty", ".target\n", []cudaptx.TargetDeclaration{{Line: 1}}},
		{"raw-alias-suffix", ".target compute_90a, texmode_independent;\n", []cudaptx.TargetDeclaration{{Line: 1, Target: "compute_90a, texmode_independent"}}},
		{"file-literal", ".file 1 \"/.target sm_70;label: // ignored.cu\"\n.target sm_90\n", []cudaptx.TargetDeclaration{{Line: 2, Target: "sm_90"}}},
		{"file-continued", `.file 1 "prefix\
.target sm_70
suffix\" // .target sm_80"
.target sm_90
`, []cudaptx.TargetDeclaration{{Line: 4, Target: "sm_90"}}},
		{"pragma-continued", `.pragma "prefix\
.target sm_70
suffix\"; phantom: .entry fake() {}";
.target sm_90
`, []cudaptx.TargetDeclaration{{Line: 4, Target: "sm_90"}}},
		{"opaque-source", ".nv_intermediate_source_section {\n.source_begin\n.target sm_70\n.source_end\n}\n.target sm_90\n", []cudaptx.TargetDeclaration{{Line: 6, Target: "sm_90"}}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			for _, parser := range parsers {
				t.Run(parser.name, func(t *testing.T) {
					text := tc.text + ".visible .entry kernel() { head: bra done; .pragma \".target sm_80; phantom:\"; done: ret; }\n"
					m, err := parser.parse(text)
					if err != nil {
						t.Fatal(err)
					}
					if !reflect.DeepEqual(m.TargetDeclarations, tc.want) {
						t.Fatalf("declarations=%+v want %+v", m.TargetDeclarations, tc.want)
					}
					last := ""
					if len(tc.want) != 0 {
						last = tc.want[len(tc.want)-1].Target
					}
					if m.Target != last {
						t.Fatalf("legacy Target=%q want %q", m.Target, last)
					}
					k := m.Kernel("kernel")
					if len(m.Kernels) != 1 || k == nil || len(k.Instructions) != 2 || len(k.Labels) != 2 || k.Labels[0].Name != "head" || k.Labels[0].Instruction != 0 || k.Labels[1].Name != "done" || k.Labels[1].Instruction != 1 {
						t.Fatalf("literal/directive framing changed labels: %+v", k)
					}
				})
			}
		})
	}
}

func TestTargetDeclarationsPerParseOwnership(t *testing.T) {
	a, err := cudaptx.ParseString(".target sm_90\n")
	if err != nil {
		t.Fatal(err)
	}
	b, err := cudaptx.ParseString(".target sm_90\n")
	if err != nil {
		t.Fatal(err)
	}
	a.TargetDeclarations[0].Target = "changed"
	if b.TargetDeclarations[0].Target != "sm_90" || a.Target != "sm_90" {
		t.Fatal("declarations alias another parse or the legacy scalar")
	}
}
