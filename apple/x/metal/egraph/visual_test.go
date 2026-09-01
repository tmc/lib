package egraph

import (
	"strings"
	"testing"
)

func TestVisualExports(t *testing.T) {
	g, root := visualTestGraph(t)
	g.Rebuild()

	opts := GraphvizOptions{
		HighlightExtraction: true,
		ShowCost:            true,
		ShowHardwareTarget:  true,
	}
	for _, test := range []struct {
		name string
		got  string
		want []string
	}{
		{
			name: "dot",
			got:  g.ToDOT(opts),
			want: []string{
				"digraph egraph {",
				"cluster_eclass_",
				"fused_rmsnorm_gemv\\nheuristic cost:",
				"gemv\\nheuristic cost:",
				"fillcolor=\"#3fb950\"",
				"-> eclass_",
			},
		},
		{
			name: "mermaid",
			got:  g.ToMermaid(opts),
			want: []string{
				"flowchart TD",
				"subgraph cluster_eclass_",
				"fused_rmsnorm_gemv&lt;br/&gt;heuristic cost:",
				"gemv&lt;br/&gt;heuristic cost:",
				"classDef apple_ane",
				"--> eclass_",
			},
		},
	} {
		t.Run(test.name, func(t *testing.T) {
			for _, want := range test.want {
				if !strings.Contains(test.got, want) {
					t.Errorf("export does not contain %q:\n%s", want, test.got)
				}
			}
		})
	}

	if got := g.ToDOT(opts); got != g.ToDOT(opts) {
		t.Error("DOT export is not deterministic")
	}
	if got := g.ToMermaid(opts); got != g.ToMermaid(opts) {
		t.Error("Mermaid export is not deterministic")
	}

	if root == 0 {
		t.Fatal("root = 0, want non-leaf root")
	}
}

func TestVisualExportsEscapeLabels(t *testing.T) {
	g := New()
	leaf := g.Add(NewLeaf("a[\"b\\c\""))
	g.Add(NewNode("use", leaf))

	if got := g.ToDOT(GraphvizOptions{}); !strings.Contains(got, `a[\"b\\c\"`) {
		t.Errorf("DOT did not escape label: %s", got)
	}
	if got := g.ToMermaid(GraphvizOptions{}); !strings.Contains(got, "a&#91;&#34;b\\c&#34;") {
		t.Errorf("Mermaid did not escape label: %s", got)
	}
}

func visualTestGraph(t *testing.T) (*EGraph, Id) {
	t.Helper()
	g := New()
	x := g.Add(NewLeaf("x"))
	wNorm := g.Add(NewLeaf("w_norm"))
	eps := g.Add(NewLeaf("eps"))
	wProj := g.Add(NewLeaf("w_proj"))
	norm := g.Add(NewNode("rmsnorm", x, wNorm, eps))
	root := g.Add(NewNode("gemv", norm, wProj))

	rule := MustRewrite("fuse_rmsnorm_gemv",
		"(gemv (rmsnorm ?x ?w_norm ?eps) ?w_proj)",
		"(fused_rmsnorm_gemv ?x ?w_norm ?eps ?w_proj)")
	Saturate(g, []*Rewrite{rule}, 10)
	return g, root
}
