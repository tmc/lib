package egraph

import (
	"fmt"
	"html"
	"sort"
	"strings"
)

// GraphvizOptions configures graph visualization exports.
type GraphvizOptions struct {
	// HighlightExtraction marks each e-class's lowest-cost known node.
	HighlightExtraction bool

	// ShowCost adds the fused-megakernel heuristic cost when it is known.
	ShowCost bool

	// ShowHardwareTarget colors nodes according to the placement heuristic.
	ShowHardwareTarget bool
}

type visualNodeID struct {
	class Id
	index int
}

// ToDOT returns g as Graphviz DOT. The result is deterministic for an
// unchanged E-Graph.
func (g *EGraph) ToDOT(opts GraphvizOptions) string {
	var b strings.Builder
	b.WriteString("digraph egraph {\n")
	b.WriteString("\trankdir=LR;\n")
	b.WriteString("\tnode [shape=box];\n")

	classes := g.Classes()
	costs, extracted := g.visualCosts(classes)
	for _, class := range classes {
		fmt.Fprintf(&b, "\tsubgraph cluster_eclass_%d {\n", class.ID)
		fmt.Fprintf(&b, "\t\tlabel=\"e-class %d\";\n", class.ID)
		fmt.Fprintf(&b, "\t\teclass_%d [shape=ellipse, label=\"e-class %d\"];\n", class.ID, class.ID)
		for i, node := range visualNodes(class.Nodes) {
			name := visualNodeName(class.ID, i)
			id := visualNodeID{class: class.ID, index: i}
			cost, known := costs[id]
			label := visualLabel(node, cost, opts.ShowCost && known)
			attrs := []string{fmt.Sprintf("label=\"%s\"", dotEscape(label))}
			if opts.ShowHardwareTarget {
				attrs = append(attrs, "style=filled", "fillcolor=\""+dotColor(visualTarget(node.Op))+"\"")
			}
			if opts.HighlightExtraction && extracted[id] {
				attrs = append(attrs, "color=\"#3fb950\"", "penwidth=3")
			}
			if node.Subsumed {
				attrs = append(attrs, "color=\"#8b949e\"", "style=dashed")
			}
			fmt.Fprintf(&b, "\t\t%s [%s];\n", name, strings.Join(attrs, ", "))
		}
		b.WriteString("\t}\n")
	}

	for _, class := range classes {
		for i, node := range visualNodes(class.Nodes) {
			for _, child := range node.Children {
				fmt.Fprintf(&b, "\t%s -> eclass_%d;\n", visualNodeName(class.ID, i), g.Find(child))
			}
		}
	}
	b.WriteString("}\n")
	return b.String()
}

// ToMermaid returns g as Mermaid flowchart source. The result is deterministic
// for an unchanged E-Graph.
func (g *EGraph) ToMermaid(opts GraphvizOptions) string {
	var b strings.Builder
	b.WriteString("flowchart TD\n")
	if opts.ShowHardwareTarget {
		b.WriteString("    classDef metal_gpu fill:#58a6ff,stroke:#58a6ff,color:#fff\n")
		b.WriteString("    classDef apple_ane fill:#3fb950,stroke:#3fb950,color:#fff\n")
		b.WriteString("    classDef host_cpu fill:#8b949e,stroke:#8b949e,color:#fff\n")
	}

	classes := g.Classes()
	costs, extracted := g.visualCosts(classes)
	for _, class := range classes {
		fmt.Fprintf(&b, "    subgraph cluster_eclass_%d[\"e-class %d\"]\n", class.ID, class.ID)
		fmt.Fprintf(&b, "        eclass_%d((\"e-class %d\"))\n", class.ID, class.ID)
		for i, node := range visualNodes(class.Nodes) {
			name := visualNodeName(class.ID, i)
			id := visualNodeID{class: class.ID, index: i}
			cost, known := costs[id]
			label := mermaidEscape(visualLabel(node, cost, opts.ShowCost && known))
			fmt.Fprintf(&b, "        %s[\"%s\"]\n", name, label)
			if opts.ShowHardwareTarget {
				fmt.Fprintf(&b, "        class %s %s\n", name, visualTarget(node.Op))
			}
			if opts.HighlightExtraction && extracted[id] {
				fmt.Fprintf(&b, "        style %s stroke:#3fb950,stroke-width:3px\n", name)
			}
			if node.Subsumed {
				fmt.Fprintf(&b, "        style %s stroke:#8b949e,stroke-dasharray:5 5\n", name)
			}
		}
		b.WriteString("    end\n")
	}

	for _, class := range classes {
		for i, node := range visualNodes(class.Nodes) {
			for _, child := range node.Children {
				fmt.Fprintf(&b, "    %s --> eclass_%d\n", visualNodeName(class.ID, i), g.Find(child))
			}
		}
	}
	return b.String()
}

func (g *EGraph) visualCosts(classes []*EClass) (map[visualNodeID]float64, map[visualNodeID]bool) {
	ext := NewExtractor(g, FusedMegakernelCost{})
	costs := make(map[visualNodeID]float64)
	extracted := make(map[visualNodeID]bool)
	for _, class := range classes {
		classID := g.Find(class.ID)
		best, hasBest := ext.bestNodes[classID]
		for i, node := range visualNodes(class.Nodes) {
			id := visualNodeID{class: class.ID, index: i}
			if node.Subsumed {
				continue
			}
			childCosts := make([]float64, len(node.Children))
			known := true
			for i, child := range node.Children {
				cost, ok := ext.costs[g.Find(child)]
				if !ok {
					known = false
					break
				}
				childCosts[i] = cost
			}
			if known {
				costs[id] = evaluateCostModel(ext.costModel, node, childCosts)
			}
			if hasBest && node.Equal(best) {
				extracted[id] = true
			}
		}
	}
	return costs, extracted
}

func visualNodes(nodes []ENode) []ENode {
	result := append([]ENode(nil), nodes...)
	sort.Slice(result, func(i, j int) bool {
		return result[i].Key() < result[j].Key()
	})
	return result
}

func visualNodeName(classID Id, index int) string {
	return fmt.Sprintf("eclass_%d_node_%d", classID, index)
}

func visualLabel(node ENode, cost float64, showCost bool) string {
	label := node.Op
	if node.Sort != "" {
		label = node.Sort + ":" + label
	}
	if node.Subsumed {
		label += "\nsubsumed"
	}
	if showCost {
		label += fmt.Sprintf("\nheuristic cost: %.3g", cost)
	}
	return label
}

func visualTarget(op string) TargetDevice {
	switch op {
	case "fused_swiglu_gemv", "fused_rmsnorm_gemv", "fused_rmsnorm_qkv", "fused_swiglu_down":
		return TargetANE
	case "causal_flash_attention", "rope", "add":
		return TargetGPU
	default:
		return TargetCPU
	}
}

func dotColor(target TargetDevice) string {
	switch target {
	case TargetGPU:
		return "#58a6ff"
	case TargetANE:
		return "#3fb950"
	default:
		return "#8b949e"
	}
}

func dotEscape(s string) string {
	s = strings.ReplaceAll(s, "\\", "\\\\")
	s = strings.ReplaceAll(s, "\"", "\\\"")
	s = strings.ReplaceAll(s, "\n", "\\n")
	s = strings.ReplaceAll(s, "\r", "\\r")
	return strings.ReplaceAll(s, "\t", "\\t")
}

func mermaidEscape(s string) string {
	s = html.EscapeString(s)
	s = strings.ReplaceAll(s, "[", "&#91;")
	s = strings.ReplaceAll(s, "]", "&#93;")
	s = strings.ReplaceAll(s, "\r\n", "&lt;br/&gt;")
	s = strings.ReplaceAll(s, "\r", "&lt;br/&gt;")
	s = strings.ReplaceAll(s, "\n", "&lt;br/&gt;")
	return strings.ReplaceAll(s, "\t", "&#9;")
}
