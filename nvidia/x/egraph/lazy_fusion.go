package egraph

import (
	"fmt"
	"strings"

	"github.com/tmc/lib/nvidia/x/cudafuse"
	"github.com/tmc/lib/nvidia/x/cudair"
)

// ExtractedFusionRegion represents a maximal contiguous subgraph of fusible ops
// extracted from the e-graph, unified into a single mega-kernel.
type ExtractedFusionRegion struct {
	ID       int
	OpNames  []string
	Root     *Expr
	Inputs   []*Expr
	Boundary string
	Kernel   *cudafuse.FusedKernel
}

// ExtractedFusionPlan holds the set of extracted fusion regions partitioned by boundaries.
type ExtractedFusionPlan struct {
	Regions []*ExtractedFusionRegion
	// RootBoundary is the root expression's operation when it is a materializing
	// or compaction boundary. It is empty when the root is not a boundary.
	RootBoundary string
}

// FusionTensor supplies the row-major FP32 shape of an expression through
// Expr.Data. Indexed fusion regions require it on every node, including external
// inputs. A nil Shape explicitly describes a scalar. Permutation is used only
// by transpose and lists each input axis in output-axis order.
type FusionTensor struct {
	Shape       []int
	Permutation []int
}

// LazyFusionRewriteRules returns rewrite rules that identify adjacent fusible elementwise
// operations and unify them into fused mega-kernel candidate nodes in the E-Graph.
func LazyFusionRewriteRules() []*Rewrite {
	return []*Rewrite{
		MustRewrite("fuse_add_mul", "(mul (add ?a ?b) ?c)", "(fused_add_mul ?a ?b ?c)"),
		MustRewrite("fuse_mul_add", "(add (mul ?a ?b) ?c)", "(fused_fma ?a ?b ?c)"),
		MustRewrite("fuse_relu_mul", "(mul (relu ?a) ?b)", "(fused_relu_mul ?a ?b)"),
		MustRewrite("fuse_silu_mul", "(mul (silu ?a) ?b)", "(fused_silu_mul ?a ?b)"),
		MustRewrite("fuse_exp_add", "(add (exp ?a) ?b)", "(fused_exp_add ?a ?b)"),
		MustRewrite("fuse_neg_add", "(add (neg ?a) ?b)", "(fused_neg_add ?a ?b)"),
	}
}

// ExtractFusionRegions discovers maximal contiguous subgraphs of lazily fusible operations
// bounded by materializing ops or conditionally fused compactions, extracting them into
// unified cudafuse fusion regions.
func ExtractFusionRegions(expr *Expr) (*ExtractedFusionPlan, error) {
	if expr == nil {
		return nil, fmt.Errorf("egraph: nil expression")
	}
	if err := validateFusionExpression(expr); err != nil {
		return nil, err
	}
	expr = expandFusionAliases(expr, make(map[*Expr]*Expr))

	plan := &ExtractedFusionPlan{}
	if cudafuse.IsMaterializingBoundary(strings.ToLower(expr.Op)) || cudafuse.IsCompactionBoundary(strings.ToLower(expr.Op)) {
		plan.RootBoundary = strings.ToLower(expr.Op)
	}
	var walkErr error
	regionID := 0
	walked := make(map[*Expr]bool)

	var walk func(e *Expr, parentBoundary string)
	walk = func(e *Expr, parentBoundary string) {
		if e == nil || walkErr != nil || walked[e] {
			return
		}
		walked[e] = true

		op := strings.ToLower(e.Op)
		if cudafuse.IsMaterializingBoundary(op) || cudafuse.IsCompactionBoundary(op) {
			for _, child := range e.Children {
				walk(child, op)
			}
			return
		}

		if cudafuse.IsFusibleElementwise(op) {
			regionID++
			region := &ExtractedFusionRegion{
				ID:       regionID,
				Root:     e,
				Boundary: parentBoundary,
			}

			// Collect all fusible nodes and external inputs within this maximal contiguous subgraph
			var opNames []string
			var inputs []*Expr
			visited := make(map[*Expr]bool)

			var collectSubtree func(cur *Expr)
			collectSubtree = func(cur *Expr) {
				if cur == nil || visited[cur] {
					return
				}
				visited[cur] = true

				curOp := strings.ToLower(cur.Op)
				if cudafuse.IsFusibleElementwise(curOp) {
					// Collect children first (bottom-up)
					for _, ch := range cur.Children {
						collectSubtree(ch)
					}
					opNames = append(opNames, curOp)
				} else {
					// External leaf or input to this fusion region
					inputs = append(inputs, cur)
					// If this input is itself a boundary (e.g. nested filter), continue walk
					if cudafuse.IsMaterializingBoundary(curOp) || cudafuse.IsCompactionBoundary(curOp) {
						walk(cur, curOp)
					}
				}
			}

			collectSubtree(e)
			region.OpNames = opNames
			region.Inputs = inputs

			// Synthesize a cudafuse.FusedKernel from the extracted subgraph
			fk, err := compileExtractedRegion(region)
			if err != nil {
				walkErr = fmt.Errorf("egraph: compile extracted region %d: %w", region.ID, err)
				return
			}
			region.Kernel = fk

			plan.Regions = append(plan.Regions, region)
			return
		}

		for _, child := range e.Children {
			walk(child, parentBoundary)
		}
	}

	walk(expr, "")
	if walkErr != nil {
		return nil, walkErr
	}
	return plan, nil
}

// compileExtractedRegion synthesizes a CUDA C++ kernel using cudafuse for the extracted fusion region.
func compileExtractedRegion(region *ExtractedFusionRegion) (*cudafuse.FusedKernel, error) {
	if region == nil || region.Root == nil {
		return nil, fmt.Errorf("egraph: invalid region")
	}
	for _, op := range region.OpNames {
		if op == "transpose" {
			return compileIndexedFusionRegion(region)
		}
	}

	g := cudair.NewGraph()
	shape := []int{1024}
	nodeMap := make(map[*Expr]*cudair.Node)

	// Add inputs
	for i, in := range region.Inputs {
		name := fmt.Sprintf("in_%d_%s", i, sanitizeIdent(in.Op))
		nodeMap[in] = g.AddInput(name, shape, cudair.Float32)
	}

	var lower func(e *Expr) (*cudair.Node, error)
	lower = func(e *Expr) (*cudair.Node, error) {
		if node, exists := nodeMap[e]; exists {
			return node, nil
		}

		op := strings.ToLower(e.Op)
		switch op {
		case "add", "sub", "mul", "div":
			if len(e.Children) < 2 {
				return nil, fmt.Errorf("egraph: binary op %s has fewer than 2 children", op)
			}
			c0, err := lower(e.Children[0])
			if err != nil {
				return nil, err
			}
			c1, err := lower(e.Children[1])
			if err != nil {
				return nil, err
			}
			res := g.AddBinary(op, c0, c1)
			nodeMap[e] = res
			return res, nil
		case "relu", "silu", "exp", "sqrt", "neg":
			if len(e.Children) < 1 {
				return nil, fmt.Errorf("egraph: unary op %s has no children", op)
			}
			c0, err := lower(e.Children[0])
			if err != nil {
				return nil, err
			}
			res := g.AddUnary(op, c0)
			nodeMap[e] = res
			return res, nil
		case "reshape":
			if len(e.Children) != 1 {
				return nil, fmt.Errorf("egraph: reshape has %d children, want 1", len(e.Children))
			}
			// The scalar kernel is linear-indexed; a reshape is a view change.
			return lower(e.Children[0])
		default:
			if len(e.Children) != 0 {
				return nil, fmt.Errorf("egraph: cannot lower operator %q inside fusion region", e.Op)
			}
			// A leaf with no children is an external tensor input.
			name := fmt.Sprintf("in_leaf_%s", sanitizeIdent(e.Op))
			node := g.AddInput(name, shape, cudair.Float32)
			nodeMap[e] = node
			return node, nil
		}
	}

	rootNode, err := lower(region.Root)
	if err != nil {
		return nil, err
	}

	if err := g.SetOutputs(rootNode); err != nil {
		return nil, err
	}

	kernelName := fmt.Sprintf("fused_egraph_region_%d", region.ID)
	fk, err := cudafuse.FuseWithSourceMap(g, kernelName)
	if err != nil {
		return nil, err
	}
	return &fk, nil
}

func validateFusionExpression(root *Expr) error {
	state := make(map[*Expr]int)
	var visit func(*Expr) error
	visit = func(expr *Expr) error {
		if expr == nil || state[expr] == 1 {
			return fmt.Errorf("egraph: nil or cyclic fusion expression")
		}
		if state[expr] == 2 {
			return nil
		}
		state[expr] = 1
		for _, child := range expr.Children {
			if err := visit(child); err != nil {
				return err
			}
		}
		state[expr] = 2
		return nil
	}
	return visit(root)
}

func expandFusionAliases(expr *Expr, expanded map[*Expr]*Expr) *Expr {
	if expr == nil {
		return nil
	}
	if result := expanded[expr]; result != nil {
		return result
	}
	children := make([]*Expr, len(expr.Children))
	for i, child := range expr.Children {
		children[i] = expandFusionAliases(child, expanded)
	}
	makeExpr := func(op string, children ...*Expr) *Expr {
		return &Expr{Op: op, Data: expr.Data, Children: children}
	}
	var result *Expr
	switch strings.ToLower(expr.Op) {
	case "fused_add_mul":
		if len(children) == 3 {
			result = makeExpr("mul", makeExpr("add", children[0], children[1]), children[2])
		}
	case "fused_fma":
		if len(children) == 3 {
			result = makeExpr("add", makeExpr("mul", children[0], children[1]), children[2])
		}
	case "fused_relu_mul", "fused_silu_mul", "fused_exp_add", "fused_neg_add":
		if len(children) == 2 {
			unary, binary := "", "mul"
			left, right := children[0], children[1]
			switch strings.ToLower(expr.Op) {
			case "fused_relu_mul":
				unary = "relu"
			case "fused_silu_mul":
				unary = "silu"
			case "fused_exp_add":
				unary, binary = "exp", "add"
			case "fused_neg_add":
				unary, binary = "neg", "add"
			}
			result = makeExpr(binary, makeExpr(unary, left), right)
		}
	}
	if result == nil {
		clone := *expr
		clone.Children = children
		result = &clone
	} else {
		result.Class, result.Origin = expr.Class, expr.Origin
	}
	expanded[expr] = result
	return result
}

func sanitizeIdent(s string) string {
	s = strings.ReplaceAll(s, "?", "")
	s = strings.ReplaceAll(s, "-", "_")
	var b strings.Builder
	for _, ch := range s {
		if ch >= 'a' && ch <= 'z' || ch >= 'A' && ch <= 'Z' || ch >= '0' && ch <= '9' || ch == '_' {
			b.WriteRune(ch)
		}
	}
	res := b.String()
	if res == "" {
		return "var"
	}
	return res
}
