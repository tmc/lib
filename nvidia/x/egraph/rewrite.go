package egraph

import (
	"fmt"
	"sort"
	"strings"
)

// Rewrite defines an equivalence rewrite rule from a left-hand pattern to a right-hand pattern.
type Rewrite struct {
	Name string
	LHS  *Pattern
	RHS  *Pattern
}

// Identity identifies the complete rule definition, including both patterns.
// Names alone are intentionally insufficient for replay across rule sets.
func (r *Rewrite) Identity() string {
	if r == nil {
		return "<nil>"
	}
	return r.Name + "|" + r.LHS.String() + "=>" + r.RHS.String()
}

// NewRewrite parses left-hand and right-hand pattern strings and creates a Rewrite rule.
func NewRewrite(name, lhsStr, rhsStr string) (*Rewrite, error) {
	lhs, err := ParsePattern(lhsStr)
	if err != nil {
		return nil, fmt.Errorf("egraph: parse lhs %q: %w", lhsStr, err)
	}
	rhs, err := ParsePattern(rhsStr)
	if err != nil {
		return nil, fmt.Errorf("egraph: parse rhs %q: %w", rhsStr, err)
	}
	lhsVars := make(map[string]bool)
	collectVariables(lhs, lhsVars)
	rhsVars := make(map[string]bool)
	collectVariables(rhs, rhsVars)
	var unbound []string
	for name := range rhsVars {
		if !lhsVars[name] {
			unbound = append(unbound, name)
		}
	}
	if len(unbound) > 0 {
		sort.Strings(unbound)
		return nil, fmt.Errorf("egraph: rhs variables not bound by lhs: %s", strings.Join(unbound, ", "))
	}
	return &Rewrite{
		Name: name,
		LHS:  lhs,
		RHS:  rhs,
	}, nil
}

func collectVariables(p *Pattern, variables map[string]bool) {
	if p.IsVar {
		variables[p.Op] = true
		return
	}
	for _, child := range p.Children {
		collectVariables(child, variables)
	}
}

// MustRewrite creates a Rewrite or panics if parsing fails (useful for table-defined constants).
func MustRewrite(name, lhsStr, rhsStr string) *Rewrite {
	r, err := NewRewrite(name, lhsStr, rhsStr)
	if err != nil {
		panic(err)
	}
	return r
}

// Apply searches the E-Graph for matches of LHS and unifies them with RHS instantiations.
func (r *Rewrite) Apply(g *EGraph) int {
	matches := r.LHS.Search(g)
	// Apply has no typed proof state.  Hardware-sensitive rules must go
	// through SearchState.ApplyAction, where their operand contracts are
	// checked, so the legacy entry point always fails closed.
	if isMandatoryRule(r.Name) || isMandatoryRewrite(r) {
		return 0
	}
	merges := 0

	for _, m := range matches {
		rhsID := r.RHS.Instantiate(g, m.Subst)
		if _, merged := g.Union(m.ClassID, rhsID); merged {
			merges++
		}
	}

	if merges > 0 {
		g.Rebuild()
	}
	return merges
}

func isMandatoryRule(name string) bool {
	switch name {
	case "hoist_q2k_scales_weights", "hoist_iq2xxs_scales_weights", "fuse_rmsnorm_quant_q8", "fuse_rmsnorm_quant_q8_default_eps", "fuse_swiglu_quant_q8_1", "vectorize_moe_gather_w16", "vectorize_moe_gather_w32", "prefill_group_expand_w256":
		return true
	default:
		return false
	}
}

func isMandatoryRewrite(r *Rewrite) bool {
	if r == nil || r.LHS == nil || r.RHS == nil {
		return false
	}
	sig := r.LHS.String() + "=>" + r.RHS.String()
	for _, known := range MandatoryRewriteRules() {
		if known.LHS.String()+"=>"+known.RHS.String() == sig {
			return true
		}
	}
	return false
}

// Saturate repeatedly applies rewrite rules until no more unions occur or maxIterations is reached.
func Saturate(g *EGraph, rules []*Rewrite, maxIterations int) (iterations int, totalMerges int) {
	if maxIterations <= 0 {
		maxIterations = 30
	}

	for iter := 0; iter < maxIterations; iter++ {
		iterations++
		iterMerges := 0
		for _, r := range rules {
			iterMerges += r.Apply(g)
		}
		totalMerges += iterMerges
		if iterMerges == 0 {
			break
		}
	}
	return iterations, totalMerges
}
