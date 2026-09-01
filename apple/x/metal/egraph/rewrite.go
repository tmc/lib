package egraph

import (
	"fmt"
)

// Rewrite defines an equivalence rewrite rule from a left-hand pattern to a right-hand pattern.
type Rewrite struct {
	Name string
	LHS  *Pattern
	RHS  *Pattern
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
	return &Rewrite{
		Name: name,
		LHS:  lhs,
		RHS:  rhs,
	}, nil
}

// MustRewrite creates a Rewrite or panics if parsing fails.
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
