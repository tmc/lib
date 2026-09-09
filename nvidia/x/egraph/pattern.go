package egraph

import (
	"fmt"
	"strings"
	"unicode"
)

// Subst maps pattern variable names (e.g. "?x") to canonical E-Class IDs.
type Subst map[string]Id

// Clone returns a shallow copy of the substitution map.
func (s Subst) Clone() Subst {
	c := make(Subst, len(s))
	for k, v := range s {
		c[k] = v
	}
	return c
}

// Pattern represents a structural tree pattern containing operators, literals, and variables (?var).
type Pattern struct {
	Op       string
	IsVar    bool
	Children []*Pattern
}

// Var creates a variable pattern node (e.g. "?a", "?x").
func Var(name string) *Pattern {
	if !strings.HasPrefix(name, "?") {
		name = "?" + name
	}
	return &Pattern{
		Op:    name,
		IsVar: true,
	}
}

// Pat creates an operator pattern node with child patterns.
func Pat(op string, children ...*Pattern) *Pattern {
	return &Pattern{
		Op:       op,
		IsVar:    false,
		Children: children,
	}
}

// ParsePattern parses an S-expression pattern string like "(+ ?a ?b)" or "(gemv (rmsnorm ?x ?w) ?wq)".
func ParsePattern(s string) (*Pattern, error) {
	tokens := tokenize(s)
	if len(tokens) == 0 {
		return nil, fmt.Errorf("egraph: empty pattern")
	}
	p, rem, err := parseExpr(tokens)
	if err != nil {
		return nil, err
	}
	if len(rem) > 0 {
		return nil, fmt.Errorf("egraph: unparsed trailing tokens in pattern: %v", rem)
	}
	return p, nil
}

func tokenize(s string) []string {
	var tokens []string
	var curr strings.Builder

	flush := func() {
		if curr.Len() > 0 {
			tokens = append(tokens, curr.String())
			curr.Reset()
		}
	}

	for _, r := range s {
		switch {
		case r == '(' || r == ')' || r == ',':
			flush()
			if r != ',' {
				tokens = append(tokens, string(r))
			}
		case unicode.IsSpace(r):
			flush()
		default:
			curr.WriteRune(r)
		}
	}
	flush()
	return tokens
}

func parseExpr(tokens []string) (*Pattern, []string, error) {
	if len(tokens) == 0 {
		return nil, nil, fmt.Errorf("egraph: unexpected end of tokens")
	}

	head := tokens[0]
	tokens = tokens[1:]

	if head == "(" {
		if len(tokens) == 0 {
			return nil, nil, fmt.Errorf("egraph: unmatched '('")
		}
		op := tokens[0]
		tokens = tokens[1:]

		var children []*Pattern
		for len(tokens) > 0 && tokens[0] != ")" {
			child, rem, err := parseExpr(tokens)
			if err != nil {
				return nil, nil, err
			}
			children = append(children, child)
			tokens = rem
		}
		if len(tokens) == 0 || tokens[0] != ")" {
			return nil, nil, fmt.Errorf("egraph: missing ')' for op %s", op)
		}
		return Pat(op, children...), tokens[1:], nil
	}

	if strings.HasPrefix(head, "?") {
		return Var(head), tokens, nil
	}

	return &Pattern{Op: head}, tokens, nil
}

// MatchMatch represents a matched pattern instance at a specific EClass.
type MatchMatch struct {
	ClassID Id
	Subst   Subst
}

// Search searches for all matches of the pattern across the entire E-Graph.
func (p *Pattern) Search(g *EGraph) []MatchMatch {
	var matches []MatchMatch
	for _, c := range g.Classes() {
		substs := p.MatchEClass(g, c.ID, make(Subst))
		for _, s := range substs {
			matches = append(matches, MatchMatch{
				ClassID: c.ID,
				Subst:   s,
			})
		}
	}
	return matches
}

// MatchEClass checks if this pattern matches against a specific e-class.
func (p *Pattern) MatchEClass(g *EGraph, classID Id, current Subst) []Subst {
	canonicalID := g.Find(classID)

	if p.IsVar {
		if boundID, ok := current[p.Op]; ok {
			if g.Find(boundID) == canonicalID {
				return []Subst{current}
			}
			return nil
		}
		// Bind variable
		s := current.Clone()
		s[p.Op] = canonicalID
		return []Subst{s}
	}

	eclass := g.Class(canonicalID)
	if eclass == nil {
		return nil
	}

	var results []Subst
	for _, node := range eclass.Nodes {
		if node.Op != p.Op || len(node.Children) != len(p.Children) {
			continue
		}

		// Match all child patterns recursively
		currentSubsts := []Subst{current}
		for i, childPat := range p.Children {
			childClassID := node.Children[i]
			var nextSubsts []Subst
			for _, sub := range currentSubsts {
				res := childPat.MatchEClass(g, childClassID, sub)
				nextSubsts = append(nextSubsts, res...)
			}
			currentSubsts = nextSubsts
			if len(currentSubsts) == 0 {
				break
			}
		}
		results = append(results, currentSubsts...)
	}

	return results
}

// Instantiate creates new ENodes in the E-Graph according to this pattern and the given substitution.
func (p *Pattern) Instantiate(g *EGraph, s Subst) Id {
	if p.IsVar {
		if id, ok := s[p.Op]; ok {
			return g.Find(id)
		}
		// Unbound variable becomes a leaf
		return g.Add(NewLeaf(p.Op))
	}

	childIDs := make([]Id, len(p.Children))
	for i, c := range p.Children {
		childIDs[i] = c.Instantiate(g, s)
	}

	return g.Add(NewNode(p.Op, childIDs...))
}

func (p *Pattern) String() string {
	if p.IsVar {
		return p.Op
	}
	if len(p.Children) == 0 {
		return p.Op
	}
	var sb strings.Builder
	sb.WriteString("(")
	sb.WriteString(p.Op)
	for _, c := range p.Children {
		sb.WriteString(" ")
		sb.WriteString(c.String())
	}
	sb.WriteString(")")
	return sb.String()
}
