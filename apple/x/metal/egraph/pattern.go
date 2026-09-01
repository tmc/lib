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
			return nil, nil, fmt.Errorf("egraph: missing operator after '('")
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
			return nil, nil, fmt.Errorf("egraph: unclosed parenthesis in pattern")
		}
		tokens = tokens[1:] // Consume ')'
		return Pat(op, children...), tokens, nil
	}

	if strings.HasPrefix(head, "?") {
		return Var(head), tokens, nil
	}

	return &Pattern{
		Op:    head,
		IsVar: false,
	}, tokens, nil
}

// Match captures the binding of a pattern to a target E-Class.
type Match struct {
	ClassID Id
	Subst   Subst
}

// Search scans all e-classes in the E-Graph for subtrees matching this pattern.
func (p *Pattern) Search(g *EGraph) []Match {
	var matches []Match
	for _, class := range g.Classes() {
		substs := p.matchClass(g, class.ID, make(Subst))
		for _, s := range substs {
			matches = append(matches, Match{
				ClassID: class.ID,
				Subst:   s,
			})
		}
	}
	return matches
}

func (p *Pattern) matchClass(g *EGraph, classID Id, subst Subst) []Subst {
	canonicalID := g.Find(classID)

	if p.IsVar {
		if boundID, ok := subst[p.Op]; ok {
			if g.Find(boundID) == canonicalID {
				return []Subst{subst}
			}
			return nil
		}
		newSubst := subst.Clone()
		newSubst[p.Op] = canonicalID
		return []Subst{newSubst}
	}

	class, ok := g.classes[canonicalID]
	if !ok {
		return nil
	}

	var results []Subst
	for _, node := range class.Nodes {
		if node.Op != p.Op || len(node.Children) != len(p.Children) {
			continue
		}

		currSubsts := []Subst{subst.Clone()}
		for i, childPat := range p.Children {
			var nextSubsts []Subst
			childID := node.Children[i]
			for _, s := range currSubsts {
				matched := childPat.matchClass(g, childID, s)
				nextSubsts = append(nextSubsts, matched...)
			}
			currSubsts = nextSubsts
			if len(currSubsts) == 0 {
				break
			}
		}
		results = append(results, currSubsts...)
	}

	return results
}

// Instantiate creates corresponding concrete nodes in the E-Graph using bindings in subst.
func (p *Pattern) Instantiate(g *EGraph, subst Subst) Id {
	if p.IsVar {
		if id, ok := subst[p.Op]; ok {
			return g.Find(id)
		}
		panic(fmt.Sprintf("egraph: unbound pattern variable %s during instantiation", p.Op))
	}

	childIDs := make([]Id, len(p.Children))
	for i, c := range p.Children {
		childIDs[i] = c.Instantiate(g, subst)
	}

	return g.Add(ENode{
		Op:       p.Op,
		Children: childIDs,
	})
}
