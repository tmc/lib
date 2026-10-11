package egraph

import "fmt"

// NodeDeleteAction removes one constructor node from the e-class matched by
// the rule's first term pattern. The term's child patterns identify the exact
// canonical arguments to delete. Deletion preserves the e-class and any other
// constructor nodes in it.
type NodeDeleteAction struct {
	Term *Pattern
}

// WithNodeDelete adds a one-shot structural node deletion to a rewrite. The
// term must be a constructor pattern whose variables are bound by the LHS.
// When the deletion removes the final node, the e-class remains with its ID
// and sort but has no extractable structural alternatives.
func (r *Rewrite) WithNodeDelete(term *Pattern) error {
	if r == nil {
		return fmt.Errorf("egraph: nil rewrite")
	}
	if err := r.validate(); err != nil {
		return err
	}
	copy := cloneNodeDeletePattern(term)
	r.nodeDeletes = append(r.nodeDeletes, NodeDeleteAction{Term: copy})
	if err := r.validate(); err != nil {
		r.nodeDeletes = r.nodeDeletes[:len(r.nodeDeletes)-1]
		return err
	}
	return nil
}

// NewNodeDeleteRule creates a rule that deletes a matched constructor node
// without merging its e-class with another class.
func NewNodeDeleteRule(name string, lhs, term *Pattern) (*Rewrite, error) {
	r := &Rewrite{Name: name, LHS: lhs}
	r.nodeDeletes = []NodeDeleteAction{{Term: cloneNodeDeletePattern(term)}}
	if err := r.validate(); err != nil {
		return nil, err
	}
	return r, nil
}

func cloneNodeDeletePattern(pattern *Pattern) *Pattern {
	if pattern == nil {
		return nil
	}
	copy := *pattern
	copy.Children = make([]*Pattern, len(pattern.Children))
	for i, child := range pattern.Children {
		copy.Children[i] = cloneNodeDeletePattern(child)
	}
	copy.FunctionColumns = append([]FunctionColumnPattern(nil), pattern.FunctionColumns...)
	copy.RelationColumns = append([]FunctionColumnPattern(nil), pattern.RelationColumns...)
	copy.int64Expr = cloneFunctionActionExprPtr(pattern.int64Expr)
	copy.float64Expr = cloneFunctionActionExprPtr(pattern.float64Expr)
	return &copy
}

func (g *EGraph) deleteNodeForMatch(pattern *Pattern, match Match) (bool, error) {
	classID, err := g.FindChecked(match.ClassID)
	if err != nil {
		return false, err
	}
	class := g.store.Class(classID)
	if class == nil {
		return false, nil
	}
	variables := make(map[string]bool)
	collectVariables(pattern, variables)
	for i, node := range class.Nodes {
		if !pattern.matchesNode(node) {
			continue
		}
		substitutions := []Subst{make(Subst)}
		for column, child := range pattern.Children {
			var next []Subst
			for _, subst := range substitutions {
				next = append(next, child.matchClass(g, node.Children[column], subst)...)
			}
			substitutions = next
			if len(substitutions) == 0 {
				break
			}
		}
		for _, subst := range substitutions {
			matches := true
			for name := range variables {
				want, wantOK := match.Subst[name]
				got, gotOK := subst[name]
				if !wantOK || !gotOK || g.Find(want) != g.Find(got) {
					matches = false
					break
				}
			}
			if !matches {
				continue
			}
			deletedNode := g.canonicalize(node)
			deletedNode.Children = append([]Id(nil), deletedNode.Children...)
			if g.deletedTerms == nil {
				g.deletedTerms = make(map[string][]deletedTerm)
			}
			key := deletedNode.hashKey()
			g.deletedTerms[key] = append(g.deletedTerms[key], deletedTerm{node: deletedNode, classID: classID})
			copy(class.Nodes[i:], class.Nodes[i+1:])
			class.Nodes[len(class.Nodes)-1] = ENode{}
			class.Nodes = class.Nodes[:len(class.Nodes)-1]
			class.indexNodes()
			if g.nodeCount > 0 {
				g.nodeCount--
			}
			g.revision++
			g.dirty = true
			g.recordTermChangeAndParents(classID)
			return true, nil
		}
	}
	// Egglog's deferred delete rule is guarded by the matching view row. If the
	// row is already absent, deleting its one-shot marker has no graph effect.
	return false, nil
}

type deletedTerm struct {
	node    ENode
	classID Id
}

// rebuildDeletedTerms canonicalizes retained constructor results after unions,
// and drops tombstones whose visible node has since been re-added by another
// path. The term table semantics are retained while deleted nodes stay out of
// matching, extraction, and the ordinary node indexes.
func (g *EGraph) rebuildDeletedTerms() {
	if len(g.deletedTerms) == 0 {
		return
	}
	rebuilt := make(map[string][]deletedTerm)
	for _, terms := range g.deletedTerms {
		for _, term := range terms {
			classID := g.Find(term.classID)
			class := g.store.Class(classID)
			if class == nil {
				continue
			}
			node := g.canonicalize(term.node)
			visible := false
			for _, existing := range class.Nodes {
				if g.canonicalize(existing).Equal(node) {
					visible = true
					break
				}
			}
			if visible {
				continue
			}
			node.Children = append([]Id(nil), node.Children...)
			key := node.hashKey()
			rebuilt[key] = append(rebuilt[key], deletedTerm{node: node, classID: classID})
		}
	}
	g.deletedTerms = rebuilt
}

func (g *EGraph) forgetDeletedTerm(node ENode) {
	key := node.hashKey()
	terms := g.deletedTerms[key]
	if len(terms) == 0 {
		return
	}
	kept := terms[:0]
	for _, term := range terms {
		if !g.canonicalize(term.node).Equal(node) {
			kept = append(kept, term)
		}
	}
	if len(kept) == 0 {
		delete(g.deletedTerms, key)
	} else {
		g.deletedTerms[key] = kept
	}
}

func (g *EGraph) restoreDeletedTerm(node ENode) (Id, bool) {
	key := node.hashKey()
	terms := g.deletedTerms[key]
	for _, term := range terms {
		if !g.canonicalize(term.node).Equal(node) {
			continue
		}
		classID := g.Find(term.classID)
		class := g.store.Class(classID)
		if class == nil || class.Sort != node.Sort {
			continue
		}
		class.Nodes = append(class.Nodes, node)
		class.indexNodes()
		g.nodeCount++
		g.revision++
		if g.analysis != nil {
			made := g.makeAnalysisData(node)
			merged, _ := mergeAnalysisValues(g.analysis, class.AnalysisData, class.analysisDataSet, made, made != nil)
			class.AnalysisData = merged
			class.analysisDataSet = class.analysisDataSet || made != nil
		}
		g.store.AddMemoCandidate(key, classID)
		opKey := nodeIndexKey{op: node.Op, arity: len(node.Children)}
		indexed := false
		for _, indexedClass := range g.opIndex[opKey] {
			if g.Find(indexedClass) == classID {
				indexed = true
				break
			}
		}
		if !indexed {
			g.opIndex[opKey] = append(g.opIndex[opKey], classID)
		}
		for _, child := range node.Children {
			g.parentIndex[g.Find(child)] = append(g.parentIndex[g.Find(child)], classID)
		}
		g.recordTermChangeAndParents(classID)
		g.forgetDeletedTerm(node)
		g.dirty = true
		return classID, true
	}
	return InvalidId, false
}
