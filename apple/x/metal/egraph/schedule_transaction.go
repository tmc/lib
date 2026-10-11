package egraph

import "github.com/tmc/lib/egraphcore"

// cloneForScheduleRun returns a graph with independent mutable graph state.
// Analysis values and node payloads are shared under their immutability
// contracts; registered callbacks are shared because they are immutable.
func (g *EGraph) cloneForScheduleRun() *EGraph {
	clone := *g
	clone.activeAnalysisBudget = nil
	clone.store = g.store.Clone(func(class *EClass) *EClass {
		copy := *class
		copy.Nodes = cloneNodes(class.Nodes)
		copy.nodesByOpArity = make(map[nodeIndexKey][]int, len(class.nodesByOpArity))
		for key, indices := range class.nodesByOpArity {
			copy.nodesByOpArity[key] = append([]int(nil), indices...)
		}
		return &copy
	})
	clone.deletedTerms = make(map[string][]deletedTerm, len(g.deletedTerms))
	for key, terms := range g.deletedTerms {
		copy := make([]deletedTerm, len(terms))
		for i, term := range terms {
			copy[i] = deletedTerm{node: cloneNode(term.node), classID: term.classID}
		}
		clone.deletedTerms[key] = copy
	}
	clone.opIndex = make(map[nodeIndexKey][]Id, len(g.opIndex))
	for key, ids := range g.opIndex {
		clone.opIndex[key] = append([]Id(nil), ids...)
	}
	clone.parentIndex = make(map[Id][]Id, len(g.parentIndex))
	for id, parents := range g.parentIndex {
		clone.parentIndex[id] = append([]Id(nil), parents...)
	}
	clone.relations = make(map[string]*relationTable, len(g.relations))
	for name, table := range g.relations {
		copy := *table
		copy.schema.Sorts = append([]string(nil), table.schema.Sorts...)
		copy.schema.Columns = append([]FunctionColumnType(nil), table.schema.Columns...)
		copy.core = table.core.Clone()
		copy.rows = make(map[string][]FunctionValue, len(table.rows))
		for key, row := range table.rows {
			copy.rows[key] = append([]FunctionValue(nil), row...)
		}
		copy.subsumed = cloneSet(table.subsumed)
		copy.index = cloneFunctionValueIndex(table.index)
		clone.relations[name] = &copy
	}
	clone.facts = g.facts.Clone()
	clone.functions = make(map[string]*functionTable, len(g.functions))
	for name, table := range g.functions {
		copy := *table
		copy.schema = cloneFunctionSchema(table.schema)
		copy.rows = table.rows.Clone()
		clone.functions[name] = &copy
	}
	clone.functionMergeOps = make(map[string]FunctionMergeOperator, len(g.functionMergeOps))
	for name, op := range g.functionMergeOps {
		op.Arguments = append([]FunctionColumnType(nil), op.Arguments...)
		clone.functionMergeOps[name] = op
	}
	clone.relationInsertions = cloneStringSlices(g.relationInsertions)
	clone.termChanges = append([]Id(nil), g.termChanges...)
	return &clone
}

func cloneNodes(nodes []ENode) []ENode {
	cloned := make([]ENode, len(nodes))
	for i, node := range nodes {
		cloned[i] = cloneNode(node)
	}
	return cloned
}

func cloneNode(node ENode) ENode {
	node.Children = append([]Id(nil), node.Children...)
	return node
}

func cloneSet[K comparable](set map[K]struct{}) map[K]struct{} {
	clone := make(map[K]struct{}, len(set))
	for key := range set {
		clone[key] = struct{}{}
	}
	return clone
}

func cloneStringSlices(src map[string][]string) map[string][]string {
	clone := make(map[string][]string, len(src))
	for key, values := range src {
		clone[key] = append([]string(nil), values...)
	}
	return clone
}

func cloneRowIndex(index rowIndex) rowIndex {
	clone := make(rowIndex, len(index))
	for i, byID := range index {
		clone[i] = make(map[Id]map[string]struct{}, len(byID))
		for id, keys := range byID {
			clone[i][id] = cloneSet(keys)
		}
	}
	return clone
}

func cloneFunctionValueIndex(index functionValueIndex) functionValueIndex {
	clone := make(functionValueIndex, len(index))
	for i, byValue := range index {
		clone[i] = make(map[egraphcore.Value[Id]]map[string]struct{}, len(byValue))
		for value, keys := range byValue {
			clone[i][value] = cloneSet(keys)
		}
	}
	return clone
}

func cloneFunctionSchema(schema FunctionSchema) FunctionSchema {
	schema.KeySorts = append([]string(nil), schema.KeySorts...)
	schema.KeyTypes = append([]FunctionColumnType(nil), schema.KeyTypes...)
	if schema.MergeExpr != nil {
		cloned := cloneFunctionMergeExprs([]FunctionMergeExpr{*schema.MergeExpr})[0]
		schema.MergeExpr = &cloned
	}
	return schema
}
