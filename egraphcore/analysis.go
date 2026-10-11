package egraphcore

import (
	"errors"
	"sort"
)

// ErrInvalidAnalysisPropagation reports a missing callback for analysis
// propagation.
var ErrInvalidAnalysisPropagation = errors.New("egraphcore: invalid analysis propagation")

// AnalysisPropagation provides backend operations used by the shared
// fixed-point worklist. Make reports whether a node contributes a fact;
// Merge combines two present facts and reports whether the value changed.
// Data and SetData read and write a class's accumulated fact and its presence
// independently, allowing an explicit nil fact. VisitParents visits classes
// whose facts may depend on the given class. Spend accounts for one Make or
// Merge callback evaluation.
type AnalysisPropagation[ID ~uint32, Node any] struct {
	Canonical    func(ID) ID
	Exists       func(ID) bool
	Nodes        func(ID) []Node
	Data         func(ID) (any, bool)
	SetData      func(ID, any, bool)
	Make         func(Node) (any, bool, error)
	Merge        func(any, any) (any, bool, error)
	VisitParents func(ID, func(ID))
	Spend        func() error
}

// PropagateAnalysis repeatedly joins node facts into seeded classes and
// revisits dependent parents until the worklist is empty. Backend callbacks
// own graph storage, node interpretation, fact semantics, and work limits.
func PropagateAnalysis[ID ~uint32, Node any](
	seeds []ID,
	ops AnalysisPropagation[ID, Node],
) error {
	if ops.Canonical == nil || ops.Exists == nil || ops.Nodes == nil ||
		ops.Data == nil || ops.SetData == nil || ops.Make == nil ||
		ops.Merge == nil || ops.VisitParents == nil || ops.Spend == nil {
		return ErrInvalidAnalysisPropagation
	}
	queue := make([]ID, 0, len(seeds))
	queued := make(map[ID]bool, len(seeds))
	for _, seed := range seeds {
		id := ops.Canonical(seed)
		if !ops.Exists(id) || queued[id] {
			continue
		}
		queue = append(queue, id)
		queued[id] = true
	}
	for len(queue) > 0 {
		id := queue[0]
		queue[0] = 0
		queue = queue[1:]
		queued[id] = false
		if !ops.Exists(id) {
			continue
		}
		data, hasData := ops.Data(id)
		changed := false
		for _, node := range ops.Nodes(id) {
			if err := ops.Spend(); err != nil {
				return err
			}
			made, madeFact, err := ops.Make(node)
			if err != nil {
				return err
			}
			if !madeFact {
				continue
			}
			if !hasData {
				data, hasData, changed = made, true, true
				continue
			}
			if err := ops.Spend(); err != nil {
				return err
			}
			merged, didChange, err := ops.Merge(data, made)
			if err != nil {
				return err
			}
			if didChange {
				data = merged
				changed = true
			}
		}
		if !changed {
			continue
		}
		ops.SetData(id, data, hasData)
		dependents := make([]ID, 0)
		seen := make(map[ID]bool)
		ops.VisitParents(id, func(parent ID) {
			parent = ops.Canonical(parent)
			if ops.Exists(parent) && !seen[parent] {
				seen[parent] = true
				dependents = append(dependents, parent)
			}
		})
		sort.Slice(dependents, func(i, j int) bool { return dependents[i] < dependents[j] })
		for _, parent := range dependents {
			if !queued[parent] {
				queue = append(queue, parent)
				queued[parent] = true
			}
		}
	}
	return nil
}
