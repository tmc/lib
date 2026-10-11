package egraph

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"fmt"
	"math/rand"
	"reflect"

	"github.com/tmc/lib/egraphcore"
)

// CandidateSnapshot is a revision-bound view of the selectable nodes reachable
// from one root class. It is independent of cost-based ExtractTopK.
//
// The snapshot is read-only by convention. The graph must not be mutated while
// methods are running; every operation checks the graph revision before use.
type CandidateSnapshot struct {
	g        *EGraph
	root     Id
	revision uint64
	classes  []Id
	nodes    map[Id][]ENode
}

// candidateMaterializer reuses traversal state for candidates from one
// immutable snapshot. Expr values and child slices are still allocated for
// every materialization; only the dense lookup and traversal scratch is reused.
type candidateMaterializer struct {
	snapshot     *CandidateSnapshot
	materializer *egraphcore.ChoiceMaterializer[Id, ENode, *Expr]
	fingerprints *egraphcore.ChoiceFingerprinter[Id, ENode]
}

func (m *candidateMaterializer) reachableClasses(choice CandidateChoice) ([]Id, error) {
	if m == nil || m.snapshot == nil || m.materializer == nil {
		return nil, fmt.Errorf("egraph: nil candidate materializer")
	}
	return m.materializer.ReachableClasses(m.snapshot.root, choice,
		func(class Id) []ENode { return m.snapshot.nodes[class] },
		func(node ENode) []Id { return node.Children })
}

// CandidateChoice selects a node index for any classes materialized from a
// CandidateSnapshot. Indices refer to the deterministic order returned by
// CandidateSnapshot.Alternatives. Extra choices for unreachable classes are
// ignored during materialization.
type CandidateChoice map[Id]int

// NewCandidateSnapshot captures canonical classes reachable from root through
// nodes that survive candidate cleanup. A node survives only when each child
// class has at least one surviving node. Alternatives are sorted
// deterministically.
func NewCandidateSnapshot(g *EGraph, root Id) (*CandidateSnapshot, error) {
	if g == nil {
		return nil, fmt.Errorf("egraph: nil graph")
	}
	if err := g.RebuildChecked(); err != nil {
		return nil, fmt.Errorf("egraph: rebuild before candidate snapshot: %w", err)
	}
	canonicalRoot, err := g.FindChecked(root)
	if err != nil {
		return nil, err
	}
	s := &CandidateSnapshot{g: g, root: canonicalRoot, revision: g.revision, nodes: make(map[Id][]ENode)}
	space, err := egraphcore.NewChoiceSpaceFrom(canonicalRoot,
		g.FindChecked,
		func(id Id) ([]ENode, error) {
			class := g.store.Class(id)
			if class == nil {
				return nil, fmt.Errorf("egraph: missing candidate class %d", id)
			}
			return class.Nodes, nil
		},
		func(node ENode) (ENode, error) {
			copy := node
			copy.Children = make([]Id, len(node.Children))
			for i, child := range node.Children {
				canonicalChild, findErr := g.FindChecked(child)
				if findErr != nil || g.store.Class(canonicalChild) == nil {
					return ENode{}, fmt.Errorf("egraph: missing candidate child class %d", child)
				}
				copy.Children[i] = canonicalChild
			}
			return copy, nil
		},
		func(node ENode) []Id { return node.Children },
		func(node ENode) bool { return !node.Subsumed },
		func(node ENode) ENode { node.Children = append([]Id(nil), node.Children...); return node },
		func(left, right ENode) bool { return candidateNodeOrder(left) < candidateNodeOrder(right) },
	)
	if err != nil {
		return nil, fmt.Errorf("egraph: build candidate choice space: %w", err)
	}
	s.nodes = make(map[Id][]ENode)
	s.classes = space.Classes()
	for _, id := range s.classes {
		s.nodes[id] = space.Alternatives(id)
	}
	return s, nil
}

func candidateNodeOrder(node ENode) string {
	// Include payload formatting as Key does. Equal textual keys retain source
	// order, which is stable for an unchanged graph revision.
	return node.Key()
}

// Root returns the canonical root class ID captured by the snapshot.
func (s *CandidateSnapshot) Root() Id {
	if s == nil {
		return InvalidId
	}
	return s.root
}

// Revision returns the graph revision captured by the snapshot.
func (s *CandidateSnapshot) Revision() uint64 {
	if s == nil {
		return 0
	}
	return s.revision
}

// Classes returns reachable canonical class IDs in ascending order.
func (s *CandidateSnapshot) Classes() []Id {
	if s == nil {
		return nil
	}
	return append([]Id(nil), s.classes...)
}

// Alternatives returns a copy of the selectable nodes for class. Subsumed
// nodes are omitted. Empty classes removed by candidate cleanup, unknown
// classes, and unreachable classes have no alternatives.
func (s *CandidateSnapshot) Alternatives(class Id) ([]ENode, error) {
	if err := s.checkCurrent(); err != nil {
		return nil, err
	}
	canonical, err := s.g.FindChecked(class)
	if err != nil {
		return nil, err
	}
	nodes, ok := s.nodes[canonical]
	if !ok {
		return nil, nil
	}
	result := make([]ENode, len(nodes))
	for i, node := range nodes {
		result[i] = node
		result[i].Children = append([]Id(nil), node.Children...)
	}
	return result, nil
}

// DefaultChoice selects the first deterministically ordered alternative in
// every reachable class. Classes with no selectable nodes are omitted; if one
// is reached during materialization, that operation returns an explicit error.
func (s *CandidateSnapshot) DefaultChoice() (CandidateChoice, error) {
	if err := s.checkCurrent(); err != nil {
		return nil, err
	}
	choice := make(CandidateChoice, len(s.classes))
	for _, id := range s.classes {
		if len(s.nodes[id]) != 0 {
			choice[id] = 0
		}
	}
	return choice, nil
}

// RandomChoice independently selects one alternative per reachable class
// using rng. Classes without alternatives are omitted. A nil random source is
// an error; no package-global randomness is used.
func (s *CandidateSnapshot) RandomChoice(rng *rand.Rand) (CandidateChoice, error) {
	choice, err := s.randomChoice(rng)
	if err != nil {
		return nil, err
	}
	return s.RepairCycles(choice, rng)
}

func (s *CandidateSnapshot) randomChoice(rng *rand.Rand) (CandidateChoice, error) {
	if err := s.checkCurrent(); err != nil {
		return nil, err
	}
	if rng == nil {
		return nil, fmt.Errorf("egraph: nil candidate random source")
	}
	choice := make(CandidateChoice, len(s.classes))
	for _, id := range s.classes {
		if n := len(s.nodes[id]); n != 0 {
			choice[id] = rng.Intn(n)
		}
	}
	return choice, nil
}

// RepairCycles returns a copy of choice with reachable selected cycles repaired.
// For each cyclic component, it considers alternatives that reduce the number
// of distinct child classes selected inside that component, choosing a
// maximum-reduction alternative per class and breaking ties with rng.
func (s *CandidateSnapshot) RepairCycles(choice CandidateChoice, rng *rand.Rand) (CandidateChoice, error) {
	if err := s.checkCurrent(); err != nil {
		return nil, err
	}
	if rng == nil {
		return nil, fmt.Errorf("egraph: nil candidate random source")
	}
	repaired, err := egraphcore.RepairChoiceCycles(
		s.root,
		map[Id]int(choice),
		func(class Id) []ENode { return s.nodes[class] },
		func(node ENode) []Id { return node.Children },
		nil,
		rng.Intn,
		128,
	)
	if err != nil {
		return nil, fmt.Errorf("egraph: repair candidate cycles: %w", err)
	}
	return CandidateChoice(repaired), nil
}

// Materialize builds the selected root expression. Repeated references to a
// class share one Expr pointer. Missing or invalid choices and selected cycles
// are errors. Choices for classes not reached by the selected alternatives are
// ignored.
func (s *CandidateSnapshot) Materialize(choice CandidateChoice) (*Expr, error) {
	if s == nil {
		return nil, fmt.Errorf("egraph: nil candidate snapshot")
	}
	if err := s.checkCurrent(); err != nil {
		return nil, err
	}
	expr, err := egraphcore.MaterializeChoice(s.root, map[Id]int(choice),
		func(id Id) []ENode { return s.nodes[id] },
		func(node ENode) []Id { return node.Children },
		func(_ Id, node ENode, children []*Expr) (*Expr, error) {
			return &Expr{Op: node.Op, Sort: node.Sort, Data: node.Data, Children: children}, nil
		})
	if err != nil {
		return nil, fmt.Errorf("egraph: materialize candidate: %w", err)
	}
	if err := s.checkCurrent(); err != nil {
		return nil, err
	}
	return expr, nil
}

func newCandidateMaterializer(snapshot *CandidateSnapshot) (*candidateMaterializer, error) {
	materializer, err := egraphcore.NewChoiceMaterializer[Id, ENode, *Expr](snapshot.classes)
	if err != nil {
		return nil, fmt.Errorf("egraph: create candidate materializer: %w", err)
	}
	return &candidateMaterializer{
		snapshot:     snapshot,
		materializer: materializer,
		fingerprints: egraphcore.NewChoiceFingerprinter[Id, ENode](),
	}, nil
}

// materialize reuses dense traversal scratch while preserving Materialize's
// fresh Expr graph and shared-child semantics.
func (m *candidateMaterializer) materialize(choice CandidateChoice) (*Expr, error) {
	if m == nil || m.snapshot == nil {
		return nil, fmt.Errorf("egraph: nil candidate materializer")
	}
	s := m.snapshot
	if err := s.checkCurrent(); err != nil {
		return nil, err
	}
	expr, err := m.materializer.Materialize(s.root, map[Id]int(choice),
		func(id Id) []ENode { return s.nodes[id] },
		func(node ENode) []Id { return node.Children },
		func(_ Id, node ENode, children []*Expr) (*Expr, error) {
			return &Expr{Op: node.Op, Sort: node.Sort, Data: node.Data, Children: children}, nil
		})
	if err != nil {
		return nil, fmt.Errorf("egraph: materialize candidate: %w", err)
	}
	if err := s.checkCurrent(); err != nil {
		return nil, err
	}
	return expr, nil
}

func (s *CandidateSnapshot) checkCurrent() error {
	if s == nil || s.g == nil {
		return fmt.Errorf("egraph: nil candidate snapshot")
	}
	if s.g.revision != s.revision {
		return fmt.Errorf("egraph: candidate snapshot is stale (revision %d, graph revision %d)", s.revision, s.g.revision)
	}
	return nil
}

// fingerprint computes structural fingerprints directly from selected
// alternatives. Cached entries are valid for the immutable snapshot when the
// class choice and each direct child fingerprint match. Search can reject
// duplicate choices before allocating their Expr graphs.
func (m *candidateMaterializer) fingerprint(choice CandidateChoice) (string, error) {
	if m == nil || m.snapshot == nil {
		return "", fmt.Errorf("egraph: nil candidate materializer")
	}
	s := m.snapshot
	if err := s.checkCurrent(); err != nil {
		return "", err
	}
	sum, err := m.fingerprints.Fingerprint(s.root, map[Id]int(choice),
		func(id Id) []ENode { return s.nodes[id] },
		func(node ENode) []Id { return node.Children },
		func(_ Id, node ENode, childHashes []egraphcore.ChoiceFingerprint) egraphcore.ChoiceFingerprint {
			h := sha256.New()
			var length [8]byte
			writeField := func(value string) {
				binary.LittleEndian.PutUint64(length[:], uint64(len(value)))
				_, _ = h.Write(length[:])
				_, _ = h.Write([]byte(value))
			}
			writeField(node.Op)
			writeField(node.Sort)
			if node.Data == nil {
				writeField("")
			} else {
				if value, ok := egraphcore.ValueFingerprint(node.Data); ok {
					writeField("value:" + value)
				} else {
					writeField(fmt.Sprintf("type:%T", node.Data))
				}
			}
			binary.LittleEndian.PutUint64(length[:], uint64(len(childHashes)))
			_, _ = h.Write(length[:])
			for _, childHash := range childHashes {
				_, _ = h.Write(childHash[:])
			}
			var fingerprint egraphcore.ChoiceFingerprint
			copy(fingerprint[:], h.Sum(nil))
			return fingerprint
		}, nil)
	if err != nil {
		return "", fmt.Errorf("egraph: fingerprint candidate: %w", err)
	}
	if err := s.checkCurrent(); err != nil {
		return "", err
	}
	return hex.EncodeToString(sum[:]), nil
}

// candidateChoiceEqualExpr compares a selected snapshot expression with an
// already materialized expression without allocating another Expr graph.
func candidateChoiceEqualExpr(s *CandidateSnapshot, choice CandidateChoice, expr *Expr) bool {
	return egraphcore.SelectedChoiceMatches(
		s.root,
		choice,
		func(class Id) []ENode { return s.nodes[class] },
		func(node ENode) []Id { return node.Children },
		expr,
		func(expr *Expr) []*Expr {
			if expr == nil {
				return nil
			}
			return expr.Children
		},
		func(_ Id, node ENode, expr *Expr) bool {
			return expr != nil && expr.Op == node.Op && expr.Sort == node.Sort && reflect.DeepEqual(expr.Data, node.Data)
		},
	)
}
