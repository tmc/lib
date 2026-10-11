package egraph

import (
	"fmt"
	"math"
	"reflect"
	"sort"
	"strings"
	"time"

	"github.com/tmc/lib/egraphcore"
)

var _ egraphcore.Graph[Id, ENode] = (*EGraph)(nil)

// Id identifies an equivalence class in an E-Graph.
type Id uint32

// InvalidId is returned by the unchecked convenience methods when an
// operation cannot be completed. Use the Checked variants to inspect errors.
const InvalidId Id = ^Id(0)

// ENode is an operator node within an equivalence class with child e-class IDs.
type ENode struct {
	Op       string
	Sort     string // Optional result sort; nodes in one class must have the same sort.
	Children []Id
	Data     any  // Payloads must not be mutated after the node is added to a graph.
	Subsumed bool // Subsumed nodes remain stored but are hidden from rule matching and extraction.
}

// NewNode creates an operator ENode with child e-class IDs.
func NewNode(op string, children ...Id) ENode {
	return ENode{
		Op:       op,
		Children: children,
	}
}

// NewSortedNode creates an operator node with a result sort and child classes.
func NewSortedNode(sort, op string, children ...Id) ENode {
	return ENode{Op: op, Sort: sort, Children: children}
}

// NewLeaf creates a leaf literal or symbol ENode with zero children.
func NewLeaf(val any) ENode {
	return ENode{
		Op:   fmt.Sprint(val),
		Data: val,
	}
}

// NewSortedLeaf creates a leaf node with a result sort and payload.
func NewSortedLeaf(sort string, val any) ENode {
	return ENode{Op: fmt.Sprint(val), Sort: sort, Data: val}
}

// Key returns a stable textual representation of an ENode with canonical child IDs.
// EGraph uses Equal to confirm identity when formatted data keys collide.
func (n ENode) Key() string {
	var sb strings.Builder
	if n.Sort != "" {
		sb.WriteString(n.Sort)
		sb.WriteByte(':')
	}
	sb.WriteString(n.Op)
	if n.Data != nil {
		sb.WriteByte('{')
		sb.WriteString(fmt.Sprintf("%T:%#v", n.Data, n.Data))
		sb.WriteByte('}')
	}
	if len(n.Children) == 0 {
		return sb.String()
	}
	sb.WriteByte('(')
	for i, c := range n.Children {
		if i > 0 {
			sb.WriteByte(',')
		}
		sb.WriteString(fmt.Sprint(c))
	}
	sb.WriteByte(')')
	return sb.String()
}

// hashKey groups nodes with the same operator, canonical children, and a
// conservative fingerprint of supported payload values. Equal still checks
// the payload because fingerprints can collide. Unsupported payload shapes
// retain the payload-blind key and its equality-checked fallback.
func (n ENode) hashKey() string {
	var sb strings.Builder
	sb.WriteString(n.Sort)
	sb.WriteByte('\x00')
	sb.WriteString(n.Op)
	sb.WriteByte('(')
	for i, child := range n.Children {
		if i > 0 {
			sb.WriteByte(',')
		}
		sb.WriteString(fmt.Sprint(child))
	}
	sb.WriteByte(')')
	if fingerprint, ok := dataFingerprint(n.Data); ok {
		sb.WriteString("\x00data:")
		sb.WriteString(fingerprint)
	}
	return sb.String()
}

func dataFingerprint(data any) (string, bool) {
	return egraphcore.ValueFingerprint(data)
}

func (n ENode) String() string {
	if len(n.Children) == 0 && n.Sort == "" {
		return n.Op
	}
	return n.Key()
}

// Equal returns true if two ENodes have identical operator, data, and children.
func (n ENode) Equal(other ENode) bool {
	if n.Op != other.Op || n.Sort != other.Sort || len(n.Children) != len(other.Children) || !reflect.DeepEqual(n.Data, other.Data) {
		return false
	}
	for i := range n.Children {
		if n.Children[i] != other.Children[i] {
			return false
		}
	}
	return true
}

// EClass represents an equivalence class of equivalent ENodes.
type EClass struct {
	ID               Id
	Sort             string
	Nodes            []ENode
	AnalysisData     any  // Immutable facts supplied by the graph's Analysis, if any.
	HasAnalysisData  bool // Whether AnalysisData is present; it may be nil.
	analysisFacts    any  // Facts explicitly merged by callers, independent of nodes.
	analysisDataSet  bool
	analysisFactsSet bool
	nodesByOpArity   map[nodeIndexKey][]int
}

type nodeIndexKey struct {
	op    string
	arity int
}

// RebuildStats describes the most recent completed or failed dirty rebuild.
type RebuildStats struct {
	Sequence            uint64
	Duration            time.Duration
	Passes              int
	ClassesVisited      int
	NodesVisited        int
	Canonicalizations   int
	ClassMerges         int
	FunctionTablePasses int
	FunctionMergeRounds int
}

// EGraph represents an equivalence graph with congruence closure and hash consing.
type EGraph struct {
	store                *egraphcore.Store[Id, *EClass]
	deletedTerms         map[string][]deletedTerm
	opIndex              map[nodeIndexKey][]Id
	relations            map[string]*relationTable
	facts                *egraphcore.FactStore[Id]
	functions            map[string]*functionTable
	functionMergeOps     map[string]FunctionMergeOperator
	parentIndex          map[Id][]Id
	signatures           map[string][]OperatorSignature
	datatypes            map[string]Datatype
	variantOwners        map[string]string
	declaredSortsStrict  bool
	analysis             Analysis
	analysisWorkLimit    int
	activeAnalysisBudget *analysisWorkBudget
	nodeCount            int
	strictSignatures     bool
	revision             uint64
	rebuildSequence      uint64
	relationLogsEnabled  bool
	relationInsertions   map[string][]string
	termChangesEnabled   bool
	termChanges          []Id
	lastRebuildStats     RebuildStats
	activeRebuildStats   *RebuildStats
	rebuildObserver      func(RebuildStats)
	dirty                bool
	rebuilding           bool
}

const defaultAnalysisWorkLimit = 10_000_000

// New creates an empty E-Graph.
func New() *EGraph {
	return NewWithAnalysis(nil)
}

// NewWithAnalysis creates an empty E-Graph with an optional e-class analysis.
// Analysis methods must satisfy the contracts documented by Analysis.
func NewWithAnalysis(analysis Analysis) *EGraph {
	return &EGraph{
		store:             egraphcore.NewStore[Id, *EClass](),
		deletedTerms:      make(map[string][]deletedTerm),
		opIndex:           make(map[nodeIndexKey][]Id),
		relations:         make(map[string]*relationTable),
		facts:             egraphcore.NewFactStore[Id](),
		functions:         make(map[string]*functionTable),
		functionMergeOps:  make(map[string]FunctionMergeOperator),
		parentIndex:       make(map[Id][]Id),
		signatures:        make(map[string][]OperatorSignature),
		datatypes:         make(map[string]Datatype),
		variantOwners:     make(map[string]string),
		analysis:          analysis,
		analysisWorkLimit: defaultAnalysisWorkLimit,
	}
}

// Find finds the canonical representative Id for an equivalence class with path compression.
// It returns InvalidId for an invalid ID or nil graph.
func (g *EGraph) Find(id Id) Id {
	if g == nil || uint64(id) >= uint64(g.store.Len()) {
		return InvalidId
	}
	return g.store.Find(id)
}

// FindChecked finds the canonical representative for a valid equivalence class ID.
func (g *EGraph) FindChecked(id Id) (Id, error) {
	if g == nil {
		return InvalidId, fmt.Errorf("egraph: nil graph")
	}
	if uint64(id) >= uint64(g.store.Len()) {
		return InvalidId, fmt.Errorf("egraph: invalid class ID %d", id)
	}
	return g.Find(id), nil
}

// Add adds an ENode to the E-Graph and returns its canonical class ID. It
// rebuilds pending unions before adding the node. Data payloads must remain
// immutable while the node is in the graph. It returns InvalidId when the
// node has an empty operator, an invalid child ID, the graph is nil, or the
// graph has exhausted its class ID space, or a registered operator signature
// rejects it. Use AddChecked to inspect errors.
func (g *EGraph) Add(n ENode) Id {
	if g == nil || !g.store.Initialized() || g.opIndex == nil || n.Op == "" {
		return InvalidId
	}
	for _, child := range n.Children {
		if uint64(child) >= uint64(g.store.Len()) {
			return InvalidId
		}
	}
	if g.validateSignature(n) != nil {
		return InvalidId
	}
	return g.add(n)
}

// AddChecked adds an ENode after validating its operator, child IDs, and any
// registered operator signature.
func (g *EGraph) AddChecked(n ENode) (Id, error) {
	if g == nil {
		return InvalidId, fmt.Errorf("egraph: nil graph")
	}
	if !g.store.Initialized() || g.opIndex == nil {
		return InvalidId, fmt.Errorf("egraph: graph is not initialized; use New")
	}
	if n.Op == "" {
		return InvalidId, fmt.Errorf("egraph: empty operator")
	}
	for _, child := range n.Children {
		if _, err := g.FindChecked(child); err != nil {
			return InvalidId, fmt.Errorf("egraph: invalid child ID %d: %w", child, err)
		}
	}
	if err := g.validateSignature(n); err != nil {
		return InvalidId, err
	}
	if g.dirty {
		if err := g.RebuildChecked(); err != nil {
			return InvalidId, fmt.Errorf("egraph: rebuild before add: %w", err)
		}
	}
	id := g.add(n)
	if id == InvalidId {
		return InvalidId, fmt.Errorf("egraph: class ID space exhausted")
	}
	return id, nil
}

func (g *EGraph) add(n ENode) Id {
	if g.dirty {
		if err := g.RebuildChecked(); err != nil {
			return InvalidId
		}
	}
	canonNode := g.canonicalize(n)
	key := canonNode.hashKey()
	classID, found, err := egraphcore.FindMemoCandidateIter(
		func(visit func(Id) bool) { g.store.VisitMemoCandidates(key, visit) },
		canonNode,
		g.Find,
		func(id Id) []ENode {
			if class := g.store.Class(id); class != nil {
				return class.Nodes
			}
			return nil
		},
		g.canonicalize,
		func(a, b ENode) bool { return a.Equal(b) },
	)
	if err != nil {
		return InvalidId
	}
	if found {
		g.forgetDeletedTerm(canonNode)
		return classID
	}
	if classID, restored := g.restoreDeletedTerm(canonNode); restored {
		return classID
	}
	if uint64(g.store.Len()) >= uint64(InvalidId) {
		return InvalidId
	}

	analysisData := g.makeAnalysisData(canonNode)
	newID, ok := g.store.Insert(key, func(newID Id) *EClass {
		return &EClass{
			ID:              newID,
			Sort:            canonNode.Sort,
			Nodes:           []ENode{canonNode},
			AnalysisData:    analysisData,
			analysisDataSet: analysisData != nil,
			nodesByOpArity:  map[nodeIndexKey][]int{{op: canonNode.Op, arity: len(canonNode.Children)}: {0}},
		}
	})
	if !ok {
		return InvalidId
	}
	g.revision++
	g.nodeCount++
	opKey := nodeIndexKey{op: canonNode.Op, arity: len(canonNode.Children)}
	g.opIndex[opKey] = append(g.opIndex[opKey], newID)
	for _, child := range canonNode.Children {
		g.parentIndex[child] = append(g.parentIndex[child], newID)
	}
	g.recordTermChange(newID)

	return newID
}

// Union merges two equivalence classes and returns the new canonical representative Id.
// Call Rebuild after one or more unions before relying on congruence closure
// and hash-consing indexes. Add rebuilds pending unions automatically.
// Union returns InvalidId and false when an ID is invalid or the graph is nil.
// It also refuses to merge classes with different sorts. Use UnionChecked to
// inspect errors, including analysis work-limit errors.
func (g *EGraph) Union(id1, id2 Id) (Id, bool) {
	if g == nil || uint64(id1) >= uint64(g.store.Len()) || uint64(id2) >= uint64(g.store.Len()) {
		return InvalidId, false
	}
	root1, root2 := g.Find(id1), g.Find(id2)
	if root1 == InvalidId || root2 == InvalidId {
		return InvalidId, false
	}
	if root1 == root2 {
		return root1, false
	}
	if g.store.Class(root1) == nil || g.store.Class(root2) == nil || g.store.Class(root1).Sort != g.store.Class(root2).Sort {
		return InvalidId, false
	}
	root, merged, err := g.union(root1, root2)
	if err != nil {
		return InvalidId, false
	}
	return root, merged
}

// UnionChecked merges two valid equivalence classes. Call Rebuild after one or
// more unions before relying on congruence closure and hash-consing indexes.
// Add rebuilds pending unions automatically.
func (g *EGraph) UnionChecked(id1, id2 Id) (Id, bool, error) {
	if g == nil {
		return InvalidId, false, fmt.Errorf("egraph: nil graph")
	}
	root1, root2, status := g.store.ResolveRoots(id1, id2)
	if status == egraphcore.RootPairInvalidID {
		if uint64(id1) >= uint64(g.store.Len()) {
			return InvalidId, false, fmt.Errorf("egraph: invalid class ID %d", id1)
		}
		return InvalidId, false, fmt.Errorf("egraph: invalid class ID %d", id2)
	}
	if status == egraphcore.RootPairSame {
		return root1, false, nil
	}
	class1, class2 := g.store.Class(root1), g.store.Class(root2)
	if status == egraphcore.RootPairMissingClass || class1 == nil || class2 == nil {
		return InvalidId, false, fmt.Errorf("egraph: missing class during union (%d, %d)", root1, root2)
	}
	if class1.Sort != class2.Sort {
		return InvalidId, false, fmt.Errorf("egraph: cannot union sort %q with sort %q", class1.Sort, class2.Sort)
	}
	root, merged, err := g.union(root1, root2)
	if err != nil {
		return InvalidId, false, err
	}
	if root == InvalidId {
		return InvalidId, false, fmt.Errorf("egraph: missing class during union (%d, %d)", root1, root2)
	}
	return root, merged, nil
}

func (g *EGraph) union(root1, root2 Id) (Id, bool, error) {
	class1, class2 := g.store.Class(root1), g.store.Class(root2)
	if class1 == nil || class2 == nil {
		return InvalidId, false, nil
	}
	budget := g.activeAnalysisBudget
	if budget == nil {
		budget = g.newAnalysisWorkBudget()
	}
	var (
		analysisData     any
		analysisDataSet  bool
		analysisFacts    any
		analysisFactsSet bool
	)
	if g.analysis != nil {
		var err error
		analysisData, _, err = mergeAnalysisValuesWithBudget(g.analysis, class1.AnalysisData, class1.analysisDataSet, class2.AnalysisData, class2.analysisDataSet, budget)
		if err != nil {
			return InvalidId, false, err
		}
		analysisDataSet = class1.analysisDataSet || class2.analysisDataSet
		analysisFacts, _, err = mergeAnalysisValuesWithBudget(g.analysis, class1.analysisFacts, class1.analysisFactsSet, class2.analysisFacts, class2.analysisFactsSet, budget)
		if err != nil {
			return InvalidId, false, err
		}
		analysisFactsSet = class1.analysisFactsSet || class2.analysisFactsSet
	}
	g.recordTermChangeAndParents(root1)
	g.recordTermChangeAndParents(root2)
	root, merged := g.store.MergeRootsBySize(root1, root2, func(class1, class2 *EClass) *EClass {
		class1.Nodes = append(class1.Nodes, class2.Nodes...)
		class1.indexNodes()
		if g.analysis != nil {
			class1.AnalysisData = analysisData
			class1.analysisDataSet = analysisDataSet
			class1.analysisFacts = analysisFacts
			class1.analysisFactsSet = analysisFactsSet
		}
		return class1
	})
	if !merged {
		return InvalidId, false, nil
	}
	g.revision++
	g.dirty = true
	if g.activeRebuildStats != nil {
		g.activeRebuildStats.ClassMerges++
	}

	return root, true, nil
}

func (g *EGraph) recordTermChange(id Id) {
	if g != nil && g.termChangesEnabled {
		g.termChanges = append(g.termChanges, id)
	}
}

func (g *EGraph) recordTermChangeAndParents(id Id) {
	if g == nil || !g.termChangesEnabled {
		return
	}
	queue := []Id{id}
	seen := make(map[Id]bool)
	for len(queue) > 0 {
		last := len(queue) - 1
		current := queue[last]
		queue = queue[:last]
		if seen[current] {
			continue
		}
		seen[current] = true
		g.recordTermChange(current)
		root := g.Find(current)
		if root != InvalidId && root != current {
			queue = append(queue, root)
		}
		for _, parent := range g.parentIndex[current] {
			queue = append(queue, parent)
		}
		if root != InvalidId && root != current {
			for _, parent := range g.parentIndex[root] {
				queue = append(queue, parent)
			}
		}
	}
}

func (g *EGraph) canonicalize(n ENode) ENode {
	if g.activeRebuildStats != nil {
		g.activeRebuildStats.Canonicalizations++
	}
	if len(n.Children) == 0 {
		return n
	}
	canon := ENode{
		Op:       n.Op,
		Sort:     n.Sort,
		Data:     n.Data,
		Children: make([]Id, len(n.Children)),
		Subsumed: n.Subsumed,
	}
	for i, c := range n.Children {
		canon.Children[i] = g.Find(c)
	}
	return canon
}

// Rebuild restores the congruence invariants and hash-consing memo table. It
// ignores function-table conflicts; use RebuildChecked to observe them.
func (g *EGraph) Rebuild() {
	_ = g.RebuildChecked()
}

// RebuildChecked restores congruence, table, and hash-consing invariants.
// A function merge conflict leaves the function tables unpublished and the
// graph dirty so callers can inspect or repair the conflicting rows. Analysis
// callback work is bounded by AnalysisWorkLimit; exhausting it returns
// ErrAnalysisWorkLimit and leaves the graph dirty.
func (g *EGraph) RebuildChecked() error {
	if g == nil || !g.store.Initialized() || g.opIndex == nil {
		return nil
	}
	if !g.dirty || g.rebuilding {
		return nil
	}
	started := time.Now()
	stats := RebuildStats{}
	g.rebuildSequence++
	stats.Sequence = g.rebuildSequence
	g.activeRebuildStats = &stats
	g.rebuilding = true
	defer func() {
		stats.Duration = time.Since(started)
		g.lastRebuildStats = stats
		g.activeRebuildStats = nil
		g.rebuilding = false
		if g.rebuildObserver != nil {
			g.rebuildObserver(stats)
		}
	}()
	var report egraphcore.CongruenceReport
	var rebuiltParents map[Id][]Id
	analysisBudget := g.newAnalysisWorkBudget()
	g.activeAnalysisBudget = analysisBudget
	defer func() { g.activeAnalysisBudget = nil }()
	maxPasses := g.store.Len()
	if maxPasses < 1 {
		maxPasses = 1
	}
	functionStages := len(g.functions)
	// Union-triggered restarts reduce the live class count. Function-table
	// canonicalization can also commit staged defaults or fresh nodes and ask
	// for another pass, so retain per-function headroom and one stable pass.
	if functionStages > math.MaxInt-maxPasses {
		maxPasses = math.MaxInt
	} else {
		maxPasses += functionStages
		if maxPasses < math.MaxInt {
			maxPasses++ // Allow function-stage propagation and the final stable pass.
		}
	}
	report, err := egraphcore.DriveCongruence[Id, ENode, string, struct{}](
		g,
		maxPasses,
		func(pass, index int) (struct{}, []egraphcore.OwnedNode[Id, ENode], bool, error) {
			if index != 0 {
				return struct{}{}, nil, false, nil
			}
			// Each outer pass observes one complete snapshot. Congruence unions
			// invalidate it, so the driver restarts from a fresh graph snapshot.
			snapshots := make([]*EClass, 0, g.store.ClassCount())
			for _, class := range g.classesList() {
				snapshots = append(snapshots, &EClass{
					ID:    class.ID,
					Nodes: append([]ENode(nil), class.Nodes...),
				})
			}
			owned := make([]egraphcore.OwnedNode[Id, ENode], 0)
			for _, snapshot := range snapshots {
				stats.ClassesVisited++
				root := g.Find(snapshot.ID)
				if g.store.Class(root) == nil {
					continue
				}
				for _, node := range snapshot.Nodes {
					stats.NodesVisited++
					cn := g.canonicalize(node)
					owned = append(owned, egraphcore.OwnedNode[Id, ENode]{Owner: root, Node: cn})
				}
			}
			return struct{}{}, owned, true, nil
		},
		func(node ENode) string { return node.hashKey() },
		func(left, right ENode) bool { return left.Equal(right) },
		func(left, right ENode) ENode {
			left.Subsumed = left.Subsumed || right.Subsumed
			return left
		},
		func(_ struct{}, byOwner map[Id][]ENode) (bool, error) {
			owners := make([]Id, 0, len(byOwner))
			for id := range byOwner {
				owners = append(owners, id)
			}
			sort.Slice(owners, func(i, j int) bool { return owners[i] < owners[j] })
			for _, id := range owners {
				root := g.Find(id)
				class := g.store.Class(root)
				if class == nil {
					continue
				}
				nodes := byOwner[id]
				sort.SliceStable(nodes, func(i, j int) bool { return nodes[i].Key() < nodes[j].Key() })
				class.Nodes = append(class.Nodes[:0], nodes...)
				class.indexNodes()
			}
			if err := g.rebuildAnalysis(analysisBudget); err != nil {
				return false, err
			}

			// Rebuild graph indexes from the stable canonical nodes.
			g.store.ResetMemo()
			g.opIndex = make(map[nodeIndexKey][]Id)
			rebuiltParents = make(map[Id][]Id)
			g.nodeCount = 0
			for _, class := range g.classesList() {
				root := g.Find(class.ID)
				indexedNodes := make(map[nodeIndexKey]bool)
				for _, node := range class.Nodes {
					g.nodeCount++
					for _, child := range node.Children {
						rebuiltParents[child] = append(rebuiltParents[child], root)
					}
					opKey := nodeIndexKey{op: node.Op, arity: len(node.Children)}
					if !indexedNodes[opKey] {
						g.opIndex[opKey] = append(g.opIndex[opKey], root)
						indexedNodes[opKey] = true
					}
					memoKey := node.hashKey()
					found := false
					g.store.VisitMemoCandidates(memoKey, func(classID Id) bool {
						if g.Find(classID) == root {
							found = true
							return false
						}
						return true
					})
					if !found {
						g.store.AddMemoCandidate(memoKey, root)
					}
				}
			}
			if err := g.canonicalizeRelations(); err != nil {
				return false, err
			}
			stats.FunctionTablePasses++
			mergedFunctions, err := g.canonicalizeFunctions()
			if err != nil {
				return false, err
			}
			if mergedFunctions {
				stats.FunctionMergeRounds++
				return true, nil
			}
			return false, nil
		},
	)
	stats.Passes = report.Passes
	if err != nil {
		return err
	}
	g.parentIndex = rebuiltParents
	g.rebuildDeletedTerms()
	g.dirty = false
	return nil
}

// LastRebuildStats returns counters for the most recent dirty rebuild. It
// returns zero values before the graph has needed a rebuild.
func (g *EGraph) LastRebuildStats() RebuildStats {
	if g == nil {
		return RebuildStats{}
	}
	return g.lastRebuildStats
}

func (c *EClass) indexNodes() {
	c.nodesByOpArity = make(map[nodeIndexKey][]int)
	for i, node := range c.Nodes {
		key := nodeIndexKey{op: node.Op, arity: len(node.Children)}
		c.nodesByOpArity[key] = append(c.nodesByOpArity[key], i)
	}
}

// Classes rebuilds pending unions and returns snapshots of all current
// canonical equivalence classes.
// Node slices and child-ID slices are copied. Node payloads and analysis data
// are shared and must remain immutable while the graph uses them.
func (g *EGraph) Classes() []*EClass {
	classes, err := g.ClassesChecked()
	if err != nil {
		return nil
	}
	return classes
}

// ClassesChecked rebuilds pending changes and returns snapshots of the
// canonical classes, reporting function-table conflicts.
func (g *EGraph) ClassesChecked() ([]*EClass, error) {
	if g == nil {
		return nil, nil
	}
	if err := g.RebuildChecked(); err != nil {
		return nil, err
	}
	classes := g.classesList()
	if classes == nil {
		return nil, nil
	}
	res := make([]*EClass, len(classes))
	for i, class := range classes {
		nodes := make([]ENode, len(class.Nodes))
		for j, node := range class.Nodes {
			nodes[j] = node
			nodes[j].Children = append([]Id(nil), node.Children...)
		}
		res[i] = &EClass{ID: class.ID, Sort: class.Sort, Nodes: nodes, AnalysisData: class.AnalysisData, HasAnalysisData: class.analysisDataSet, analysisFacts: class.analysisFacts, analysisDataSet: class.analysisDataSet, analysisFactsSet: class.analysisFactsSet}
	}
	return res, nil
}

func (g *EGraph) classesList() []*EClass {
	if g == nil || !g.store.Initialized() {
		return nil
	}
	res := make([]*EClass, 0, g.store.ClassCount())
	g.store.RangeClasses(func(_ Id, c *EClass) bool {
		res = append(res, c)
		return true
	})
	sort.Slice(res, func(i, j int) bool {
		return res[i].ID < res[j].ID
	})
	return res
}
