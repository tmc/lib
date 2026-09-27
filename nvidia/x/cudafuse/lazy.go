package cudafuse

import (
	"fmt"
	"slices"
	"sync"

	"github.com/tmc/lib/nvidia/x/cudair"
)

// MemoryTracker records device buffer allocation metrics across lazy and eager pipelines.
type MemoryTracker struct {
	mu             sync.Mutex
	Allocations    int
	AllocatedBytes int64
	FreedBytes     int64
}

// RecordAlloc records an allocation of the given byte size.
func (mt *MemoryTracker) RecordAlloc(bytes int64) {
	if mt == nil {
		return
	}
	mt.mu.Lock()
	defer mt.mu.Unlock()
	mt.Allocations++
	mt.AllocatedBytes += bytes
}

// RecordFree records a deallocation of the given byte size.
func (mt *MemoryTracker) RecordFree(bytes int64) {
	if mt == nil {
		return
	}
	mt.mu.Lock()
	defer mt.mu.Unlock()
	mt.FreedBytes += bytes
}

// LazyContext manages DAG construction, allocation metrics, and fusion pipelines.
type LazyContext struct {
	Tracker *MemoryTracker
	mu      sync.Mutex
	nextID  int
}

// NewLazyContext creates a new LazyContext with an initialized MemoryTracker.
func NewLazyContext() *LazyContext {
	return &LazyContext{
		Tracker: &MemoryTracker{},
	}
}

func (ctx *LazyContext) newID() int {
	ctx.mu.Lock()
	defer ctx.mu.Unlock()
	ctx.nextID++
	return ctx.nextID
}

// LazyNode represents an operation node in a deferred execution DAG.
type LazyNode struct {
	ID        int
	Op        string
	Inputs    []*LazyNode
	Shape     []int
	DType     cudair.DataType
	Evaluated bool
	Attrs     map[string]any
}

// Name returns a unique identifier for the node in generated kernels.
func (n *LazyNode) Name() string {
	if n == nil {
		return "nil"
	}
	return fmt.Sprintf("node_%d_%s", n.ID, n.Op)
}

// LazyArray represents a deferred tensor whose evaluation is postponed until a boundary is reached.
type LazyArray struct {
	Ctx   *LazyContext
	Node  *LazyNode
	Shape []int
	DType cudair.DataType
}

// NewLazyArray creates an un-evaluated leaf LazyArray representing an input tensor.
func NewLazyArray(ctx *LazyContext, shape []int, dtype cudair.DataType) *LazyArray {
	if ctx == nil {
		ctx = NewLazyContext()
	}
	node := &LazyNode{
		ID:    ctx.newID(),
		Op:    "input",
		Shape: slices.Clone(shape),
		DType: dtype,
		Attrs: make(map[string]any),
	}
	return &LazyArray{
		Ctx:   ctx,
		Node:  node,
		Shape: slices.Clone(shape),
		DType: dtype,
	}
}

// Add chains a lazy elementwise addition without allocating intermediate GPU memory.
func (la *LazyArray) Add(other *LazyArray) *LazyArray {
	return la.binaryOp("add", other)
}

// Mul chains a lazy elementwise multiplication without allocating intermediate GPU memory.
func (la *LazyArray) Mul(other *LazyArray) *LazyArray {
	return la.binaryOp("mul", other)
}

// Sub chains a lazy elementwise subtraction without allocating intermediate GPU memory.
func (la *LazyArray) Sub(other *LazyArray) *LazyArray {
	return la.binaryOp("sub", other)
}

// Div chains a lazy elementwise division without allocating intermediate GPU memory.
func (la *LazyArray) Div(other *LazyArray) *LazyArray {
	return la.binaryOp("div", other)
}

// Map chains a unary elementwise operation (e.g. "relu", "silu", "exp", "sqrt", "neg").
func (la *LazyArray) Map(op string) *LazyArray {
	node := &LazyNode{
		ID:     la.Ctx.newID(),
		Op:     op,
		Inputs: []*LazyNode{la.Node},
		Shape:  slices.Clone(la.Shape),
		DType:  la.DType,
		Attrs:  make(map[string]any),
	}
	return &LazyArray{
		Ctx:   la.Ctx,
		Node:  node,
		Shape: slices.Clone(la.Shape),
		DType: la.DType,
	}
}

// Map2 chains a binary elementwise operation over two lazy arrays.
func (la *LazyArray) Map2(op string, other *LazyArray) *LazyArray {
	return la.binaryOp(op, other)
}

// Reshape records a lazy reshape operation.
func (la *LazyArray) Reshape(newShape []int) *LazyArray {
	node := &LazyNode{
		ID:     la.Ctx.newID(),
		Op:     "reshape",
		Inputs: []*LazyNode{la.Node},
		Shape:  slices.Clone(newShape),
		DType:  la.DType,
		Attrs:  map[string]any{"target_shape": slices.Clone(newShape)},
	}
	return &LazyArray{
		Ctx:   la.Ctx,
		Node:  node,
		Shape: slices.Clone(newShape),
		DType: la.DType,
	}
}

// Transpose records a lazy transpose permutation.
func (la *LazyArray) Transpose(perm []int) *LazyArray {
	node := &LazyNode{
		ID:     la.Ctx.newID(),
		Op:     "transpose",
		Inputs: []*LazyNode{la.Node},
		Shape:  slices.Clone(la.Shape),
		DType:  la.DType,
		Attrs:  map[string]any{"permutation": slices.Clone(perm)},
	}
	return &LazyArray{
		Ctx:   la.Ctx,
		Node:  node,
		Shape: slices.Clone(la.Shape),
		DType: la.DType,
	}
}

// Gather records a lazy gather operation using index offsets.
func (la *LazyArray) Gather(indices *LazyArray) *LazyArray {
	inputs := []*LazyNode{la.Node}
	if indices != nil {
		inputs = append(inputs, indices.Node)
	}
	node := &LazyNode{
		ID:     la.Ctx.newID(),
		Op:     "gather",
		Inputs: inputs,
		Shape:  slices.Clone(la.Shape),
		DType:  la.DType,
		Attrs:  make(map[string]any),
	}
	return &LazyArray{
		Ctx:   la.Ctx,
		Node:  node,
		Shape: slices.Clone(la.Shape),
		DType: la.DType,
	}
}

// Cycle records a lazy periodic repeating operation.
func (la *LazyArray) Cycle(count int) *LazyArray {
	node := &LazyNode{
		ID:     la.Ctx.newID(),
		Op:     "cycle",
		Inputs: []*LazyNode{la.Node},
		Shape:  slices.Clone(la.Shape),
		DType:  la.DType,
		Attrs:  map[string]any{"count": count},
	}
	return &LazyArray{
		Ctx:   la.Ctx,
		Node:  node,
		Shape: slices.Clone(la.Shape),
		DType: la.DType,
	}
}

// Replicate records a lazy tensor replication operation.
func (la *LazyArray) Replicate(count int) *LazyArray {
	node := &LazyNode{
		ID:     la.Ctx.newID(),
		Op:     "replicate",
		Inputs: []*LazyNode{la.Node},
		Shape:  slices.Clone(la.Shape),
		DType:  la.DType,
		Attrs:  map[string]any{"count": count},
	}
	return &LazyArray{
		Ctx:   la.Ctx,
		Node:  node,
		Shape: slices.Clone(la.Shape),
		DType: la.DType,
	}
}

// Where records a conditionally fused compaction node using a boolean predicate mask.
func (la *LazyArray) Where(pred, other *LazyArray) *LazyArray {
	inputs := []*LazyNode{la.Node}
	if pred != nil {
		inputs = append(inputs, pred.Node)
	}
	if other != nil {
		inputs = append(inputs, other.Node)
	}
	node := &LazyNode{
		ID:     la.Ctx.newID(),
		Op:     "where",
		Inputs: inputs,
		Shape:  slices.Clone(la.Shape),
		DType:  la.DType,
		Attrs:  make(map[string]any),
	}
	return &LazyArray{
		Ctx:   la.Ctx,
		Node:  node,
		Shape: slices.Clone(la.Shape),
		DType: la.DType,
	}
}

// Filter records a predicate-based compaction boundary that alters dynamic output size.
func (la *LazyArray) Filter(pred *LazyArray) *LazyArray {
	inputs := []*LazyNode{la.Node}
	if pred != nil {
		inputs = append(inputs, pred.Node)
	}
	node := &LazyNode{
		ID:     la.Ctx.newID(),
		Op:     "filter",
		Inputs: inputs,
		Shape:  slices.Clone(la.Shape),
		DType:  la.DType,
		Attrs:  make(map[string]any),
	}
	return &LazyArray{
		Ctx:   la.Ctx,
		Node:  node,
		Shape: slices.Clone(la.Shape),
		DType: la.DType,
	}
}

// Keep records a conditionally fused predicate-masking boundary.
func (la *LazyArray) Keep(pred *LazyArray) *LazyArray {
	inputs := []*LazyNode{la.Node}
	if pred != nil {
		inputs = append(inputs, pred.Node)
	}
	node := &LazyNode{
		ID:     la.Ctx.newID(),
		Op:     "keep",
		Inputs: inputs,
		Shape:  slices.Clone(la.Shape),
		DType:  la.DType,
		Attrs:  make(map[string]any),
	}
	return &LazyArray{
		Ctx:   la.Ctx,
		Node:  node,
		Shape: slices.Clone(la.Shape),
		DType: la.DType,
	}
}

// Sum triggers materialization by appending a reduction boundary and flushing the pipeline.
func (la *LazyArray) Sum() (*MaterializedPlan, error) {
	return la.Reduce("sum", 0)
}

// Reduce appends a reduction boundary node and compiles the pending fusion graph.
func (la *LazyArray) Reduce(op string, axis int) (*MaterializedPlan, error) {
	node := &LazyNode{
		ID:     la.Ctx.newID(),
		Op:     op,
		Inputs: []*LazyNode{la.Node},
		Shape:  []int{1},
		DType:  la.DType,
		Attrs:  map[string]any{"axis": axis},
	}
	res := &LazyArray{
		Ctx:   la.Ctx,
		Node:  node,
		Shape: []int{1},
		DType: la.DType,
	}
	return res.Materialize()
}

// Scan triggers materialization of an associative prefix scan.
func (la *LazyArray) Scan(op string) (*MaterializedPlan, error) {
	node := &LazyNode{
		ID:     la.Ctx.newID(),
		Op:     "scan",
		Inputs: []*LazyNode{la.Node},
		Shape:  slices.Clone(la.Shape),
		DType:  la.DType,
		Attrs:  map[string]any{"scan_op": op},
	}
	res := &LazyArray{
		Ctx:   la.Ctx,
		Node:  node,
		Shape: slices.Clone(la.Shape),
		DType: la.DType,
	}
	return res.Materialize()
}

// Sort triggers materialization of a key-value or array sort.
func (la *LazyArray) Sort() (*MaterializedPlan, error) {
	node := &LazyNode{
		ID:     la.Ctx.newID(),
		Op:     "sort",
		Inputs: []*LazyNode{la.Node},
		Shape:  slices.Clone(la.Shape),
		DType:  la.DType,
		Attrs:  make(map[string]any),
	}
	res := &LazyArray{
		Ctx:   la.Ctx,
		Node:  node,
		Shape: slices.Clone(la.Shape),
		DType: la.DType,
	}
	return res.Materialize()
}

// ToHost triggers complete materialization and returns the deferred execution plan.
func (la *LazyArray) ToHost() (*MaterializedPlan, error) {
	return la.Materialize()
}

// Materialize flushes the deferred DAG into compiled fusion regions, eliminating intermediate allocations.
func (la *LazyArray) Materialize() (*MaterializedPlan, error) {
	return CompileDeferredGraph(la.Ctx, la.Node)
}

func (la *LazyArray) binaryOp(op string, other *LazyArray) *LazyArray {
	inputs := []*LazyNode{la.Node}
	if other != nil {
		inputs = append(inputs, other.Node)
	}
	node := &LazyNode{
		ID:     la.Ctx.newID(),
		Op:     op,
		Inputs: inputs,
		Shape:  slices.Clone(la.Shape),
		DType:  la.DType,
		Attrs:  make(map[string]any),
	}
	return &LazyArray{
		Ctx:   la.Ctx,
		Node:  node,
		Shape: slices.Clone(la.Shape),
		DType: la.DType,
	}
}
