package cudafuse

import (
	"fmt"
	"slices"
	"strings"
	"sync"

	"github.com/tmc/lib/nvidia/x/cudair"
)

// MemoryTracker records host-side allocation bookkeeping for a deferred graph.
type MemoryTracker struct {
	mu             sync.Mutex
	allocations    int
	allocatedBytes int64
	freedBytes     int64
}

// AllocationStats contains host-side counts recorded by the deferred graph compiler.
type AllocationStats struct {
	Allocations    int
	AllocatedBytes int64
	FreedBytes     int64
}

// RecordAlloc records an allocation count and byte size in host bookkeeping.
func (mt *MemoryTracker) RecordAlloc(bytes int64) {
	if mt == nil {
		return
	}
	mt.mu.Lock()
	defer mt.mu.Unlock()
	mt.allocations++
	mt.allocatedBytes += bytes
}

// RecordFree records a deallocation byte size in host bookkeeping.
func (mt *MemoryTracker) RecordFree(bytes int64) {
	if mt == nil {
		return
	}
	mt.mu.Lock()
	defer mt.mu.Unlock()
	mt.freedBytes += bytes
}

// Snapshot returns allocation counts and byte totals under the tracker's lock.
func (mt *MemoryTracker) Snapshot() AllocationStats {
	if mt == nil {
		return AllocationStats{}
	}
	mt.mu.Lock()
	defer mt.mu.Unlock()
	return AllocationStats{Allocations: mt.allocations, AllocatedBytes: mt.allocatedBytes, FreedBytes: mt.freedBytes}
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
	Err       error
}

// Name returns a unique identifier for the node in generated kernels.
func (n *LazyNode) Name() string {
	if n == nil {
		return "nil"
	}
	return fmt.Sprintf("node_%d_%s", n.ID, n.Op)
}

// LazyArray represents a deferred tensor whose evaluation is postponed until a boundary is reached.
// Its zero value is not usable; construct arrays with NewLazyArray.
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
	_, shapeErr := elementCount(shape)
	if dtype < cudair.Float32 || dtype > cudair.Int8 {
		shapeErr = cudair.ErrTypeMismatch
	}
	node := &LazyNode{
		ID:    ctx.newID(),
		Op:    "input",
		Shape: slices.Clone(shape),
		DType: dtype,
		Attrs: make(map[string]any),
		Err:   shapeErr,
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
	if la == nil || la.Node == nil || la.Ctx == nil {
		return failedArray(nil, "map", fmt.Errorf("invalid receiver"))
	}
	op = strings.ToLower(strings.TrimSpace(op))
	if !isUnaryOperator(op) {
		return failedArray(la.Ctx, op, fmt.Errorf("unknown unary operator %q", op))
	}
	if err := arrayError(op, la); err != nil {
		return failedArray(la.Ctx, op, err)
	}
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
	if la == nil || la.Node == nil || la.Ctx == nil {
		return failedArray(nil, "reshape", fmt.Errorf("invalid receiver"))
	}
	err := validateReshape(la.Shape, newShape)
	if inputErr := arrayError("reshape", la); inputErr != nil {
		err = inputErr
	}
	node := &LazyNode{
		ID:     la.Ctx.newID(),
		Op:     "reshape",
		Inputs: []*LazyNode{la.Node},
		Shape:  slices.Clone(newShape),
		DType:  la.DType,
		Attrs:  map[string]any{"target_shape": slices.Clone(newShape)},
		Err:    err,
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
	if la == nil || la.Node == nil || la.Ctx == nil {
		return failedArray(nil, "transpose", fmt.Errorf("invalid receiver"))
	}
	shape, err := transposeShape(la.Shape, perm)
	if inputErr := arrayError("transpose", la); inputErr != nil {
		err = inputErr
	}
	node := &LazyNode{
		ID:     la.Ctx.newID(),
		Op:     "transpose",
		Inputs: []*LazyNode{la.Node},
		Shape:  shape,
		DType:  la.DType,
		Attrs:  map[string]any{"permutation": slices.Clone(perm)},
		Err:    err,
	}
	return &LazyArray{
		Ctx:   la.Ctx,
		Node:  node,
		Shape: slices.Clone(shape),
		DType: la.DType,
	}
}

// Gather records a rank-one result with one element per integer index.
// An empty input permits only empty indices. Index values are not checked here.
func (la *LazyArray) Gather(indices *LazyArray) *LazyArray {
	if err := arrayError("gather", la, indices); err != nil {
		return failedArray(contextOf(la), "gather", err)
	}
	if !isIndexType(indices.DType) {
		return failedArray(la.Ctx, "gather", cudair.ErrTypeMismatch)
	}
	size, _ := elementCount(la.Shape)
	count, _ := elementCount(indices.Shape)
	if size == 0 && count != 0 {
		return failedArray(la.Ctx, "gather", fmt.Errorf("gather from empty input"))
	}
	node := &LazyNode{ID: la.Ctx.newID(), Op: "gather", Inputs: []*LazyNode{la.Node, indices.Node}, Shape: []int{count}, DType: la.DType, Attrs: make(map[string]any)}
	return &LazyArray{Ctx: la.Ctx, Node: node, Shape: []int{count}, DType: la.DType}
}

// Cycle repeats the receiver's row-major elements to fill shape, truncating
// when the target is smaller. The target must be ranked and nonempty, and the
// receiver must have at least one element. This records a materializing boundary.
func (la *LazyArray) Cycle(shape []int) *LazyArray {
	if la == nil || la.Node == nil || la.Ctx == nil {
		return failedArray(nil, "cycle", fmt.Errorf("invalid receiver"))
	}
	node := &LazyNode{
		ID:     la.Ctx.newID(),
		Op:     "cycle",
		Inputs: []*LazyNode{la.Node},
		Shape:  slices.Clone(shape),
		DType:  la.DType,
		Attrs:  map[string]any{"target_shape": slices.Clone(shape)},
	}
	node.Err = arrayError("cycle", la)
	if node.Err == nil {
		node.Err = validateLazyShape(node)
	}
	return &LazyArray{
		Ctx:   la.Ctx,
		Node:  node,
		Shape: slices.Clone(shape),
		DType: la.DType,
	}
}

// Replicate records a rank-one view repeating each input element count times.
// Count must be positive. Device lowering of this boundary is unsupported.
func (la *LazyArray) Replicate(count int) *LazyArray {
	if err := arrayError("replicate", la); err != nil {
		return failedArray(contextOf(la), "replicate", err)
	}
	if count <= 0 {
		return failedArray(la.Ctx, "replicate", fmt.Errorf("replicate count must be positive"))
	}
	size, _ := elementCount(la.Shape)
	if size > int(^uint(0)>>1)/count {
		return failedArray(la.Ctx, "replicate", fmt.Errorf("replicate count overflows"))
	}
	shape := []int{size * count}
	node := &LazyNode{ID: la.Ctx.newID(), Op: "replicate", Inputs: []*LazyNode{la.Node}, Shape: slices.Clone(shape), DType: la.DType, Attrs: map[string]any{"count": count}}
	return &LazyArray{Ctx: la.Ctx, Node: node, Shape: shape, DType: la.DType}
}

// Where selects between this array and other using a predicate mask. It preserves
// the element count; it is distinct from Filter's stable compaction.
func (la *LazyArray) Where(pred, other *LazyArray) *LazyArray {
	if la == nil || la.Node == nil || la.Ctx == nil {
		return failedArray(nil, "where", fmt.Errorf("invalid receiver"))
	}
	inputs := []*LazyNode{la.Node}
	if pred != nil {
		inputs = append(inputs, pred.Node)
	}
	if other != nil {
		inputs = append(inputs, other.Node)
	}
	var nodeErr error
	nodeErr = arrayError("where", la, pred, other)
	if pred == nil || pred.Node == nil || other == nil || other.Node == nil {
		nodeErr = fmt.Errorf("where requires predicate and alternate inputs")
	}
	node := &LazyNode{
		ID:     la.Ctx.newID(),
		Op:     "where",
		Inputs: inputs,
		Shape:  slices.Clone(la.Shape),
		DType:  la.DType,
		Attrs:  make(map[string]any),
		Err:    nodeErr,
	}
	return &LazyArray{
		Ctx:   la.Ctx,
		Node:  node,
		Shape: slices.Clone(la.Shape),
		DType: la.DType,
	}
}

// Filter stably selects rank-one FP32 values whose FP32 predicate is exactly one.
// Predicates must be zero or one; both signs of zero are false. Invalid or
// nonfinite predicates are refused during execution. Shape [-1] marks an
// unresolved planning extent, never an allocation size. Execution.Shape returns
// the selected extent after completed count validation. This initial execution
// supports one Filter with unary/scalar pointwise, Sum and sum Scan consumers.
func (la *LazyArray) Filter(pred *LazyArray) *LazyArray {
	if la == nil || la.Node == nil || la.Ctx == nil {
		return failedArray(nil, "filter", fmt.Errorf("invalid receiver"))
	}
	inputs := []*LazyNode{la.Node}
	if pred != nil {
		inputs = append(inputs, pred.Node)
	}
	var nodeErr error
	nodeErr = arrayError("filter", la, pred)
	if pred == nil || pred.Node == nil {
		nodeErr = fmt.Errorf("filter requires a predicate")
	}
	node := &LazyNode{
		ID:     la.Ctx.newID(),
		Op:     "filter",
		Inputs: inputs,
		Shape:  []int{-1},
		DType:  la.DType,
		Attrs:  make(map[string]any),
		Err:    nodeErr,
	}
	if node.Err == nil {
		node.Err = validateFilterShape(node)
	}
	return &LazyArray{
		Ctx:   la.Ctx,
		Node:  node,
		Shape: []int{-1},
		DType: la.DType,
	}
}

// Keep records a conditionally fused predicate-masking boundary.
func (la *LazyArray) Keep(pred *LazyArray) *LazyArray {
	if la == nil || la.Node == nil || la.Ctx == nil {
		return failedArray(nil, "keep", fmt.Errorf("invalid receiver"))
	}
	inputs := []*LazyNode{la.Node}
	if pred != nil {
		inputs = append(inputs, pred.Node)
	}
	var nodeErr error
	nodeErr = arrayError("keep", la, pred)
	if pred == nil || pred.Node == nil {
		nodeErr = fmt.Errorf("keep requires a predicate")
	}
	node := &LazyNode{
		ID:     la.Ctx.newID(),
		Op:     "keep",
		Inputs: inputs,
		Shape:  slices.Clone(la.Shape),
		DType:  la.DType,
		Attrs:  make(map[string]any),
		Err:    nodeErr,
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
	if err := arrayError("reduce", la); err != nil {
		return nil, err
	}
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
	if err := arrayError("scan", la); err != nil {
		return nil, err
	}
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
	if err := arrayError("sort", la); err != nil {
		return nil, err
	}
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

// Materialize compiles the deferred DAG into fusion regions and records host-side
// bookkeeping for the output allocation. It does not allocate device memory.
func (la *LazyArray) Materialize() (*MaterializedPlan, error) {
	if err := arrayError("materialize", la); err != nil {
		return nil, err
	}
	if la == nil || la.Node == nil {
		return nil, fmt.Errorf("cudafuse: invalid lazy array")
	}
	return CompileDeferredGraph(la.Ctx, la.Node)
}

func (la *LazyArray) binaryOp(op string, other *LazyArray) *LazyArray {
	if la == nil || la.Node == nil || la.Ctx == nil {
		return failedArray(nil, op, fmt.Errorf("invalid receiver"))
	}
	op = strings.ToLower(strings.TrimSpace(op))
	if !isBinaryOperator(op) {
		return failedArray(la.Ctx, op, fmt.Errorf("unknown binary operator %q", op))
	}
	inputs := []*LazyNode{la.Node}
	if other != nil {
		inputs = append(inputs, other.Node)
	}
	var nodeErr error
	if other == nil || other.Node == nil {
		nodeErr = fmt.Errorf("%s requires a non-nil operand", op)
	}
	shape := slices.Clone(la.Shape)
	if nodeErr == nil {
		nodeErr = arrayError(op, la, other)
	}
	if nodeErr == nil && la.DType != other.DType {
		nodeErr = cudair.ErrTypeMismatch
	}
	if nodeErr == nil {
		shape, nodeErr = binaryShape(la.Shape, other.Shape)
	}
	if nodeErr != nil {
		shape = nil
	}
	node := &LazyNode{
		ID:     la.Ctx.newID(),
		Op:     op,
		Inputs: inputs,
		Shape:  slices.Clone(shape),
		DType:  la.DType,
		Attrs:  make(map[string]any),
		Err:    nodeErr,
	}
	return &LazyArray{
		Ctx:   la.Ctx,
		Node:  node,
		Shape: slices.Clone(shape),
		DType: la.DType,
	}
}

func isUnaryOperator(op string) bool {
	switch op {
	case "relu", "silu", "exp", "sqrt", "neg":
		return true
	default:
		return false
	}
}

func isBinaryOperator(op string) bool {
	switch op {
	case "add", "sub", "mul", "div":
		return true
	default:
		return false
	}
}

func failedArray(ctx *LazyContext, op string, err error) *LazyArray {
	if ctx == nil {
		ctx = NewLazyContext()
	}
	node := &LazyNode{ID: ctx.newID(), Op: op, Err: err}
	return &LazyArray{Ctx: ctx, Node: node}
}

func validateReshape(oldShape, newShape []int) error {
	oldCount, err := shapeElements(oldShape)
	if err != nil {
		return err
	}
	newCount, err := shapeElements(newShape)
	if err != nil {
		return err
	}
	if oldCount != newCount {
		return fmt.Errorf("reshape changes element count from %d to %d", oldCount, newCount)
	}
	return nil
}

func transposeShape(shape, perm []int) ([]int, error) {
	if len(shape) != len(perm) {
		return nil, fmt.Errorf("transpose permutation length %d does not match rank %d", len(perm), len(shape))
	}
	seen := make([]bool, len(shape))
	out := make([]int, len(perm))
	for i, axis := range perm {
		if axis < 0 || axis >= len(shape) || seen[axis] {
			return nil, fmt.Errorf("invalid transpose axis %d", axis)
		}
		seen[axis] = true
		out[i] = shape[axis]
	}
	return out, nil
}
