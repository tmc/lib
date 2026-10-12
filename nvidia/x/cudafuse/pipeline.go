package cudafuse

import (
	"fmt"
	"math"
	"strings"

	"github.com/tmc/lib/nvidia/x/cudadevicetable"
	"github.com/tmc/lib/nvidia/x/cudaerrors"
	"github.com/tmc/lib/nvidia/x/cudair"
)

// FusionRegion represents a contiguous subgraph of operations fused into a single mega-kernel.
type FusionRegion struct {
	ID         int
	Nodes      []*LazyNode
	Inputs     []*LazyNode
	Outputs    []*LazyNode
	Kernel     *FusedKernel
	Compaction bool
	Boundary   string
	Reason     string
}

// MaterializedPlan contains the compiled fusion regions and allocation receipts.
type MaterializedPlan struct {
	Regions        []*FusionRegion
	AllocatedBytes int64
	OutputBuffers  int
	RootNode       *LazyNode
}

// CheckTarget returns a fixed result based on target: known targets return
// ErrHardwarePending and unknown targets return ErrUnsupported. It does not
// probe hardware or execute a kernel.
func (p *MaterializedPlan) CheckTarget(target string) error {
	if !isKnownTarget(target) {
		return fmt.Errorf("cudafuse: target %q: %w", target, cudaerrors.ErrUnsupported)
	}
	return fmt.Errorf("cudafuse: device execution on %s: %w", target, cudaerrors.ErrHardwarePending)
}

func isKnownTarget(target string) bool {
	target = strings.ToLower(strings.TrimSpace(target))
	if target == "" {
		return false
	}
	for _, known := range cudadevicetable.Targets() {
		if strings.ToLower(strings.TrimSpace(known)) == target {
			return true
		}
	}
	return false
}

// CompileDeferredGraph flushes a deferred execution DAG into compiled fusion regions.
func CompileDeferredGraph(ctx *LazyContext, root *LazyNode) (*MaterializedPlan, error) {
	if root == nil {
		return nil, fmt.Errorf("cudafuse: nil root node")
	}
	if ctx == nil {
		ctx = NewLazyContext()
	}

	ordered, err := checkedTopologicalOrder(root)
	if err != nil {
		return nil, err
	}
	for _, n := range ordered {
		if n.Err != nil {
			return nil, fmt.Errorf("cudafuse: node %s: %w", n.Name(), n.Err)
		}
	}
	regions, err := partitionIntoRegions(ordered)
	if err != nil {
		return nil, fmt.Errorf("cudafuse: partition fusion regions: %w", err)
	}

	// For regions of fusible elementwise nodes, compile them via FuseWithSourceMap.
	for _, reg := range regions {
		if reg.Compaction || IsMaterializingBoundary(reg.Boundary) {
			continue
		}
		if len(reg.Nodes) == 0 {
			continue
		}
		hasCompute := false
		for _, n := range reg.Nodes {
			if n.Op != "input" {
				hasCompute = true
				break
			}
		}
		if !hasCompute {
			continue
		}

		kernelName := fmt.Sprintf("fused_region_%d", reg.ID)
		indexed := false
		for _, node := range reg.Nodes {
			indexed = indexed || node.Op == "transpose"
		}
		if indexed {
			kernel, err := compileIndexedLazyRegion(reg.Inputs, reg.Outputs, kernelName)
			if err != nil {
				return nil, fmt.Errorf("cudafuse: compile indexed region %d: %w", reg.ID, err)
			}
			reg.Kernel = &kernel
			continue
		}
		g := cudair.NewGraph()
		nodeMap := make(map[*LazyNode]*cudair.Node)

		for _, in := range reg.Inputs {
			if _, exists := nodeMap[in]; !exists {
				elements, shapeErr := shapeElements(in.Shape)
				if shapeErr != nil || elements > math.MaxInt {
					return nil, fmt.Errorf("cudafuse: input %s has invalid shape %v", in.Name(), in.Shape)
				}
				nodeMap[in] = g.AddInput(in.Name(), []int{int(elements)}, in.DType)
			}
		}

		for _, n := range reg.Nodes {
			if n.Op == "input" {
				continue
			}
			switch n.Op {
			case "add", "sub", "mul", "div":
				if len(n.Inputs) != 2 {
					return nil, fmt.Errorf("cudafuse: %s requires two inputs", n.Op)
				}
				in1 := nodeMap[n.Inputs[0]]
				in2 := nodeMap[n.Inputs[1]]
				if in1 == nil || in2 == nil {
					return nil, fmt.Errorf("cudafuse: missing input for %s", n.Op)
				}
				nodeMap[n] = g.AddBinary(n.Op, in1, in2)
			case "relu", "silu", "exp", "sqrt", "neg":
				if len(n.Inputs) != 1 {
					return nil, fmt.Errorf("cudafuse: %s requires one input", n.Op)
				}
				in1 := nodeMap[n.Inputs[0]]
				if in1 == nil {
					return nil, fmt.Errorf("cudafuse: missing input for %s", n.Op)
				}
				nodeMap[n] = g.AddUnary(n.Op, in1)
			case "reshape":
				if len(n.Inputs) != 1 || nodeMap[n.Inputs[0]] == nil {
					return nil, fmt.Errorf("cudafuse: reshape is missing its input")
				}
				// The emitter is a flat elementwise kernel. A reshape with an
				// unchanged element count changes only the view, not the scalar
				// expression or its linear index.
				nodeMap[n] = nodeMap[n.Inputs[0]]
			default:
				return nil, fmt.Errorf("cudafuse: unsupported operator %q in fusion region: %w", n.Op, ErrUnsupportedOperator)
			}
		}

		var outNodes []*cudair.Node
		for _, out := range reg.Outputs {
			if cNode, ok := nodeMap[out]; ok {
				outNodes = append(outNodes, cNode)
			}
		}
		if len(outNodes) > 0 {
			fk, err := compileRegionKernel(g, kernelName, outNodes...)
			if err != nil {
				return nil, fmt.Errorf("cudafuse: compile region %d: %w", reg.ID, err)
			}
			reg.Kernel = &fk
		}
	}

	// Calculate output allocation size for the materialized result
	allocBytes, err := outputBytes(root.Shape, root.DType)
	if err != nil {
		return nil, fmt.Errorf("cudafuse: output allocation size: %w", err)
	}

	// Record the single output buffer allocation
	ctx.Tracker.RecordAlloc(allocBytes)

	// Mark nodes as evaluated
	for _, n := range ordered {
		n.Evaluated = true
	}

	return &MaterializedPlan{
		Regions:        regions,
		AllocatedBytes: allocBytes,
		OutputBuffers:  1,
		RootNode:       root,
	}, nil
}

func compileRegionKernel(g *cudair.Graph, kernelName string, outputs ...*cudair.Node) (FusedKernel, error) {
	if err := g.SetOutputs(outputs...); err != nil {
		return FusedKernel{}, fmt.Errorf("set outputs: %w", err)
	}
	fk, err := FuseWithSourceMap(g, kernelName)
	if err != nil {
		return FusedKernel{}, fmt.Errorf("fuse graph: %w", err)
	}
	return fk, nil
}

func outputBytes(shape []int, dtype cudair.DataType) (int64, error) {
	if dtype < cudair.Float32 || dtype > cudair.Int8 {
		return 0, fmt.Errorf("unsupported data type %s", dtype)
	}
	empty := false
	for _, dim := range shape {
		if dim < 0 {
			return 0, fmt.Errorf("negative output dimension %d", dim)
		}
		empty = empty || dim == 0
	}
	if empty {
		return 0, nil
	}
	elements := uint64(1)
	for _, dim := range shape {
		if elements > math.MaxInt64/uint64(dim) {
			return 0, fmt.Errorf("output element count overflows int64")
		}
		elements *= uint64(dim)
	}
	bytes := uint64(dtype.SizeBytes())
	if elements > math.MaxInt64/bytes {
		return 0, fmt.Errorf("output byte count overflows int64")
	}
	return int64(elements * bytes), nil
}

func checkedTopologicalOrder(root *LazyNode) ([]*LazyNode, error) {
	var ordered []*LazyNode
	state := make(map[*LazyNode]int)
	var visit func(*LazyNode) error
	visit = func(n *LazyNode) error {
		if n == nil || state[n] == 1 {
			return fmt.Errorf("cudafuse: nil or cyclic deferred node")
		}
		if state[n] == 2 {
			return nil
		}
		state[n] = 1
		for _, in := range n.Inputs {
			if err := visit(in); err != nil {
				return err
			}
		}
		state[n] = 2
		ordered = append(ordered, n)
		return nil
	}
	if err := visit(root); err != nil {
		return nil, err
	}
	return ordered, nil
}

func partitionIntoRegions(nodes []*LazyNode) ([]*FusionRegion, error) {
	var regions []*FusionRegion
	var currentRegion *FusionRegion
	regionCounter := 0

	startNewRegion := func(compaction bool, boundary string) *FusionRegion {
		regionCounter++
		r := &FusionRegion{
			ID:         regionCounter,
			Compaction: compaction,
			Boundary:   boundary,
		}
		regions = append(regions, r)
		return r
	}

	currentRegion = startNewRegion(false, "")

	for _, n := range nodes {
		if IsCompactionBoundary(n.Op) {
			// Seal current region if it has operations
			if len(currentRegion.Nodes) > 0 {
				currentRegion.Outputs = findOutputs(currentRegion.Nodes, nodes)
			}
			// Compaction gets its own isolated boundary region
			compReg := startNewRegion(true, n.Op)
			compReg.Nodes = append(compReg.Nodes, n)
			compReg.Inputs = uniqueExecutionNodes(n.Inputs)
			compReg.Outputs = []*LazyNode{n}

			// Start clean new downstream region
			currentRegion = startNewRegion(false, "")
			continue
		}

		if IsMaterializingBoundary(n.Op) {
			if len(currentRegion.Nodes) > 0 {
				currentRegion.Outputs = findOutputs(currentRegion.Nodes, nodes)
			}
			matReg := startNewRegion(false, n.Op)
			if n.Op == "gather" || n.Op == "cycle" || n.Op == "replicate" {
				matReg.Reason = "operation is not lowered into the flat elementwise kernel"
			}
			matReg.Nodes = append(matReg.Nodes, n)
			matReg.Inputs = uniqueExecutionNodes(n.Inputs)
			matReg.Outputs = []*LazyNode{n}

			currentRegion = startNewRegion(false, "")
			continue
		}

		// Separate iteration domains even when their DFS positions are adjacent.
		// Reshape and transpose preserve element count within an indexed region.
		if len(currentRegion.Nodes) != 0 {
			previous, err := shapeElements(currentRegion.Nodes[0].Shape)
			if err != nil {
				return nil, err
			}
			count, err := shapeElements(n.Shape)
			if err != nil {
				return nil, err
			}
			if count != previous {
				currentRegion.Outputs = findOutputs(currentRegion.Nodes, nodes)
				currentRegion = startNewRegion(false, "")
			}
		}
		// Fusible elementwise or input node
		currentRegion.Nodes = append(currentRegion.Nodes, n)
		for _, in := range n.Inputs {
			if in.Op == "input" || !containsNode(currentRegion.Nodes, in) {
				if !containsNode(currentRegion.Inputs, in) {
					currentRegion.Inputs = append(currentRegion.Inputs, in)
				}
			}
		}
	}

	if len(currentRegion.Nodes) > 0 {
		currentRegion.Outputs = findOutputs(currentRegion.Nodes, nodes)
	}

	// Filter out empty regions
	var filtered []*FusionRegion
	for _, r := range regions {
		if len(r.Nodes) > 0 {
			filtered = append(filtered, r)
		}
	}

	return filtered, nil
}

func containsNode(list []*LazyNode, n *LazyNode) bool {
	for _, item := range list {
		if item == n {
			return true
		}
	}
	return false
}

func findOutputs(regionNodes []*LazyNode, allNodes []*LazyNode) []*LazyNode {
	var outputs []*LazyNode
	regionMap := make(map[*LazyNode]bool)
	for _, n := range regionNodes {
		regionMap[n] = true
	}

	for _, n := range regionNodes {
		isOutput := false
		// If consumed by a node outside this region
		for _, other := range allNodes {
			if !regionMap[other] {
				for _, in := range other.Inputs {
					if in == n {
						isOutput = true
						break
					}
				}
			}
			if isOutput {
				break
			}
		}
		// Or if it is a terminal node
		if otherUsersCount(n, allNodes) == 0 {
			isOutput = true
		}
		if isOutput && !containsNode(outputs, n) {
			outputs = append(outputs, n)
		}
	}

	if len(outputs) == 0 && len(regionNodes) > 0 {
		outputs = append(outputs, regionNodes[len(regionNodes)-1])
	}
	return outputs
}

func otherUsersCount(n *LazyNode, all []*LazyNode) int {
	users := 0
	for _, other := range all {
		for _, in := range other.Inputs {
			if in == n {
				users++
			}
		}
	}
	return users
}
