package cudafuse

import (
	"fmt"
	"strconv"
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
}

// MaterializedPlan contains the compiled fusion regions and allocation receipts.
type MaterializedPlan struct {
	Regions        []*FusionRegion
	AllocatedBytes int64
	OutputBuffers  int
	RootNode       *LazyNode
}

// Execute checks hardware capability for target architecture and returns ErrHardwarePending or ErrUnsupported.
// It strictly prohibits faking GPU execution.
func (p *MaterializedPlan) Execute(target string) error {
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
	if _, ok := cudadevicetable.LookupTarget(target); ok {
		return true
	}
	if _, ok := cudadevicetable.Lookup(target); ok {
		return true
	}
	if strings.HasPrefix(target, "sm_") {
		numPart := strings.TrimPrefix(target, "sm_")
		numPart = strings.TrimRight(numPart, "abcdefghijklmnopqrstuvwxyz")
		if n, err := strconv.Atoi(numPart); err == nil && n >= 35 && n <= 121 {
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

	ordered := topologicalOrder(root)
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
		g := cudair.NewGraph()
		nodeMap := make(map[*LazyNode]*cudair.Node)

		for _, in := range reg.Inputs {
			if _, exists := nodeMap[in]; !exists {
				nodeMap[in] = g.AddInput(in.Name(), in.Shape, in.DType)
			}
		}

		for _, n := range reg.Nodes {
			if n.Op == "input" {
				continue
			}
			switch n.Op {
			case "add", "sub", "mul", "div":
				in1 := nodeMap[n.Inputs[0]]
				in2 := nodeMap[n.Inputs[1]]
				if in1 == nil || in2 == nil {
					return nil, fmt.Errorf("cudafuse: missing input for %s", n.Op)
				}
				nodeMap[n] = g.AddBinary(n.Op, in1, in2)
			case "relu", "silu", "exp", "sqrt", "neg":
				in1 := nodeMap[n.Inputs[0]]
				if in1 == nil {
					return nil, fmt.Errorf("cudafuse: missing input for %s", n.Op)
				}
				nodeMap[n] = g.AddUnary(n.Op, in1)
			default:
				// If not a scalar C++ op recognized by Fuse, mark as custom or lowerable
			}
		}

		var outNodes []*cudair.Node
		for _, out := range reg.Outputs {
			if cNode, ok := nodeMap[out]; ok {
				outNodes = append(outNodes, cNode)
			}
		}
		if len(outNodes) > 0 {
			if err := g.SetOutputs(outNodes...); err == nil {
				if fk, fuseErr := FuseWithSourceMap(g, kernelName); fuseErr == nil {
					reg.Kernel = &fk
				}
			}
		}
	}

	// Calculate output allocation size for the materialized result
	count := 1
	for _, dim := range root.Shape {
		if dim > 0 {
			count *= dim
		}
	}
	elemBytes := 4 // float32
	if root.DType == cudair.Float16 {
		elemBytes = 2
	}
	allocBytes := int64(count * elemBytes)

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

func topologicalOrder(root *LazyNode) []*LazyNode {
	var ordered []*LazyNode
	visited := make(map[*LazyNode]bool)

	var visit func(n *LazyNode)
	visit = func(n *LazyNode) {
		if n == nil || visited[n] {
			return
		}
		visited[n] = true
		for _, in := range n.Inputs {
			visit(in)
		}
		ordered = append(ordered, n)
	}

	visit(root)
	return ordered
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
			compReg.Inputs = append(compReg.Inputs, n.Inputs...)
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
			matReg.Nodes = append(matReg.Nodes, n)
			matReg.Inputs = append(matReg.Inputs, n.Inputs...)
			matReg.Outputs = []*LazyNode{n}

			currentRegion = startNewRegion(false, "")
			continue
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
