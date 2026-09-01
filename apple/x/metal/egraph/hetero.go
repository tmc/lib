package egraph

// TargetDevice specifies the hardware execution target for a partition or e-class.
type TargetDevice string

const (
	TargetGPU TargetDevice = "metal_gpu" // Apple Silicon Metal GPU Cores
	TargetANE TargetDevice = "apple_ane" // Apple Neural Engine Cores
	TargetCPU TargetDevice = "host_cpu"  // Host Apple CPU
)

// PartitionedGraph represents a heterogeneous pipeline partitioned across GPU and ANE.
type PartitionedGraph struct {
	ANEExprs []*Expr
	GPUExprs []*Expr
}

// HeterogeneousPartition splits an optimized E-Graph expression DAG into ANE and GPU subgraphs.
//
// Heuristic Cost Model:
// 1. Heavy MLP / Feed-Forward Network GEMVs and SwiGLU activations (fixed shape) -> Apple Neural Engine (ANE)
// 2. Causal Self-Attention, Dynamic KV Cache, and RoPE rotations -> Apple Metal GPU
func HeterogeneousPartition(root *Expr) *PartitionedGraph {
	pg := &PartitionedGraph{}
	partitionNode(root, pg)
	return pg
}

func partitionNode(e *Expr, pg *PartitionedGraph) {
	if e == nil {
		return
	}

	switch e.Op {
	case "fused_swiglu_gemv", "fused_rmsnorm_gemv":
		pg.ANEExprs = append(pg.ANEExprs, e)
	case "causal_flash_attention", "rope":
		pg.GPUExprs = append(pg.GPUExprs, e)
	default:
		if e.Op == "add" {
			pg.GPUExprs = append(pg.GPUExprs, e)
		}
		for _, child := range e.Children {
			partitionNode(child, pg)
		}
	}
}
