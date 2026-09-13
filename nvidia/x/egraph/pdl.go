package egraph

import (
	"fmt"
	"sort"
	"strings"
)

// PDLBarrierWait is the PTX inline assembly barrier that synchronizes
// CTA execution with predecessor grid completion.
const PDLBarrierWait = `asm volatile("griddepcontrol.wait;":::"memory");`

// PDLBarrierLaunch is the PTX inline assembly trigger that unblocks
// dependent grid CTAs.
const PDLBarrierLaunch = `asm volatile("griddepcontrol.launch_dependents;":::);`

// KernelSplit represents a kernel partitioned into an independent preamble,
// a synchronization barrier, and a dependent body.
type KernelSplit struct {
	// Preamble contains independent sub-expressions that do not depend on
	// predecessor kernel output (e.g. weight loads, bias additions, constants, unpacks).
	Preamble []*Expr

	// Barrier is the synchronization barrier emitted between preamble and body.
	// Defaults to griddepcontrol.wait;.
	Barrier string

	// Body is the dependent computation consuming predecessor output.
	Body *Expr

	// IndependentInputs lists leaf variable names loaded or transformed during the preamble.
	IndependentInputs []string

	// DependentInputs lists variable names that depend on predecessor kernel completion.
	DependentInputs []string
}

// AST converts the split kernel into an AST Expr representation with Op "pdl_split"
// structured around the griddepcontrol_wait barrier.
func (ks *KernelSplit) AST() *Expr {
	var preambleNode *Expr
	switch len(ks.Preamble) {
	case 0:
		preambleNode = &Expr{Op: "pdl_nop"}
	case 1:
		preambleNode = ks.Preamble[0]
	default:
		preambleNode = &Expr{
			Op:       "pdl_preamble",
			Children: ks.Preamble,
		}
	}

	return &Expr{
		Op: "pdl_split",
		Children: []*Expr{
			preambleNode,
			&Expr{Op: "griddepcontrol_wait"},
			ks.Body,
		},
	}
}

// GenerateCUDA generates a complete CUDA C++ kernel partitioned into preamble,
// barrier (griddepcontrol.wait;), and dependent body.
func (ks *KernelSplit) GenerateCUDA(opts CodegenOptions) (string, error) {
	opts.EnablePDL = true
	return GenerateCUDA(ks.AST(), opts)
}

// GeneratePDLCUDA generates a complete, compilable CUDA C++ kernel partitioned into:
// (a) Preamble: executes independent loads/computations (weights, biases, constants).
// (b) Barrier: emits cudaGridDependencySynchronize() / griddepcontrol.wait;.
// (c) Dependent Body: consumes predecessor kernel output.
func GeneratePDLCUDA(expr *Expr, opts CodegenOptions) (string, error) {
	opts.EnablePDL = true
	return GenerateCUDA(expr, opts)
}

// SplitKernel partitions expr into an independent preamble and a dependent body.
// Variables matching depInputs (default: "x", "in", "input", "prev") are treated
// as arriving from the predecessor kernel.
//
// Independent inputs (e.g. weight loads, bias additions, constants, unpacks)
// are isolated into the Preamble so they execute concurrently before the
// griddepcontrol.wait barrier.
func SplitKernel(expr *Expr, depInputs ...string) (*KernelSplit, error) {
	if expr == nil {
		return nil, fmt.Errorf("egraph: nil expr passed to SplitKernel")
	}

	depSet := make(map[string]bool)
	if len(depInputs) == 0 {
		depSet["x"] = true
		depSet["in"] = true
		depSet["input"] = true
		depSet["prev"] = true
	} else {
		for _, d := range depInputs {
			depSet[d] = true
		}
	}

	// Handle existing pdl_split expression
	if expr.Op == "pdl_split" && len(expr.Children) >= 2 {
		preamble := []*Expr{expr.Children[0]}
		if expr.Children[0].Op == "pdl_preamble" {
			preamble = expr.Children[0].Children
		}
		body := expr.Children[len(expr.Children)-1]
		return &KernelSplit{
			Preamble:          preamble,
			Barrier:           PDLBarrierWait,
			Body:              body,
			IndependentInputs: collectLeafNames(&Expr{Children: preamble}),
			DependentInputs:   collectDependentLeaves(body, depSet),
		}, nil
	}

	// Handle pipeline expressions (e.g. pdl_gemv_pipeline ?x ?w1 ?w2)
	if expr.Op == "pdl_gemv_pipeline" && len(expr.Children) >= 3 {
		preamble := []*Expr{expr.Children[2]}
		body := &Expr{
			Op:       "gemv",
			Children: []*Expr{{Op: "prev"}, expr.Children[2]},
		}
		return &KernelSplit{
			Preamble:          preamble,
			Barrier:           PDLBarrierWait,
			Body:              body,
			IndependentInputs: collectLeafNames(expr.Children[2]),
			DependentInputs:   []string{"prev"},
		}, nil
	}

	preamble, deps, indeps := collectSplitExprs(expr, depSet)

	ks := &KernelSplit{
		Preamble:          preamble,
		Barrier:           PDLBarrierWait,
		Body:              expr,
		IndependentInputs: indeps,
		DependentInputs:   deps,
	}
	return ks, nil
}

// SplitPreambleBody is a convenience wrapper around SplitKernel returning
// the AST representation with Op "pdl_split".
func SplitPreambleBody(expr *Expr, depInputs ...string) (*Expr, error) {
	ks, err := SplitKernel(expr, depInputs...)
	if err != nil {
		return nil, err
	}
	return ks.AST(), nil
}

// isExprDependent reports whether expr or any of its descendants references
// any variable in depSet.
func isExprDependent(expr *Expr, depSet map[string]bool) bool {
	if expr == nil {
		return false
	}
	if len(expr.Children) == 0 {
		return depSet[expr.Op]
	}
	for _, c := range expr.Children {
		if isExprDependent(c, depSet) {
			return true
		}
	}
	return false
}

// collectSplitExprs extracts independent subtrees and leaf variable categories.
func collectSplitExprs(expr *Expr, depSet map[string]bool) (preamble []*Expr, deps []string, indeps []string) {
	if expr == nil {
		return nil, nil, nil
	}

	// If the entire expression does not depend on predecessor output,
	// it can execute in the preamble.
	if !isExprDependent(expr, depSet) {
		return []*Expr{expr}, nil, collectLeafNames(expr)
	}

	// Leaf dependent node
	if len(expr.Children) == 0 {
		return nil, []string{expr.Op}, nil
	}

	for _, c := range expr.Children {
		if !isExprDependent(c, depSet) {
			preamble = append(preamble, c)
			indeps = append(indeps, collectLeafNames(c)...)
		} else {
			p, d, ind := collectSplitExprs(c, depSet)
			preamble = append(preamble, p...)
			deps = append(deps, d...)
			indeps = append(indeps, ind...)
		}
	}

	return preamble, uniqueStrings(deps), uniqueStrings(indeps)
}

func collectLeafNames(expr *Expr) []string {
	if expr == nil {
		return nil
	}
	var leaves []string
	var walk func(e *Expr)
	walk = func(e *Expr) {
		if e == nil {
			return
		}
		if len(e.Children) == 0 {
			leaves = append(leaves, e.Op)
			return
		}
		for _, c := range e.Children {
			walk(c)
		}
	}
	walk(expr)
	return uniqueStrings(leaves)
}

func collectDependentLeaves(expr *Expr, depSet map[string]bool) []string {
	leaves := collectLeafNames(expr)
	var deps []string
	for _, l := range leaves {
		if depSet[l] {
			deps = append(deps, l)
		}
	}
	return uniqueStrings(deps)
}

func uniqueStrings(s []string) []string {
	if len(s) == 0 {
		return nil
	}
	seen := make(map[string]bool, len(s))
	var result []string
	for _, v := range s {
		if !seen[v] {
			seen[v] = true
			result = append(result, v)
		}
	}
	sort.Strings(result)
	return result
}

// PDLRules returns equality saturation rewrite rules for Programmatic
// Dependent Launch (PDL) pipelines and preamble/body splitting.
func PDLRules() []*Rewrite {
	return []*Rewrite{
		// 1. Sequential GEMVs with independent weight load preamble
		MustRewrite("pdl_overlap_gemv",
			"(gemv (gemv ?x ?w1) ?w2)",
			"(pdl_gemv_pipeline ?x ?w1 ?w2)"),

		// 2. Sequential GEMVs with quantized Q8_0 unpacking preamble
		MustRewrite("pdl_overlap_gemv_q8_0",
			"(gemv (gemv ?x ?w1) (ptx_unpack_q8_0 ?w2 ?scale))",
			"(pdl_gemv_pipeline ?x ?w1 (ptx_unpack_q8_0 ?w2 ?scale))"),

		// 3. Sequential GEMV with bias addition preamble
		MustRewrite("pdl_overlap_gemv_bias",
			"(add (gemv (gemv ?x ?w1) ?w2) ?bias)",
			"(pdl_pipeline (gemv ?x ?w1) (add ?w2 ?bias))"),

		// 4. RMSNorm followed by GEMV projection with weight fetch preamble
		MustRewrite("pdl_overlap_rmsnorm_gemv",
			"(gemv (rmsnorm ?x ?w_norm ?eps) ?w_proj)",
			"(pdl_pipeline (rmsnorm ?x ?w_norm ?eps) ?w_proj)"),
	}
}

// hasPDLPreambleWork reports whether a PDL node has independent preamble work
// (such as weight fetches, bias loads, unpacking, or dequantization) available
// to overlap before the grid dependency synchronization barrier.
func hasPDLPreambleWork(n ENode, childCosts []float64) bool {
	// Explicitly tagged as having no preamble or entry-syncing
	if strings.Contains(n.Op, "nopreamble") || strings.Contains(n.Op, "entry_sync") {
		return false
	}

	// For explicit split nodes: child 0 is the preamble
	if strings.HasPrefix(n.Op, "pdl_split") {
		return len(childCosts) >= 2 && childCosts[0] > 0
	}

	// For pipeline nodes: at least one operand after the primary input represents
	// independent preamble work (weight matrix, bias, constants, unpacking)
	// that can be preloaded/precomputed before waiting on the predecessor.
	if strings.HasPrefix(n.Op, "pdl_") {
		if len(n.Children) >= 2 {
			for i := 1; i < len(childCosts); i++ {
				if childCosts[i] > 0 {
					return true
				}
			}
		}
	}
	return false
}

// generatePDLKernel produces CUDA C++ source code partitioned into:
// (a) Preamble: executes independent loads / computations.
// (b) Barrier: emits cudaGridDependencySynchronize() / griddepcontrol.wait;.
// (c) Dependent Body: consumes predecessor output.
func generatePDLKernel(expr *Expr, opts CodegenOptions) string {
	scalarType := "float"
	if opts.Precision == "half" || opts.Precision == "float16" {
		scalarType = "uint16_t"
	}

	switch {
	case expr.Op == "pdl_gemv_pipeline":
		return fmt.Sprintf(`extern "C" __global__ void %s(
    const %s* __restrict__ x,
    const %s* __restrict__ W1,
    const %s* __restrict__ W2,
    %s* __restrict__ out,
    int in_dim,
    int hidden_dim,
    int out_dim
) {
    extern __shared__ float s_stage1[];

    // Stage 1: Producer GEMV: y = W1 * x
    int row1 = blockIdx.x * blockDim.x + threadIdx.x;
    if (row1 < hidden_dim) {
        float sum = 0.0f;
        for (int k = 0; k < in_dim; k++) {
            sum += (float)x[k] * (float)W1[row1 * in_dim + k];
        }
        s_stage1[row1] = sum;
    }
    __syncthreads();

    // Trigger launch of dependent grids
    cudaTriggerProgrammaticLaunchCompletion();

    // =========================================================================
    // (a) PDL Preamble: Independent Prologue Work (Stage 2 Weight Fetch)
    // Executes concurrently with predecessor CTA execution.
    // =========================================================================
    int row2 = blockIdx.x * blockDim.x + threadIdx.x;
    float w2_prefetch = 0.0f;
    if (row2 < out_dim && hidden_dim > 0) {
        w2_prefetch = (float)W2[row2 * hidden_dim];
    }

    // =========================================================================
    // (b) PDL Barrier: Wait for predecessor completion
    // =========================================================================
    cudaGridDependencySynchronize();
    asm volatile("griddepcontrol.wait;":::"memory");

    // =========================================================================
    // (c) PDL Dependent Body: Consumes stage 1 producer output
    // =========================================================================
    if (row2 < out_dim) {
        float acc = 0.0f;
        if (hidden_dim > 0) {
            acc = s_stage1[0] * w2_prefetch;
        }
        for (int k = 1; k < hidden_dim; k++) {
            acc += s_stage1[k] * (float)W2[row2 * hidden_dim + k];
        }
        out[row2] = acc;
    }
}
`, opts.KernelName, scalarType, scalarType, scalarType, scalarType)

	case expr.Op == "fused_swiglu_gemv" || expr.Op == "tile_swiglu" || expr.Op == "pdl_swiglu":
		return fmt.Sprintf(`extern "C" __global__ void %s(
    const %s* __restrict__ x,
    const %s* __restrict__ W_gate,
    const %s* __restrict__ W_up,
    const %s* __restrict__ W_down,
    %s* __restrict__ out,
    int hidden_dim,
    int intermediate_dim
) {
    extern __shared__ float s_act[];

    // =========================================================================
    // (a) PDL Preamble: Independent weight prefetch / indexing prologue
    // Executes independent loads before predecessor kernel completes.
    // =========================================================================
    int tid = threadIdx.x;
    float w_prefetch = 0.0f;
    if (tid < intermediate_dim && hidden_dim > 0) {
        w_prefetch = (float)W_gate[tid * hidden_dim];
    }

    // =========================================================================
    // (b) PDL Barrier: Synchronize on predecessor completion
    // =========================================================================
    cudaGridDependencySynchronize();
    asm volatile("griddepcontrol.wait;":::"memory");

    // =========================================================================
    // (c) PDL Dependent Body: Consumes predecessor output x
    // =========================================================================
    for (int j = threadIdx.x; j < intermediate_dim; j += blockDim.x) {
        float gate = (hidden_dim > 0) ? (float)x[0] * w_prefetch : 0.0f;
        float up = 0.0f;
        for (int k = (hidden_dim > 0 ? 1 : 0); k < hidden_dim; k++) {
            float xv = (float)x[k];
            gate += xv * (float)W_gate[j * hidden_dim + k];
            up += xv * (float)W_up[j * hidden_dim + k];
        }
        float silu = gate / (1.0f + expf(-gate));
        s_act[j] = silu * up;
    }
    __syncthreads();

    for (int out_idx = blockIdx.x * blockDim.x + threadIdx.x; out_idx < hidden_dim; out_idx += gridDim.x * blockDim.x) {
        float sum = 0.0f;
        for (int j = 0; j < intermediate_dim; j++) {
            sum += s_act[j] * (float)W_down[out_idx * intermediate_dim + j];
        }
        out[out_idx] = sum;
    }
}
`, opts.KernelName, scalarType, scalarType, scalarType, scalarType, scalarType)

	case expr.Op == "fused_rmsnorm_gemv" || expr.Op == "tile_rmsnorm_qkv" || expr.Op == "tile_rmsnorm_proj" || expr.Op == "pdl_rmsnorm_gemv":
		return fmt.Sprintf(`extern "C" __global__ void %s(
    const %s* __restrict__ x,
    const %s* __restrict__ W_norm,
    const %s* __restrict__ W_proj,
    %s* __restrict__ out,
    int hidden_dim,
    int proj_dim,
    float eps
) {
    // =========================================================================
    // (a) PDL Preamble: Independent weight and parameter loading
    // =========================================================================
    int tid = threadIdx.x;
    float w_norm_val = (tid < hidden_dim) ? (float)W_norm[tid] : 0.0f;

    // =========================================================================
    // (b) PDL Barrier: Synchronize on predecessor completion
    // =========================================================================
    cudaGridDependencySynchronize();
    asm volatile("griddepcontrol.wait;":::"memory");

    // =========================================================================
    // (c) PDL Dependent Body: RMSNorm reduction + GEMV projection on x
    // =========================================================================
    extern __shared__ float s_norm[];
    float sum_sq = 0.0f;
    for (int k = threadIdx.x; k < hidden_dim; k += blockDim.x) {
        float val = (float)x[k];
        sum_sq += val * val;
    }
    sum_sq = warp_sum(sum_sq);
    if (threadIdx.x == 0) {
        s_norm[0] = rsqrtf(sum_sq / (float)hidden_dim + eps);
    }
    __syncthreads();
    float rms_scale = s_norm[0];

    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < proj_dim) {
        float acc = 0.0f;
        for (int k = 0; k < hidden_dim; k++) {
            float normed_x = (float)x[k] * rms_scale * ((k == tid) ? w_norm_val : (float)W_norm[k]);
            acc += normed_x * (float)W_proj[row * hidden_dim + k];
        }
        out[row] = acc;
    }
}
`, opts.KernelName, scalarType, scalarType, scalarType, scalarType)

	case expr.Op == "q8_0_matvec" || expr.Op == "pdl_q8_0_matvec":
		return fmt.Sprintf(`extern "C" __global__ void %s(
    const block_q8_0* __restrict__ W,
    const float* __restrict__ x,
    float* __restrict__ y,
    int rows,
    int cols
) {
    // =========================================================================
    // (a) PDL Preamble: Fetch quantized block scales and unswizzle parameters
    // =========================================================================
    int warp_id = (blockDim.x * blockIdx.x + threadIdx.x) / WARP_SIZE;
    int lane = threadIdx.x %% WARP_SIZE;
    float scale_prefetch = 0.0f;
    if (warp_id < rows && cols >= 32) {
        scale_prefetch = half_to_float(W[warp_id * (cols / 32)].d);
    }

    // =========================================================================
    // (b) PDL Barrier: Synchronize on predecessor completion
    // =========================================================================
    cudaGridDependencySynchronize();
    asm volatile("griddepcontrol.wait;":::"memory");

    // =========================================================================
    // (c) PDL Dependent Body: Matrix-vector dot product with activation x
    // =========================================================================
    if (warp_id >= rows) return;
    int blocks_per_row = cols / 32;
    float row_sum = 0.0f;

    for (int b = 0; b < blocks_per_row; b++) {
        const block_q8_0* blk = &W[warp_id * blocks_per_row + b];
        float d = (b == 0) ? scale_prefetch : half_to_float(blk->d);
        int8_t q = blk->qs[lane];
        float xv = x[b * 32 + lane];
        float prod = (float)q * d * xv;
        row_sum += warp_sum(prod);
    }
    if (lane == 0) {
        y[warp_id] = row_sum;
    }
}
`, opts.KernelName)

	default:
		// Default partitioned GEMV / composite kernel with preamble and barrier
		return fmt.Sprintf(`extern "C" __global__ void %s(
    const %s* __restrict__ in_dep,
    const %s* __restrict__ in_indep,
    const %s* __restrict__ bias,
    %s* __restrict__ out,
    int rows,
    int cols
) {
    // =========================================================================
    // (a) PDL Preamble: Independent prologue loads & bias staging
    // Executes concurrently with predecessor CTA execution.
    // =========================================================================
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    float b_val = (bias != nullptr && idx < rows) ? (float)bias[idx] : 0.0f;
    float indep_val = (in_indep != nullptr && idx < rows) ? (float)in_indep[idx] : 0.0f;

    // =========================================================================
    // (b) PDL Barrier: Synchronize on predecessor grid completion
    // =========================================================================
    cudaGridDependencySynchronize();
    asm volatile("griddepcontrol.wait;":::"memory");

    // =========================================================================
    // (c) PDL Dependent Body: Consumes predecessor output in_dep
    // =========================================================================
    if (idx < rows) {
        float sum = b_val;
        if (cols <= 0) {
            sum += (float)in_dep[idx] + indep_val;
        } else {
            for (int k = 0; k < cols; k++) {
                sum += (float)in_dep[k] * (float)in_indep[idx * cols + k];
            }
        }
        out[idx] = sum;
    }
}
`, opts.KernelName, scalarType, scalarType, scalarType, scalarType)
	}
}
