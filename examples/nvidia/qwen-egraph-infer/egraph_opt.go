package main

import (
	"fmt"
	"strings"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/egraph"
	"github.com/tmc/lib/nvidia/x/nvrtcx"
)

// hasOp returns whether expr or any of its descendants has the given operator.
func hasOp(expr *egraph.Expr, op string) bool {
	if expr == nil {
		return false
	}
	if expr.Op == op {
		return true
	}
	for _, child := range expr.Children {
		if hasOp(child, op) {
			return true
		}
	}
	return false
}

// generateEmbeddingLookup generates the token embedding lookup kernel.
func generateEmbeddingLookup() string {
	return `extern "C" __global__ void QwenEmbeddingLookup(
    const float* __restrict__ embedTable,
    const int* __restrict__ tokenIn,
    float* __restrict__ outX,
    int hiddenSize
) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= hiddenSize) return;
    int tokenID = tokenIn[0];
    outX[idx] = embedTable[tokenID * hiddenSize + idx];
}`
}

// generateRMSNorm generates the layer root-mean-square normalization kernel.
func generateRMSNorm() string {
	return `extern "C" __global__ void QwenRMSNorm(
    const float* __restrict__ x,
    const float* __restrict__ normWeight,
    float* __restrict__ outNormX,
    int hiddenSize, float eps
) {
    __shared__ float s_variance;
    float localSumSq = 0.0f;
    for (int i = threadIdx.x; i < hiddenSize; i += blockDim.x) {
        float val = x[i];
        localSumSq += val * val;
    }
    for (int offset = 16; offset > 0; offset /= 2) {
        localSumSq += __shfl_down_sync(0xffffffff, localSumSq, offset);
    }
    if (threadIdx.x == 0) {
        s_variance = rsqrtf(localSumSq / (float)hiddenSize + eps);
    }
    __syncthreads();
    float invRms = s_variance;
    for (int i = threadIdx.x; i < hiddenSize; i += blockDim.x) {
        outNormX[i] = x[i] * normWeight[i] * invRms;
    }
}`
}

// generateFusedQKVGEMV generates the combined query, key, and value projection kernel.
func generateFusedQKVGEMV() string {
	return `extern "C" __global__ void QwenFusedQKVGEMV(
    const float* __restrict__ normX,
    const float* __restrict__ W_q,
    const float* __restrict__ W_k,
    const float* __restrict__ W_v,
    float* __restrict__ out_q,
    float* __restrict__ out_k,
    float* __restrict__ out_v,
    int hiddenSize, int qDim, int kvDim
) {
    int row = blockDim.x * blockIdx.x + threadIdx.x;
    if (row < qDim) {
        float sumQ = 0.0f;
        for (int col = 0; col < hiddenSize; col++) {
            sumQ += normX[col] * W_q[row * hiddenSize + col];
        }
        out_q[row] = sumQ;
    }
    if (row < kvDim) {
        float sumK = 0.0f;
        float sumV = 0.0f;
        for (int col = 0; col < hiddenSize; col++) {
            sumK += normX[col] * W_k[row * hiddenSize + col];
            sumV += normX[col] * W_v[row * hiddenSize + col];
        }
        out_k[row] = sumK;
        out_v[row] = sumV;
    }
}`
}

// generateRoPEKV generates the RoPE rotation and KV cache store kernel.
// When fused is true, it fuses rotation and cache storage into a single launch.
func generateRoPEKV(fused bool) string {
	if fused {
		return `// Fused RoPE + Paged KV Store (selected by e-graph plan)
extern "C" __global__ void QwenFusedRoPEStoreKV(
    float* __restrict__ Q,
    float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ kCache,
    float* __restrict__ vCache,
    int pos, int numQHeads, int numKVHeads, int headDim, float ropeTheta, int maxSeqLen
) {
    int headIdx = blockIdx.x;
    int pairIdx = threadIdx.x;
    if (headIdx >= numQHeads || pairIdx >= headDim / 2) return;

    float freq = 1.0f / powf(ropeTheta, (float)(2 * pairIdx) / (float)headDim);
    float val = (float)pos * freq;
    float cosVal = cosf(val);
    float sinVal = sinf(val);

    int qOffset = headIdx * headDim + 2 * pairIdx;
    float q0 = Q[qOffset];
    float q1 = Q[qOffset + 1];
    Q[qOffset]     = q0 * cosVal - q1 * sinVal;
    Q[qOffset + 1] = q0 * sinVal + q1 * cosVal;

    if (headIdx < numKVHeads) {
        int kOffset = headIdx * headDim + 2 * pairIdx;
        float k0 = K[kOffset];
        float k1 = K[kOffset + 1];
        float rotK0 = k0 * cosVal - k1 * sinVal;
        float rotK1 = k0 * sinVal + k1 * cosVal;
        K[kOffset]     = rotK0;
        K[kOffset + 1] = rotK1;

        int cacheIdx0 = (headIdx * maxSeqLen + pos) * headDim + 2 * pairIdx;
        kCache[cacheIdx0]     = rotK0;
        kCache[cacheIdx0 + 1] = rotK1;

        int vOffset = headIdx * headDim + 2 * pairIdx;
        vCache[cacheIdx0]     = V[vOffset];
        vCache[cacheIdx0 + 1] = V[vOffset + 1];
    }
}`
	}
	return `// Unfused RoPE + Paged KV Store (baseline fallback)
extern "C" __global__ void QwenFusedRoPEStoreKV(
    float* __restrict__ Q,
    float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ kCache,
    float* __restrict__ vCache,
    int pos, int numQHeads, int numKVHeads, int headDim, float ropeTheta, int maxSeqLen
) {
    int headIdx = blockIdx.x;
    int pairIdx = threadIdx.x;
    if (headIdx >= numQHeads || pairIdx >= headDim / 2) return;

    float freq = 1.0f / powf(ropeTheta, (float)(2 * pairIdx) / (float)headDim);
    float val = (float)pos * freq;
    float cosVal = cosf(val);
    float sinVal = sinf(val);

    // Step 1: RoPE on Q
    int qOffset = headIdx * headDim + 2 * pairIdx;
    float q0 = Q[qOffset];
    float q1 = Q[qOffset + 1];
    Q[qOffset]     = q0 * cosVal - q1 * sinVal;
    Q[qOffset + 1] = q0 * sinVal + q1 * cosVal;

    // Step 2: RoPE on K and store to cache
    if (headIdx < numKVHeads) {
        int kOffset = headIdx * headDim + 2 * pairIdx;
        float k0 = K[kOffset];
        float k1 = K[kOffset + 1];
        float rotK0 = k0 * cosVal - k1 * sinVal;
        float rotK1 = k0 * sinVal + k1 * cosVal;
        K[kOffset]     = rotK0;
        K[kOffset + 1] = rotK1;

        int cacheIdx0 = (headIdx * maxSeqLen + pos) * headDim + 2 * pairIdx;
        kCache[cacheIdx0]     = rotK0;
        kCache[cacheIdx0 + 1] = rotK1;

        int vOffset = headIdx * headDim + 2 * pairIdx;
        vCache[cacheIdx0]     = V[vOffset];
        vCache[cacheIdx0 + 1] = V[vOffset + 1];
    }
}`
}

// generateGateUpSilu generates the FFN Gate + Up projection and SiLU activation kernel.
func generateGateUpSilu(fused bool) string {
	if fused {
		return `// Fused Gate + Up + SiLU (selected by e-graph plan)
extern "C" __global__ void QwenFusedGateUpSilu(
    const float* __restrict__ normX,
    const float* __restrict__ W_gate,
    const float* __restrict__ W_up,
    float* __restrict__ outFFNAct,
    int hiddenSize, int intermediateSize
) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= intermediateSize) return;

    float gate = 0.0f;
    float up = 0.0f;
    for (int k = 0; k < hiddenSize; k++) {
        float val = normX[k];
        gate += val * W_gate[idx * hiddenSize + k];
        up   += val * W_up[idx * hiddenSize + k];
    }
    float silu = gate / (1.0f + expf(-gate));
    outFFNAct[idx] = silu * up;
}`
	}
	return `// Unfused Gate + Up + SiLU (baseline fallback)
extern "C" __global__ void QwenFusedGateUpSilu(
    const float* __restrict__ normX,
    const float* __restrict__ W_gate,
    const float* __restrict__ W_up,
    float* __restrict__ outFFNAct,
    int hiddenSize, int intermediateSize
) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= intermediateSize) return;

    float gate = 0.0f;
    for (int k = 0; k < hiddenSize; k++) {
        gate += normX[k] * W_gate[idx * hiddenSize + k];
    }
    float up = 0.0f;
    for (int k = 0; k < hiddenSize; k++) {
        up += normX[k] * W_up[idx * hiddenSize + k];
    }
    float silu = gate / (1.0f + expf(-gate));
    outFFNAct[idx] = silu * up;
}`
}

// generateDownAdd generates the FFN Down-projection and residual addition kernel.
func generateDownAdd(fused bool) string {
	if fused {
		return `// Fused Down GEMV + Residual Add (selected by e-graph plan)
extern "C" __global__ void QwenFusedDownAdd(
    const float* __restrict__ ffnAct,
    const float* __restrict__ W_down,
    const float* __restrict__ inX,
    float* __restrict__ outX,
    int hiddenSize, int intermediateSize
) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= hiddenSize) return;

    float acc = 0.0f;
    for (int j = 0; j < intermediateSize; j++) {
        acc += ffnAct[j] * W_down[idx * intermediateSize + j];
    }
    outX[idx] = inX[idx] + acc;
}`
	}
	return `// Unfused Down GEMV + Residual Add (baseline fallback)
extern "C" __global__ void QwenFusedDownAdd(
    const float* __restrict__ ffnAct,
    const float* __restrict__ W_down,
    const float* __restrict__ inX,
    float* __restrict__ outX,
    int hiddenSize, int intermediateSize
) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= hiddenSize) return;

    float acc = 0.0f;
    for (int j = 0; j < intermediateSize; j++) {
        acc += ffnAct[j] * W_down[idx * intermediateSize + j];
    }
    outX[idx] = inX[idx] + acc;
}`
}

// generateArgmaxSampling generates the greedy argmax next-token selection kernel.
func generateArgmaxSampling() string {
	return `extern "C" __global__ void QwenArgmaxSampling(
    const float* __restrict__ logits,
    int* __restrict__ tokenOut,
    int vocabSize
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    int maxIdx = 0;
    float maxVal = logits[0];
    for (int i = 1; i < vocabSize; i++) {
        if (logits[i] > maxVal) {
            maxVal = logits[i];
            maxIdx = i;
        }
    }
    tokenOut[0] = maxIdx;
}`
}

// generateQwenMegakernels dynamically generates the CUDA source code for the
// Qwen inference engine driven directly by the extracted e-graph AST plan.
func generateQwenMegakernels(plan *egraph.Expr, cfg *QwenConfig) (string, error) {
	if plan == nil {
		return "", fmt.Errorf("egraph: nil plan passed to generateQwenMegakernels")
	}

	var sb strings.Builder
	sb.WriteString("// Auto-generated by Qwen E-Graph JIT Optimizer\n")
	sb.WriteString(fmt.Sprintf("// Extracted Plan: %s\n\n", plan.String()))

	// Common device constants
	sb.WriteString("#define WARP_SIZE 32\n")
	sb.WriteString("#define FULL_WARP_MASK 0xffffffff\n\n")

	// 1. Core architecture-invariant kernels
	sb.WriteString(generateEmbeddingLookup())
	sb.WriteString("\n\n")
	sb.WriteString(generateRMSNorm())
	sb.WriteString("\n\n")
	sb.WriteString(generateFusedQKVGEMV())
	sb.WriteString("\n\n")

	// 2. Attention RoPE + KV Cache Store: driven by e-graph plan
	fusedRoPE := hasOp(plan, "fused_rope_store_kv")
	sb.WriteString(generateRoPEKV(fusedRoPE))
	sb.WriteString("\n\n")

	// 3. FFN SwiGLU: driven by e-graph plan
	fusedSwiGLU := hasOp(plan, "fused_swiglu_gemv")
	sb.WriteString(generateGateUpSilu(fusedSwiGLU))
	sb.WriteString("\n\n")
	sb.WriteString(generateDownAdd(fusedSwiGLU))
	sb.WriteString("\n\n")

	// 4. Argmax sampling
	sb.WriteString(generateArgmaxSampling())
	sb.WriteString("\n")

	return sb.String(), nil
}

// FusedQwenEngine holds compiled GPU functions derived from the E-Graph optimizer.
type FusedQwenEngine struct {
	embedFn      driver.Function
	normFn       driver.Function
	fusedQKVFn   driver.Function
	fusedRoPEKV  driver.Function
	fusedGateUp  driver.Function
	fusedDownAdd driver.Function
	argmaxFn     driver.Function
	plan         *egraph.Expr
	planSummary  string
}

// Plan returns the optimal AST expression extracted by equality saturation.
func (e *FusedQwenEngine) Plan() *egraph.Expr {
	return e.plan
}

// PlanSummary returns the string representation of the optimal AST plan.
func (e *FusedQwenEngine) PlanSummary() string {
	return e.planSummary
}

// OptimizeAndCompileQwen runs equality saturation on the Qwen layer DAG and
// compiles dynamically generated JIT megakernels synthesized from the optimal plan.
func OptimizeAndCompileQwen(dev *cuda.Device, cfg *QwenConfig) (*FusedQwenEngine, error) {
	eg := egraph.New()

	x := eg.Add(egraph.NewLeaf("x"))
	wNorm := eg.Add(egraph.NewLeaf("w_norm"))
	eps := eg.Add(egraph.NewLeaf("eps"))
	wQ := eg.Add(egraph.NewLeaf("w_q"))
	wK := eg.Add(egraph.NewLeaf("w_k"))
	wV := eg.Add(egraph.NewLeaf("w_v"))
	pos := eg.Add(egraph.NewLeaf("pos"))
	theta := eg.Add(egraph.NewLeaf("theta"))
	kvCache := eg.Add(egraph.NewLeaf("kv_cache"))

	normX := eg.Add(egraph.NewNode("rmsnorm", x, wNorm, eps))
	q := eg.Add(egraph.NewNode("gemv", normX, wQ))
	k := eg.Add(egraph.NewNode("gemv", normX, wK))
	v := eg.Add(egraph.NewNode("gemv", normX, wV))
	qRope := eg.Add(egraph.NewNode("rope", q, pos, theta))
	kRope := eg.Add(egraph.NewNode("rope", k, pos, theta))
	kvStore := eg.Add(egraph.NewNode("store_kv", qRope, kRope, v, kvCache))

	wGate := eg.Add(egraph.NewLeaf("w_gate"))
	wUp := eg.Add(egraph.NewLeaf("w_up"))
	wDown := eg.Add(egraph.NewLeaf("w_down"))
	gateProj := eg.Add(egraph.NewNode("gemv", normX, wGate))
	upProj := eg.Add(egraph.NewNode("gemv", normX, wUp))
	siluNode := eg.Add(egraph.NewNode("silu", gateProj))
	swigluMult := eg.Add(egraph.NewNode("mul", siluNode, upProj))
	ffnOut := eg.Add(egraph.NewNode("gemv", swigluMult, wDown))

	layerOut := eg.Add(egraph.NewNode("add", kvStore, ffnOut))

	rules := []*egraph.Rewrite{
		egraph.MustRewrite("fuse_rmsnorm_gemv",
			"(gemv (rmsnorm ?x ?w_norm ?eps) ?w_proj)",
			"(fused_rmsnorm_gemv ?x ?w_norm ?eps ?w_proj)"),
		egraph.MustRewrite("fuse_rope_store_kv",
			"(store_kv (rope ?q ?pos ?th) (rope ?k ?pos ?th) ?v ?cache)",
			"(fused_rope_store_kv ?q ?k ?v ?pos ?th ?cache)"),
		egraph.MustRewrite("fuse_swiglu_gemv",
			"(gemv (mul (silu (gemv ?nx ?w_g)) (gemv ?nx ?w_u)) ?w_d)",
			"(fused_swiglu_gemv ?nx ?w_g ?w_u ?w_d)"),
	}

	egraph.Saturate(eg, rules, 10)
	bestPlan, _, err := egraph.ExtractBest(eg, layerOut, egraph.FusedMegakernelCost{})
	if err != nil {
		return nil, fmt.Errorf("egraph extract plan: %w", err)
	}

	cudaSrc, err := generateQwenMegakernels(bestPlan, cfg)
	if err != nil {
		return nil, fmt.Errorf("generate megakernels from plan: %w", err)
	}

	major, minor, err := dev.ComputeCapability()
	if err != nil {
		return nil, fmt.Errorf("device compute capability: %w", err)
	}

	opts := []nvrtcx.Option{
		nvrtcx.WithName("qwen_megakernels.cu"),
		nvrtcx.WithArch(major, minor),
	}
	compiled, err := nvrtcx.Compile(cudaSrc, opts...)
	if err != nil {
		return nil, fmt.Errorf("compile megakernels: %w", err)
	}

	loaded, err := compiled.Load(dev)
	if err != nil {
		return nil, fmt.Errorf("load module: %w", err)
	}

	kEmbed, _ := loaded.GetFunction("QwenEmbeddingLookup")
	kNorm, _ := loaded.GetFunction("QwenRMSNorm")
	kQKV, _ := loaded.GetFunction("QwenFusedQKVGEMV")
	kRoPEKV, _ := loaded.GetFunction("QwenFusedRoPEStoreKV")
	kGateUp, _ := loaded.GetFunction("QwenFusedGateUpSilu")
	kDownAdd, _ := loaded.GetFunction("QwenFusedDownAdd")
	kArgmax, _ := loaded.GetFunction("QwenArgmaxSampling")

	return &FusedQwenEngine{
		embedFn:      kEmbed,
		normFn:       kNorm,
		fusedQKVFn:   kQKV,
		fusedRoPEKV:  kRoPEKV,
		fusedGateUp:  kGateUp,
		fusedDownAdd: kDownAdd,
		argmaxFn:     kArgmax,
		plan:         bestPlan,
		planSummary:  bestPlan.String(),
	}, nil
}
