# Design Document: Closed-Loop Adaptive Profile-Guided Optimization (PGO) for Pure-Go GPU Inference

**Status:** Draft for Peer Review  
**Target Subsystems:** `apple/x/metal/egraph`, `apple/x/metal/metalprof`, `nvidia/x/egraph`, `nvidia/x/cudaprof`, `examples/llama-infer`  
**Authors:** tmc/lib Architecture Team  

---

## 1. Executive Summary & Core Problem

Existing high-performance inference runtimes (e.g. vLLM, TensorRT-LLM, PyTorch Inductor) rely almost exclusively on **Static Ahead-Of-Time (AOT) Compilation and Tuning**. Prior to deployment, they run heavy offline search scripts for specific batch sizes and hardware architectures, generating frozen compute graphs.

When deployed in dynamic production environments, static engines suffer from fundamental performance cliffs:
1. **The Batch-Size Crossover ($B=1 \to B=32$)**: At $B=1$ (single-token decode), memory bandwidth dominates; a custom fused GEMV + in-register activation kernel is optimal. At $B \ge 16$, compute dominates; discrete matrix-multiplication kernels utilizing hardware matrix cores (MPS / Tensor Cores) surpass fused memory-streaming kernels.
2. **Context Horizon Expansion ($S=128 \to S=32\text{k}$)**: Short contexts are dominated by FFN execution ($<10\%$ attention time); long contexts shift $>60\%$ of execution time into attention and KV-cache reads, requiring sliding window or chunked tree-attention kernels.
3. **Speculative Decoding Acceptance Volatility**: High acceptance rates favor parallel multi-token tree verification; low acceptance rates favor low-overhead greedy decode.
4. **Thermal Throttling & DVFS Frequency Modulation**: When mobile or workstation GPUs thermally throttle, the ratio of compute throughput to memory bandwidth shifts dynamically.

### The Pure-Go Closed-Loop Solution
Because `tmc/lib` is built on a pure-Go architecture with dynamic JIT compilation (`metal.CompileLibrary` on Apple Silicon, `nvrtcx.Compile` on NVIDIA), JIT-compiling a specialized compute shader takes **$<5\text{ ms}$** in background memory without touching disk or invoking heavy external C++ toolchains.

By closing the loop between our **E-Graph Equality Saturation Optimizer (`x/egraph`)** and our **Low-Overhead GPU Hardware Telemetry (`metalprof` / `cudaprof`)**, we can create an **Autonomous, Self-Healing Dynamic Optimization Runtime** that re-evaluates execution graphs live and hot-swaps optimal GPU kernels with zero stall bubbles.

---

## 2. System Architecture & The Feedback Loop

```mermaid
graph TD
    subgraph "1. GPU Inference Fast-Path (0-Overhead)"
        A["Inference Step (Prefill / Decode)"] --> B["Command Encoder Dispatch"]
        B --> C["FlightRecorder (Hardware Timestamps & Dimensions)"]
    end

    subgraph "2. Continuous Telemetry & Shift Detection"
        C --> D{"Workload Regime Shift Detected?"}
        D -->|No: Latency on Budget| A
        D -->|Yes: (Batch, SeqLen, Drift)| E["Adaptive Optimizer Controller"]
    end

    subgraph "3. E-Graph Pareto Extraction & Background JIT"
        E --> F["E-Graph: Extract Top-K Structurally Distinct ASTs"]
        F --> G["Asynchronous JIT Compiler (MSL / PTX in Background)"]
        G --> H["Warmup Micro-Benchmark (metalprof.SampleBuffer)"]
        H --> I["PGO Calibration Cache (Device, Batch, SeqLen)"]
    end

    subgraph "4. Atomic Hot-Swap"
        I --> J["Atomic Pointer Swap (ModelEngine.activePipeline)"]
        J --> A
    end
```

---

## 3. Detailed Subsystem Specifications

### 3.1 E-Graph Pareto Subgraph Extractor (`TopKExtractor`)

**Limitation in Current Design:**  
The existing `ExtractBest(g, root, costModel)` computes a single AST based on static heuristic scalar costs (e.g. `fused_silu_mul = 1.0`, `matmul = 12.0`). If the heuristic misjudges register spilling or memory bandwidth limits, the optimizer gets locked into the wrong graph.

**Proposed Enhancement (`TopKExtractor`):**  
Extract the top-$K$ structurally distinct expressions from the root E-Class:
1. **Candidate 1 (Full Fusion Megakernel)**: Fused RMSNorm + GEMV + RoPE + Attention + SwiGLU + Residual.
2. **Candidate 2 (3-Stage Discrete Pipeline)**: RMSNorm+GEMV $\to$ Discrete FlashAttention $\to$ Fused SwiGLU FFN.
3. **Candidate 3 (Tensor/Matrix Core Hybrid)**: MPS/cuBLASLt Matrix Multiplication + Fused Epilogue Activations.

```go
type CandidateAST struct {
    Expr         *Expr
    StructuralID string
    HeuristicCost float64
}

// ExtractTopK returns the K lowest-cost structurally unique ASTs for an E-Class.
func (ext *TopKExtractor) ExtractTopK(root Id, k int) ([]CandidateAST, error)
```

---

### 3.2 Hardware Telemetry Bridging (`metalprof` & `cudaprof`)

The profiling packages provide the exact empirical ground truth needed to calibrate the E-Graph:

1. **In-Flight Duration & Hardware Counters**:
   - [`metalprof.SampleBuffer`](file:///Users/tmc/go/src/github.com/tmc/lib/apple/x/metal/metalprof/counters.go): Resolves exact GPU timestamps per dispatch without CPU stalls.
   - [`metalprof.FlightRecorder`](file:///Users/tmc/go/src/github.com/tmc/lib/apple/x/metal/metalprof/flightrecorder.go): Tracks rolling execution latency and dimensions in a cyclic ring buffer.
2. **pprof Context Label Integration**:
   - `pprof.Labels` (`model`, `layer`, `batch_size`, `seq_len`) are bound to GPU execution events, allowing the tuner to attribute latency directly to specific operator nodes in the graph.

---

### 3.3 Dynamic PGO Calibration Table (`CalibrationCache`)

The runtime maintains a thread-safe, content-addressed calibration cache:

```go
type WorkloadSignature struct {
    DeviceIdentifier string // e.g. "Apple M4 Max / 128GB"
    BatchSize        int    // e.g. 1, 4, 16, 32
    SeqLenBucket     int    // e.g. 512, 2048, 8192, 32768
    HiddenDim        int    // e.g. 2048, 4096
    DType            string // "F16", "BF16", "MXFP4"
}

type CalibrationCache struct {
    mu      sync.RWMutex
    winners map[WorkloadSignature]*CompiledPipeline
}
```

- **Lookup ($O(1)$)**: Fast-path execution queries the cache in $<50\text{ ns}$ using read-locks or atomic maps.
- **Cache Miss**: If a novel workload signature appears (e.g. batch size expands from 1 to 16 for the first time), the engine immediately uses the heuristic default graph, while queuing a background calibration task.

---

### 3.4 Concurrency & Atomic Hot-Swapping

To maintain zero stalls on the active inference stream:

```go
type ModelEngine struct {
    // Active pipeline states accessed by concurrent forward passes
    activePipeline atomic.Pointer[CompiledPipeline]
    
    // Background tuner queue
    tuningQueue chan WorkloadSignature
    recorder    *metalprof.FlightRecorder
}

func (eng *ModelEngine) DecodeStep(tokenID int32, pos int) (time.Duration, error) {
    pipe := eng.activePipeline.Load()
    
    // Execute current pipeline on GPU command queue
    dur, err := pipe.Dispatch(tokenID, pos)
    
    // Record telemetry into ring buffer
    eng.recorder.Record(pipe.Name, dur, pipe.Grid, pipe.TG, nil)
    
    return dur, err
}
```

1. Background goroutine JIT-compiles new candidate shaders using `metal.CompileLibrary`.
2. Evaluates candidates in a private scratch command buffer.
3. Once verified, atomically updates `activePipeline.Store(winner)`.
4. Subsequent decode iterations seamlessly execute the new optimal kernel without synchronization locks or GC pauses.

---

## 4. Specific Discussion Points for Peer Review

We solicit feedback and peer evaluation on the following design choices:

1. **Top-K Extraction Complexity**:
   - *Question:* What is the optimal value of $K$? Is $K=3\dots 5$ sufficient to capture the Pareto frontier of full-fusion vs. discrete vs. tensor-core matrix multiplication without bloating JIT memory?
2. **Warmup & Micro-Benchmarking Overhead**:
   - *Question:* Should candidate micro-benchmarking be performed during server startup / warm-up prompts, or opportunistically interleaved into idle GPU intervals between user requests?
3. **Thresholds for Dynamic Re-Optimization**:
   - *Question:* How should we parameterize drift detection? (e.g. $\ge 20\%$ latency increase over 5 consecutive steps, or strict bucketing on batch size $\Delta \ge 4$ and sequence length $2\times$ boundaries)?
4. **Cross-Platform Parity**:
   - *Question:* Can the exact same `TopKExtractor` and `CalibrationCache` structs be shared between `apple/x/metal` and `nvidia/x/cuda` via a unified `x/egraph` package?

---

## 5. Reviewer Sign-Off & Verdict Template

```markdown
### Review Feedback
- **Reviewer:** [Session ID / Name]
- **Verdict:** [GREEN / AMENDMENT / RED]
- **Comments on Section 3 (Architecture):**
- **Comments on Section 4 (Questions):**
```
