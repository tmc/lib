package main

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"strconv"
	"strings"
)

// ExportedBundleArtifact represents a self-contained, cryptographically hashed
// e-graph kernel optimization bundle (.egbundle) and standalone CUDA C++
// reproducer (.cu) ready for 1-click deployment or offline qualification.
type ExportedBundleArtifact struct {
	FormatVersion        string                   `json:"format_version"`
	Arch                 string                   `json:"arch"`
	Provenance           string                   `json:"provenance"`
	KernelName           string                   `json:"kernel_name"`
	Iteration            int                      `json:"iteration"`
	Phase                string                   `json:"phase"`
	Promoted             bool                     `json:"promoted"`
	HoldoutSpeedup       float64                  `json:"holdout_speedup"`
	LatencyUs            float64                  `json:"latency_us"`
	MaxRegs              int                      `json:"max_regs"`
	TotalSpillBytes      int                      `json:"total_spill_bytes"`
	Verification         SimpleKernelVerification `json:"verification"`
	CompileCommand       string                   `json:"compile_command"`
	SHA256Hex            string                   `json:"sha256_hex"`
	CUDASource           string                   `json:"cuda_source"`
	StandaloneReproducer string                   `json:"standalone_reproducer"`
	Stages               []AnnotatedStageSummary  `json:"stages"`
}

func defaultExportArch(arch, iterArch string) string {
	if a := strings.TrimSpace(arch); a != "" {
		return a
	}
	if a := strings.TrimSpace(iterArch); a != "" {
		return a
	}
	return "sm_121a"
}

func defaultExportProvenance(provenance, arch string) string {
	if p := strings.TrimSpace(provenance); p != "" {
		return p
	}
	return fmt.Sprintf("static-verified + egraph-calibrated (%s)", arch)
}

func standaloneCompileCommand(iter int, arch string) string {
	fileBase := fmt.Sprintf("reproducer_cand_%02d", iter)
	return fmt.Sprintf("nvcc -O3 -arch=%s -lineinfo -Xptxas -v %s.cu -o %s", arch, fileBase, fileBase)
}

// RenderStandaloneCUReproducer renders a complete, zero-dependency, compilable
// CUDA C++ (.cu) source file containing the candidate kernel, its formal
// barrier verification certificate, PTXAS/SASS resource summary comments, and
// a standalone CUDA Graph capture + timing + numerical verification harness.
func RenderStandaloneCUReproducer(it KernelStreamIteration, arch string, provenance string) string {
	arch = defaultExportArch(arch, it.Arch)
	provenance = defaultExportProvenance(provenance, arch)
	ver := VerifySimpleKernelCandidate(it.CUDALines)
	compileCmd := standaloneCompileCommand(it.Iteration, arch)
	kernelName := strings.TrimSpace(it.KernelName)
	if kernelName == "" {
		kernelName = "fused_rmsnorm_swiglu_gemv"
	}

	var b strings.Builder
	fmt.Fprintf(&b, "// ============================================================================\n")
	fmt.Fprintf(&b, "// E-Graph Standalone Production Reproducer — Candidate #%02d (%s)\n", it.Iteration, it.Phase)
	fmt.Fprintf(&b, "// Build: %s\n", compileCmd)
	fmt.Fprintf(&b, "// ============================================================================\n")
	fmt.Fprintf(&b, "// Formal Barrier Verification Certificate (Passed=%t):\n", ver.Passed())
	fmt.Fprintf(&b, "//   RAWBarrierOK:      %t\n", ver.RAWBarrierOK)
	fmt.Fprintf(&b, "//   UniformBarrierOK:  %t\n", ver.UniformBarrierOK)
	fmt.Fprintf(&b, "//   AsyncWaitGroupOK:  %t\n", ver.AsyncWaitGroupOK)
	fmt.Fprintf(&b, "//   CrossWarpReduceOK: %t\n", ver.CrossWarpReduceOK)
	fmt.Fprintf(&b, "//   BankConflictRisk:  %t\n", ver.BankConflictRisk)
	if ver.ViolationReason != "" {
		fmt.Fprintf(&b, "//   ViolationReason:   %s\n", ver.ViolationReason)
	}
	fmt.Fprintf(&b, "//\n")
	fmt.Fprintf(&b, "// PTXAS & SASS Resource / Holdout Summary:\n")
	fmt.Fprintf(&b, "//   KernelName:        %s\n", kernelName)
	fmt.Fprintf(&b, "//   Arch:              %s\n", arch)
	fmt.Fprintf(&b, "//   Provenance:        %s\n", provenance)
	fmt.Fprintf(&b, "//   Promoted:          %t\n", it.Promoted)
	fmt.Fprintf(&b, "//   MaxRegs:           %d\n", it.MaxRegs)
	fmt.Fprintf(&b, "//   TotalSpillBytes:   %d\n", it.TotalSpillBytes)
	fmt.Fprintf(&b, "//   HoldoutSpeedup:    %.2fx\n", it.HoldoutSpeedup)
	fmt.Fprintf(&b, "//   LatencyUs:         %.2f us\n", it.LatencyUs)
	for _, st := range it.Stages {
		fmt.Fprintf(&b, "//   Stage s%d (%s): regs=%d shared=%dB spills=%dB wait_weight=%d l2=%s\n",
			st.StageIndex, st.StageName, st.RegistersPerThread, st.SharedBytes,
			st.SpillStoreBytes+st.SpillLoadBytes, st.WaitStallWeight, st.L2Policy)
	}
	fmt.Fprintf(&b, "// ============================================================================\n\n")

	b.WriteString("#include <cuda_runtime.h>\n")
	b.WriteString("#include <stdio.h>\n")
	b.WriteString("#include <stdlib.h>\n")
	b.WriteString("#include <math.h>\n")
	b.WriteString("#include <stdint.h>\n\n")

	b.WriteString("#define CHECK_CUDA(call) do { \\\n")
	b.WriteString("    cudaError_t err__ = (call); \\\n")
	b.WriteString("    if (err__ != cudaSuccess) { \\\n")
	b.WriteString("        fprintf(stderr, \"CUDA error at %s:%d: %s\\n\", __FILE__, __LINE__, cudaGetErrorString(err__)); \\\n")
	b.WriteString("        exit(EXIT_FAILURE); \\\n")
	b.WriteString("    } \\\n")
	b.WriteString("} while (0)\n\n")

	// Emit the candidate's 30-line CUDA kernel verbatim.
	for _, line := range it.CUDALines {
		b.WriteString(line)
		b.WriteByte('\n')
	}
	b.WriteByte('\n')

	// Emit standalone main() with CUDA Graph capture, 200 timed launches, and numerical check.
	fmt.Fprintf(&b, `int main(int argc, char** argv) {
    (void)argc;
    (void)argv;
    const int kThreads = 256;
    const int kHidden = 1024;          // 256 threads * float4
    const int kWeightElems = 256 * 1024; // 256 rows * 256 float4 columns
    const int kOutElems = 256;

    size_t bytes_x = (size_t)kHidden * sizeof(float);
    size_t bytes_w = (size_t)kHidden * sizeof(float);
    size_t bytes_W_up = (size_t)kWeightElems * sizeof(float);
    size_t bytes_out = (size_t)kOutElems * sizeof(float);

    float* h_x = (float*)malloc(bytes_x);
    float* h_w = (float*)malloc(bytes_w);
    float* h_W_up = (float*)malloc(bytes_W_up);
    float* h_out = (float*)malloc(bytes_out);
    if (!h_x || !h_w || !h_W_up || !h_out) {
        fprintf(stderr, "host allocation failed\n");
        return EXIT_FAILURE;
    }

    for (int i = 0; i < kHidden; ++i) {
        h_x[i] = 0.05f * sinf((float)i * 0.13f) + 0.1f;
        h_w[i] = 0.85f + 0.15f * cosf((float)i * 0.07f);
    }
    for (int i = 0; i < kWeightElems; ++i) {
        h_W_up[i] = 0.02f * cosf((float)(i %% 997) * 0.019f);
    }

    float *d_x = NULL, *d_w = NULL, *d_W_up = NULL, *d_out = NULL;
    CHECK_CUDA(cudaMalloc((void**)&d_x, bytes_x));
    CHECK_CUDA(cudaMalloc((void**)&d_w, bytes_w));
    CHECK_CUDA(cudaMalloc((void**)&d_W_up, bytes_W_up));
    CHECK_CUDA(cudaMalloc((void**)&d_out, bytes_out));

    CHECK_CUDA(cudaMemcpy(d_x, h_x, bytes_x, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_w, h_w, bytes_w, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_W_up, h_W_up, bytes_W_up, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(d_out, 0, bytes_out));

    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    // Capture kernel into a zero-host-update CUDA Graph executable.
    cudaGraph_t graph;
    cudaGraphExec_t graphExec;
    CHECK_CUDA(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    fused_rmsnorm_swiglu_gemv<<<1, kThreads, 0, stream>>>(d_x, d_w, d_W_up, d_out);
    CHECK_CUDA(cudaStreamEndCapture(stream, &graph));
    CHECK_CUDA(cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0));

    const int kWarmup = 20;
    const int kTimedIters = 200;
    for (int i = 0; i < kWarmup; ++i) {
        CHECK_CUDA(cudaGraphLaunch(graphExec, stream));
    }
    CHECK_CUDA(cudaStreamSynchronize(stream));

    cudaEvent_t evStart, evStop;
    CHECK_CUDA(cudaEventCreate(&evStart));
    CHECK_CUDA(cudaEventCreate(&evStop));

    CHECK_CUDA(cudaEventRecord(evStart, stream));
    for (int i = 0; i < kTimedIters; ++i) {
        CHECK_CUDA(cudaGraphLaunch(graphExec, stream));
    }
    CHECK_CUDA(cudaEventRecord(evStop, stream));
    CHECK_CUDA(cudaEventSynchronize(evStop));

    float elapsedMs = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&elapsedMs, evStart, evStop));
    double avgLatencyUs = ((double)elapsedMs * 1000.0) / (double)kTimedIters;
    double totalBytesPerLaunch = (double)(bytes_x + bytes_w + bytes_W_up + bytes_out);
    double effectiveGBps = (totalBytesPerLaunch / (avgLatencyUs * 1e-6)) / 1e9;

    CHECK_CUDA(cudaMemcpy(h_out, d_out, bytes_out, cudaMemcpyDeviceToHost));

    // Verify finite numerical output across all 256 output channels.
    int nonFinite = 0;
    double checksum = 0.0;
    for (int i = 0; i < kOutElems; ++i) {
        if (!isfinite(h_out[i])) {
            nonFinite++;
        } else {
            checksum += (double)h_out[i];
        }
    }
    if (nonFinite > 0) {
        fprintf(stderr, "FAIL: %%d non-finite output elements detected\n", nonFinite);
        return EXIT_FAILURE;
    }

    printf("cand_%02d (%s) arch=%s latency_us=%%.2f throughput_gbps=%%.2f checksum=%%.6f status=PASS\n",
           avgLatencyUs, effectiveGBps, checksum);

    CHECK_CUDA(cudaEventDestroy(evStart));
    CHECK_CUDA(cudaEventDestroy(evStop));
    CHECK_CUDA(cudaGraphExecDestroy(graphExec));
    CHECK_CUDA(cudaGraphDestroy(graph));
    CHECK_CUDA(cudaStreamDestroy(stream));
    CHECK_CUDA(cudaFree(d_x));
    CHECK_CUDA(cudaFree(d_w));
    CHECK_CUDA(cudaFree(d_W_up));
    CHECK_CUDA(cudaFree(d_out));
    free(h_x);
    free(h_w);
    free(h_W_up);
    free(h_out);
    return EXIT_SUCCESS;
}
`, it.Iteration, it.Phase, arch)

	return b.String()
}

// BuildExportedEGBundle constructs an [ExportedBundleArtifact] for a candidate
// iteration, computing the SHA-256 digest of its CUDA source, running static
// barrier and dataflow verification, and rendering its standalone .cu reproducer.
func BuildExportedEGBundle(it KernelStreamIteration, arch string, provenance string) (*ExportedBundleArtifact, error) {
	if len(it.CUDALines) == 0 {
		return nil, errors.New("export bundle: candidate has empty CUDALines")
	}
	arch = defaultExportArch(arch, it.Arch)
	provenance = defaultExportProvenance(provenance, arch)

	cudaSource := strings.Join(it.CUDALines, "\n")
	sum := sha256.Sum256([]byte(cudaSource))
	shaHex := hex.EncodeToString(sum[:])

	ver := VerifySimpleKernelCandidate(it.CUDALines)
	reproducer := RenderStandaloneCUReproducer(it, arch, provenance)

	kernelName := strings.TrimSpace(it.KernelName)
	if kernelName == "" {
		kernelName = "fused_rmsnorm_swiglu_gemv"
	}

	return &ExportedBundleArtifact{
		FormatVersion:        "egbundle.v1",
		Arch:                 arch,
		Provenance:           provenance,
		KernelName:           kernelName,
		Iteration:            it.Iteration,
		Phase:                it.Phase,
		Promoted:             it.Promoted,
		HoldoutSpeedup:       it.HoldoutSpeedup,
		LatencyUs:            it.LatencyUs,
		MaxRegs:              it.MaxRegs,
		TotalSpillBytes:      it.TotalSpillBytes,
		Verification:         ver,
		CompileCommand:       standaloneCompileCommand(it.Iteration, arch),
		SHA256Hex:            shaHex,
		CUDASource:           cudaSource,
		StandaloneReproducer: reproducer,
		Stages:               it.Stages,
	}, nil
}

// handleKernelsExport serves GET /v1/kernels/export with query parameters:
//   - iter: candidate iteration index (default: latest promoted Champion in simple trajectory)
//   - format: "cu"|"cuda" for standalone .cu reproducer, or "egbundle"|"json" (default)
func (s *EGServer) handleKernelsExport(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet && r.Method != http.MethodHead {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	iters, err := s.BuildKernelStreamIterations("simple", 1)
	if err != nil || len(iters) == 0 {
		http.Error(w, "failed to build kernel iterations", http.StatusInternalServerError)
		return
	}

	// Default to the latest promoted Champion in the trajectory.
	selectedIdx := len(iters) - 1
	for i := len(iters) - 1; i >= 0; i-- {
		if iters[i].Promoted {
			selectedIdx = i
			break
		}
	}

	q := r.URL.Query()
	if iterStr := strings.TrimSpace(q.Get("iter")); iterStr != "" {
		idx, parseErr := strconv.Atoi(iterStr)
		if parseErr != nil || idx < 0 || idx >= len(iters) {
			http.Error(w, fmt.Sprintf("invalid iter %q (must be 0..%d)", iterStr, len(iters)-1), http.StatusBadRequest)
			return
		}
		selectedIdx = idx
	}

	s.liveMu.RLock()
	prov := s.liveProvenance
	s.liveMu.RUnlock()
	if prov == "" {
		prov = defaultExportProvenance("", s.cfg.Arch)
	}

	bundle, err := BuildExportedEGBundle(iters[selectedIdx], s.cfg.Arch, prov)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	format := strings.ToLower(strings.TrimSpace(q.Get("format")))
	switch format {
	case "cu", "cuda":
		filename := fmt.Sprintf("reproducer_cand_%02d.cu", bundle.Iteration)
		w.Header().Set("Content-Type", "text/x-cuda; charset=utf-8")
		w.Header().Set("Content-Disposition", fmt.Sprintf("attachment; filename=%q", filename))
		_, _ = w.Write([]byte(bundle.StandaloneReproducer))
	default:
		filename := fmt.Sprintf("cand_%02d.egbundle.json", bundle.Iteration)
		w.Header().Set("Content-Type", "application/json")
		w.Header().Set("Content-Disposition", fmt.Sprintf("attachment; filename=%q", filename))
		enc := json.NewEncoder(w)
		enc.SetIndent("", "  ")
		_ = enc.Encode(bundle)
	}
}
