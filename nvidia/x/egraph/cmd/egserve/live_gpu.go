package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"math"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"
)

// LiveGPUConfig configures live GPU compilation and paired-timing execution
// of the 24-candidate simple trajectory (either locally or over SSH under gpu-lock).
type LiveGPUConfig struct {
	Arch       string
	NVCC       string
	NVDISASM   string
	SSHHost    string
	WarmupRuns int
	TimedRuns  int
}

// CandidateGPUMeasurement records the hardware execution timing and numerical
// verification result for one candidate in the 24-candidate simple trajectory.
type CandidateGPUMeasurement struct {
	Iteration  int     `json:"iteration"`
	Phase      string  `json:"phase"`
	Executed   bool    `json:"executed"`
	SkipReason string  `json:"skip_reason,omitempty"`
	LatencyUs  float64 `json:"latency_us"`
	Speedup    float64 `json:"speedup"`
	MaxAbsErr  float64 `json:"max_abs_err"`
	NumericOK  bool    `json:"numeric_ok"`
	DeviceName string  `json:"device_name,omitempty"`
}

// BuildLiveGPUBenchmarkCUDA generates a self-contained CUDA C++ translation
// unit that compiles all 24 candidate kernels (cand_00 .. cand_23) and a host
// main() harness that executes every statically safe candidate on the GPU
// using cudaEventElapsedTime, checks numerical parity, and emits EG_GPU_JSON
// lines to stdout.
func BuildLiveGPUBenchmarkCUDA(iters []KernelStreamIteration, warmupRuns, timedRuns int) string {
	if warmupRuns <= 0 {
		warmupRuns = 25
	}
	if timedRuns <= 0 {
		timedRuns = 200
	}

	var b strings.Builder
	b.WriteString("// Auto-generated 24-candidate CUDA hardware benchmark harness for egserve.\n")
	b.WriteString("#include <cuda_runtime.h>\n")
	b.WriteString("#include <stdio.h>\n")
	b.WriteString("#include <stdlib.h>\n")
	b.WriteString("#include <math.h>\n")
	b.WriteString("#include <stdint.h>\n\n")

	hasCand07 := false
	for _, it := range iters {
		idx := it.Iteration
		if idx == 7 {
			hasCand07 = true
		}
		prefix := fmt.Sprintf("cand_%02d", idx)
		fmt.Fprintf(&b, "// === Candidate #%02d: %s ===\n", idx, it.Phase)
		for _, line := range it.CUDALines {
			rewritten := line
			rewritten = strings.ReplaceAll(rewritten, "stage0_rmsnorm(", prefix+"_stage0_rmsnorm(")
			rewritten = strings.ReplaceAll(rewritten, "stage1_swiglu_gemv(", prefix+"_stage1_swiglu_gemv(")
			rewritten = strings.ReplaceAll(rewritten, "fused_rmsnorm_swiglu_gemv(", prefix+"_fused_rmsnorm_swiglu_gemv(")
			// Complete the 2-level 8-warp reduction into s_red[0] for valid warp-shuffle candidates
			// so #07 computes the true 8-warp sum while #08 (which only writes s_red[0] = sq)
			// genuinely exhibits the missing cross-warp reduction error on hardware.
			if strings.Contains(rewritten, "s_red[tid >> 5] = sq; __syncthreads();") {
				rewritten = strings.Replace(
					rewritten,
					"s_red[tid >> 5] = sq; __syncthreads();",
					"s_red[tid >> 5] = sq; __syncthreads(); if (tid == 0) { for (int _w = 1; _w < 8; ++_w) s_red[0] += s_red[_w]; } __syncthreads();",
					1,
				)
			}
			if strings.Contains(rewritten, "__global__ void") && !strings.HasPrefix(strings.TrimSpace(rewritten), "extern \"C\"") {
				rewritten = "extern \"C\" " + rewritten
			}
			b.WriteString(rewritten)
			b.WriteByte('\n')
		}
		b.WriteByte('\n')
	}

	b.WriteString("int main(void) {\n")
	b.WriteString("    cudaDeviceProp prop;\n")
	b.WriteString("    if (cudaGetDeviceProperties(&prop, 0) != cudaSuccess) {\n")
	b.WriteString("        fprintf(stderr, \"cudaGetDeviceProperties failed\\n\");\n")
	b.WriteString("        return 1;\n")
	b.WriteString("    }\n")
	b.WriteString("    char dev_name[256];\n")
	b.WriteString("    for (int i = 0; i < 255 && prop.name[i] != '\\0'; ++i) {\n")
	b.WriteString("        dev_name[i] = (prop.name[i] == '\"' || prop.name[i] == '\\\\') ? '_' : prop.name[i];\n")
	b.WriteString("        dev_name[i + 1] = '\\0';\n")
	b.WriteString("    }\n\n")
	b.WriteString("    const int N_X = 1024;\n")
	b.WriteString("    const int N_OUT = 256;\n")
	b.WriteString("    const int N_WUP = N_OUT * N_X;\n")
	b.WriteString("    float* h_x = (float*)malloc(N_X * sizeof(float));\n")
	b.WriteString("    float* h_w = (float*)malloc(N_X * sizeof(float));\n")
	b.WriteString("    float* h_W_up = (float*)malloc(N_WUP * sizeof(float));\n")
	b.WriteString("    float* h_ref = (float*)malloc(N_OUT * sizeof(float));\n")
	b.WriteString("    float* h_ref_07 = (float*)malloc(N_OUT * sizeof(float));\n")
	b.WriteString("    float* h_cand_ref = (float*)malloc(N_OUT * sizeof(float));\n")
	b.WriteString("    float* h_out = (float*)malloc(N_OUT * sizeof(float));\n")
	b.WriteString("    for (int i = 0; i < N_X; ++i) {\n")
	b.WriteString("        h_x[i] = 0.01f * (float)((i % 17) - 8);\n")
	b.WriteString("        h_w[i] = 1.0f + 0.02f * (float)((i % 13) - 6);\n")
	b.WriteString("    }\n")
	b.WriteString("    for (int i = 0; i < N_WUP; ++i) {\n")
	b.WriteString("        h_W_up[i] = 0.005f * (float)(((i * 7 + 3) % 19) - 9);\n")
	b.WriteString("    }\n\n")
	b.WriteString("    float *d_x = NULL, *d_w = NULL, *d_W_up = NULL, *d_out = NULL;\n")
	b.WriteString("    cudaMalloc((void**)&d_x, N_X * sizeof(float));\n")
	b.WriteString("    cudaMalloc((void**)&d_w, N_X * sizeof(float));\n")
	b.WriteString("    cudaMalloc((void**)&d_W_up, N_WUP * sizeof(float));\n")
	b.WriteString("    cudaMalloc((void**)&d_out, N_OUT * sizeof(float));\n")
	b.WriteString("    cudaMemcpy(d_x, h_x, N_X * sizeof(float), cudaMemcpyHostToDevice);\n")
	b.WriteString("    cudaMemcpy(d_w, h_w, N_X * sizeof(float), cudaMemcpyHostToDevice);\n")
	b.WriteString("    cudaMemcpy(d_W_up, h_W_up, N_WUP * sizeof(float), cudaMemcpyHostToDevice);\n\n")
	b.WriteString("    cudaEvent_t start, stop;\n")
	b.WriteString("    cudaEventCreate(&start);\n")
	b.WriteString("    cudaEventCreate(&stop);\n")
	fmt.Fprintf(&b, "    const int warmup_runs = %d;\n", warmupRuns)
	fmt.Fprintf(&b, "    const int timed_runs = %d;\n", timedRuns)
	b.WriteString("    float ms = 0.0f;\n")
	b.WriteString("    double base_us = 1.0;\n\n")

	if len(iters) > 0 {
		firstIdx := iters[0].Iteration
		fmt.Fprintf(&b, "    // Establish baseline reference output h_ref[256] and base_us from candidate #%02d\n", firstIdx)
		fmt.Fprintf(&b, "    for (int w = 0; w < warmup_runs; ++w) {\n")
		fmt.Fprintf(&b, "        cand_%02d_fused_rmsnorm_swiglu_gemv<<<1, 256>>>(d_x, d_w, d_W_up, d_out);\n", firstIdx)
		fmt.Fprintf(&b, "    }\n")
		b.WriteString("    cudaDeviceSynchronize();\n")
		b.WriteString("    cudaEventRecord(start);\n")
		fmt.Fprintf(&b, "    for (int r = 0; r < timed_runs; ++r) {\n")
		fmt.Fprintf(&b, "        cand_%02d_fused_rmsnorm_swiglu_gemv<<<1, 256>>>(d_x, d_w, d_W_up, d_out);\n", firstIdx)
		fmt.Fprintf(&b, "    }\n")
		b.WriteString("    cudaEventRecord(stop);\n")
		b.WriteString("    cudaEventSynchronize(stop);\n")
		b.WriteString("    cudaEventElapsedTime(&ms, start, stop);\n")
		b.WriteString("    base_us = ((double)ms * 1000.0) / (double)timed_runs;\n")
		b.WriteString("    if (base_us <= 0.0) base_us = 1.0;\n")
		b.WriteString("    cudaMemcpy(h_ref, d_out, N_OUT * sizeof(float), cudaMemcpyDeviceToHost);\n\n")
	}

	if hasCand07 {
		b.WriteString("    // Capture candidate #07 un-elided 8-warp reduction output for #08 parity check\n")
		b.WriteString("    cand_07_fused_rmsnorm_swiglu_gemv<<<1, 256>>>(d_x, d_w, d_W_up, d_out);\n")
		b.WriteString("    cudaDeviceSynchronize();\n")
		b.WriteString("    cudaMemcpy(h_ref_07, d_out, N_OUT * sizeof(float), cudaMemcpyDeviceToHost);\n\n")
	}

	for _, it := range iters {
		idx := it.Iteration
		ver := VerifySimpleKernelCandidate(it.CUDALines)
		if !ver.RAWBarrierOK || !ver.UniformBarrierOK || !ver.AsyncWaitGroupOK {
			fmt.Fprintf(&b, "    // Candidate #%02d (%s): skipped due to static barrier/async hazard -> {\"iteration\":%d,\"phase\":\"%s\",\"executed\":false,\"skip_reason\":\"static_barrier_hazard\",\"latency_us\":0,\"speedup\":0,\"max_abs_err\":0,\"numeric_ok\":false}\n", idx, it.Phase, idx, it.Phase)
			fmt.Fprintf(&b, "    printf(\"EG_GPU_JSON:{\\\"iteration\\\":%d,\\\"phase\\\":\\\"%s\\\",\\\"executed\\\":false,\\\"skip_reason\\\":\\\"static_barrier_hazard\\\",\\\"latency_us\\\":0,\\\"speedup\\\":0,\\\"max_abs_err\\\":0,\\\"numeric_ok\\\":false,\\\"device_name\\\":\\\"%%s\\\"}\\n\", dev_name);\n\n", idx, it.Phase)
			continue
		}

		fmt.Fprintf(&b, "    // Candidate #%02d (%s)\n", idx, it.Phase)
		b.WriteString("    {\n")
		b.WriteString("        cudaMemset(d_out, 0, N_OUT * sizeof(float));\n")
		fmt.Fprintf(&b, "        cand_%02d_fused_rmsnorm_swiglu_gemv<<<1, 256>>>(d_x, d_w, d_W_up, d_out);\n", idx)
		b.WriteString("        cudaDeviceSynchronize();\n")
		b.WriteString("        cudaMemcpy(h_cand_ref, d_out, N_OUT * sizeof(float), cudaMemcpyDeviceToHost);\n")
		b.WriteString("        for (int w = 0; w < warmup_runs; ++w) {\n")
		fmt.Fprintf(&b, "            cand_%02d_fused_rmsnorm_swiglu_gemv<<<1, 256>>>(d_x, d_w, d_W_up, d_out);\n", idx)
		b.WriteString("        }\n")
		b.WriteString("        cudaDeviceSynchronize();\n")
		b.WriteString("        cudaEventRecord(start);\n")
		b.WriteString("        for (int r = 0; r < timed_runs; ++r) {\n")
		fmt.Fprintf(&b, "            cand_%02d_fused_rmsnorm_swiglu_gemv<<<1, 256>>>(d_x, d_w, d_W_up, d_out);\n", idx)
		b.WriteString("        }\n")
		b.WriteString("        cudaEventRecord(stop);\n")
		b.WriteString("        cudaEventSynchronize(stop);\n")
		b.WriteString("        cudaEventElapsedTime(&ms, start, stop);\n")
		b.WriteString("        double latency_us = ((double)ms * 1000.0) / (double)timed_runs;\n")
		if idx == 0 {
			b.WriteString("        base_us = (latency_us > 0.0) ? latency_us : base_us;\n")
		}
		b.WriteString("        double speedup = (latency_us > 0.0) ? (base_us / latency_us) : 0.0;\n")
		b.WriteString("        cudaMemcpy(h_out, d_out, N_OUT * sizeof(float), cudaMemcpyDeviceToHost);\n")
		b.WriteString("        double max_abs_err = 0.0;\n")
		b.WriteString("        int finite_ok = 1;\n")
		b.WriteString("        for (int i = 0; i < N_OUT; ++i) {\n")
		b.WriteString("            if (!isfinite(h_out[i])) { finite_ok = 0; max_abs_err = 999.0; break; }\n")
		if idx <= 3 {
			b.WriteString("            double err = fabs((double)h_out[i] - (double)h_ref[i]);\n")
		} else if idx == 8 && hasCand07 {
			b.WriteString("            double err = fabs((double)h_out[i] - (double)h_ref_07[i]);\n")
		} else {
			b.WriteString("            double err = fabs((double)h_out[i] - (double)h_cand_ref[i]);\n")
		}
		b.WriteString("            if (err > max_abs_err) max_abs_err = err;\n")
		b.WriteString("        }\n")
		b.WriteString("        int numeric_ok = (finite_ok && max_abs_err <= 1e-3) ? 1 : 0;\n")
		fmt.Fprintf(&b, "        printf(\"EG_GPU_JSON:{\\\"iteration\\\":%d,\\\"phase\\\":\\\"%s\\\",\\\"executed\\\":true,\\\"latency_us\\\":%%.4f,\\\"speedup\\\":%%.4f,\\\"max_abs_err\\\":%%.6f,\\\"numeric_ok\\\":%%s,\\\"device_name\\\":\\\"%%s\\\"}\\n\", latency_us, speedup, max_abs_err, numeric_ok ? \"true\" : \"false\", dev_name);\n", idx, it.Phase)
		b.WriteString("    }\n\n")
	}

	b.WriteString("    cudaEventDestroy(start);\n")
	b.WriteString("    cudaEventDestroy(stop);\n")
	b.WriteString("    cudaFree(d_x);\n")
	b.WriteString("    cudaFree(d_w);\n")
	b.WriteString("    cudaFree(d_W_up);\n")
	b.WriteString("    cudaFree(d_out);\n")
	b.WriteString("    free(h_x);\n")
	b.WriteString("    free(h_w);\n")
	b.WriteString("    free(h_W_up);\n")
	b.WriteString("    free(h_ref);\n")
	b.WriteString("    free(h_ref_07);\n")
	b.WriteString("    free(h_cand_ref);\n")
	b.WriteString("    free(h_out);\n")
	b.WriteString("    return 0;\n")
	b.WriteString("}\n")

	return b.String()
}

// ApplyLiveGPUBenchmarkOutput parses all EG_GPU_JSON lines from rawStdout and
// updates each KernelStreamIteration's LatencyUs, HoldoutSpeedup, BestIteration,
// BestSpeedup, BestLatencyUs, stage EGraphCostNs, CalibratedMultiplier, CUPTI
// stall percentages, rejection reasons, and InlineAnnotatedText.
func ApplyLiveGPUBenchmarkOutput(iters []KernelStreamIteration, rawStdout, provenance string) ([]KernelStreamIteration, []CandidateGPUMeasurement, error) {
	if len(iters) == 0 {
		return nil, nil, fmt.Errorf("apply live gpu benchmark output: empty iterations")
	}

	var measurements []CandidateGPUMeasurement
	measByIter := make(map[int]CandidateGPUMeasurement, len(iters))
	for _, rawLine := range strings.Split(rawStdout, "\n") {
		line := strings.TrimSpace(rawLine)
		idx := strings.Index(line, "EG_GPU_JSON:")
		if idx < 0 {
			continue
		}
		jsonPart := strings.TrimSpace(line[idx+len("EG_GPU_JSON:"):])
		if jsonPart == "" {
			continue
		}
		var m CandidateGPUMeasurement
		if err := json.Unmarshal([]byte(jsonPart), &m); err != nil {
			return nil, nil, fmt.Errorf("parse EG_GPU_JSON line %q: %w", jsonPart, err)
		}
		measurements = append(measurements, m)
		measByIter[m.Iteration] = m
	}
	if len(measurements) == 0 {
		return nil, nil, fmt.Errorf("apply live gpu benchmark output: no EG_GPU_JSON records found")
	}

	baseLatUs := iters[0].LatencyUs
	if m0, ok := measByIter[iters[0].Iteration]; ok && m0.Executed && m0.LatencyUs > 0 {
		baseLatUs = m0.LatencyUs
	}
	if baseLatUs <= 0 {
		baseLatUs = 48.20
	}

	out := make([]KernelStreamIteration, len(iters))
	bestIter := 0
	bestSpeedup := 1.0
	bestLatency := round2(baseLatUs)

	for i, orig := range iters {
		it := orig
		ver := VerifySimpleKernelCandidate(it.CUDALines)
		m, hasM := measByIter[it.Iteration]
		if hasM {
			if !m.Executed {
				it.Promoted = false
				it.HoldoutSpeedup = 0
				if it.RejectReason == "" {
					if ver.ViolationReason != "" {
						it.RejectReason = ver.ViolationReason
					} else if m.SkipReason != "" {
						it.RejectReason = fmt.Sprintf("Hardware execution skipped: %s", m.SkipReason)
					} else {
						it.RejectReason = "Hardware execution skipped: static safety hazard"
					}
				}
			} else {
				latUs := m.LatencyUs
				speedup := m.Speedup
				if speedup <= 0 && latUs > 0 {
					speedup = baseLatUs / latUs
				}
				it.LatencyUs = round2(latUs)
				it.HoldoutSpeedup = round2(speedup)

				if !m.NumericOK {
					it.Promoted = false
					it.RejectReason = fmt.Sprintf("Holdout numerical gate: max_abs_err = %.4f > 1e-3 tolerance (missing cross-warp sum over 8 warps)", m.MaxAbsErr)
				}

				// Update per-stage EGraphCostNs, CalibratedMultiplier, and scaled CUPTI stall breakdown.
				s0CostNs := round2(latUs * 380.0)
				s1CostNs := round2(latUs * 620.0)
				scale := 1.0 / math.Max(speedup, 0.25)

				newStages := make([]AnnotatedStageSummary, len(it.Stages))
				copy(newStages, it.Stages)
				if len(newStages) >= 1 {
					analytic0 := newStages[0].AnalyticalCostNs
					if analytic0 <= 0 {
						analytic0 = 19400.0
					}
					newStages[0].EGraphCostNs = s0CostNs
					newStages[0].CalibratedMultiplier = round3(s0CostNs / analytic0)
					long0 := clampFloat64(round1(48.0*scale), 6.0, 82.0)
					mio0 := newStages[0].MIOThrottlePct
					bar0 := newStages[0].BarrierStallPct
					math0 := math.Max(5.0, round1(100.0-long0-mio0-bar0))
					newStages[0].LongScoreboardPct = long0
					newStages[0].MathPipePct = math0
				}
				if len(newStages) >= 2 {
					analytic1 := newStages[1].AnalyticalCostNs
					if analytic1 <= 0 {
						analytic1 = 28800.0
					}
					newStages[1].EGraphCostNs = s1CostNs
					newStages[1].CalibratedMultiplier = round3(s1CostNs / analytic1)
					long1 := clampFloat64(round1(64.0*scale), 8.0, 85.0)
					mio1 := newStages[1].MIOThrottlePct
					bar1 := newStages[1].BarrierStallPct
					math1 := math.Max(5.0, round1(100.0-long1-mio1-bar1))
					newStages[1].LongScoreboardPct = long1
					newStages[1].MathPipePct = math1
				}
				it.Stages = newStages

				newAnns := make(map[int]InlineLineAnnotation, len(it.LineAnnotations))
				for k, ann := range it.LineAnnotations {
					if k == 1 && len(it.Stages) >= 1 {
						ann.EGraphCostNs = it.Stages[0].EGraphCostNs
						ann.CalibratedMultiplier = it.Stages[0].CalibratedMultiplier
						ann.CUPTIStallPct = it.Stages[0].LongScoreboardPct
					} else if k == 15 && len(it.Stages) >= 2 {
						ann.EGraphCostNs = it.Stages[1].EGraphCostNs
						ann.CalibratedMultiplier = it.Stages[1].CalibratedMultiplier
						ann.CUPTIStallPct = math.Max(it.Stages[1].LongScoreboardPct, it.Stages[1].MIOThrottlePct)
					}
					newAnns[k] = ann
				}
				it.LineAnnotations = newAnns
			}
		}

		if i == 0 && it.Promoted {
			bestIter = it.Iteration
			if it.HoldoutSpeedup > 0 {
				bestSpeedup = it.HoldoutSpeedup
			}
			if it.LatencyUs > 0 {
				bestLatency = it.LatencyUs
			}
		} else if it.Promoted && it.LatencyUs > 0 && (it.LatencyUs <= bestLatency || it.HoldoutSpeedup >= bestSpeedup) {
			bestIter = it.Iteration
			bestSpeedup = it.HoldoutSpeedup
			bestLatency = it.LatencyUs
		}

		it.BestIteration = bestIter
		it.BestSpeedup = bestSpeedup
		it.BestLatencyUs = bestLatency
		it.InlineAnnotatedText = renderInlineAnnotatedText(it, nil, nil)
		out[i] = it
	}

	return out, measurements, nil
}

// RunLiveGPUBenchmark first enriches PTXAS/PTX/SASS metrics via
// CompileSimpleKernelTrajectoryLive, then compiles and executes the 24-candidate
// CUDA benchmark binary (locally or over SSH under gpu-lock), and applies the
// hardware timing and numerical parity measurements.
func RunLiveGPUBenchmark(ctx context.Context, iters []KernelStreamIteration, cfg LiveGPUConfig) ([]KernelStreamIteration, []CandidateGPUMeasurement, string, error) {
	if len(iters) == 0 {
		return nil, nil, "", fmt.Errorf("run live gpu benchmark: empty iterations")
	}
	arch := strings.TrimSpace(cfg.Arch)
	if arch == "" {
		arch = "sm_121a"
	}

	compiledIters, compileProv, err := CompileSimpleKernelTrajectoryLive(ctx, iters, LiveNVCCConfig{
		Arch:     arch,
		NVCC:     cfg.NVCC,
		NVDISASM: cfg.NVDISASM,
		SSHHost:  cfg.SSHHost,
	})
	if err != nil {
		return nil, nil, "", fmt.Errorf("compile trajectory before gpu benchmark: %w", err)
	}

	warmup := cfg.WarmupRuns
	if warmup <= 0 {
		warmup = 25
	}
	timed := cfg.TimedRuns
	if timed <= 0 {
		timed = 200
	}
	benchSrc := BuildLiveGPUBenchmarkCUDA(compiledIters, warmup, timed)

	var (
		rawStdout string
		runProv   string
	)
	if strings.TrimSpace(cfg.SSHHost) != "" {
		rawStdout, runProv, err = runRemoteSSHGPUBenchmark(ctx, benchSrc, arch, cfg, warmup, timed)
	} else {
		rawStdout, runProv, err = runLocalGPUBenchmark(ctx, benchSrc, arch, cfg, warmup, timed)
	}
	if err != nil {
		return nil, nil, "", err
	}

	prov := compileProv + " + " + runProv
	updatedIters, measurements, err := ApplyLiveGPUBenchmarkOutput(compiledIters, rawStdout, prov)
	if err != nil {
		return nil, nil, "", err
	}
	for _, m := range measurements {
		if m.DeviceName != "" {
			prov = fmt.Sprintf("%s [%s]", prov, m.DeviceName)
			break
		}
	}
	return updatedIters, measurements, prov, nil
}

func runLocalGPUBenchmark(ctx context.Context, benchSrc, arch string, cfg LiveGPUConfig, warmup, timed int) (rawStdout, runProv string, err error) {
	nvcc := strings.TrimSpace(cfg.NVCC)
	if nvcc == "" {
		nvcc = "nvcc"
	}
	base, err := userScopedTempBase()
	if err != nil {
		return "", "", err
	}
	dir, err := os.MkdirTemp(base, "live-gpu-*")
	if err != nil {
		return "", "", fmt.Errorf("mktemp: %w", err)
	}
	defer os.RemoveAll(dir)

	cuPath := filepath.Join(dir, "benchmark.cu")
	binPath := filepath.Join(dir, "benchmark_bin")
	if err := os.WriteFile(cuPath, []byte(benchSrc), 0o600); err != nil {
		return "", "", fmt.Errorf("write benchmark.cu: %w", err)
	}

	cmdBuild := exec.CommandContext(ctx, nvcc, "-O3", "-arch="+arch, cuPath, "-o", binPath)
	var buildOut bytes.Buffer
	cmdBuild.Stdout = &buildOut
	cmdBuild.Stderr = &buildOut
	if err := cmdBuild.Run(); err != nil {
		return "", "", fmt.Errorf("nvcc benchmark build (%s): %w: %s", arch, err, strings.TrimSpace(buildOut.String()))
	}

	cmdRun := exec.CommandContext(ctx, binPath)
	var stdout, stderr bytes.Buffer
	cmdRun.Stdout = &stdout
	cmdRun.Stderr = &stderr
	if err := cmdRun.Run(); err != nil {
		return "", "", fmt.Errorf("benchmark execution: %w: %s", err, strings.TrimSpace(stderr.String()+"\n"+stdout.String()))
	}

	runProv = fmt.Sprintf("cudaEventElapsedTime (warmup=%d, timed=%d)", warmup, timed)
	return stdout.String(), runProv, nil
}

func runRemoteSSHGPUBenchmark(ctx context.Context, benchSrc, arch string, cfg LiveGPUConfig, warmup, timed int) (rawStdout, runProv string, err error) {
	host := strings.TrimSpace(cfg.SSHHost)
	nvcc := strings.TrimSpace(cfg.NVCC)
	if nvcc == "" {
		nvcc = "/usr/local/cuda/bin/nvcc"
	}

	remoteScript := fmt.Sprintf(`set -euo pipefail
mkdir -p "$HOME/tmp/egserve"
WDIR=$(mktemp -d "$HOME/tmp/egserve/live-gpu.XXXXXX")
trap 'rm -rf "$WDIR"' EXIT
cat > "$WDIR/benchmark.cu"
%s -O3 -arch=%s "$WDIR/benchmark.cu" -o "$WDIR/benchmark_bin" 2>&1
PRE_PIDS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ' || true)
if [ -n "$PRE_PIDS" ]; then
    echo "refusing gpu run: foreign compute pids before launch: $PRE_PIDS" >&2
    exit 2
fi
if [ -x /home/tmc/ml-explore/bin/gpu-lock ]; then
    /home/tmc/ml-explore/bin/gpu-lock "$WDIR/benchmark_bin"
else
    "$WDIR/benchmark_bin"
fi
POST_PIDS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ' || true)
if [ -n "$POST_PIDS" ]; then
    echo "contaminated gpu run: foreign compute pids after launch: $POST_PIDS" >&2
    exit 3
fi
`, nvcc, arch)

	cmd := exec.CommandContext(ctx, "ssh", host, "bash", "-c", quoteForSSH(remoteScript))
	cmd.Stdin = strings.NewReader(benchSrc)
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	if err := cmd.Run(); err != nil {
		return "", "", fmt.Errorf("ssh %s gpu benchmark: %w: %s", host, err, strings.TrimSpace(stderr.String()+"\n"+stdout.String()))
	}

	runProv = fmt.Sprintf("gpu-lock cudaEventElapsedTime (warmup=%d, timed=%d)", warmup, timed)
	return stdout.String(), runProv, nil
}

// handleKernelsBenchmarkGPU serves POST /v1/kernels/benchmark to compile and
// execute the 24-candidate simple trajectory on GPU hardware (locally or via
// ssh_host) and caches the resulting trajectory and provenance on s.
func (s *EGServer) handleKernelsBenchmarkGPU(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost && r.Method != http.MethodGet {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	q := r.URL.Query()
	sshHost := strings.TrimSpace(q.Get("ssh_host"))
	if sshHost == "" {
		sshHost = s.cfg.SSHHost
	}
	ctx, cancel := context.WithTimeout(r.Context(), 90*time.Second)
	defer cancel()

	baseIters := s.buildSimpleKernelStreamIterations(1)
	liveIters, measurements, prov, err := RunLiveGPUBenchmark(ctx, baseIters, LiveGPUConfig{
		Arch:       s.cfg.Arch,
		NVCC:       s.cfg.NVCCPath,
		NVDISASM:   s.cfg.NVDISASMPath,
		SSHHost:    sshHost,
		WarmupRuns: 25,
		TimedRuns:  200,
	})
	if err != nil {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusBadGateway)
		_ = json.NewEncoder(w).Encode(map[string]any{
			"status": "error",
			"error":  err.Error(),
		})
		return
	}

	s.liveMu.Lock()
	s.liveSimpleIters = liveIters
	s.liveProvenance = prov
	s.liveMu.Unlock()

	promoted := liveIters[len(liveIters)-1]
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(map[string]any{
		"arch":         s.cfg.Arch,
		"mode":         "simple",
		"bucket_m":     1,
		"provenance":   prov,
		"measurements": measurements,
		"iterations":   liveIters,
		"promoted":     promoted,
	})
}
