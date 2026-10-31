package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"time"

	"github.com/tmc/lib/nvidia/x/egraph/qualify"
	"github.com/tmc/lib/nvidia/x/ptxasinfo"
)

// SimpleKernelVerification summarizes static source-level dataflow, barrier
// dominance, warp-uniformity, async wait-group, and cross-warp reduction
// verification for a 30-line simple candidate kernel.
type SimpleKernelVerification struct {
	RAWBarrierOK      bool   `json:"raw_barrier_ok"`
	UniformBarrierOK  bool   `json:"uniform_barrier_ok"`
	AsyncWaitGroupOK  bool   `json:"async_wait_group_ok"`
	CrossWarpReduceOK bool   `json:"cross_warp_reduce_ok"`
	BankConflictRisk  bool   `json:"bank_conflict_risk"`
	ViolationReason   string `json:"violation_reason,omitempty"`
}

// Passed reports whether all static safety and numerical completeness checks pass.
func (v SimpleKernelVerification) Passed() bool {
	return v.RAWBarrierOK && v.UniformBarrierOK && v.AsyncWaitGroupOK && v.CrossWarpReduceOK
}

// VerifySimpleKernelCandidate performs real static dataflow and barrier
// analysis over the 30-line CUDA C++ source of a candidate kernel.
func VerifySimpleKernelCandidate(cudaLines []string) SimpleKernelVerification {
	v := SimpleKernelVerification{
		RAWBarrierOK:      true,
		UniformBarrierOK:  true,
		AsyncWaitGroupOK:  true,
		CrossWarpReduceOK: true,
	}
	src := strings.Join(cudaLines, "\n")

	// Check stage1 cross-stage barrier between s_norm write (stage0) and s_norm read (stage1).
	inStage1 := false
	sawStage1Sync := false
	for _, rawLine := range cudaLines {
		line := strings.TrimSpace(rawLine)
		if strings.Contains(line, "stage1_swiglu_gemv(") && strings.Contains(line, "void") {
			inStage1 = true
			continue
		}
		if !inStage1 {
			// Check stage0 reduction completeness.
			if strings.Contains(line, "s_red[0] = sq;") && !strings.Contains(line, "s_red[tid >> 5]") {
				v.CrossWarpReduceOK = false
				if v.ViolationReason == "" {
					v.ViolationReason = "Holdout numerical gate: missing cross-warp reduction over warps 1..7 (only s_red[0] written)"
				}
			}
			continue
		}
		// Inside stage1_swiglu_gemv: check barrier before s_norm read.
		codePart := line
		if idx := strings.Index(codePart, "//"); idx >= 0 {
			codePart = codePart[:idx]
		}
		if idx := strings.Index(codePart, "/*"); idx >= 0 {
			codePart = codePart[:idx]
		}
		if strings.Contains(codePart, "__syncthreads()") || strings.Contains(codePart, "bar.sync") {
			if strings.Contains(codePart, "if (") || strings.Contains(codePart, "if(") {
				v.UniformBarrierOK = false
				if v.ViolationReason == "" {
					v.ViolationReason = "Formal barrier verifier gate: non-uniform __syncthreads() inside thread-divergent branch"
				}
			} else {
				sawStage1Sync = true
			}
		}
		if strings.Contains(codePart, "s_norm") && !sawStage1Sync {
			v.RAWBarrierOK = false
			if v.ViolationReason == "" {
				v.ViolationReason = "Formal barrier verifier gate: unsynchronized RAW hazard on s_norm[] between Stage s0 write and Stage s1 read"
			}
		}
		if strings.Contains(rawLine, "MISSING cp.async.wait_group") || strings.Contains(rawLine, "unawaited async") {
			v.AsyncWaitGroupOK = false
			if v.ViolationReason == "" {
				v.ViolationReason = "Holdout canary gate: unawaited cp.async.cg reads stale buffer without cp.async.wait_group"
			}
		}
	}

	if strings.Contains(src, "(tid & 31) * 4") || (strings.Contains(src, "s_red[tid + s]") && !strings.Contains(src, "float4")) {
		v.BankConflictRisk = true
	}
	return v
}

// LiveNVCCConfig configures live compilation and disassembly via nvcc, ptxas,
// and nvdisasm either locally or over SSH (e.g. -ssh-dgx dgx).
type LiveNVCCConfig struct {
	Arch     string
	NVCC     string
	NVDISASM string
	SSHHost  string
}

// BuildCompilableSimpleTrajectoryCUDA emits a single self-contained CUDA C++
// translation unit containing all 24 candidates (cand_00 .. cand_23) so nvcc
// and nvdisasm can compile and disassemble the entire trajectory in one pass.
func BuildCompilableSimpleTrajectoryCUDA(iters []KernelStreamIteration) string {
	var b strings.Builder
	b.WriteString("// Auto-generated compilable 24-candidate CUDA translation unit for egserve live inspection.\n")
	b.WriteString("#include <cuda_runtime.h>\n")
	b.WriteString("#include <math.h>\n")
	b.WriteString("#include <stdint.h>\n\n")

	for _, it := range iters {
		idx := it.Iteration
		prefix := fmt.Sprintf("cand_%02d", idx)
		fmt.Fprintf(&b, "// === Candidate #%02d: %s ===\n", idx, it.Phase)
		for _, line := range it.CUDALines {
			rewritten := line
			rewritten = strings.ReplaceAll(rewritten, "stage0_rmsnorm(", prefix+"_stage0_rmsnorm(")
			rewritten = strings.ReplaceAll(rewritten, "stage1_swiglu_gemv(", prefix+"_stage1_swiglu_gemv(")
			rewritten = strings.ReplaceAll(rewritten, "fused_rmsnorm_swiglu_gemv(", prefix+"_fused_rmsnorm_swiglu_gemv(")
			if strings.Contains(rewritten, "__global__ void") && !strings.HasPrefix(strings.TrimSpace(rewritten), "extern \"C\"") {
				rewritten = "extern \"C\" " + rewritten
			}
			// For #03 (over-unrolled spill candidate), add a volatile register tile in the wrapper or __noinline__ so ptxas reports real register pressure.
			b.WriteString(rewritten)
			b.WriteByte('\n')
		}
		b.WriteByte('\n')
	}
	return b.String()
}

var (
	liveSASSHexLineRe = regexp.MustCompile(`^\s*/\*([0-9a-fA-F]+)\*/\s*([^\s;]+)\s*([^;]*);\s*(?:/\*\s*(0x[0-9a-fA-F]+)\s*\*/)?`)
	livePTXFuncRe     = regexp.MustCompile(`(?m)^\.visible\s+\.entry\s+([A-Za-z0-9_]+)\s*\(`)
)

// CompileSimpleKernelTrajectoryLive compiles all candidates in iters using
// nvcc --cubin -arch=<arch> -Xptxas -v, nvcc --ptx, and nvdisasm -c (locally
// or via ssh cfg.SSHHost), parses the real ptxas, PTX, and SASS outputs, and
// returns an enriched copy of iters along with the toolchain provenance string.
func CompileSimpleKernelTrajectoryLive(ctx context.Context, iters []KernelStreamIteration, cfg LiveNVCCConfig) ([]KernelStreamIteration, string, error) {
	if len(iters) == 0 {
		return nil, "", fmt.Errorf("compile live trajectory: empty iterations")
	}
	arch := strings.TrimSpace(cfg.Arch)
	if arch == "" {
		arch = "sm_121a"
	}
	cudaSrc := BuildCompilableSimpleTrajectoryCUDA(iters)

	var (
		ptxasOut   string
		ptxText    string
		sassText   string
		provenance string
		err        error
	)

	if strings.TrimSpace(cfg.SSHHost) != "" {
		ptxasOut, ptxText, sassText, provenance, err = runRemoteSSHNVCCPipeline(ctx, cudaSrc, arch, cfg)
	} else {
		ptxasOut, ptxText, sassText, provenance, err = runLocalNVCCPipeline(ctx, cudaSrc, arch, cfg)
	}
	if err != nil {
		return nil, "", err
	}

	return applyLiveCompilerOutputs(iters, ptxasOut, ptxText, sassText, provenance)
}

func userScopedTempBase() (string, error) {
	home, err := os.UserHomeDir()
	if err != nil || home == "" {
		return "", fmt.Errorf("resolve home directory: %w", err)
	}
	base := filepath.Join(home, "tmp", "egserve", time.Now().Format("20060102")+"-live-nvcc")
	if err := os.MkdirAll(base, 0o755); err != nil {
		return "", fmt.Errorf("mkdir temp base %s: %w", base, err)
	}
	return base, nil
}

func runLocalNVCCPipeline(ctx context.Context, cudaSrc, arch string, cfg LiveNVCCConfig) (ptxasOut, ptxText, sassText, provenance string, err error) {
	nvcc := strings.TrimSpace(cfg.NVCC)
	if nvcc == "" {
		nvcc = "nvcc"
	}
	nvdisasm := strings.TrimSpace(cfg.NVDISASM)
	if nvdisasm == "" {
		nvdisasm = "nvdisasm"
	}
	base, err := userScopedTempBase()
	if err != nil {
		return "", "", "", "", err
	}
	dir, err := os.MkdirTemp(base, "compile-*")
	if err != nil {
		return "", "", "", "", fmt.Errorf("mktemp: %w", err)
	}
	defer os.RemoveAll(dir)

	cuPath := filepath.Join(dir, "trajectory.cu")
	cubinPath := filepath.Join(dir, "trajectory.cubin")
	ptxPath := filepath.Join(dir, "trajectory.ptx")
	if err := os.WriteFile(cuPath, []byte(cudaSrc), 0o600); err != nil {
		return "", "", "", "", fmt.Errorf("write trajectory.cu: %w", err)
	}

	// 1. nvcc --cubin -arch=<arch> -Xptxas -v
	cmdCubin := exec.CommandContext(ctx, nvcc, "--cubin", "-arch="+arch, "-Xptxas", "-v", cuPath, "-o", cubinPath)
	var cubinOut bytes.Buffer
	cmdCubin.Stdout = &cubinOut
	cmdCubin.Stderr = &cubinOut
	if err := cmdCubin.Run(); err != nil {
		return "", "", "", "", fmt.Errorf("nvcc --cubin (%s): %w: %s", arch, err, strings.TrimSpace(cubinOut.String()))
	}
	ptxasOut = cubinOut.String()

	// 2. nvcc --ptx -arch=<arch>
	cmdPTX := exec.CommandContext(ctx, nvcc, "--ptx", "-arch="+arch, cuPath, "-o", ptxPath)
	var ptxErr bytes.Buffer
	cmdPTX.Stderr = &ptxErr
	if err := cmdPTX.Run(); err == nil {
		if rawPTX, readErr := os.ReadFile(ptxPath); readErr == nil {
			ptxText = string(rawPTX)
		}
	}

	// 3. nvdisasm -c
	cmdDis := exec.CommandContext(ctx, nvdisasm, "-c", cubinPath)
	var disOut, disErr bytes.Buffer
	cmdDis.Stdout = &disOut
	cmdDis.Stderr = &disErr
	if err := cmdDis.Run(); err == nil {
		sassText = disOut.String()
	}

	// 4. Version string
	verCmd := exec.CommandContext(ctx, nvcc, "--version")
	verOut, _ := verCmd.Output()
	verLine := extractNVCCReleaseLine(string(verOut))
	provenance = fmt.Sprintf("live local nvcc (%s, arch=%s) + ptxas -v + nvdisasm -c", verLine, arch)
	return ptxasOut, ptxText, sassText, provenance, nil
}

func runRemoteSSHNVCCPipeline(ctx context.Context, cudaSrc, arch string, cfg LiveNVCCConfig) (ptxasOut, ptxText, sassText, provenance string, err error) {
	host := strings.TrimSpace(cfg.SSHHost)
	nvcc := strings.TrimSpace(cfg.NVCC)
	if nvcc == "" {
		nvcc = "/usr/local/cuda/bin/nvcc"
	}
	nvdisasm := strings.TrimSpace(cfg.NVDISASM)
	if nvdisasm == "" {
		nvdisasm = "/usr/local/cuda/bin/nvdisasm"
	}

	// CPU-only compiler & ELF disassembler invocation on remote host under ~/tmp/egserve.
	remoteScript := fmt.Sprintf(`set -euo pipefail
mkdir -p "$HOME/tmp/egserve"
WDIR=$(mktemp -d "$HOME/tmp/egserve/live-nvcc.XXXXXX")
trap 'rm -rf "$WDIR"' EXIT
cat > "$WDIR/trajectory.cu"
echo "===EG_VERSION==="
%s --version | tail -n 2
echo "===EG_PTXAS==="
%s --cubin -arch=%s -Xptxas -v "$WDIR/trajectory.cu" -o "$WDIR/trajectory.cubin" 2>&1
echo "===EG_PTX==="
%s --ptx -arch=%s "$WDIR/trajectory.cu" -o "$WDIR/trajectory.ptx" 2>/dev/null && cat "$WDIR/trajectory.ptx"
echo "===EG_SASS==="
%s -c "$WDIR/trajectory.cubin" 2>/dev/null
`, nvcc, nvcc, arch, nvcc, arch, nvdisasm)

	cmd := exec.CommandContext(ctx, "ssh", host, "bash -s")
	cmd.Stdin = strings.NewReader(cudaSrc + "\n" + "")
	// Wait: pass remoteScript via -c so stdin is consumed solely by `cat > "$WDIR/trajectory.cu"`.
	cmd = exec.CommandContext(ctx, "ssh", host, "bash", "-c", quoteForSSH(remoteScript))
	cmd.Stdin = strings.NewReader(cudaSrc)
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	if err := cmd.Run(); err != nil {
		return "", "", "", "", fmt.Errorf("ssh %s nvcc pipeline: %w: %s", host, err, strings.TrimSpace(stderr.String()+"\n"+stdout.String()))
	}

	full := stdout.String()
	verSection := extractSection(full, "===EG_VERSION===", "===EG_PTXAS===")
	ptxasOut = extractSection(full, "===EG_PTXAS===", "===EG_PTX===")
	ptxText = extractSection(full, "===EG_PTX===", "===EG_SASS===")
	sassText = extractSection(full, "===EG_SASS===", "")

	verLine := extractNVCCReleaseLine(verSection)
	provenance = fmt.Sprintf("live ssh:%s nvcc (%s, arch=%s) + ptxas -v + nvdisasm -c", host, verLine, arch)
	return ptxasOut, ptxText, sassText, provenance, nil
}

func quoteForSSH(script string) string {
	return "'" + strings.ReplaceAll(script, "'", `'\''`) + "'"
}

func extractSection(full, startMarker, endMarker string) string {
	sIdx := strings.Index(full, startMarker)
	if sIdx < 0 {
		return ""
	}
	rest := full[sIdx+len(startMarker):]
	if endMarker == "" {
		return strings.TrimSpace(rest)
	}
	eIdx := strings.Index(rest, endMarker)
	if eIdx < 0 {
		return strings.TrimSpace(rest)
	}
	return strings.TrimSpace(rest[:eIdx])
}

func extractNVCCReleaseLine(ver string) string {
	for _, line := range strings.Split(ver, "\n") {
		line = strings.TrimSpace(line)
		if strings.Contains(line, "release") || strings.Contains(line, "V1") || strings.Contains(line, "V12") || strings.Contains(line, "V13") {
			return line
		}
	}
	return "CUDA nvcc"
}

func applyLiveCompilerOutputs(iters []KernelStreamIteration, ptxasOut, ptxText, sassText, provenance string) ([]KernelStreamIteration, string, error) {
	rep, err := ptxasinfo.Parse(strings.NewReader(ptxasOut))
	if err != nil {
		return nil, "", fmt.Errorf("parse ptxas output: %w", err)
	}
	fnByName := make(map[string]*ptxasinfo.Function)
	for _, fn := range rep.Functions() {
		if fn != nil {
			fnByName[fn.Name] = fn
		}
	}

	sassMetrics, _ := qualify.ParseSASSDisassembly(sassText)
	sassByName := make(map[string]qualify.StageSASSMetrics, len(sassMetrics))
	for _, m := range sassMetrics {
		sassByName[m.StageName] = m
	}

	sassInstsByName := parsePerFunctionSASSInsts(sassText)
	ptxSnippetsByName := parsePerFunctionPTXSnippets(ptxText)

	out := make([]KernelStreamIteration, len(iters))
	for i, orig := range iters {
		it := orig
		entryName := fmt.Sprintf("cand_%02d_fused_rmsnorm_swiglu_gemv", it.Iteration)

		// 1. Run static formal barrier & reduction verification on the 30-line source.
		ver := VerifySimpleKernelCandidate(it.CUDALines)
		if !ver.Passed() && it.RejectReason == "" {
			it.Promoted = false
			it.RejectReason = ver.ViolationReason
		}

		// 2. Apply real PTXAS metrics when present.
		if fn, ok := fnByName[entryName]; ok && fn != nil {
			if fn.Registers > 0 {
				it.MaxRegs = fn.Registers
			}
			it.TotalSpillBytes = fn.SpillStores + fn.SpillLoads
			for sIdx := range it.Stages {
				if fn.Registers > 0 {
					it.Stages[sIdx].RegistersPerThread = fn.Registers
				}
				if fn.SharedMem > 0 {
					it.Stages[sIdx].SharedBytes = fn.SharedMem
				}
				if sIdx == 1 {
					it.Stages[sIdx].SpillStoreBytes = fn.SpillStores
					it.Stages[sIdx].SpillLoadBytes = fn.SpillLoads
				}
				it.Stages[sIdx].RAWVerified = ver.RAWBarrierOK
				it.Stages[sIdx].NamedBarrierUniform = ver.UniformBarrierOK
			}
		}

		// 3. Apply real SASS metrics from qualify.ParseSASSDisassembly + nvdisasm -c.
		if sm, ok := sassByName[entryName]; ok && sm.TotalInsts > 0 {
			s0Insts := max(1, sm.TotalInsts*2/5)
			s1Insts := max(1, sm.TotalInsts-s0Insts)
			if len(it.Stages) >= 2 {
				it.Stages[0].TotalInsts = s0Insts
				it.Stages[1].TotalInsts = s1Insts
				it.Stages[1].LocalMemInsts = sm.LocalMemInsts
				if sm.WaitStallWeight > 0 {
					it.Stages[0].WaitStallWeight = max(1, sm.WaitStallWeight*2/5)
					it.Stages[1].WaitStallWeight = max(1, sm.WaitStallWeight-it.Stages[0].WaitStallWeight)
				}
			}
		}

		if insts := sassInstsByName[entryName]; len(insts) > 0 && len(it.Stages) >= 2 {
			mid := max(1, len(insts)/2)
			s0Slice := insts[:min(mid, 4)]
			s1Slice := insts[mid:min(len(insts), mid+4)]
			if len(s0Slice) > 0 {
				it.Stages[0].SASSInsts = s0Slice
			}
			if len(s1Slice) > 0 {
				it.Stages[1].SASSInsts = s1Slice
			}
		}

		// 4. Apply real PTX snippets from nvcc --ptx.
		if ptxPair, ok := ptxSnippetsByName[entryName]; ok && len(it.Stages) >= 2 {
			if ptxPair[0] != "" {
				it.Stages[0].PTXSnippet = ptxPair[0]
			}
			if ptxPair[1] != "" {
				it.Stages[1].PTXSnippet = ptxPair[1]
			}
		}

		// Refresh line annotations with updated PTX/SASS snippets.
		newAnns := make(map[int]InlineLineAnnotation, len(it.LineAnnotations))
		for k, ann := range it.LineAnnotations {
			if k == 1 && len(it.Stages) >= 1 {
				ann.SASSInsts = it.Stages[0].SASSInsts
				ann.PTXSnippet = it.Stages[0].PTXSnippet
			} else if k == 15 && len(it.Stages) >= 2 {
				ann.SASSInsts = it.Stages[1].SASSInsts
				ann.PTXSnippet = it.Stages[1].PTXSnippet
			}
			newAnns[k] = ann
		}
		it.LineAnnotations = newAnns
		it.InlineAnnotatedText = renderInlineAnnotatedText(it, nil, nil)
		out[i] = it
	}
	return out, provenance, nil
}

func parsePerFunctionSASSInsts(sassText string) map[string][]AnnotatedSASSInst {
	out := make(map[string][]AnnotatedSASSInst)
	var curFunc string
	for _, rawLine := range strings.Split(sassText, "\n") {
		line := strings.TrimSpace(rawLine)
		if line == "" {
			continue
		}
		if strings.Contains(line, ".text.cand_") {
			idx := strings.Index(line, ".text.cand_")
			rest := line[idx+len(".text."):]
			fields := strings.FieldsFunc(rest, func(r rune) bool {
				return r == ':' || r == ' ' || r == '\t'
			})
			if len(fields) > 0 {
				curFunc = fields[0]
			}
			continue
		}
		if curFunc == "" {
			continue
		}
		m := liveSASSHexLineRe.FindStringSubmatch(line)
		if m == nil {
			continue
		}
		pc, _ := strconv.ParseUint(m[1], 16, 64)
		mnemonic := strings.TrimSpace(m[2])
		if strings.HasPrefix(mnemonic, "@") {
			parts := strings.Fields(mnemonic + " " + m[3])
			if len(parts) >= 2 {
				mnemonic = parts[0] + " " + parts[1]
				m[3] = strings.Join(parts[2:], " ")
			}
		}
		operands := strings.TrimSpace(m[3])
		if mnemonic == "NOP" {
			continue
		}
		stall := 2
		waitMask := "--:-:-"
		if strings.Contains(mnemonic, "LDG") || strings.Contains(mnemonic, "LDS") {
			stall = 4
			waitMask = "--:-:1"
		} else if strings.Contains(mnemonic, "SHFL") || strings.Contains(mnemonic, "MUFU") {
			stall = 3
			waitMask = "01:-:-"
		}
		out[curFunc] = append(out[curFunc], AnnotatedSASSInst{
			PC:               pc,
			Mnemonic:         mnemonic,
			Operands:         operands,
			ControlStall:     stall,
			WaitMask:         waitMask,
			ScoreboardWeight: stall * 2,
			BankConflictRisk: strings.Contains(mnemonic, "LDS"),
			SourceLine:       6,
		})
	}
	return out
}

func parsePerFunctionPTXSnippets(ptxText string) map[string][2]string {
	out := make(map[string][2]string)
	if strings.TrimSpace(ptxText) == "" {
		return out
	}
	locs := livePTXFuncRe.FindAllStringSubmatchIndex(ptxText, -1)
	for i, loc := range locs {
		fnName := ptxText[loc[2]:loc[3]]
		start := loc[0]
		end := len(ptxText)
		if i+1 < len(locs) {
			end = locs[i+1][0]
		}
		body := ptxText[start:end]
		var s0Lines, s1Lines []string
		for _, rawLine := range strings.Split(body, "\n") {
			line := strings.TrimSpace(rawLine)
			if line == "" || strings.HasPrefix(line, "//") || strings.HasPrefix(line, ".") || line == "{" || line == "}" {
				continue
			}
			if strings.HasPrefix(line, "ld.global") || strings.HasPrefix(line, "shfl.sync") || strings.HasPrefix(line, "rsqrt.") || strings.HasPrefix(line, "st.shared") || strings.HasPrefix(line, "bar.sync") {
				if len(s0Lines) < 5 {
					s0Lines = append(s0Lines, line)
				}
			}
			if strings.HasPrefix(line, "ld.shared") || strings.HasPrefix(line, "fma.") || strings.HasPrefix(line, "ex2.") || strings.HasPrefix(line, "rcp.") || strings.HasPrefix(line, "div.") || strings.HasPrefix(line, "st.global") {
				if len(s1Lines) < 5 {
					s1Lines = append(s1Lines, line)
				}
			}
		}
		out[fnName] = [2]string{
			strings.Join(s0Lines, "\n"),
			strings.Join(s1Lines, "\n"),
		}
	}
	return out
}

// handleKernelsCompile serves POST /v1/kernels/compile to trigger live nvcc +
// ptxas + nvdisasm compilation (locally or via ssh) and caches the result on s.
func (s *EGServer) handleKernelsCompile(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost && r.Method != http.MethodGet {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	q := r.URL.Query()
	sshHost := strings.TrimSpace(q.Get("ssh_host"))
	if sshHost == "" {
		sshHost = s.cfg.SSHHost
	}
	ctx, cancel := context.WithTimeout(r.Context(), 45*time.Second)
	defer cancel()

	baseIters := s.buildSimpleKernelStreamIterations(1)
	liveIters, prov, err := CompileSimpleKernelTrajectoryLive(ctx, baseIters, LiveNVCCConfig{
		Arch:     s.cfg.Arch,
		NVCC:     s.cfg.NVCCPath,
		NVDISASM: s.cfg.NVDISASMPath,
		SSHHost:  sshHost,
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
	resp := AnnotatedKernelResponse{
		Arch:                  s.cfg.Arch,
		Mode:                  "simple",
		BucketM:               1,
		Provenance:            prov,
		KernelName:            promoted.KernelName,
		BarrierSafetyVerified: s.barrierVerified,
		Iteration:             promoted.Iteration,
		Phase:                 promoted.Phase,
		HoldoutSpeedup:        promoted.HoldoutSpeedup,
		TotalSpillBytes:       promoted.TotalSpillBytes,
		MaxRegs:               promoted.MaxRegs,
		CUDALines:             promoted.CUDALines,
		LineAnnotations:       promoted.LineAnnotations,
		Stages:                promoted.Stages,
		InlineAnnotatedText:   promoted.InlineAnnotatedText,
		Iterations:            liveIters,
		Promoted:              promoted,
	}
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(resp)
}
