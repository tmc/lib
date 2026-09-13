//go:build linux

package main

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/google/pprof/profile"
	"github.com/tmc/lib/nvidia/cupti"
	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudacapture"
	"github.com/tmc/lib/nvidia/x/cudaprof"
	"github.com/tmc/lib/nvidia/x/cudaptx"
	"github.com/tmc/lib/nvidia/x/cudasass"
	"github.com/tmc/lib/nvidia/x/cuptipcsampling"
)

func findNvdisasm() string {
	candidates := []string{
		"/usr/local/cuda-13.0/bin/nvdisasm",
		"/usr/local/cuda-13/bin/nvdisasm",
		"/usr/local/cuda/bin/nvdisasm",
		"nvdisasm",
	}
	for _, p := range candidates {
		if _, err := os.Stat(p); err == nil {
			return p
		}
	}
	return "nvdisasm"
}

func rewriteSASSSourcePath(sass []byte, name string) []byte {
	const marker = `//## File "`
	out := make([]byte, 0, len(sass))
	for _, line := range strings.SplitAfter(string(sass), "\n") {
		i := strings.Index(line, marker)
		if i < 0 {
			out = append(out, line...)
			continue
		}
		rest := line[i+len(marker):]
		j := strings.IndexByte(rest, '"')
		if j < 0 {
			out = append(out, line...)
			continue
		}
		out = append(out, line[:i+len(marker)]...)
		out = append(out, name...)
		out = append(out, rest[j:]...)
	}
	return out
}

func isArithmeticOpcode(op string) bool {
	lower := strings.ToLower(op)
	for _, prefix := range []string{
		"add", "sub", "mul", "fma", "mad", "dp4a", "shfl", "rsqrt", "div", "cvt", "ld", "st", "idp",
	} {
		if strings.HasPrefix(lower, prefix) {
			return true
		}
	}
	return false
}

func pcSamplingBudget(reasons int, period uint32) (cuptipcsampling.Config, error) {
	if reasons < 1 {
		return cuptipcsampling.Config{}, errors.New("pc sampling: empty stall reason inventory")
	}
	if period < 5 || period > 31 {
		return cuptipcsampling.Config{}, fmt.Errorf("pc sampling: period exponent %d outside 5 through 31", period)
	}
	const (
		outputBytes = 4 << 20
		nameBytes   = 256
		maxNames    = 4096
	)
	stride := uint64(64) + uint64(reasons)*8
	capacity := (outputBytes/2 - 80) / stride
	if capacity > maxNames/2 {
		capacity = maxNames / 2
	}
	if capacity < 1 {
		return cuptipcsampling.Config{}, fmt.Errorf("pc sampling: %d reasons exceed the output budget", reasons)
	}
	return cuptipcsampling.Config{
		Period:            period,
		WorkerSleepMS:     10,
		MaxReasons:        reasons,
		ScratchBytes:      8 << 20,
		HardwareBytes:     8 << 20,
		MaxNativeBytes:    32 << 20,
		PCsPerBuffer:      int(capacity),
		MaxOutputBytes:    outputBytes,
		MaxReads:          1 << 12,
		MaxNames:          maxNames,
		MaxNameBytes:      nameBytes,
		MaxTotalNameBytes: maxNames * nameBytes,
	}, nil
}

func queryStallReasons(cuCtx cupti.Context) (inv cuptipcsampling.ReasonInventory, err error) {
	disable, err := cuptipcsampling.Enable(cuCtx)
	if err != nil {
		return cuptipcsampling.ReasonInventory{}, err
	}
	defer func() { err = errors.Join(err, disable()) }()
	return cuptipcsampling.StallReasons(cuCtx, 1<<10)
}

func TestCUDAPCProfile(t *testing.T) {
	modelPath := os.Getenv("DS4_MODEL")
	if modelPath == "" {
		modelPath = "/home/tmc/ds4-models/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf"
	}
	if _, err := os.Stat(modelPath); err != nil {
		t.Skipf("model not accessible: %v", err)
	}

	dev, err := cuda.NewDevice(0)
	if err != nil {
		t.Skipf("cuda device 0 not available: %v", err)
	}
	defer dev.Close()

	devName, err := dev.Name()
	if err != nil {
		t.Fatal(err)
	}
	major, minor, err := dev.ComputeCapability()
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("device: %s (sm_%d%d)", devName, major, minor)

	// Retain primary context for CUPTI PC sampling
	cuCtx, err := dev.PrimaryContext()
	if err != nil {
		t.Skipf("retain primary context: %v", err)
	}
	defer cuCtx.Release()

	// Check CUPTI stall reasons
	inv, err := queryStallReasons(cupti.Context(cuCtx.Raw()))
	if err != nil {
		t.Skipf("query stall reasons: %v", err)
	}
	t.Logf("CUPTI stall reason inventory: %d reasons", len(inv.Reasons))

	// Instantiate runtime
	t.Logf("loading DS4 model from %s...", modelPath)
	r, err := newCUDARuntime(modelPath, 1024)
	if err != nil {
		t.Fatalf("newCUDARuntime: %v", err)
	}
	defer r.Close()

	// Warmup 1 step to trigger JIT compilation of ds4_math.cu
	t.Log("warming up 1 step to compile kernels...")
	if err := r.Eval(100, 0); err != nil {
		t.Fatalf("warmup step: %v", err)
	}
	if err := r.Reset(); err != nil {
		t.Fatalf("reset: %v", err)
	}

	compiled := r.MathCompiled()
	if compiled == nil {
		t.Fatal("mathCompiled is nil after warmup")
	}
	cubin := compiled.CUBIN()
	ptx := compiled.PTX()
	if len(cubin) == 0 || ptx == "" {
		t.Fatalf("compiled module missing cubin (%d bytes) or ptx (%d bytes)", len(cubin), len(ptx))
	}
	t.Logf("compiled ds4_math: cubin %d bytes, ptx %d bytes", len(cubin), len(ptx))

	// Disassemble CUBIN using nvdisasm
	nvdisasmPath := findNvdisasm()
	t.Logf("disassembling CUBIN using %s...", nvdisasmPath)
	d := cudasass.Disassembler{Path: nvdisasmPath, TempDir: t.TempDir()}
	disCtx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	assembly, err := d.Disassemble(disCtx, cubin, cudasass.Options{SourceLines: true})
	if err != nil {
		t.Fatalf("nvdisasm failed: %v", err)
	}
	t.Logf("disassembled SASS: %d bytes", len(assembly.Stdout))

	sass := rewriteSASSSourcePath(assembly.Stdout, "ds4_math.cu")

	capFiles := map[string][]byte{
		"ds4_math.cu":    []byte(cudaMathSource),
		"ds4_math.ptx":   []byte(ptx),
		"ds4_math.cubin": cubin,
		"ds4_math.sass":  sass,
	}

	cap := &cudacapture.Capture{
		Files: capFiles,
		Code: cudacapture.CodeFiles{
			Source: "ds4_math.cu",
			PTX:    "ds4_math.ptx",
			CUBIN:  "ds4_math.cubin",
			SASS:   "ds4_math.sass",
		},
	}

	code, err := cudacapture.LoadCode(cap)
	if err != nil {
		t.Fatalf("load code: %v", err)
	}
	t.Logf("loaded code: %d kernels decoded", len(code.Kernels))

	virtual, err := cudaptx.ParseString(ptx)
	if err != nil {
		t.Fatalf("parse virtual ptx: %v", err)
	}

	for _, k := range code.Kernels {
		if k == nil {
			continue
		}
		pk := virtual.Kernel(k.Name)
		if pk != nil && len(pk.Instructions) > 0 {
			k.PTXLine = pk.Instructions[0].Line
			for _, inst := range pk.Instructions {
				if isArithmeticOpcode(inst.Opcode) {
					k.PTXLine = inst.Line
					break
				}
			}
		}
	}

	// Create pprof session
	var profBuf bytes.Buffer
	prof := cudaprof.NewSession(&profBuf)
	prof.SetMappingFile("ds4_math.cubin", fmt.Sprintf("%x", sha256.Sum256(cubin)))

	sampleTypes := []string{"samples", "inst_executed", "stall_scoreboard", "stall_wait", "stall_mem", "stall_math_pipe_throttle", "gpu_time"}
	if err := prof.SetSampleTypes(sampleTypes...); err != nil {
		t.Fatalf("set sample types: %v", err)
	}
	if err := prof.SetDefaultSampleType("samples"); err != nil {
		t.Fatalf("set default sample type: %v", err)
	}
	prof.SetPeriod("gpu_time", "nanoseconds", 1)

	ptxBase := uint64(0x200000)
	prof.AddMapping("ds4_math.ptx", "", ptxBase, ptxBase+uint64(len(ptx)))

	prof.AddComment("model: DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8")
	prof.AddComment("device: " + devName)
	prof.AddComment("measurement: hardware PC sampling")
	prof.AddComment("attribution: hardware PC samples attributed to SASS, PTX and source")

	// Start PC sampling
	period := uint32(10) // 2^10 = 1024 cycles
	config, err := pcSamplingBudget(len(inv.Reasons), period)
	if err != nil {
		t.Fatalf("pc sampling budget: %v", err)
	}
	for _, reason := range inv.Reasons {
		config.ReasonIDs = append(config.ReasonIDs, reason.Index)
	}

	session, err := cuptipcsampling.Start(cupti.Context(cuCtx.Raw()), config)
	if err != nil {
		t.Fatalf("start pc sampling: %v", err)
	}

	resolver := cuptipcsampling.NewCaptureResolverFromCubin(code, cubin)
	bridge := &cuptipcsampling.ProfileBridge{
		Session:   prof,
		Inventory: inv,
		Resolver:  resolver,
	}

	const numTokens = 8
	t.Logf("running %d decode steps under hardware PC sampling...", numTokens)
	workloadStart := time.Now()

	for step := 0; step < numTokens; step++ {
		tok := 1000 + step*17
		if err := r.Eval(tok, step); err != nil {
			t.Fatalf("eval step %d: %v", step, err)
		}
	}
	if err := driver.CtxSynchronize(); err != nil {
		t.Fatalf("synchronize: %v", err)
	}
	workloadDur := time.Since(workloadStart)
	t.Logf("decode completed in %v (%.2f ms/token)", workloadDur, float64(workloadDur.Milliseconds())/float64(numTokens))

	// Stop sampling and drain
	if err := session.Stop(); err != nil {
		t.Fatalf("stop pc sampling: %v", err)
	}

	var totalSamples, droppedSamples, nonUserSamples uint64
	var pcsResolvedToPTX, totalPCs int

	drainBatch := func(b cuptipcsampling.Batch) {
		droppedSamples += b.DroppedSamples
		nonUserSamples += b.NonUserSamples
		if !b.Valid {
			return
		}
		totalSamples += b.TotalSamples
		for _, pc := range b.PCs {
			if pc.Name == "" {
				continue
			}
			totalPCs++
			loc := resolver.ResolvePC(cuptipcsampling.ModuleID{CRC: pc.CRC}, pc.Name, pc.Offset)
			if loc.PTXLine > 0 {
				pcsResolvedToPTX++
			}
		}
		_ = bridge.RecordBatch(context.Background(), b)
	}

	for {
		batch, readErr := session.Read()
		drainBatch(batch)
		if readErr != nil {
			t.Fatalf("read batch: %v", readErr)
		}
		if batch.RemainingPCs == 0 {
			break
		}
	}
	if err := session.Close(); err != nil {
		t.Fatalf("close session: %v", err)
	}
	if final, ok := session.Final(); ok {
		drainBatch(final)
	}

	resolution := float64(0)
	if totalPCs > 0 {
		resolution = float64(pcsResolvedToPTX) / float64(totalPCs)
	}

	t.Logf("PC sampling results:")
	t.Logf("  total samples:    %d", totalSamples)
	t.Logf("  unique PCs:       %d", totalPCs)
	t.Logf("  resolved to PTX:  %d (%.1f%%)", pcsResolvedToPTX, 100.0*resolution)
	t.Logf("  dropped samples:  %d", droppedSamples)
	t.Logf("  non-user samples: %d", nonUserSamples)

	prof.SetTimespan(workloadStart, workloadDur)
	if err := prof.Close(); err != nil {
		t.Fatalf("close profiler session: %v", err)
	}

	parsedProf, err := profile.ParseData(profBuf.Bytes())
	if err != nil {
		t.Fatalf("parse generated profile: %v", err)
	}
	cap.Profile = parsedProf

	meta := map[string]any{
		"created":            time.Now().UTC().Format(time.RFC3339),
		"device":             devName,
		"architecture":       fmt.Sprintf("sm_%d%d", major, minor),
		"tokens":             numTokens,
		"duration_ms":        workloadDur.Milliseconds(),
		"total_samples":      totalSamples,
		"unique_pcs":         totalPCs,
		"pcs_resolved":       pcsResolvedToPTX,
		"resolution_percent": 100.0 * resolution,
		"dropped_samples":    droppedSamples,
	}
	metaBytes, _ := json.MarshalIndent(meta, "", "  ")
	cap.Metadata = metaBytes

	// Write to destination
	outDir := filepath.Join(os.Getenv("HOME"), "tmp")
	_ = os.MkdirAll(outDir, 0755)
	outFile := filepath.Join(outDir, "ds4-decode-pcsampled.pprof")

	fOut, err := os.Create(outFile)
	if err != nil {
		t.Fatalf("create %s: %v", outFile, err)
	}
	if err := cudacapture.Write(fOut, cap); err != nil {
		fOut.Close()
		t.Fatalf("write capture: %v", err)
	}
	if err := fOut.Close(); err != nil {
		t.Fatalf("close output: %v", err)
	}

	t.Logf("successfully wrote PC-attributed profile to %s (%d bytes)", outFile, len(profBuf.Bytes()))

	if totalSamples == 0 {
		t.Fatal("expected >0 PC samples")
	}
	if resolution < 0.95 {
		t.Errorf("PTX resolution %.1f%% below 95%% threshold", 100.0*resolution)
	}
}
