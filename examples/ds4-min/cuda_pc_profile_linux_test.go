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
	"sort"
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
		outputBytes = 2 << 20
		nameBytes   = 256
		maxNames    = 65536
	)
	stride := uint64(64) + uint64(reasons)*8
	capacity := (outputBytes/2 - 80) / stride
	if capacity > 512 {
		capacity = 512
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

	// Start PC sampling before the runtime loads and JIT-compiles anything.
	// CUPTI attributes samples to a module only when it can see that module
	// load; a module compiled before this call has its kernels counted in
	// NonUsrKernelsTotalSamples instead, with no PC records delivered. An
	// earlier ordering started sampling after the warmup and attributed 364
	// of 478168 samples.
	period := uint32(16) // 2^16 = 65,536 cycles, suited for full multi-layer inference
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

	assembly, err := d.Disassemble(disCtx, cubin, cudasass.Options{PrintCode: true, InlineLines: true, PTXLines: true})
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

	// Kernels this capture has source for. cuBLAS, cublasLt and CUTLASS
	// kernels run in the same context and are sampled, but they are not in
	// ds4_math.cubin and can never resolve to its PTX. Counting them in the
	// resolution denominator measures module coverage, not attribution.
	ownKernels := make(map[string]bool, len(code.Kernels))
	for _, k := range code.Kernels {
		if k != nil {
			ownKernels[k.Name] = true
		}
	}

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

	// Every stall type ProfileBridge can emit has to be declared here:
	// SetSampleTypes selects what gets written, so an undeclared type is
	// dropped silently rather than reported, and the acceptance identity
	// below would then fail for a bookkeeping reason instead of a real one.
	stallTypes := []string{
		"stall_scoreboard", "stall_wait", "stall_math", "stall_mem",
		"stall_branch", "stall_fetch", "stall_drain", "stall_dispatch",
		"stall_not_selected",
	}
	// The bridge fills PCSamples, which cudaprof writes as "pc_samples";
	// "samples" is the launch-count metric it never sets. Declaring only
	// "samples" projects the entire PC sample payload out of the profile.
	sampleTypes := append([]string{"pc_samples", "inst_executed"}, stallTypes...)
	sampleTypes = append(sampleTypes, "gpu_time")
	if err := prof.SetSampleTypes(sampleTypes...); err != nil {
		t.Fatalf("set sample types: %v", err)
	}
	if err := prof.SetDefaultSampleType("pc_samples"); err != nil {
		t.Fatalf("set default sample type: %v", err)
	}
	prof.SetPeriod("gpu_time", "nanoseconds", 1)

	ptxBase := uint64(0x200000)
	prof.AddMapping("ds4_math.ptx", "", ptxBase, ptxBase+uint64(len(ptx)))

	prof.AddComment("model: DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8")
	prof.AddComment("device: " + devName)
	prof.AddComment("measurement: hardware PC sampling")
	prof.AddComment("attribution: hardware PC samples attributed to SASS, PTX and source")

	resolver := cuptipcsampling.NewCaptureResolverFromCubin(code, cubin)
	bridge := &cuptipcsampling.ProfileBridge{
		Session:   prof,
		Inventory: inv,
		Resolver:  resolver,
	}

	var totalSamples, droppedSamples, nonUserSamples uint64
	reasonTotals := make(map[string]uint64)
	ownReasonTotals := make(map[string]uint64)
	var hardwareBufferFull bool
	crcsSeen := make(map[uint64]int)
	var ownPCs, foreignPCs int
	var foreignSamples uint64
	var pcsResolvedToPTX, totalPCs int

	kernelCounts := make(map[string]int)
	kernelSampleCounts := make(map[string]uint64)

	drainBatch := func(b cuptipcsampling.Batch) {
		if b.HardwareBufferFull != 0 {
			hardwareBufferFull = true
		}
		droppedSamples += b.DroppedSamples
		nonUserSamples += b.NonUserSamples
		if !b.Valid {
			return
		}
		totalSamples += b.TotalSamples
		for _, pc := range b.PCs {
			kernelCounts[pc.Name]++
			if pc.Name == "" {
				continue
			}
			own := ownKernels[pc.Name]
			for _, r := range pc.Reasons {
				name := inv.Lookup(r.Index)
				reasonTotals[name] += uint64(r.Samples)
				// The same count, restricted to kernels this capture compiled.
				// Every reason percentage taken over reasonTotals alone is a
				// property of the CUDA context, which also runs cuBLAS,
				// cublasLt and CUTLASS -- not a property of ds4's kernels.
				if own {
					ownReasonTotals[name] += uint64(r.Samples)
				}
				// A _not_issued companion counts a subset of the cycles its
				// base reason already counted; summing both is what made an
				// earlier capture's stall total run 1.88x the collected
				// sample count.
				if !strings.HasSuffix(strings.ToLower(name), "_not_issued") {
					kernelSampleCounts[pc.Name] += uint64(r.Samples)
				}
			}
			totalPCs++
			crcsSeen[pc.CRC]++
			if !own {
				foreignPCs++
				foreignSamples += kernelPCSamples(inv, pc)
				continue
			}
			ownPCs++
			loc := resolver.ResolvePC(cuptipcsampling.ModuleID{CRC: pc.CRC}, pc.Name, pc.Offset)
			if loc.PTXLine > 0 {
				pcsResolvedToPTX++
			}
		}
		if err := bridge.RecordBatch(context.Background(), b); err != nil {
			// Discarding this error leaves every counter below looking
			// healthy over an empty profile.
			t.Errorf("record batch: %v", err)
		}
	}

	// Sampling has been running since before the model load, so everything
	// collected up to here belongs to load, JIT and warmup. Discard it: the
	// profile below is the decode arm, and these batches would otherwise be
	// recorded into it by the first inter-step drain.
	preDecodeBatches, preDecodeSamples := 0, uint64(0)
	for {
		b, readErr := session.Read()
		if readErr != nil || !b.Valid {
			break
		}
		preDecodeBatches++
		preDecodeSamples += b.TotalSamples
		if b.RemainingPCs == 0 {
			break
		}
	}
	t.Logf("discarded %d pre-decode batches (%d samples: load, JIT, warmup)",
		preDecodeBatches, preDecodeSamples)

	const numTokens = 4
	t.Logf("running %d decode steps under hardware PC sampling (period exponent %d)...", numTokens, period)
	workloadStart := time.Now()

	for step := 0; step < numTokens; step++ {
		tok := 1000 + step*17
		if err := r.Eval(tok, step); err != nil {
			t.Fatalf("eval step %d: %v", step, err)
		}
		// Drain any ready batches between steps
		for {
			b, readErr := session.Read()
			if readErr != nil || !b.Valid {
				break
			}
			drainBatch(b)
			if b.RemainingPCs == 0 {
				break
			}
		}
	}
	if err := r.device.WithContext(func() error { return driver.CtxSynchronize() }); err != nil {
		t.Fatalf("synchronize: %v", err)
	}
	workloadDur := time.Since(workloadStart)
	t.Logf("decode completed in %v (%.2f ms/token)", workloadDur, float64(workloadDur.Milliseconds())/float64(numTokens))

	// Stop sampling and drain
	if err := session.Stop(); err != nil {
		t.Fatalf("stop pc sampling: %v", err)
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
	if ownPCs > 0 {
		resolution = float64(pcsResolvedToPTX) / float64(ownPCs)
	}

	t.Logf("PC sampling results:")
	t.Logf("  total samples:    %d", totalSamples)
	t.Logf("  unique PCs:       %d", totalPCs)
	t.Logf("  ds4_math PCs:     %d", ownPCs)
	t.Logf("  resolved to PTX:  %d of %d (%.1f%%)", pcsResolvedToPTX, ownPCs, 100.0*resolution)
	t.Logf("  resolver CUBINCRC: %#x (0 disables the module guard)", resolver.CUBINCRC)
	for crc, n := range crcsSeen {
		t.Logf("  module CRC %#x: %d PCs", crc, n)
	}
	t.Logf("  foreign PCs:      %d (%d samples; cuBLAS/cublasLt/CUTLASS, no PTX here)", foreignPCs, foreignSamples)
	t.Logf("  dropped samples:  %d", droppedSamples)
	t.Logf("  hw buffer full:   %v", hardwareBufferFull)
	t.Logf("  non-user samples: %d", nonUserSamples)
	for k, count := range kernelCounts {
		t.Logf("  kernel %q: %d PCs, %d samples", k, count, kernelSampleCounts[k])
	}

	prof.SetTimespan(workloadStart, workloadDur)
	if err := prof.Close(); err != nil {
		t.Fatalf("close profiler session: %v", err)
	}

	parsedProf, err := profile.ParseData(profBuf.Bytes())
	if err != nil {
		t.Fatalf("parse generated profile: %v", err)
	}
	cap.Profile = parsedProf

	// Acceptance identity. Every base (non-companion) PC sample must land in
	// exactly one bucket: a stall type the bridge models, inst_executed (the
	// scheduler selected the warp), or the unmapped tally under its own name.
	// If the three do not sum to the "samples" total, a reason is being
	// dropped, double counted, or classified twice -- and no instruction-level
	// claim from this capture is safe until that is explained.
	var baseReasonSum uint64
	for name, n := range reasonTotals {
		if !strings.HasSuffix(strings.ToLower(name), "_not_issued") {
			baseReasonSum += n
		}
	}
	// The native reason ledger, by CUPTI's own names. The modelled stall
	// types are unions -- stall_scoreboard covers both short_scoreboard and
	// long_scoreboard, and only the long one is the memory-dependency term --
	// so a claim about what the workload waits on cannot be read off the
	// sample types alone. Emit the ledger or it exists only in this function.
	reasonNames := make([]string, 0, len(reasonTotals))
	for name := range reasonTotals {
		reasonNames = append(reasonNames, name)
	}
	sort.Slice(reasonNames, func(i, j int) bool {
		return reasonTotals[reasonNames[i]] > reasonTotals[reasonNames[j]]
	})
	// Two populations, printed side by side, because a percentage is only
	// meaningful with its subject named. "all" is every sampled kernel in the
	// CUDA context, cuBLAS and cublasLt and CUTLASS included; "ds4" is only the
	// kernels compiled from ds4_math.cu. A reason share quoted off the first
	// column describes the context this process happened to run, not ds4.
	var allBase, ownBase uint64
	for _, name := range reasonNames {
		if strings.HasSuffix(strings.ToLower(name), "_not_issued") {
			continue
		}
		allBase += reasonTotals[name]
		ownBase += ownReasonTotals[name]
	}
	pct := func(v, total uint64) float64 {
		if total == 0 {
			return 0
		}
		return 100.0 * float64(v) / float64(total)
	}
	t.Logf("native reason ledger (%d reasons); base-reason totals: all %d, ds4 %d",
		len(reasonNames), allBase, ownBase)
	t.Logf("  %-48s %12s %7s %12s %7s", "reason", "all", "%", "ds4", "%")
	for _, name := range reasonNames {
		a, o := reasonTotals[name], ownReasonTotals[name]
		if strings.HasSuffix(strings.ToLower(name), "_not_issued") {
			// Companions are excluded from both denominators above, so
			// printing a share for them would divide by the wrong total.
			t.Logf("  %-48s %12d %7s %12d %7s", name, a, "-", o, "-")
			continue
		}
		t.Logf("  %-48s %12d %6.2f%% %12d %6.2f%%", name, a, pct(a, allBase), o, pct(o, ownBase))
	}
	if allBase != ownBase {
		t.Logf("  the two columns differ by %d base samples (%.2f%% of all): every "+
			"reason share quoted without a column named is ambiguous by that much",
			allBase-ownBase, pct(allBase-ownBase, allBase))
	}

	// CUPTI counts non-user kernel samples in TotalSamples but never delivers
	// them as PC records, so they are the third term here, not a discrepancy.
	if baseReasonSum+nonUserSamples != totalSamples {
		t.Errorf("base reasons %d + non-user %d = %d != CUPTI total_samples %d (delta %d)",
			baseReasonSum, nonUserSamples, baseReasonSum+nonUserSamples, totalSamples,
			int64(baseReasonSum+nonUserSamples)-int64(totalSamples))
	}
	userFraction := float64(0)
	if totalSamples > 0 {
		userFraction = float64(baseReasonSum) / float64(totalSamples)
	}
	// Reported, not gated: there is no established floor for how much of a ds4
	// decode should land in NVRTC-compiled user kernels. A PTX resolution rate
	// is a percentage of the PCs that arrived, and says nothing about this.
	t.Logf("user-kernel samples: %d of %d (%.3f%%); non-user %d",
		baseReasonSum, totalSamples, 100*userFraction, nonUserSamples)

	byType := make(map[string]int64, len(parsedProf.SampleType))
	for i, st := range parsedProf.SampleType {
		for _, smp := range parsedProf.Sample {
			byType[st.Type] += smp.Value[i]
		}
	}
	var stallSum int64
	for _, name := range stallTypes {
		stallSum += byType[name]
	}
	var unmappedBase, unmappedCompanion int64
	for name, n := range bridge.UnmappedStalls {
		if strings.HasSuffix(strings.ToLower(name), "_not_issued") {
			unmappedCompanion += n
			continue
		}
		unmappedBase += n
	}
	accepted := stallSum + byType["inst_executed"] + unmappedBase

	t.Logf("acceptance identity (decode arm, %d steps):", numTokens)
	for _, name := range stallTypes {
		t.Logf("  %-22s %d", name, byType[name])
	}
	t.Logf("  %-22s %d", "inst_executed", byType["inst_executed"])
	t.Logf("  %-22s %d", "unmapped (base)", unmappedBase)
	for name, n := range bridge.UnmappedStalls {
		t.Logf("      %q: %d", name, n)
	}
	t.Logf("  %-22s %d", "= accepted", accepted)
	t.Logf("  %-22s %d", "pc_samples (profile)", byType["pc_samples"])
	t.Logf("  %-22s %d (excluded: subset of the base reasons)", "companions", unmappedCompanion)
	if accepted != byType["pc_samples"] {
		t.Errorf("acceptance identity failed: stalls+inst_executed+unmapped = %d, samples = %d (delta %d)",
			accepted, byType["pc_samples"], accepted-byType["pc_samples"])
	}

	meta := map[string]any{
		"created":                        time.Now().UTC().Format(time.RFC3339),
		"device":                         devName,
		"architecture":                   fmt.Sprintf("sm_%d%d", major, minor),
		"tokens":                         numTokens,
		"duration_ms":                    workloadDur.Milliseconds(),
		"total_samples":                  totalSamples,
		"unique_pcs":                     totalPCs,
		"pcs_resolved":                   pcsResolvedToPTX,
		"resolution_percent":             100.0 * resolution,
		"dropped_samples":                droppedSamples,
		"acceptance_accepted":            accepted,
		"acceptance_samples":             byType["pc_samples"],
		"acceptance_stalls":              stallSum,
		"acceptance_inst_executed":       byType["inst_executed"],
		"acceptance_unmapped_base":       unmappedBase,
		"acceptance_companions_excluded": unmappedCompanion,
		"base_reason_sum":                baseReasonSum,
		"reason_samples":                 reasonTotals,
		"ds4_reason_samples":             ownReasonTotals,
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
		t.Errorf("PTX resolution %.1f%% of %d ds4_math PCs below 95%% threshold", 100.0*resolution, ownPCs)
	}
}

// kernelPCSamples reports the base-reason sample total for one PC. The
// _not_issued companions count a subset of the cycles their base reason
// already counted, so including them would double the total.
func kernelPCSamples(inv cuptipcsampling.ReasonInventory, pc cuptipcsampling.PC) uint64 {
	var n uint64
	for _, r := range pc.Reasons {
		if !strings.HasSuffix(strings.ToLower(inv.Lookup(r.Index)), "_not_issued") {
			n += uint64(r.Samples)
		}
	}
	return n
}
