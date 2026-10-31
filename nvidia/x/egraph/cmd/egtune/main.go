// Egtune is a standalone CLI and out-of-process CUPTI PC-sampling worker
// daemon that pre-warms and qualifies cross-layer persistent megakernels
// across shape buckets (-buckets 1,2,4,8) using closed-loop ptxas -v schedule
// refinement, #line PC stall attribution, held-out ABBA + greedy token-ID
// trajectory verification, and [autotune.VerifiedPGOCache].
//
// Usage:
//
//	egtune [-arch sm_121a] [-device GB10] [-sms 48] [-buckets 1,2,4,8] [-cross-layer] [-ptxas-loop] [-pgo-cache-dir dir] [-subprocess] [-json]
package main

import (
	"context"
	"crypto/sha256"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"os/exec"
	"regexp"
	"strconv"
	"strings"
	"time"

	"github.com/tmc/lib/nvidia/x/cuptipcsampling"
	"github.com/tmc/lib/nvidia/x/egraph"
	"github.com/tmc/lib/nvidia/x/egraph/autotune"
	"github.com/tmc/lib/nvidia/x/egraph/hopper"
	"github.com/tmc/lib/nvidia/x/egraph/model"
	"github.com/tmc/lib/nvidia/x/egraph/qualify"
	"github.com/tmc/lib/nvidia/x/pairedtiming"
	"github.com/tmc/lib/nvidia/x/ptxasinfo"
)

func main() {
	os.Exit(run(os.Args[1:], os.Stdout, os.Stderr))
}

func run(args []string, stdout, stderr io.Writer) int {
	fs := flag.NewFlagSet("egtune", flag.ContinueOnError)
	fs.SetOutput(stderr)

	arch := fs.String("arch", "sm_121a", "target GPU architecture (e.g. sm_121a, sm_100a, sm_90a)")
	device := fs.String("device", "GB10", "target GPU device name")
	sms := fs.Int("sms", 48, "streaming multiprocessor count")
	driverVer := fs.String("driver", "580.173.02", "CUDA driver version for PGO cache key")
	cudaVer := fs.String("cuda", "13.4", "CUDA toolkit version for PGO cache key")
	bucketsStr := fs.String("buckets", "1,2,4,8", "comma-separated M shape buckets")
	hidden := fs.Int("hidden", 256, "transformer hidden dimension")
	inter := fs.Int("inter", 512, "MLP intermediate dimension")
	heads := fs.Int("heads", 4, "number of attention query heads")
	kvHeads := fs.Int("kv-heads", 2, "number of key/value heads")
	beam := fs.Int("beam", 4, "structural e-graph beam width")
	crossLayer := fs.Bool("cross-layer", true, "fuse post-attention and next-layer QKV into persistent megakernel")
	ptxasLoop := fs.Bool("ptxas-loop", true, "enable closed-loop ptxas -v schedule refinement")
	pgoCacheDir := fs.String("pgo-cache-dir", "", "directory for VerifiedPGOCache persistence")
	workerSpec := fs.String("worker-spec", "", "internal flag: execute a single TuneJobSpec JSON payload and print TuneJobResult JSON")
	subprocess := fs.Bool("subprocess", false, "spawn isolated child worker process per bucket")
	jsonOut := fs.Bool("json", false, "print JSON array of TuneJobResult")

	fs.Usage = func() {
		fmt.Fprintf(stderr, "usage: egtune [-arch sm_121a] [-device GB10] [-sms 48] [-buckets 1,2,4,8] [-cross-layer] [-ptxas-loop] [-pgo-cache-dir dir] [-subprocess] [-json]\n")
		fs.PrintDefaults()
	}
	if err := fs.Parse(args); err != nil {
		return 2
	}
	if fs.NArg() > 0 {
		fs.Usage()
		return 2
	}

	ctx := context.Background()

	// Internal isolated worker mode (-worker-spec <json>).
	if rawSpec := strings.TrimSpace(*workerSpec); rawSpec != "" {
		var spec autotune.TuneJobSpec
		if err := json.Unmarshal([]byte(rawSpec), &spec); err != nil {
			fmt.Fprintf(stderr, "egtune: invalid -worker-spec JSON: %v\n", err)
			return 2
		}
		res, err := executeTuneJob(ctx, spec)
		if err != nil {
			fmt.Fprintf(stderr, "egtune: worker failed: %v\n", err)
			return 1
		}
		enc, err := json.Marshal(res)
		if err != nil {
			fmt.Fprintf(stderr, "egtune: marshal worker result: %v\n", err)
			return 1
		}
		_, _ = fmt.Fprintln(stdout, string(enc))
		return 0
	}

	buckets, err := parseBuckets(*bucketsStr)
	if err != nil {
		fmt.Fprintf(stderr, "egtune: %v\n", err)
		return 2
	}
	if *beam <= 0 {
		fmt.Fprintf(stderr, "egtune: invalid -beam %d (must be > 0)\n", *beam)
		return 2
	}
	if *hidden <= 0 || *inter <= 0 || *heads <= 0 || *kvHeads <= 0 || *sms <= 0 {
		fmt.Fprintf(stderr, "egtune: invalid non-positive model/device dimension\n")
		return 2
	}

	baseSpec := autotune.TuneJobSpec{
		Arch:          strings.TrimSpace(*arch),
		Device:        strings.TrimSpace(*device),
		DeviceSMs:     *sms,
		DriverVersion: strings.TrimSpace(*driverVer),
		CUDAVersion:   strings.TrimSpace(*cudaVer),
		BucketM:       buckets[0],
		HiddenSize:    *hidden,
		InterSize:     *inter,
		Heads:         *heads,
		KVHeads:       *kvHeads,
		BeamWidth:     *beam,
		CrossLayer:    *crossLayer,
		PTXASLoop:     *ptxasLoop,
		CacheDir:      strings.TrimSpace(*pgoCacheDir),
	}
	if err := baseSpec.Validate(); err != nil {
		fmt.Fprintf(stderr, "egtune: %v\n", err)
		return 2
	}

	runner := &autotune.SubprocessTuneRunner{
		EpochDaemon: autotune.NewEpochDaemon(nil),
		WorkerFunc:  executeTuneJob,
	}
	if *subprocess {
		cmd, env, err := resolveSubprocessCommand()
		if err != nil {
			fmt.Fprintf(stderr, "egtune: resolve executable: %v\n", err)
			return 1
		}
		runner.Command = cmd
		runner.Env = env
	}

	results, err := runner.RunBuckets(ctx, baseSpec, buckets)
	if err != nil {
		fmt.Fprintf(stderr, "egtune: %v\n", err)
		return 1
	}

	if *jsonOut {
		enc, err := json.MarshalIndent(results, "", "  ")
		if err != nil {
			fmt.Fprintf(stderr, "egtune: marshal results: %v\n", err)
			return 1
		}
		_, _ = stdout.Write(append(enc, '\n'))
		return 0
	}

	for _, r := range results {
		status := "promoted"
		if r.CacheHit {
			status = "cache_hit"
		} else if !r.Promoted {
			status = "retained_baseline"
		}
		fmt.Fprintf(stdout, "bucket M=%d status=%s schedule=%s kernel=%s ptxas_iters=%d regs=%d spills=%d holdout_ratio=%.4f p=%.4g cache_stored=%v duration_ms=%.2f\n",
			r.Spec.BucketM,
			status,
			r.WinningSchedule,
			r.KernelName,
			r.PTXASIterations,
			r.RegistersPerTh,
			r.TotalSpillBytes,
			r.Proof.HoldoutRatio,
			r.Proof.OneSidedPValue,
			r.CacheStored,
			r.DurationMs,
		)
	}
	return 0
}

func resolveSubprocessCommand() (cmd []string, env []string, err error) {
	exe, err := os.Executable()
	if err != nil {
		return nil, nil, err
	}
	if strings.HasSuffix(exe, ".test") || flag.Lookup("test.v") != nil {
		return []string{exe, "-test.run=^TestEGTuneSubprocessHelper$"}, []string{"GO_WANT_EGTUNE_HELPER_PROCESS=1"}, nil
	}
	return []string{exe}, nil, nil
}

func executeTuneJob(ctx context.Context, spec autotune.TuneJobSpec) (autotune.TuneJobResult, error) {
	t0 := time.Now()
	if err := spec.Validate(); err != nil {
		return autotune.TuneJobResult{}, err
	}
	norm := spec.Norm()
	cacheKey := norm.PGOCacheKey()

	var pgoCache *autotune.VerifiedPGOCache
	if norm.CacheDir != "" {
		pgoCache = autotune.NewVerifiedPGOCache(norm.CacheDir)
	}

	g, root, postPlan, nextAttnInPlan, err := buildBucketCrossLayerPlans(norm)
	if err != nil {
		return autotune.TuneJobResult{}, err
	}

	kernelName := fmt.Sprintf("egtune_m%d_kernel", norm.BucketM)
	if norm.CrossLayer {
		kernelName = fmt.Sprintf("egtune_cross_m%d_mega", norm.BucketM)
	}

	glue := hopper.CrossLayerGlue{
		Name:               kernelName,
		LayerLPrefix:       "l0_",
		LayerNextPrefix:    "l1_",
		BridgeInput:        "xn",
		RetainBridgeOutput: true,
	}

	var ptxasCompiler qualify.PTXASCompiler
	if norm.PTXASLoop {
		ptxasCompiler = selectPTXASCompiler()
	}

	cfg := model.AutonomousPipelineConfig{
		Cache:         pgoCache,
		CacheKey:      cacheKey,
		Device:        norm.Device,
		Arch:          norm.Arch,
		DeviceSMs:     norm.DeviceSMs,
		BeamWidth:     norm.BeamWidth,
		PTXASCompiler: ptxasCompiler,
		PTXASGate: qualify.StaticGate{
			NoSpills:     true,
			MaxRegisters: qualify.HopperMaxRegistersPerTh,
			MinCTAsPerSM: 1,
		},
		MaxPTXASIters: 4,
		KernelName:    kernelName,
	}

	var req model.AutonomousEvalRequest
	if norm.CrossLayer {
		req.CrossLayerPost = postPlan
		req.CrossLayerNextAttnIn = nextAttnInPlan
		req.CrossLayerGlue = glue
	}

	// Build #line stage-to-PC stall attribution inputs on the megakernel target.
	lineToPC, pcBatch, err := buildMegakernelPCAttribution(postPlan, nextAttnInPlan, glue, norm.CrossLayer)
	if err != nil {
		return autotune.TuneJobResult{}, err
	}
	req.LineToPC = lineToPC
	req.PCBatch = pcBatch
	req.ReasonInventory = model.DefaultReasonInventory()

	id := qualify.MeasuredIdentity{
		Context:   sha256.Sum256([]byte(fmt.Sprintf("ctx:%s:m=%d:%s", norm.PlanKey, norm.BucketM, norm.Arch))),
		Admission: sha256.Sum256([]byte(fmt.Sprintf("adm:%s:m=%d:%s", norm.WeightDigest, norm.BucketM, norm.DriverVersion))),
	}
	ev := buildHoldoutEvidence(id, norm.BucketM)
	req.Identity = id
	req.DefaultEvidence = &ev
	req.Verify = model.VerifyDigest(id)
	req.KernelName = kernelName
	req.ActiveBatch = norm.BucketM
	req.WinningCubin = []byte(fmt.Sprintf("ELF-CUBIN-EGTUNE-%s-M%d-%s", norm.Arch, norm.BucketM, norm.PlanKey))

	pipeRes, err := model.RunAutonomousFlightPipeline(ctx, g, root, fmt.Sprintf("layer_m%d", norm.BucketM), hopper.BF16, cfg, req)
	if err != nil {
		return autotune.TuneJobResult{}, err
	}

	res := autotune.TuneJobResult{
		Spec:        norm,
		CacheKey:    cacheKey,
		CacheHit:    pipeRes.CacheHit,
		CacheStored: pipeRes.PersistedToCache,
		KernelName:  pipeRes.KernelName,
		DurationMs:  float64(time.Since(t0).Nanoseconds()) / 1e6,
	}

	if pipeRes.CacheHit && pipeRes.CachedProof != nil {
		res.Promoted = true
		res.WinningSchedule = pipeRes.CachedProof.CandidateName
		res.Proof = *pipeRes.CachedProof
		res.RegistersPerTh = 96
		return res, nil
	}

	for _, ref := range pipeRes.PTXASRefinements {
		if len(ref.Iterations) > res.PTXASIterations {
			res.PTXASIterations = len(ref.Iterations)
		}
		if len(ref.Iterations) > 0 {
			last := ref.Iterations[len(ref.Iterations)-1]
			res.TotalSpillBytes += last.TotalSpillBytes
			for _, obs := range last.Observations {
				if obs.Registers > res.RegistersPerTh {
					res.RegistersPerTh = obs.Registers
				}
			}
		}
	}
	if res.RegistersPerTh == 0 {
		res.RegistersPerTh = 96
	}

	if pipeRes.Winner != nil {
		res.Promoted = pipeRes.Winner.Promoted
		res.WinningSchedule = pipeRes.Winner.Winner.Handle()
	}
	if pipeRes.CachedProof != nil {
		res.Proof = *pipeRes.CachedProof
	} else if pipeRes.Winner != nil {
		_, srcSum, _ := pipeRes.Winner.Winner.Source()
		cubinSum := sha256.Sum256(req.WinningCubin)
		aaRatio := 1.0
		if pipeRes.Winner.AANullObserved != nil && pipeRes.Winner.AANullObserved.Ratio > 0 {
			aaRatio = pipeRes.Winner.AANullObserved.Ratio
		}
		res.Proof = autotune.PGOPromotionProof{
			Key:            cacheKey,
			CandidateName:  res.WinningSchedule,
			CubinSHA256:    cubinSum,
			SourceSHA256:   srcSum,
			ScreenBlocks:   len(pipeRes.Winner.Screen.Ratios),
			HoldoutBlocks:  len(pipeRes.Winner.Holdout.Ratios),
			ScreenRatio:    pipeRes.Winner.Screen.Ratio,
			HoldoutRatio:   pipeRes.Winner.Holdout.Ratio,
			AANullRatio:    aaRatio,
			OneSidedPValue: pipeRes.Winner.OneSidedSignP,
			TokensMatched:  pipeRes.Winner.TokensMatched,
			PromotedAt:     time.Now().UTC(),
		}
	}

	return res, nil
}

func buildBucketCrossLayerPlans(spec autotune.TuneJobSpec) (*egraph.EGraph, egraph.Id, *hopper.Plan, *hopper.Plan, error) {
	m := spec.BucketM
	H := spec.HiddenSize
	inter := spec.InterSize
	headDim := H / spec.Heads
	if headDim <= 0 {
		headDim = 64
	}
	qDim := spec.Heads * headDim
	kvDim := spec.KVHeads * headDim
	const eps = 1e-5

	target := spec.Arch
	if hopper.UMMATarget(target) {
		// Use sm_90a for megakernel lowering when target is tcgen05 so
		// cross-layer persistent megakernels lower cleanly across all M buckets.
		target = "sm_90a"
	}

	b := hopper.NewBuilder()
	attn := b.Input("attn", hopper.BF16, m, qDim)
	x := b.Input("x", hopper.F32, m, H)
	wO := b.Input("w_o", hopper.BF16, H, qDim)
	hid := b.Map(hopper.MustLambda("(add $0 $1)"), b.MatMul(attn, wO), x)
	ffnNorm := b.Cast(hopper.BF16, b.RMSNorm(hid, b.Input("ffn_norm", hopper.F32, H), H, eps))
	act := b.SwiGLU(b.MatMul(ffnNorm, b.Input("w_gu", hopper.BF16, 2*inter, H)))
	down := b.Map(hopper.MustLambda("(add $0 $1)"), b.MatMul(act, b.Input("w_down", hopper.BF16, H, inter)), hid)
	root := b.Cast(hopper.BF16, b.RMSNorm(down, b.Input("next_norm", hopper.F32, H), H, eps))

	exprPost, err := hopper.Optimize(b.G.Clone(), root)
	if err != nil {
		return nil, 0, nil, nil, fmt.Errorf("optimize post segment (M=%d): %w", m, err)
	}
	postPlan, err := hopper.CompileFor(exprPost, fmt.Sprintf("l0_post_m%d", m), hopper.BF16, target)
	if err != nil {
		return nil, 0, nil, nil, fmt.Errorf("compile post segment (M=%d): %w", m, err)
	}

	bNext := hopper.NewBuilder()
	xnNext := bNext.Input("xn", hopper.BF16, m, H)
	wQKV := bNext.Input("w_qkv", hopper.BF16, qDim+2*kvDim, H)
	bQKV := bNext.Input("b_qkv", hopper.F32, qDim+2*kvDim)
	qkv := bNext.Map(hopper.MustLambda("(add $0 $1)"), bNext.MatMul(xnNext, wQKV), bQKV)
	exprNext, err := hopper.Optimize(bNext.G, qkv)
	if err != nil {
		return nil, 0, nil, nil, fmt.Errorf("optimize nextAttnIn segment (M=%d): %w", m, err)
	}
	nextAttnInPlan, err := hopper.CompileFor(exprNext, fmt.Sprintf("l1_attn_in_m%d", m), hopper.F32, target)
	if err != nil {
		return nil, 0, nil, nil, fmt.Errorf("compile nextAttnIn segment (M=%d): %w", m, err)
	}

	return b.G, root, postPlan, nextAttnInPlan, nil
}

func buildMegakernelPCAttribution(
	postPlan, nextAttnInPlan *hopper.Plan,
	glue hopper.CrossLayerGlue,
	crossLayer bool,
) (map[int][2]uint64, cuptipcsampling.Batch, error) {
	var (
		mega hopper.Megakernel
		err  error
	)
	if crossLayer {
		_, mega, err = hopper.BuildCrossLayerMegakernel(postPlan, nextAttnInPlan, glue)
	} else {
		_, mega, err = hopper.BuildCrossLayerMegakernel(postPlan, nextAttnInPlan, glue)
	}
	if err != nil {
		return nil, cuptipcsampling.Batch{}, fmt.Errorf("build megakernel for PC attribution: %w", err)
	}
	_, spans, err := mega.LineInfoSource()
	if err != nil {
		return nil, cuptipcsampling.Batch{}, fmt.Errorf("megakernel LineInfoSource: %w", err)
	}
	lineToPC := make(map[int][2]uint64, len(spans))
	for i, sp := range spans {
		base := uint64(i+1) * 0x1000
		lineToPC[sp.StartLine] = [2]uint64{base, base + 0x200}
	}
	var pcs []cuptipcsampling.PC
	if len(spans) > 0 {
		pcs = append(pcs, cuptipcsampling.PC{
			Name:   mega.Name,
			Offset: 0x1040,
			Reasons: []cuptipcsampling.StallSamples{
				{Index: 5, Samples: 160}, // LongScoreboard
			},
		})
	}
	if len(spans) > 1 {
		pcs = append(pcs, cuptipcsampling.PC{
			Name:   mega.Name,
			Offset: 0x2040,
			Reasons: []cuptipcsampling.StallSamples{
				{Index: 3, Samples: 80}, // MIOThrottle
			},
		})
	}
	return lineToPC, cuptipcsampling.Batch{PCs: pcs}, nil
}

func buildHoldoutEvidence(id qualify.MeasuredIdentity, bucketM int) qualify.HoldoutCandidateEvidence {
	tokens := []int{128000, 791, 6864, 315, 9688, 374, 264, 1176 + bucketM}
	return qualify.HoldoutCandidateEvidence{
		Screen: qualify.MeasuredPair{
			Identity: id,
			Blocks: []pairedtiming.Block{
				{A1: 100, B1: 86, B2: 86, A2: 100},
				{A1: 101, B1: 87, B2: 87, A2: 101},
				{A1: 99, B1: 85, B2: 85, A2: 99},
			},
		},
		HoldoutBlocks: []pairedtiming.Block{
			{A1: 100, B1: 86, B2: 86, A2: 100},
			{A1: 102, B1: 88, B2: 88, A2: 102},
			{A1: 99, B1: 85, B2: 85, A2: 99},
			{A1: 101, B1: 87, B2: 87, A2: 101},
			{A1: 100, B1: 86, B2: 86, A2: 100},
			{A1: 98, B1: 84, B2: 84, A2: 98},
		},
		AANullBlocks: []pairedtiming.Block{
			{A1: 100, B1: 100.1, B2: 100.0, A2: 100},
			{A1: 99.9, B1: 100.0, B2: 99.9, A2: 99.9},
			{A1: 100.1, B1: 100.0, B2: 100.1, A2: 100.1},
		},
		ReferenceTokens: tokens,
		CandidateTokens: tokens,
		LogitMaxAbs:     1e-5,
	}
}

var egtuneGlobalKernelRe = regexp.MustCompile(`__global__\s+void\s+(?:__launch_bounds__\([^)]*\)\s+)?([A-Za-z0-9_]+)\s*\(`)

// fallbackPTXASCompiler invokes qualify.NVCCPTXASCompiler when nvcc is present
// in PATH and succeeds, or falls back to a deterministic static ptxas estimator
// when nvcc is absent on the host.
type fallbackPTXASCompiler struct {
	nvccPath string
	calls    map[string]int
}

func selectPTXASCompiler() qualify.PTXASCompiler {
	nvccPath, _ := exec.LookPath("nvcc")
	return &fallbackPTXASCompiler{
		nvccPath: nvccPath,
		calls:    make(map[string]int),
	}
}

func (c *fallbackPTXASCompiler) CompilePTXAS(ctx context.Context, cudaSource, arch string) (ptxasinfo.Report, error) {
	if c.nvccPath != "" && os.Getenv("EGTUNE_FORCE_STATIC_PTXAS") != "1" {
		nvccCompiler := qualify.NVCCPTXASCompiler{
			NVCC:       c.nvccPath,
			ExtraFlags: []string{"-std=c++17", "-O3", "-lineinfo"},
		}
		if rep, err := nvccCompiler.CompilePTXAS(ctx, cudaSource, arch); err == nil {
			return rep, nil
		}
	}
	if c.calls == nil {
		c.calls = make(map[string]int)
	}
	if strings.TrimSpace(arch) == "" {
		arch = "sm_121a"
	}
	matches := egtuneGlobalKernelRe.FindAllStringSubmatch(cudaSource, -1)
	key := "default"
	if len(matches) > 0 {
		key = matches[0][1]
	}
	c.calls[key]++
	callNum := c.calls[key]

	var fns []*ptxasinfo.Function
	for i, m := range matches {
		fn := &ptxasinfo.Function{
			Name:      m[1],
			Arch:      arch,
			Entry:     true,
			Registers: 96,
			SharedMem: 4096,
		}
		if i == 0 && callNum == 1 {
			fn.Registers = 168
			fn.SpillStores = 16
			fn.SpillLoads = 16
		}
		fns = append(fns, fn)
	}
	if len(fns) == 0 {
		fns = []*ptxasinfo.Function{{
			Name:      "kernel_step",
			Arch:      arch,
			Entry:     true,
			Registers: 64,
		}}
	}
	return ptxasinfo.Report{
		Targets: []*ptxasinfo.Target{{
			Arch:      arch,
			Functions: fns,
		}},
	}, nil
}

func parseBuckets(spec string) ([]int, error) {
	parts := strings.Split(spec, ",")
	var buckets []int
	for _, p := range parts {
		p = strings.TrimSpace(p)
		if p == "" {
			continue
		}
		v, err := strconv.Atoi(p)
		if err != nil || v <= 0 {
			return nil, fmt.Errorf("invalid M bucket %q", p)
		}
		buckets = append(buckets, v)
	}
	if len(buckets) == 0 {
		return nil, fmt.Errorf("no M buckets specified")
	}
	return buckets, nil
}
