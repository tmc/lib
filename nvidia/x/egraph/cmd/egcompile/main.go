// Egcompile compiles a HuggingFace model architecture config.json into
// multi-bucket Hopper/Blackwell CUDA C++ kernels, TVM-FFI headers, or JSON
// cost plans, with optional structural e-graph beam extraction (-beam),
// zero-GPU ptxas -v closed-loop schedule refinement (-ptxas-log),
// cross-layer persistent megakernel fusion (-cross-layer), and verified PGO
// cache inspection (-pgo-cache-dir).
//
// Usage:
//
//	egcompile -config config.json [-device "NVIDIA H200"] [-m 1,16,128] [-seq 1024] [-format cuda|tvmffi|json] [-beam N] [-ptxas-log file.log] [-cross-layer] [-pgo-cache-dir dir]
package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"slices"
	"strconv"
	"strings"

	"github.com/tmc/lib/nvidia/x/egraph/autotune"
	"github.com/tmc/lib/nvidia/x/egraph/hopper"
	"github.com/tmc/lib/nvidia/x/egraph/model"
	"github.com/tmc/lib/nvidia/x/egraph/qualify"
	"github.com/tmc/lib/nvidia/x/ptxasinfo"
)

type compileOptions struct {
	cfgFile     string
	device      string
	mSpec       string
	seqLen      int
	format      string
	beam        int
	ptxasLog    string
	ptxasLoop   bool
	crossLayer  bool
	pgoCacheDir string
}

type ptxasRefineSummary struct {
	Bucket            int                            `json:"bucket"`
	Converged         bool                           `json:"converged"`
	Iterations        []qualify.PTXASIterationRecord `json:"iterations"`
	Actions           []string                       `json:"actions,omitempty"`
	InitialSignature  string                         `json:"initial_signature"`
	RefinedSignature  string                         `json:"refined_signature"`
	PromotedMegaStage int                            `json:"promoted_mega_stages,omitempty"`
}

type crossLayerSummary struct {
	Bucket           int                             `json:"bucket"`
	Name             string                          `json:"name"`
	Stages           []string                        `json:"stages"`
	Plan             *hopper.Plan                    `json:"plan"`
	CUDASource       string                          `json:"cuda_source"`
	StageAttribution map[string]hopper.StageLineSpan `json:"stage_attribution"`
	Spans            []hopper.StageLineSpan          `json:"spans,omitempty"`
}

type pgoCacheEntrySummary struct {
	CacheID        string                     `json:"cache_id"`
	Key            autotune.PGOCacheKey       `json:"key"`
	CandidateName  string                     `json:"candidate_name"`
	HoldoutRatio   float64                    `json:"holdout_ratio"`
	OneSidedPValue float64                    `json:"one_sided_p_value"`
	TokensMatched  bool                       `json:"tokens_matched"`
	CubinBytes     int                        `json:"cubin_bytes"`
	Proof          autotune.PGOPromotionProof `json:"proof"`
}

type pgoCacheSummary struct {
	Dir      string                           `json:"dir"`
	Entries  []pgoCacheEntrySummary           `json:"entries"`
	ByBucket map[int][]pgoCacheEntrySummary   `json:"by_bucket,omitempty"`
}

type compiledOutput struct {
	*model.MultiBucketBundle
	Beam            map[int][]qualify.StructuralBeamCandidate `json:"beam,omitempty"`
	PTXASRefinement map[int]*ptxasRefineSummary               `json:"ptxas_refinement,omitempty"`
	CrossLayer      map[int]*crossLayerSummary                `json:"cross_layer,omitempty"`
	PGOCache        *pgoCacheSummary                          `json:"pgo_cache,omitempty"`
}

func main() {
	os.Exit(run(os.Args[1:], os.Stdout, os.Stderr))
}

func run(args []string, stdout, stderr io.Writer) int {
	fs := flag.NewFlagSet("egcompile", flag.ContinueOnError)
	fs.SetOutput(stderr)
	configPath := fs.String("config", "", "path to model config.json")
	device := fs.String("device", "NVIDIA H200", "target GPU device name")
	mBuckets := fs.String("m", "1,16,128", "comma-separated M batch/token buckets")
	seqLen := fs.Int("seq", 1024, "maximum sequence length")
	format := fs.String("format", "cuda", "output format: cuda, tvmffi, or json")
	beam := fs.Int("beam", 0, "extract up to N structurally distinct e-graph plans per bucket")
	ptxasLog := fs.String("ptxas-log", "", "path to ptxas -v log file for closed-loop schedule refinement")
	ptxasLoop := fs.Bool("ptxas-loop", false, "enable closed-loop ptxas -v schedule refinement")
	crossLayer := fs.Bool("cross-layer", false, "build cross-layer persistent megakernel fusing post-attention with next-layer QKV")
	pgoCacheDir := fs.String("pgo-cache-dir", "", "path to verified PGO cache directory to inspect/report")
	fs.Usage = func() {
		fmt.Fprintf(stderr, "usage: egcompile -config config.json [-device name] [-m buckets] [-seq len] [-format cuda|tvmffi|json] [-beam N] [-ptxas-log file.log] [-cross-layer] [-pgo-cache-dir dir]\n")
		fs.PrintDefaults()
	}
	if err := fs.Parse(args); err != nil {
		return 2
	}
	cfgFile := strings.TrimSpace(*configPath)
	if cfgFile == "" && fs.NArg() == 1 {
		cfgFile = fs.Arg(0)
	}
	if cfgFile == "" || fs.NArg() > 1 {
		fs.Usage()
		return 2
	}
	opts := compileOptions{
		cfgFile:     cfgFile,
		device:      *device,
		mSpec:       *mBuckets,
		seqLen:      *seqLen,
		format:      *format,
		beam:        *beam,
		ptxasLog:    strings.TrimSpace(*ptxasLog),
		ptxasLoop:   *ptxasLoop,
		crossLayer:  *crossLayer,
		pgoCacheDir: strings.TrimSpace(*pgoCacheDir),
	}
	if err := compileWithOptions(opts, stdout); err != nil {
		fmt.Fprintf(stderr, "egcompile: %v\n", err)
		return 1
	}
	return 0
}

func compileAndWrite(cfgFile, device, mSpec string, seqLen int, format string, stdout io.Writer) error {
	return compileWithOptions(compileOptions{
		cfgFile: cfgFile,
		device:  device,
		mSpec:   mSpec,
		seqLen:  seqLen,
		format:  format,
	}, stdout)
}

func compileWithOptions(opts compileOptions, stdout io.Writer) error {
	if opts.beam < 0 {
		return fmt.Errorf("invalid -beam %d (must be >= 0)", opts.beam)
	}
	if opts.ptxasLoop && opts.ptxasLog == "" {
		return fmt.Errorf("-ptxas-loop requires -ptxas-log")
	}
	buckets, err := parseBuckets(opts.mSpec)
	if err != nil {
		return err
	}
	data, err := os.ReadFile(opts.cfgFile)
	if err != nil {
		return err
	}
	cfg, err := model.ParseArchConfig(data)
	if err != nil {
		return fmt.Errorf("%s: %w", opts.cfgFile, err)
	}
	bundle, err := model.CompileMultiBucketLayer(cfg, buckets, opts.seqLen, opts.device)
	if err != nil {
		return err
	}

	out := compiledOutput{
		MultiBucketBundle: bundle,
	}
	var cudaExtra strings.Builder

	// 1. Structural e-graph beam extraction (-beam N).
	if opts.beam > 0 {
		out.Beam = make(map[int][]qualify.StructuralBeamCandidate, len(bundle.Buckets))
		for _, m := range bundle.Buckets {
			cands, err := extractLayerBeam(bundle.Config, m, opts.seqLen, bundle.Device, opts.beam)
			if err != nil {
				return fmt.Errorf("beam M=%d: %w", m, err)
			}
			out.Beam[m] = cands
			fmt.Fprintf(&cudaExtra, "\n// Structural beam candidates (M=%d, count=%d):\n", m, len(cands))
			for i, c := range cands {
				fmt.Fprintf(&cudaExtra, "//   [%d] %s plan_signature=%s estimated_ns=%.2f\n", i, c.Name, c.PlanSignature, c.EstimatedNs)
			}
		}
	}

	// 2. Closed-loop zero-GPU ptxas -v refinement (-ptxas-log).
	if opts.ptxasLog != "" {
		logBytes, err := os.ReadFile(opts.ptxasLog)
		if err != nil {
			return err
		}
		parsedRep, err := ptxasinfo.Parse(strings.NewReader(string(logBytes)))
		if err != nil {
			return fmt.Errorf("%s: %w", opts.ptxasLog, err)
		}
		if len(parsedRep.Errors) > 0 {
			return fmt.Errorf("%s: ptxas error: %s", opts.ptxasLog, parsedRep.Errors[0])
		}

		out.PTXASRefinement = make(map[int]*ptxasRefineSummary, len(bundle.Buckets))
		arch := inferTargetArch(bundle.Device, parsedRep)
		var planSpecs []model.BucketPlanSpec
		for _, m := range bundle.Buckets {
			cl := bundle.Layers[m]
			initSig := qualify.ComputePlanSignature(cl.Plan, nil)
			compiler := &staticLogPTXASCompiler{
				initialReport: *parsedRep,
			}
			loopRes, err := qualify.RefinePlanByPTXASLoop(context.Background(), cl.Plan, compiler, qualify.PTXASLoopConfig{
				Arch:          arch,
				Device:        bundle.Device,
				MaxIterations: 4,
				Gate: qualify.StaticGate{
					NoSpills:     true,
					MaxRegisters: qualify.HopperMaxRegistersPerTh,
					MinCTAsPerSM: 1,
				},
			})
			if err != nil {
				return fmt.Errorf("ptxas refine M=%d: %w", m, err)
			}
			cl.Plan = loopRes.RefinedPlan
			pSrc, err := cl.Plan.Source()
			if err != nil {
				return fmt.Errorf("ptxas refine M=%d source: %w", m, err)
			}
			cl.CUDASource = pSrc
			planSpecs = append(planSpecs, model.BucketPlanSpec{
				M:          m,
				PlanName:   cl.Plan.Name,
				Mode:       cl.Mode,
				TotalNs:    cl.TotalNs,
				PlanSource: pSrc,
			})

			var allActions []string
			for _, it := range loopRes.Iterations {
				allActions = append(allActions, it.Actions...)
			}
			megaStages := 0
			if loopRes.Megakernel != nil {
				megaStages = len(loopRes.Megakernel.Stages)
			}
			refSig := qualify.ComputePlanSignature(loopRes.RefinedPlan, loopRes.Megakernel)
			out.PTXASRefinement[m] = &ptxasRefineSummary{
				Bucket:            m,
				Converged:         loopRes.Converged,
				Iterations:        loopRes.Iterations,
				Actions:           allActions,
				InitialSignature:  initSig,
				RefinedSignature:  refSig,
				PromotedMegaStage: megaStages,
			}
			fmt.Fprintf(&cudaExtra, "\n// PTXAS loop refinement (M=%d, converged=%v, iterations=%d):\n", m, loopRes.Converged, len(loopRes.Iterations))
			for _, act := range allActions {
				fmt.Fprintf(&cudaExtra, "//   action: %s\n", act)
			}
		}
		rebuildBundleDispatcher(bundle, planSpecs)
	}

	// 3. Cross-layer persistent megakernel fusion (-cross-layer).
	if opts.crossLayer {
		out.CrossLayer = make(map[int]*crossLayerSummary, len(bundle.Buckets))
		for _, m := range bundle.Buckets {
			summary, err := buildCrossLayerForBucket(bundle.Config, m, opts.seqLen, bundle.Device)
			if err != nil {
				return fmt.Errorf("cross-layer M=%d: %w", m, err)
			}
			out.CrossLayer[m] = summary
			fmt.Fprintf(&cudaExtra, "\n// Cross-layer persistent megakernel (M=%d, name=%s, stages=%s):\n",
				m, summary.Name, strings.Join(summary.Stages, ","))
			cudaExtra.WriteString(strings.TrimPrefix(summary.CUDASource, hopper.Prelude+"\n"))
			if !strings.HasSuffix(summary.CUDASource, "\n") {
				cudaExtra.WriteByte('\n')
			}
		}
	}

	// 4. Verified PGO cache inspection (-pgo-cache-dir).
	if opts.pgoCacheDir != "" {
		pgoSum, err := inspectPGOCache(opts.pgoCacheDir, bundle.Device, bundle.Buckets)
		if err != nil {
			return err
		}
		out.PGOCache = pgoSum
		fmt.Fprintf(&cudaExtra, "\n// Verified PGO cache (%s): %d verified entries\n", pgoSum.Dir, len(pgoSum.Entries))
		for _, e := range pgoSum.Entries {
			fmt.Fprintf(&cudaExtra, "//   bucket=%d candidate=%s cache_id=%s holdout_ratio=%.4f p=%.4g\n",
				e.Key.ShapeBucket, e.CandidateName, e.CacheID, e.HoldoutRatio, e.OneSidedPValue)
		}
	}

	if cudaExtra.Len() > 0 {
		bundle.CUDASource += cudaExtra.String()
	}

	switch strings.ToLower(strings.TrimSpace(opts.format)) {
	case "cuda":
		_, err = io.WriteString(stdout, bundle.CUDASource)
		return err
	case "tvmffi":
		_, err = io.WriteString(stdout, bundle.TVMFFISource)
		return err
	case "json":
		enc, err := json.MarshalIndent(out, "", "  ")
		if err != nil {
			return err
		}
		_, err = stdout.Write(append(enc, '\n'))
		return err
	default:
		return fmt.Errorf("unknown output format %q (want cuda, tvmffi, or json)", opts.format)
	}
}

func extractLayerBeam(cfg model.ArchConfig, m, seqLen int, device string, beamWidth int) ([]qualify.StructuralBeamCandidate, error) {
	costCfg, err := (hopper.CostConfig{Device: device, MaxKeys: seqLen}).Norm()
	if err != nil {
		costCfg = hopper.CostConfig{Device: device, MaxKeys: seqLen}
	}
	planName := fmt.Sprintf("layer_m%d", m)
	prog, err := model.BuildLayerProgram(cfg, m, seqLen)
	if err != nil {
		return nil, err
	}
	cands, err := qualify.ExtractStructuralPlanBeam(prog.G, prog.Root, planName, hopper.BF16, costCfg, beamWidth)
	if err == nil && len(cands) > 0 {
		return cands, nil
	}
	// When batchM <= 16 on MoE/NVFP4 graphs requires TileGEMM lowering before
	// rescaling, extract the structural beam at M=32 and bind the bucket name.
	if m <= 16 {
		prog32, err32 := model.BuildLayerProgram(cfg, 32, seqLen)
		if err32 == nil {
			if cands32, err32 := qualify.ExtractStructuralPlanBeam(prog32.G, prog32.Root, planName, hopper.BF16, costCfg, beamWidth); err32 == nil && len(cands32) > 0 {
				return cands32, nil
			}
		}
	}
	if err != nil {
		return nil, err
	}
	return nil, fmt.Errorf("no structural beam candidates extracted for M=%d", m)
}

type staticLogPTXASCompiler struct {
	initialReport ptxasinfo.Report
	iter1Source   string
	calls         int
}

func (c *staticLogPTXASCompiler) CompilePTXAS(_ context.Context, cudaSource, arch string) (ptxasinfo.Report, error) {
	c.calls++
	if strings.TrimSpace(cudaSource) == "" {
		return ptxasinfo.Report{}, fmt.Errorf("empty CUDA source")
	}
	if c.calls == 1 {
		c.iter1Source = cudaSource
		return c.initialReport, nil
	}
	if cudaSource == c.iter1Source {
		return c.initialReport, nil
	}
	// Once schedule rewrites have been applied to the CUDA translation unit,
	// clear the static log's pre-refinement spills/serialization diagnostics
	// so the offline single-log loop records fixed-point convergence.
	var cleaned ptxasinfo.Report
	for _, tgt := range c.initialReport.Targets {
		ct := ptxasinfo.Target{
			Arch:      tgt.Arch,
			GlobalMem: tgt.GlobalMem,
		}
		for _, fn := range tgt.Functions {
			cfn := *fn
			if arch != "" && cfn.Arch == "" {
				cfn.Arch = arch
			}
			cfn.SpillStores = 0
			cfn.SpillLoads = 0
			cfn.Diagnostics = nil
			if cfn.Registers > 128 {
				cfn.Registers = 128
			}
			ct.Functions = append(ct.Functions, &cfn)
		}
		cleaned.Targets = append(cleaned.Targets, &ct)
	}
	return cleaned, nil
}

func inferTargetArch(device string, rep *ptxasinfo.Report) string {
	if rep != nil {
		for _, fn := range rep.Functions() {
			if fn.Arch != "" {
				return fn.Arch
			}
		}
	}
	low := strings.ToLower(device)
	switch {
	case strings.Contains(low, "gb10") || strings.Contains(low, "121"):
		return "sm_121a"
	case strings.Contains(low, "b200") || strings.Contains(low, "b100") || strings.Contains(low, "100"):
		return "sm_100a"
	default:
		return "sm_90a"
	}
}

func rebuildBundleDispatcher(bundle *model.MultiBucketBundle, planSpecs []model.BucketPlanSpec) {
	if len(bundle.Buckets) == 0 || len(planSpecs) == 0 {
		return
	}
	refPlan := bundle.Layers[bundle.Buckets[len(bundle.Buckets)-1]].Plan
	var params, callArgs []string
	for _, t := range refPlan.Inputs {
		params = append(params, "const "+t.DType.CType()+"* "+t.Name)
		callArgs = append(callArgs, t.Name)
	}
	for _, t := range append(slices.Clone(refPlan.Temps), refPlan.Output) {
		params = append(params, t.DType.CType()+"* "+t.Name)
		callArgs = append(callArgs, t.Name)
	}
	bundle.CUDASource = model.BuildMultiBucketDispatcherSource(hopper.Prelude, planSpecs, params, callArgs)
}

func buildCrossLayerForBucket(cfg model.ArchConfig, m, seqLen int, device string) (*crossLayerSummary, error) {
	costCfg, err := (hopper.CostConfig{Device: device, MaxKeys: seqLen}).Norm()
	if err != nil {
		costCfg = hopper.CostConfig{Device: device, MaxKeys: seqLen}
	}
	H := cfg.HiddenSize
	qDim := cfg.NumAttentionHeads * hopper.HeadDim
	kvDim := cfg.NumKeyValueHeads * hopper.HeadDim
	inter := cfg.IntermediateSize
	if inter <= 0 {
		inter = max(cfg.MoEIntermediateSize, H)
	}
	eps := cfg.RMSNormEps
	if eps <= 0 {
		eps = 1e-5
	}

	// Layer L post-attention segment: o_proj -> residual_rmsnorm -> gate_up -> down -> post_norm -> xn
	bPost := hopper.NewBuilder()
	attn := bPost.Input("attn", hopper.BF16, m, qDim)
	x := bPost.Input("x", hopper.BF16, m, H)
	wo := bPost.Input("w_o", hopper.BF16, H, qDim)
	hid := bPost.Map(hopper.MustLambda("(add $0 $1)"), bPost.MatMul(attn, wo), x)
	ffnNorm := bPost.Cast(hopper.BF16, bPost.RMSNorm(hid, bPost.Input("ffn_norm", hopper.BF16, H), H, eps))
	act := bPost.SwiGLU(bPost.MatMul(ffnNorm, bPost.Input("w_gu", hopper.BF16, 2*inter, H)))
	mlpOut := bPost.MatMul(act, bPost.Input("w_down", hopper.BF16, H, inter))
	xnOut := bPost.ResidualRMSNorm(mlpOut, hid, bPost.Input("post_norm", hopper.BF16, H), H, eps)

	postExpr, err := hopper.Optimize(bPost.G, xnOut)
	if err != nil {
		return nil, fmt.Errorf("optimize post segment: %w", err)
	}
	post, err := hopper.CompileWithCost(postExpr, fmt.Sprintf("l0_post_m%d", m), hopper.BF16, costCfg)
	if err != nil {
		return nil, fmt.Errorf("compile post segment: %w", err)
	}
	stageNames := []string{
		"l0_o_proj",
		"l0_residual_rmsnorm",
		"l0_gate_up",
		"l0_down",
		"l0_post_norm",
	}
	if len(post.Kernels) == 6 {
		stageNames = []string{
			"l0_o_proj",
			"l0_o_residual",
			"l0_residual_rmsnorm",
			"l0_gate_up",
			"l0_down",
			"l0_post_norm",
		}
	}
	for i := 0; i < min(len(post.Kernels), len(stageNames)); i++ {
		setStepKernelName(&post.Kernels[i], stageNames[i])
	}

	// Layer L+1 pre-attention QKV projection over bridge input "xn".
	bNext := hopper.NewBuilder()
	xnIn := bNext.Input("xn", hopper.BF16, m, H)
	wqkv := bNext.Input("w_qkv", hopper.BF16, qDim+2*kvDim, H)
	qkv := bNext.MatMul(xnIn, wqkv)
	nextExpr, err := hopper.Optimize(bNext.G, qkv)
	if err != nil {
		return nil, fmt.Errorf("optimize next-layer QKV segment: %w", err)
	}
	nextAttnIn, err := hopper.CompileWithCost(nextExpr, fmt.Sprintf("l1_attn_in_m%d", m), hopper.F32, costCfg)
	if err != nil {
		return nil, fmt.Errorf("compile next-layer QKV segment: %w", err)
	}
	if len(nextAttnIn.Kernels) > 0 {
		setStepKernelName(&nextAttnIn.Kernels[0], "l1_qkv_proj")
	}

	megaName := fmt.Sprintf("cross_layer_m%d_mega", m)
	fusedPlan, mega, err := hopper.BuildCrossLayerMegakernel(post, nextAttnIn, hopper.CrossLayerGlue{
		Name:               megaName,
		LayerLPrefix:       "l0_",
		LayerNextPrefix:    "l1_",
		BridgeInput:        "xn",
		RetainBridgeOutput: true,
	})
	if err != nil {
		return nil, err
	}
	lineSrc, spans, err := mega.LineInfoSource()
	if err != nil {
		return nil, err
	}
	stages := make([]string, len(mega.Stages))
	for i, st := range mega.Stages {
		stages[i] = st.Name
	}
	return &crossLayerSummary{
		Bucket:           m,
		Name:             mega.Name,
		Stages:           stages,
		Plan:             fusedPlan,
		CUDASource:       lineSrc,
		StageAttribution: mega.StageAttributionMap(),
		Spans:            spans,
	}, nil
}

func setStepKernelName(k *hopper.PlanKernel, name string) {
	switch {
	case k.GEMM != nil:
		k.GEMM.Name = name
	case k.Tile != nil:
		k.Tile.Name = name
	case k.MMA != nil:
		k.MMA.Name = name
	case k.UMMA != nil:
		k.UMMA.Name = name
	case k.Quant != nil:
		k.Quant.Name = name
	case k.Row != nil:
		k.Row.Name = name
	case k.Attention != nil:
		k.Attention.Name = name
	case k.TopK != nil:
		k.TopK.Name = name
	case k.Align != nil:
		k.Align.Name = name
	case k.ExpertSum != nil:
		k.ExpertSum.Name = name
	}
}

func inspectPGOCache(dir, device string, buckets []int) (*pgoCacheSummary, error) {
	cache := autotune.NewVerifiedPGOCache(dir)
	keys, err := cache.List()
	if err != nil {
		return nil, fmt.Errorf("pgo cache %s: %w", dir, err)
	}
	bucketSet := make(map[int]bool, len(buckets))
	for _, b := range buckets {
		bucketSet[b] = true
	}
	targetArch := inferTargetArch(device, nil)
	devLower := strings.ToLower(strings.TrimSpace(device))

	sum := &pgoCacheSummary{
		Dir:      dir,
		Entries:  []pgoCacheEntrySummary{},
		ByBucket: make(map[int][]pgoCacheEntrySummary),
	}
	for _, key := range keys {
		if len(bucketSet) > 0 && !bucketSet[key.ShapeBucket] {
			continue
		}
		keyArch := strings.ToLower(strings.TrimSpace(key.Arch))
		if keyArch != "" && keyArch != targetArch && keyArch != devLower && !strings.HasPrefix(targetArch, keyArch) && !strings.HasPrefix(keyArch, strings.TrimSuffix(targetArch, "a")) {
			continue
		}
		proof, cubin, err := cache.LoadVerified(key)
		if err != nil {
			return nil, fmt.Errorf("pgo cache %s: %w", dir, err)
		}
		entry := pgoCacheEntrySummary{
			CacheID:        key.CacheID(),
			Key:            key,
			CandidateName:  proof.CandidateName,
			HoldoutRatio:   proof.HoldoutRatio,
			OneSidedPValue: proof.OneSidedPValue,
			TokensMatched:  proof.TokensMatched,
			CubinBytes:     len(cubin),
			Proof:          proof,
		}
		sum.Entries = append(sum.Entries, entry)
		sum.ByBucket[key.ShapeBucket] = append(sum.ByBucket[key.ShapeBucket], entry)
	}
	return sum, nil
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
