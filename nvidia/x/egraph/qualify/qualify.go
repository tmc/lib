package qualify

import (
	"context"
	"fmt"
	"math"
	"strings"

	"github.com/tmc/lib/nvidia/x/cudahostemu"
	"github.com/tmc/lib/nvidia/x/cudatmem"
	"github.com/tmc/lib/nvidia/x/egraph/bitopt"
	"github.com/tmc/lib/nvidia/x/egraph/tilebridge"
	"github.com/tmc/lib/nvidia/x/sassequiv"
	"github.com/tmc/lib/nvidia/x/tcgen05"
	"github.com/tmc/lib/nvidia/x/tileir"
)

// StageStatus reports the outcome of a single qualification stage.
type StageStatus string

const (
	StagePassed  StageStatus = "passed"
	StageFailed  StageStatus = "failed"
	StageSkipped StageStatus = "skipped"
)

// StageResult records the status and diagnostic detail of one qualification stage.
type StageResult struct {
	Stage   string      `json:"stage"`
	Status  StageStatus `json:"status"`
	Summary string      `json:"summary,omitempty"`
}

// QualifyOptions configures optional verification stages in [QualifySource].
type QualifyOptions struct {
	CheckHazards   bool
	CheckNVFP4SASS bool
	TMEMArch       cudatmem.Target
	TMEMEvents     []cudatmem.TraceEvent
	TileIRStep     *tilebridge.Step
	DualM          int
	DualN          int
	DualK          int
	Eps            float32
}

// QualifyReport summarizes a multi-stage qualification run from [QualifySource].
type QualifyReport struct {
	Name    string               `json:"name"`
	Receipt QualificationReceipt `json:"receipt"`
	Stages  []StageResult        `json:"stages"`
}

// Passed reports whether the qualification report passed all active checks.
func (r QualifyReport) Passed() bool {
	for _, s := range r.Stages {
		if s.Status == StageFailed {
			return false
		}
	}
	return r.Receipt.Passed()
}

// QualificationReceipt records the outcome of all host-side formal and
// adversarial checks for a compiled kernel Step or Plan.
type QualificationReceipt struct {
	Name              string
	CUDAValid         bool
	HazardsChecked    bool
	HazardCount       int
	Hazards           []cudahostemu.Hazard
	SASSEquivVerified bool
	TileIRAgreement   bool
	TileIRMaxAbsDiff  float64
	TMEMTraceVerified bool
	TMEMPeakColumns   uint32
}

// Passed reports whether the qualification receipt passed all applicable checks
// with zero concurrency/uniformity hazards.
func (r QualificationReceipt) Passed() bool {
	if !r.CUDAValid {
		return false
	}
	if r.HazardCount != 0 || len(r.Hazards) != 0 {
		return false
	}
	if !r.HazardsChecked && !r.TMEMTraceVerified {
		return false
	}
	if r.TileIRAgreement {
		if math.IsNaN(r.TileIRMaxAbsDiff) || math.IsInf(r.TileIRMaxAbsDiff, 0) || r.TileIRMaxAbsDiff >= 2e-2 {
			return false
		}
	}
	if r.TMEMTraceVerified {
		if r.TMEMPeakColumns == 0 || r.TMEMPeakColumns > 512 {
			return false
		}
	}
	return true
}

// KernelStepSpec describes a kernel step to qualify via [QualifyStepSpec].
type KernelStepSpec struct {
	Name             string
	Kind             string
	CUDASource       string
	LauncherSource   string
	Prelude          string
	Launch           cudahostemu.Launch
	TemplateM        int
	Args             []cudahostemu.Arg
	LeadingIntArgs   []int
	ParamDecls       []string
	DefaultBufBytes  int
	NamedBuffers     map[string][]byte
	NamedInts        map[string]int
	CheckNVFP4SASS   bool
	TileIRBytecode   *tilebridge.Step
	TileIRDualTarget *tilebridge.Step
	DualM            int
	DualN            int
	DualK            int
	Eps              float32
}

// QualifyKernelSpec is an alias for [KernelStepSpec].
type QualifyKernelSpec = KernelStepSpec

// PlanSpec describes a multi-step execution plan to qualify via [QualifyPlanSpec].
type PlanSpec struct {
	Name               string
	SourceFn           func() (string, error)
	Steps              []KernelStepSpec
	PDLCUDAFn          func() (string, error)
	MegakernelSourceFn func() (string, error)
}

// QualifySource runs a configurable qualification pipeline over raw CUDA source.
func QualifySource(ctx context.Context, name, src string, launch cudahostemu.Launch, opts QualifyOptions, args ...cudahostemu.Arg) (*QualifyReport, error) {
	if strings.TrimSpace(src) == "" {
		return nil, fmt.Errorf("qualify source %s: empty CUDA source", name)
	}
	rep := &QualifyReport{
		Name: name,
		Receipt: QualificationReceipt{
			Name:      name,
			CUDAValid: true,
		},
		Stages: []StageResult{
			{Stage: "cuda_emit", Status: StagePassed, Summary: "non-empty CUDA translation unit"},
		},
	}

	if opts.CheckHazards {
		hazRep, err := cudahostemu.CheckHazards(ctx, src, launch, args...)
		if err != nil {
			rep.Stages = append(rep.Stages, StageResult{Stage: "hazards", Status: StageFailed, Summary: err.Error()})
			return rep, fmt.Errorf("qualify source %s CheckHazards: %w", name, err)
		}
		rep.Receipt.HazardsChecked = true
		rep.Receipt.HazardCount = len(hazRep.Hazards)
		rep.Receipt.Hazards = hazRep.Hazards
		if rep.Receipt.HazardCount > 0 {
			rep.Stages = append(rep.Stages, StageResult{Stage: "hazards", Status: StageFailed, Summary: fmt.Sprintf("%d hazards", rep.Receipt.HazardCount)})
			return rep, fmt.Errorf("qualify source %s: detected %d hazards: %+v", name, rep.Receipt.HazardCount, hazRep.Hazards)
		}
		rep.Stages = append(rep.Stages, StageResult{Stage: "hazards", Status: StagePassed, Summary: "0 hazards"})
	}

	if len(opts.TMEMEvents) > 0 {
		arch := opts.TMEMArch
		if arch == "" {
			arch = cudatmem.SM100A
		}
		sum, err := cudatmem.VerifyTrace(arch, opts.TMEMEvents)
		if err != nil {
			rep.Stages = append(rep.Stages, StageResult{Stage: "tmem", Status: StageFailed, Summary: err.Error()})
			return rep, fmt.Errorf("qualify source %s VerifyTrace: %w", name, err)
		}
		rep.Receipt.TMEMTraceVerified = true
		rep.Receipt.TMEMPeakColumns = sum.MaxAllocatedColumns
		rep.Stages = append(rep.Stages, StageResult{Stage: "tmem", Status: StagePassed, Summary: fmt.Sprintf("peak_cols=%d", sum.MaxAllocatedColumns)})
	}

	if opts.CheckNVFP4SASS {
		if err := VerifyNVFP4SASS(); err != nil {
			rep.Stages = append(rep.Stages, StageResult{Stage: "sass_equiv", Status: StageFailed, Summary: err.Error()})
			return rep, fmt.Errorf("qualify source %s SASS: %w", name, err)
		}
		rep.Receipt.SASSEquivVerified = true
		rep.Stages = append(rep.Stages, StageResult{Stage: "sass_equiv", Status: StagePassed, Summary: "NVFP4 dequant & E4M3 scale proven"})
	}

	if opts.TileIRStep != nil {
		if err := VerifyStepTileIRBytecode(*opts.TileIRStep); err != nil {
			rep.Stages = append(rep.Stages, StageResult{Stage: "tileir", Status: StageFailed, Summary: err.Error()})
			return rep, fmt.Errorf("qualify source %s TileIR bytecode: %w", name, err)
		}
		eps := opts.Eps
		if eps <= 0 {
			eps = 1e-5
		}
		dtRep, err := tilebridge.VerifyDualTargetAgreement(ctx, *opts.TileIRStep, opts.DualM, opts.DualN, opts.DualK, eps)
		if err != nil {
			rep.Stages = append(rep.Stages, StageResult{Stage: "tileir", Status: StageFailed, Summary: err.Error()})
			return rep, fmt.Errorf("qualify source %s VerifyDualTargetAgreement: %w", name, err)
		}
		rep.Receipt.TileIRAgreement = true
		rep.Receipt.TileIRMaxAbsDiff = dtRep.MaxAbsDiff
		rep.Stages = append(rep.Stages, StageResult{Stage: "tileir", Status: StagePassed, Summary: fmt.Sprintf("max_abs_diff=%g", dtRep.MaxAbsDiff)})
	}

	return rep, nil
}

// QualifyStepSpec runs the host-side static and dynamic qualification gate on spec:
//  1. Validates the emitted CUDA C++ kernel and launcher source.
//  2. Runs cudahostemu.CheckHazards on a synthetic launch of spec, verifying
//     0 shared-memory races, 0 DSMEM-before-wait violations, 0 C7519 WGMMA
//     warpgroup divergences, and 0 named barrier mismatches.
//  3. When spec.CheckNVFP4SASS is true, verifies symbolic SASS equivalence via
//     [VerifyDequantProgramsSASS] and [ProveScaleConversionSASS].
//  4. When spec.TileIRBytecode or spec.TileIRDualTarget is non-nil, lowers the
//     step to a *tileir.Module (\x7fTileIR\x00 bytecode) and verifies
//     dual-target numerical agreement against cudahostemu.Run.
func QualifyStepSpec(ctx context.Context, spec KernelStepSpec) (*QualificationReceipt, error) {
	kind := spec.Kind
	if kind == "" {
		kind = "kernel"
	}
	receipt := &QualificationReceipt{Name: spec.Name}
	if strings.TrimSpace(spec.CUDASource) == "" {
		return receipt, fmt.Errorf("%s %s CUDA: empty source", kind, spec.Name)
	}
	if strings.TrimSpace(spec.LauncherSource) == "" {
		return receipt, fmt.Errorf("%s %s Launcher: empty source", kind, spec.Name)
	}
	receipt.CUDAValid = true

	args := spec.Args
	var argNames []string
	if len(args) == 0 {
		bufBytes := max(spec.DefaultBufBytes, 65536)
		for _, v := range spec.LeadingIntArgs {
			args = append(args, cudahostemu.IntArg(v))
		}
		for _, p := range spec.ParamDecls {
			name, ctype := ParseParamDecl(p)
			argNames = append(argNames, name)
			if v, ok := spec.NamedInts[name]; ok {
				args = append(args, cudahostemu.IntArg(v))
				continue
			}
			if b, ok := spec.NamedBuffers[name]; ok {
				args = append(args, cudahostemu.BufferArg(name, ctype, b))
				continue
			}
			args = append(args, cudahostemu.ZeroBufferArg(name, ctype, bufBytes))
		}
	}

	launch := spec.Launch
	if spec.TemplateM > 0 && launch.Kernel == "" {
		launch.Kernel = fmt.Sprintf("eg_emu_launch(dim3(%d, %d, 1), dim3(%d, 1, 1), 0, %s<%d>, %s)",
			launch.Grid.X, launch.Grid.Y, launch.Block.X, spec.Name, spec.TemplateM, strings.Join(argNames, ", "))
		launch.Grid = cudahostemu.Dim3{}
		launch.Block = cudahostemu.Dim3{}
	} else if launch.Kernel == "" {
		launch.Kernel = spec.Name
	}

	fullSrc := spec.CUDASource
	if spec.Prelude != "" && !strings.Contains(fullSrc, spec.Prelude) {
		fullSrc = spec.Prelude + "\n" + fullSrc
	}

	rep, err := cudahostemu.CheckHazards(ctx, fullSrc, launch, args...)
	if err != nil {
		return receipt, fmt.Errorf("%s %s CheckHazards: %w", kind, spec.Name, err)
	}
	receipt.HazardsChecked = true
	receipt.HazardCount = len(rep.Hazards)
	receipt.Hazards = rep.Hazards
	if receipt.HazardCount > 0 {
		return receipt, fmt.Errorf("%s %s: detected %d hazards: %+v", kind, spec.Name, receipt.HazardCount, rep.Hazards)
	}

	if spec.CheckNVFP4SASS {
		if err := VerifyNVFP4SASS(); err != nil {
			return receipt, fmt.Errorf("%s %s SASS: %w", kind, spec.Name, err)
		}
		receipt.SASSEquivVerified = true
	}

	if spec.TileIRBytecode != nil {
		if err := VerifyStepTileIRBytecode(*spec.TileIRBytecode); err != nil {
			return receipt, fmt.Errorf("%s %s TileIR bytecode: %w", kind, spec.Name, err)
		}
	}
	if spec.TileIRDualTarget != nil {
		eps := spec.Eps
		if eps <= 0 {
			eps = 1e-5
		}
		dtRep, err := tilebridge.VerifyDualTargetAgreement(ctx, *spec.TileIRDualTarget, spec.DualM, spec.DualN, spec.DualK, eps)
		if err != nil {
			return receipt, fmt.Errorf("%s %s VerifyDualTargetAgreement: %w", kind, spec.Name, err)
		}
		receipt.TileIRAgreement = true
		receipt.TileIRMaxAbsDiff = dtRep.MaxAbsDiff
	}

	return receipt, nil
}

// QualifyPlanSpec runs [QualifyStepSpec] across every step in plan.Steps and
// verifies optional plan-level source, PDL, and Megakernel emission callbacks.
func QualifyPlanSpec(ctx context.Context, plan PlanSpec) ([]QualificationReceipt, error) {
	if len(plan.Steps) == 0 {
		return nil, fmt.Errorf("qualify plan: empty plan")
	}
	if plan.SourceFn != nil {
		if _, err := plan.SourceFn(); err != nil {
			return nil, fmt.Errorf("qualify plan %s source: %w", plan.Name, err)
		}
	}

	receipts := make([]QualificationReceipt, 0, len(plan.Steps))
	for i, st := range plan.Steps {
		r, err := QualifyStepSpec(ctx, st)
		if err != nil {
			name := st.Name
			if name == "" {
				name = "unknown"
			}
			return receipts, fmt.Errorf("qualify plan %s step %d (%s): %w", plan.Name, i, name, err)
		}
		receipts = append(receipts, *r)
	}

	if plan.PDLCUDAFn != nil {
		if code, err := plan.PDLCUDAFn(); err != nil || strings.TrimSpace(code) == "" {
			return receipts, fmt.Errorf("qualify plan %s EmitPDL: %w", plan.Name, err)
		}
	}

	if plan.MegakernelSourceFn != nil {
		if code, err := plan.MegakernelSourceFn(); err != nil || strings.TrimSpace(code) == "" {
			return receipts, fmt.Errorf("qualify plan %s Megakernel: %w", plan.Name, err)
		}
	}

	return receipts, nil
}

// QualifySM100Kernels verifies the sm_100a Blackwell tcgen05 kernel suite
// (TCGen05Cluster2GEMM, TCGen05BlockScaledGEMM, and TCGen05TMEMFlashAttention)
// through CUDA C++ emission, NVFP4/UE4M3 SASS symbolic equivalence, and
// cudatmem.VerifyTrace(cudatmem.SM100A, ...).
func QualifySM100Kernels(ctx context.Context) (*QualificationReceipt, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	receipt := &QualificationReceipt{
		Name: "sm100a_tcgen05_suite",
	}

	c2 := tcgen05.TCGen05Cluster2GEMM{Name: "tcgen05_cluster2_gemm", M: 256, N: 128, K: 32}
	bs := tcgen05.TCGen05BlockScaledGEMM{Name: "tcgen05_block_scaled_gemm", M: 128, N: 128, K: 64}
	fa := tcgen05.TCGen05TMEMFlashAttention{Name: "tcgen05_tmem_flash_attention", M: 128, KV: 64, D: 64}

	for _, item := range []struct {
		name   string
		cudaFn func() (string, error)
		events []cudatmem.TraceEvent
	}{
		{name: c2.Name, cudaFn: c2.CUDA, events: c2.TraceEvents()},
		{name: bs.Name, cudaFn: bs.CUDA, events: bs.TraceEvents()},
		{name: fa.Name, cudaFn: fa.CUDA, events: fa.TraceEvents()},
	} {
		src, err := item.cudaFn()
		if err != nil || strings.TrimSpace(src) == "" {
			return receipt, fmt.Errorf("qualify sm100 %s CUDA: %w", item.name, err)
		}
		sum, err := cudatmem.VerifyTrace(cudatmem.SM100A, item.events)
		if err != nil {
			return receipt, fmt.Errorf("qualify sm100 %s VerifyTrace: %w", item.name, err)
		}
		if sum.MaxAllocatedColumns > receipt.TMEMPeakColumns {
			receipt.TMEMPeakColumns = sum.MaxAllocatedColumns
		}
	}
	receipt.CUDAValid = true
	receipt.TMEMTraceVerified = true

	if err := VerifyNVFP4SASS(); err != nil {
		return receipt, fmt.Errorf("qualify sm100 SASS equivalence: %w", err)
	}
	receipt.SASSEquivVerified = true

	return receipt, nil
}

// QualifySM100Suite is an alias for [QualifySM100Kernels].
func QualifySM100Suite(ctx context.Context) (*QualificationReceipt, error) {
	return QualifySM100Kernels(ctx)
}

// VerifyNVFP4SASS verifies both the 1-word and 2-word NVFP4 dequantization
// bit-routing programs and positive-normal E4M3 to BF16 scale conversion at
// the symbolic SASS level.
func VerifyNVFP4SASS() error {
	if err := VerifyDequantProgramsSASS(); err != nil {
		return fmt.Errorf("verify dequant programs SASS: %w", err)
	}
	if _, err := ProveScaleConversionSASS(); err != nil {
		return fmt.Errorf("prove scale conversion SASS: %w", err)
	}
	return nil
}

// VerifyDequantProgramsSASS verifies the synthesized 1-word and 2-word NVFP4
// dequantization bit-routing programs ([bitopt.NVFP4Program1Word] and
// [bitopt.NVFP4Program2Word]) at both the bit-routing and symbolic SASS levels.
func VerifyDequantProgramsSASS() error {
	spec1 := bitopt.E2M1ToBF16Spec(1)
	p1 := bitopt.NVFP4Program1Word()
	if err := bitopt.Check(spec1, p1); err != nil {
		return fmt.Errorf("verify dequant9 1-word spec: %w", err)
	}
	p1Verbose := bitopt.NVFP4Program1Word()
	p1Verbose.Moves[1].Sel = 0x081c081c
	p1Verbose.Moves[3].Sel = 0x02030203
	opt1, res1, err := bitopt.SuperoptimizeSASS(&p1Verbose, spec1, 9)
	if err != nil {
		return fmt.Errorf("superoptimize dequant9 1-word SASS: %w", err)
	}
	if res1.Verdict != sassequiv.Equal {
		return fmt.Errorf("dequant9 1-word SASS verdict %v: %s", res1.Verdict, res1.Reason)
	}
	if err := bitopt.VerifySASSEquiv(&p1, opt1, spec1); err != nil {
		return fmt.Errorf("verify dequant9 1-word SASS equivalence: %w", err)
	}

	sass1Default, err := p1.SASS([2]string{}, nil)
	if err != nil {
		return fmt.Errorf("lower dequant9 1-word default SASS: %w", err)
	}
	sass1Renamed, err := p1.SASS([2]string{"R4", "R5"}, []string{"R12"})
	if err != nil {
		return fmt.Errorf("lower dequant9 1-word renamed SASS: %w", err)
	}
	cmp1, err := bitopt.CompareSASS(sass1Default, sass1Renamed)
	if err != nil {
		return fmt.Errorf("compare dequant9 1-word renamed SASS: %w", err)
	}
	if cmp1.Verdict != sassequiv.Equal || len(cmp1.Assumes) > 0 {
		return fmt.Errorf("dequant9 1-word renamed SASS comparison: %v", cmp1)
	}

	spec2 := bitopt.E2M1ToBF16Spec(2)
	p2 := bitopt.NVFP4Program2Word()
	if err := bitopt.Check(spec2, p2); err != nil {
		return fmt.Errorf("verify dequant 2-word spec: %w", err)
	}
	p2Verbose := bitopt.NVFP4Program2Word()
	p2Verbose.Moves[0].Sel = 0x20702070
	p2Verbose.Moves[3].Sel = 0x081c081c
	opt2, res2, err := bitopt.SuperoptimizeSASS(&p2Verbose, spec2, 16)
	if err != nil {
		return fmt.Errorf("superoptimize dequant 2-word SASS: %w", err)
	}
	if res2.Verdict != sassequiv.Equal {
		return fmt.Errorf("dequant 2-word SASS verdict %v: %s", res2.Verdict, res2.Reason)
	}
	if err := bitopt.VerifySASSEquiv(&p2, opt2, spec2); err != nil {
		return fmt.Errorf("verify dequant 2-word SASS equivalence: %w", err)
	}

	sass2Default, err := p2.SASS([2]string{}, nil)
	if err != nil {
		return fmt.Errorf("lower dequant 2-word default SASS: %w", err)
	}
	sass2Renamed, err := p2.SASS([2]string{"R4", "R5"}, []string{"R14", "R15"})
	if err != nil {
		return fmt.Errorf("lower dequant 2-word renamed SASS: %w", err)
	}
	cmp2, err := bitopt.CompareSASS(sass2Default, sass2Renamed)
	if err != nil {
		return fmt.Errorf("compare dequant 2-word renamed SASS: %w", err)
	}
	if cmp2.Verdict != sassequiv.Equal || len(cmp2.Assumes) > 0 {
		return fmt.Errorf("dequant 2-word renamed SASS comparison: %v", cmp2)
	}

	return nil
}

// ProveScaleConversionSASS proves SASS-level symbolic equivalence between the
// mask-before-shift and shift-before-mask formulations of positive-normal E4M3
// to BF16 scale conversion.
func ProveScaleConversionSASS() (sassequiv.Result, error) {
	maskThenShift := []bitopt.Inst{
		{Op: bitopt.OpAndImm, Dst: "R7", Src0: "R6", Imm: 0x7f},
		{Op: bitopt.OpShlImm, Dst: "R8", Src0: "R7", Imm: 4},
		{Op: bitopt.OpAddImm, Dst: "R9", Src0: "R8", Imm: 0x7700},
	}
	shiftThenMask := []bitopt.Inst{
		{Op: bitopt.OpShlImm, Dst: "R10", Src0: "R6", Imm: 4},
		{Op: bitopt.OpAndImm, Dst: "R11", Src0: "R10", Imm: 0x7f0},
		{Op: bitopt.OpAddImm, Dst: "R12", Src0: "R11", Imm: 0x7700},
	}

	sassA, err := bitopt.FormatSASSListing("scale_e4m3_to_bf16", [2]string{"R4", "R5"}, []string{"R6"}, maskThenShift, []string{"R9"})
	if err != nil {
		return sassequiv.Result{}, fmt.Errorf("format mask-then-shift SASS: %w", err)
	}
	sassB, err := bitopt.FormatSASSListing("scale_e4m3_to_bf16", [2]string{"R4", "R5"}, []string{"R6"}, shiftThenMask, []string{"R12"})
	if err != nil {
		return sassequiv.Result{}, fmt.Errorf("format shift-then-mask SASS: %w", err)
	}

	res, err := bitopt.CompareSASS(sassA, sassB)
	if err != nil {
		return sassequiv.Result{}, err
	}
	if res.Verdict != sassequiv.Equal {
		return res, fmt.Errorf("scale conversion SASS verdict %v: %s", res.Verdict, res.Reason)
	}
	if len(res.Assumes) > 0 {
		return res, fmt.Errorf("scale conversion SASS assumes uninitialized %v", res.Assumes)
	}

	for b := range 256 {
		eb := uint8(b)
		if !(eb&0x80 == 0 && (eb>>3)&15 != 0 && eb != 0x7f) {
			continue
		}
		gotA := uint16(((uint32(eb) & 0x7f) << 4) + 0x7700)
		gotB := uint16(((uint32(eb) << 4) & 0x7f0) + 0x7700)
		want := scaleE4M3ToBF16(eb)
		if gotA != want || gotB != want || (uint16(eb)*16+0x7700) != want {
			return res, fmt.Errorf("scale byte %#02x: got (%#04x, %#04x), want %#04x", eb, gotA, gotB, want)
		}
	}
	return res, nil
}

func scaleE4M3ToBF16(b uint8) uint16 {
	if b&0x7f == 0x7f {
		return 0x7fc0
	}
	v := math.Ldexp(float64(tilebridge.FromE4M3Bits(b)), 118)
	return tilebridge.BF16Bits(float32(v))
}

// VerifyStepTileIRBytecode lowers step to Tile IR, round-trips it through
// FormatTileIR and ParseTileIR, and verifies that it encodes to valid
// \x7fTileIR\x00 bytecode.
func VerifyStepTileIRBytecode(step tilebridge.Step) error {
	mod, err := tilebridge.StepToTileIR(step)
	if err != nil {
		return err
	}
	text := tilebridge.FormatTileIR(mod)
	parsed, err := tilebridge.ParseTileIR(text)
	if err != nil {
		return err
	}
	bc, err := tilebridge.EncodeTileIR(parsed)
	if err != nil {
		return err
	}
	if !tileir.IsTileIRBytecode(bc) {
		return fmt.Errorf("missing TileIR bytecode magic header")
	}
	return nil
}

// ParseParamDecl extracts the parameter identifier and base C element type from
// a C/CUDA parameter declaration such as "const uint16_t* __restrict__ x".
func ParseParamDecl(decl string) (name, ctype string) {
	fields := strings.Fields(decl)
	if len(fields) == 0 {
		return "arg", "uint8_t"
	}
	name = strings.TrimLeft(fields[len(fields)-1], "*")
	if idx := strings.LastIndexByte(decl, '*'); idx >= 0 {
		prefix := decl[:idx]
		var kept []string
		for _, f := range strings.Fields(prefix) {
			if f == "const" || f == "volatile" || f == "__restrict__" || f == "__restrict" {
				continue
			}
			kept = append(kept, f)
		}
		if len(kept) > 0 {
			ctype = strings.Join(kept, " ")
		}
	}
	if ctype == "" {
		ctype = "uint8_t"
	}
	return name, ctype
}
