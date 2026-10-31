package qualify

import (
	"bytes"
	"context"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"strings"

	"github.com/tmc/lib/nvidia/x/egraph/hopper"
	"github.com/tmc/lib/nvidia/x/ptxasinfo"
)

// PTXASCompiler compiles CUDA C++ source for arch and returns its parsed
// ptxas -v resource and diagnostic report.
type PTXASCompiler interface {
	CompilePTXAS(ctx context.Context, cudaSource, arch string) (ptxasinfo.Report, error)
}

// NVCCPTXASCompiler implements [PTXASCompiler] by invoking
// nvcc --cubin -arch=<arch> -Xptxas -v in a temporary directory and parsing
// stderr with [ptxasinfo.Parse].
type NVCCPTXASCompiler struct {
	// NVCC is the path or name of the nvcc binary (empty defaults to "nvcc").
	NVCC string

	// ExtraFlags holds optional additional compiler flags passed to nvcc.
	ExtraFlags []string
}

// CompilePTXAS invokes nvcc --cubin -arch=<arch> -Xptxas -v in a temporary
// directory and parses the resulting ptxas -v diagnostic output.
func (c NVCCPTXASCompiler) CompilePTXAS(ctx context.Context, cudaSource, arch string) (ptxasinfo.Report, error) {
	if strings.TrimSpace(cudaSource) == "" {
		return ptxasinfo.Report{}, fmt.Errorf("compile ptxas: empty CUDA source")
	}
	arch = strings.TrimSpace(arch)
	if arch == "" {
		arch = "sm_90a"
	}
	nvcc := strings.TrimSpace(c.NVCC)
	if nvcc == "" {
		nvcc = "nvcc"
	}
	dir, err := os.MkdirTemp("", "qualify-ptxas-*")
	if err != nil {
		return ptxasinfo.Report{}, fmt.Errorf("compile ptxas: mkdir temp: %w", err)
	}
	defer os.RemoveAll(dir)

	cuPath := filepath.Join(dir, "plan.cu")
	cubinPath := filepath.Join(dir, "plan.cubin")
	if err := os.WriteFile(cuPath, []byte(cudaSource), 0o600); err != nil {
		return ptxasinfo.Report{}, fmt.Errorf("compile ptxas: write source: %w", err)
	}

	args := []string{"--cubin", "-arch=" + arch, "-Xptxas", "-v"}
	args = append(args, c.ExtraFlags...)
	args = append(args, cuPath, "-o", cubinPath)

	cmd := exec.CommandContext(ctx, nvcc, args...)
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	runErr := cmd.Run()

	combined := stderr.String()
	if stdout.Len() > 0 {
		if combined != "" && !strings.HasSuffix(combined, "\n") {
			combined += "\n"
		}
		combined += stdout.String()
	}
	rep, parseErr := ptxasinfo.Parse(strings.NewReader(combined))
	if runErr != nil {
		return ptxasinfo.Report{}, fmt.Errorf("compile ptxas (%s): %w: %s", arch, runErr, strings.TrimSpace(combined))
	}
	if parseErr != nil {
		return ptxasinfo.Report{}, fmt.Errorf("compile ptxas (%s): parse output: %w", arch, parseErr)
	}
	return *rep, nil
}

// PTXASLoopConfig configures the closed-loop compile-time refinement pass in
// [RefinePlanByPTXASLoop].
type PTXASLoopConfig struct {
	Arch          string     // e.g. "sm_90a", "sm_121a"
	Device        string     // e.g. "NVIDIA H200", "NVIDIA GB10"
	DeviceSMs     int        // streaming multiprocessor count (0 resolves from Device/Arch)
	MaxIterations int        // maximum compile-refine iterations (0 defaults to 4)
	Gate          StaticGate // static resource constraints (spills, serialization, registers, CTAs/SM)
}

// PTXASIterationRecord records the observations, classified bottlenecks, and
// schedule rewrite actions taken during one iteration of [RefinePlanByPTXASLoop].
type PTXASIterationRecord struct {
	Iteration         int
	Observations      []KernelObservation
	Bottlenecks       map[string][]BottleneckKind
	Actions           []string
	TotalSpillBytes   int
	SerializedKernels int
	Clean             bool
}

// PTXASLoopResult holds the initial and refined plans, optional persistent
// megakernel, per-iteration telemetry records, and convergence status from
// [RefinePlanByPTXASLoop].
type PTXASLoopResult struct {
	InitialPlan *hopper.Plan
	RefinedPlan *hopper.Plan
	Megakernel  *hopper.Megakernel
	Iterations  []PTXASIterationRecord
	Converged   bool
}

// RefinePlanByPTXASLoop runs a closed-loop, zero-GPU compile-time schedule
// refinement pass over plan using compiler and cfg:
//  1. Emits plan.Source() and compiles it via compiler.CompilePTXAS(ctx, src, cfg.Arch).
//  2. Matches each step in plan.Kernels to its [ptxasinfo.Function] entry in
//     the report, computes its launch grid block count, and constructs a
//     [KernelObservation] via [ObservationFromPTXAS].
//  3. Checks whether any step exhibits compile-time bottlenecks
//     ([BottleneckRegisterSpill], [BottleneckWGMMASerial],
//     [BottleneckLowOccupancy], [BottleneckWaveTail]) or violates cfg.Gate.
//  4. If all steps are clean, marks Converged = true and returns.
//  5. Otherwise invokes [RefinePlan] (with fallback WGMMA/tile step-downs when
//     needed), records the rewrite actions in [PTXASIterationRecord], and
//     repeats until converged, no further rewrites occur, or cfg.MaxIterations
//     is reached.
func RefinePlanByPTXASLoop(ctx context.Context, plan *hopper.Plan, compiler PTXASCompiler, cfg PTXASLoopConfig) (*PTXASLoopResult, error) {
	if plan == nil || len(plan.Kernels) == 0 {
		return nil, fmt.Errorf("refine plan by ptxas loop: empty plan")
	}
	if compiler == nil {
		return nil, fmt.Errorf("refine plan by ptxas loop: nil compiler")
	}

	maxIter := cfg.MaxIterations
	if maxIter <= 0 {
		maxIter = 4
	}
	arch := strings.TrimSpace(cfg.Arch)
	deviceSMs := resolveLoopDeviceSMs(cfg)

	maxRegs := cfg.Gate.MaxRegisters
	if maxRegs <= 0 {
		maxRegs = HopperMaxRegistersPerTh
	}
	minCTAs := max(cfg.Gate.MinCTAsPerSM, 1)
	noSpills := cfg.Gate.NoSpills || !cfg.Gate.AllowSpills

	initialPlan := clonePlan(plan)
	currentPlan := clonePlan(plan)
	res := &PTXASLoopResult{
		InitialPlan: initialPlan,
		RefinedPlan: currentPlan,
	}

	for iter := 1; iter <= maxIter; iter++ {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		src, err := currentPlan.Source()
		if err != nil {
			return nil, fmt.Errorf("refine plan by ptxas loop (iteration %d): emit source: %w", iter, err)
		}
		report, err := compiler.CompilePTXAS(ctx, src, arch)
		if err != nil {
			return nil, fmt.Errorf("refine plan by ptxas loop (iteration %d): compile ptxas: %w", iter, err)
		}

		allFns := report.Functions()
		usedFns := make([]bool, len(allFns))
		obsList := make([]KernelObservation, len(currentPlan.Kernels))
		refineObs := make([]KernelObservation, len(currentPlan.Kernels))
		bottlenecks := make(map[string][]BottleneckKind, len(currentPlan.Kernels))

		var (
			totalSpills       int
			serializedKernels int
			anyDirty          bool
		)

		for i, step := range currentPlan.Kernels {
			name := stepKernelName(step)
			fns := lookupStepPTXASFunctions(report, name, arch, i, allFns, usedFns)
			fn := mergeStepPTXASFunctions(name, fns)
			gridBlocks := stepGridBlocks(step)
			obs := ObservationFromPTXAS(fn, gridBlocks, deviceSMs)
			if name != "" {
				obs.Name = name
			}

			totalSpills += obs.SpillStoreBytes + obs.SpillLoadBytes
			if obs.WGMMASerialized {
				serializedKernels++
			}

			v := stepToQualifyVariant(step)
			ctasPerSM := ResidentCTAsPerSM(v, fn.Registers, fn.SharedMem)
			spillsGate := noSpills && (fn.SpillStores > 0 || fn.SpillLoads > 0)
			serialGate := obs.WGMMASerialized
			regsGate := fn.Registers > 0 && fn.Registers > maxRegs
			ctasGate := (fn.Registers > 0 || fn.SharedMem > 0 || v.Launch.SharedBytes > 0) && ctasPerSM < minCTAs
			gateViolated := spillsGate || serialGate || regsGate || ctasGate

			if (regsGate || ctasGate) && obs.Registers > 128 && obs.AchievedOccupancy == 0 {
				obs.AchievedOccupancy = min(float64(max(ctasPerSM, 1))/float64(max(minCTAs*4, 8)), 0.20)
			}

			rawBns := ClassifyBottlenecks(obs)
			var stepBns []BottleneckKind
			for _, b := range rawBns {
				switch b {
				case BottleneckRegisterSpill:
					if noSpills {
						stepBns = append(stepBns, b)
					}
				case BottleneckWGMMASerial, BottleneckLowOccupancy:
					stepBns = append(stepBns, b)
				case BottleneckWaveTail:
					if isWaveTailActionable(step, obs) {
						stepBns = append(stepBns, b)
					}
				}
			}
			if (regsGate || ctasGate) && !slices.Contains(stepBns, BottleneckLowOccupancy) && !slices.Contains(stepBns, BottleneckRegisterSpill) {
				stepBns = append(stepBns, BottleneckLowOccupancy)
			}

			if len(stepBns) > 0 {
				key := obs.Name
				if key == "" {
					key = fmt.Sprintf("step_%d", i)
				}
				bottlenecks[key] = stepBns
			}
			if len(stepBns) > 0 || gateViolated {
				anyDirty = true
			}

			obsList[i] = obs
			rObs := obs
			if !noSpills {
				rObs.SpillStoreBytes = 0
				rObs.SpillLoadBytes = 0
			}
			if !slices.Contains(stepBns, BottleneckWaveTail) {
				rObs.GridBlocks = 0
			}
			if regsGate || ctasGate {
				if step.Tile != nil {
					if rObs.Registers <= 128 {
						rObs.Registers = 168
					}
					if rObs.AchievedOccupancy <= 0 || rObs.AchievedOccupancy >= 0.25 {
						rObs.AchievedOccupancy = 0.15
					}
				} else if rObs.SpillStoreBytes == 0 && rObs.SpillLoadBytes == 0 {
					rObs.SpillStoreBytes = 4
				}
			}
			refineObs[i] = rObs
		}

		rec := PTXASIterationRecord{
			Iteration:         iter,
			Observations:      obsList,
			Bottlenecks:       bottlenecks,
			TotalSpillBytes:   totalSpills,
			SerializedKernels: serializedKernels,
			Clean:             !anyDirty,
		}

		if !anyDirty {
			res.Iterations = append(res.Iterations, rec)
			res.RefinedPlan = currentPlan
			res.Converged = true
			return res, nil
		}

		nextPlan, mega, actions, err := RefinePlan(currentPlan, ProfileReceipt{
			Device:       cfg.Device,
			Observations: refineObs,
		})
		if err != nil {
			return nil, fmt.Errorf("refine plan by ptxas loop (iteration %d): %w", iter, err)
		}

		fallbackActions := applyPTXASFallbackRewrites(currentPlan, nextPlan, bottlenecks)
		actions = append(actions, fallbackActions...)

		rec.Actions = actions
		res.Iterations = append(res.Iterations, rec)
		if mega != nil {
			res.Megakernel = mega
		}
		res.RefinedPlan = nextPlan

		if len(actions) == 0 && plansSameSchedule(currentPlan, nextPlan) {
			break
		}
		currentPlan = nextPlan
	}

	return res, nil
}

func resolveLoopDeviceSMs(cfg PTXASLoopConfig) int {
	if cfg.DeviceSMs > 0 {
		return cfg.DeviceSMs
	}
	for _, dev := range []string{cfg.Device, cfg.Arch} {
		dev = strings.TrimSpace(dev)
		if dev == "" {
			continue
		}
		if normed, err := (hopper.CostConfig{Device: dev}).Norm(); err == nil && normed.SMs > 0 {
			return normed.SMs
		}
	}
	arch := strings.ToLower(strings.TrimSpace(cfg.Arch))
	dev := strings.ToLower(strings.TrimSpace(cfg.Device))
	if strings.HasPrefix(arch, "sm_121") || strings.Contains(dev, "gb10") {
		return 48
	}
	return 132
}

func lookupStepPTXASFunctions(report ptxasinfo.Report, stepName, arch string, stepIdx int, allFns []*ptxasinfo.Function, usedFns []bool) []*ptxasinfo.Function {
	if stepName != "" {
		fns := lookupVariantFunctions(report, stepName, arch)
		for _, suffix := range []string{"_cluster", "_wide"} {
			fns = append(fns, lookupVariantFunctions(report, stepName+suffix, arch)...)
		}
		if len(fns) == 0 && arch != "" {
			fns = lookupVariantFunctions(report, stepName, "")
			for _, suffix := range []string{"_cluster", "_wide"} {
				fns = append(fns, lookupVariantFunctions(report, stepName+suffix, "")...)
			}
		}
		if len(fns) > 0 {
			for _, f := range fns {
				for j, af := range allFns {
					if af == f {
						usedFns[j] = true
					}
				}
			}
			return fns
		}
	}
	if stepIdx >= 0 && stepIdx < len(allFns) && !usedFns[stepIdx] {
		usedFns[stepIdx] = true
		return []*ptxasinfo.Function{allFns[stepIdx]}
	}
	return nil
}

func mergeStepPTXASFunctions(stepName string, fns []*ptxasinfo.Function) ptxasinfo.Function {
	out := ptxasinfo.Function{
		Name:  stepName,
		Entry: true,
	}
	for _, f := range fns {
		if f == nil {
			continue
		}
		if out.Name == "" && f.Name != "" {
			out.Name = f.Name
		}
		if out.Arch == "" && f.Arch != "" {
			out.Arch = f.Arch
		}
		if f.Registers > out.Registers {
			out.Registers = f.Registers
		}
		if f.SharedMem > out.SharedMem {
			out.SharedMem = f.SharedMem
		}
		if f.StackFrame > out.StackFrame {
			out.StackFrame = f.StackFrame
		}
		if f.MaxRegisters > out.MaxRegisters {
			out.MaxRegisters = f.MaxRegisters
		}
		out.SpillStores += f.SpillStores
		out.SpillLoads += f.SpillLoads
		out.Diagnostics = append(out.Diagnostics, f.Diagnostics...)
	}
	return out
}

func stepGridBlocks(k hopper.PlanKernel) int {
	switch {
	case k.Tile != nil:
		g := *k.Tile
		if normed, err := g.Norm(); err == nil {
			g = normed
		}
		rows := k.M
		if g.Grouped && k.MaxRows > 0 {
			rows = k.MaxRows
		}
		if rows <= 0 {
			rows = max(g.BM, 64)
		}
		experts := 1
		if g.Grouped {
			experts = max(k.Experts, 1)
		}
		grid, _ := g.Grid(rows, experts)
		return max(grid[0], 1) * max(grid[1], 1) * max(grid[2], 1)

	case k.GEMM != nil:
		g := *k.GEMM
		if g.MaxM <= 0 {
			g.MaxM = max(k.M, 1)
		}
		if normed, err := g.Norm(); err == nil {
			g = normed
		}
		if grid, _, err := g.Grid(); err == nil {
			return max(grid[0], 1) * max(grid[1], 1) * max(grid[2], 1)
		}
		rpb := max(g.RowsPerBlock, 1)
		sk := max(g.SplitK, 1)
		return max(((max(g.N, 1)+rpb-1)/rpb)*sk, 1)

	case k.Attention != nil:
		a := *k.Attention
		if normed, err := a.Norm(); err == nil {
			a = normed
		}
		batch := max(k.Batch, 1)
		maxKeys := k.MaxKeys
		if maxKeys <= 0 {
			if a.Window > 0 {
				maxKeys = a.Window + max(a.QLen, 1) - 1
			} else {
				maxKeys = 1024
			}
		}
		grid, _ := a.Grid(batch, maxKeys)
		return max(grid[0], 1) * max(grid[1], 1) * max(grid[2], 1)

	case k.Row != nil:
		r := *k.Row
		rows := max(k.Rows, 1)
		var grid [3]int
		if r.Cluster && r.CanCluster() {
			grid, _ = r.ClusterGrid(rows)
		} else {
			grid, _ = r.Grid(rows)
		}
		return max(grid[0], 1) * max(grid[1], 1) * max(grid[2], 1)

	case k.MMA != nil:
		if grid, _, err := k.MMA.Grid(); err == nil {
			return max(grid[0], 1) * max(grid[1], 1) * max(grid[2], 1)
		}
		return 1

	case k.UMMA != nil:
		grid, _ := k.UMMA.Grid(max(k.M, 1))
		return max(grid[0], 1) * max(grid[1], 1) * max(grid[2], 1)

	case k.Quant != nil:
		grid, _ := k.Quant.Grid(max(k.M, 1))
		return max(grid[0], 1) * max(grid[1], 1) * max(grid[2], 1)

	case k.TopK != nil:
		tk := *k.TopK
		if normed, err := tk.Norm(); err == nil {
			tk = normed
		}
		grid, _ := tk.Grid(max(k.Rows, 1))
		return max(grid[0], 1) * max(grid[1], 1) * max(grid[2], 1)

	case k.Align != nil:
		return 1

	case k.ExpertSum != nil:
		es := *k.ExpertSum
		if normed, err := es.Norm(); err == nil {
			es = normed
		}
		grid, _ := es.Grid(max(k.Rows, 1))
		return max(grid[0], 1) * max(grid[1], 1) * max(grid[2], 1)

	default:
		return 1
	}
}

func stepToQualifyVariant(k hopper.PlanKernel) Variant {
	if hv, ok := stepToHopperVariant(k); ok {
		v := ToQualifyVariant(hv)
		if k.Attention != nil && v.Launch.SharedBytes == 0 {
			v.Launch.SharedBytes = k.Attention.SharedBytes()
		}
		if k.Row != nil && k.Row.Cluster && k.Row.CanCluster() {
			_, block := k.Row.ClusterGrid(max(k.Rows, 1))
			threads := max(block[0], 128)
			v.Launch.Block = threads
			v.Launch.Warps = (threads + 31) / 32
		}
		return v
	}
	return Variant{
		Name:   stepKernelName(k),
		Launch: Launch{Block: 256, Warps: 8},
	}
}

func isWaveTailActionable(step hopper.PlanKernel, obs KernelObservation) bool {
	trial := KernelObservation{
		Name:       stepKernelName(step),
		GridBlocks: obs.GridBlocks,
		DeviceSMs:  obs.DeviceSMs,
	}
	cand, reasons := RefineStep(step, trial)
	return len(reasons) > 0 && !stepSameSchedule(cand, step)
}

func applyPTXASFallbackRewrites(prevPlan, nextPlan *hopper.Plan, bottlenecks map[string][]BottleneckKind) []string {
	var extra []string
	for i := range nextPlan.Kernels {
		if i >= len(prevPlan.Kernels) || !stepSameSchedule(prevPlan.Kernels[i], nextPlan.Kernels[i]) {
			continue
		}
		name := stepKernelName(nextPlan.Kernels[i])
		key := name
		if key == "" {
			key = fmt.Sprintf("step_%d", i)
		}
		bns := bottlenecks[key]
		if len(bns) == 0 {
			continue
		}
		hasSerial := slices.Contains(bns, BottleneckWGMMASerial)
		hasRegPressure := slices.Contains(bns, BottleneckRegisterSpill) || slices.Contains(bns, BottleneckLowOccupancy)

		switch {
		case nextPlan.Kernels[i].Tile != nil:
			t := *nextPlan.Kernels[i].Tile
			if hasSerial && t.WGMMA && t.Weight != hopper.WeightE4M3 {
				cand := t
				cand.WGMMA = false
				cand.WarpSpecialized = false
				cand.Repacked = false
				cand.DequantWords = 0
				if cand.BN > 128 {
					cand.BN = 128
				}
				if _, err := cand.Norm(); err == nil {
					nextPlan.Kernels[i].Tile = &cand
					extra = append(extra, fmt.Sprintf("%s: disable WGMMA to eliminate WGMMA serialization", cand.Name))
					continue
				}
			}
			if hasRegPressure && !t.WGMMA && (t.BM == 0 || t.BM > 64) {
				cand := t
				cand.BM = 64
				if _, err := cand.Norm(); err == nil {
					nextPlan.Kernels[i].Tile = &cand
					extra = append(extra, fmt.Sprintf("%s: step down BM -> 64 to reduce register pressure", cand.Name))
					continue
				}
			}

		case nextPlan.Kernels[i].Attention != nil:
			a := *nextPlan.Kernels[i].Attention
			if hasSerial && a.WGMMA {
				cand := a
				cand.WGMMA = false
				if _, err := cand.Norm(); err == nil {
					nextPlan.Kernels[i].Attention = &cand
					extra = append(extra, fmt.Sprintf("%s: disable WGMMA to eliminate WGMMA serialization", cand.Name))
				}
			}
		}
	}
	return extra
}

func plansSameSchedule(a, b *hopper.Plan) bool {
	if a == nil || b == nil {
		return a == b
	}
	if len(a.Kernels) != len(b.Kernels) {
		return false
	}
	for i := range a.Kernels {
		if !stepSameSchedule(a.Kernels[i], b.Kernels[i]) {
			return false
		}
	}
	return true
}
