package hopper

import (
	"fmt"
	"slices"
	"strings"
)

// PDLHelpers returns the device and host-emulated C++ helpers for Hopper/Blackwell
// Programmatic Dependent Launch (griddepcontrol.launch_dependents and
// griddepcontrol.wait).
//
// On real hardware (__CUDA_ARCH__ >= 900), eg_pdl_trigger emits
// griddepcontrol.launch_dependents and eg_pdl_wait emits griddepcontrol.wait.
// Under host emulation (EG_EMU / EG_HOST_EMU / CUDAHOSTEMU), they delegate to
// eg_emu_pdl_launch_dependents and eg_emu_pdl_wait.
func PDLHelpers() string {
	return pdlPrelude
}

const pdlPrelude = `#ifndef EG_PDL_PRELUDE
#define EG_PDL_PRELUDE

#if defined(EG_HOST_EMU) || defined(EG_EMU) || defined(CUDAHOSTEMU)
#ifndef EG_EMU_PDL_H
inline void eg_emu_pdl_register_buffer(void* ptr, size_t bytes, uint32_t stage_id = 1) {
    (void)ptr;
    (void)bytes;
    (void)stage_id;
}
inline void eg_emu_pdl_launch_dependents(uint32_t stage_id = 1) { (void)stage_id; }
inline void eg_emu_pdl_wait(uint32_t stage_id = 1) { (void)stage_id; }
inline int eg_emu_pdl_violation_count() { return 0; }
#endif

#ifndef EG_EMU_PDL_LAUNCH_TYPES
#define EG_EMU_PDL_LAUNCH_TYPES 1
enum cudaLaunchAttributeID {
    cudaLaunchAttributeProgrammaticStreamSerialization = 9,
};
union cudaLaunchAttributeValue {
    int programmaticStreamSerializationAllowed;
};
struct cudaLaunchAttribute {
    cudaLaunchAttributeID id;
    cudaLaunchAttributeValue val;
};
struct cudaLaunchConfig_t {
    dim3 gridDim;
    dim3 blockDim;
    size_t dynamicSmemBytes;
    cudaStream_t stream;
    cudaLaunchAttribute* attrs;
    unsigned int numAttrs;
};
template <typename K, typename... Args>
inline cudaError_t cudaLaunchKernelEx(const cudaLaunchConfig_t* cfg, K kernel, Args... args) {
    eg_emu_launch(cfg->gridDim, cfg->blockDim, cfg->dynamicSmemBytes, kernel, args...);
    return cudaSuccess;
}
#endif
#endif

#ifndef EG_PDL_DEVICE_HELPERS
#define EG_PDL_DEVICE_HELPERS 1
EG_DEV void eg_pdl_trigger(uint32_t stage) {
#if defined(EG_NO_PDL_TRIGGER)
    (void)stage;
#elif defined(EG_HOST_EMU) || defined(EG_EMU) || defined(CUDAHOSTEMU)
    eg_emu_pdl_launch_dependents(stage);
#elif defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    (void)stage;
    asm volatile("griddepcontrol.launch_dependents;\n" ::: "memory");
#else
    (void)stage;
#endif
}

EG_DEV void eg_pdl_wait(uint32_t stage) {
#if defined(EG_NO_PDL_WAIT)
    (void)stage;
#elif defined(EG_HOST_EMU) || defined(EG_EMU) || defined(CUDAHOSTEMU)
    eg_emu_pdl_wait(stage);
#elif defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    (void)stage;
    asm volatile("griddepcontrol.wait;\n" ::: "memory");
#else
    (void)stage;
#endif
}
#endif

#endif // EG_PDL_PRELUDE
`

// PDLStageDesc describes one kernel stage in a [PDLPlanSource].
type PDLStageDesc struct {
	Name         string
	Kernel       PlanKernel
	Grid         [3]int
	Block        [3]int
	SharedBytes  int
	WaitStage    uint32   // stage ID waited on before dependent global loads (0 for stage 0)
	TriggerStage uint32   // stage ID triggered after global output stores complete (0 for final stage)
	Args         []string // buffer parameter names in kernel call order
	Outputs      []Tensor // intermediate tensors produced by this stage
}

// PDLPlanSource holds the emitted CUDA source and per-stage launch descriptors
// for a [Plan] lowered to Programmatic Dependent Launch (PDL).
type PDLPlanSource struct {
	Name     string
	Source   string
	Helpers  string
	Kernels  string
	Launcher string
	Inputs   []Tensor
	Temps    []Tensor
	Output   Tensor
	Stages   []PDLStageDesc
}

// CUDA returns the complete CUDA translation unit (Prelude, PDLHelpers, stage
// kernels, and the launch_<name>_pdl host launcher).
func (s *PDLPlanSource) CUDA() (string, error) {
	if s == nil {
		return "", fmt.Errorf("pdl: nil plan source")
	}
	return s.Source, nil
}

// String returns the complete CUDA source.
func (s *PDLPlanSource) String() string {
	if s == nil {
		return ""
	}
	return s.Source
}

// Steps returns the sequential kernel steps of p.
func (p *Plan) Steps() []PlanKernel {
	if p == nil {
		return nil
	}
	return p.Kernels
}

// EmitPDL lowers p into a Programmatic Dependent Launch (PDL) kernel stack and
// host launcher helper launch_<name>_pdl(...) using
// cudaLaunchAttributeProgrammaticStreamSerialization.
//
// Each step i > 0 calls eg_pdl_wait(uint32_t(i)) before its first dependent
// global load from step i-1's intermediate output, and each step
// i < len(p.Kernels)-1 calls eg_pdl_trigger(uint32_t(i+1)) after its global
// output stores complete.
func (p *Plan) EmitPDL(name string) (*PDLPlanSource, error) {
	if p == nil || len(p.Kernels) == 0 {
		return nil, fmt.Errorf("pdl: empty plan")
	}
	if name == "" {
		name = p.Name
	}
	if name == "" {
		name = "plan"
	}
	if !cIdent(name) {
		return nil, fmt.Errorf("pdl: bad name %q", name)
	}

	tensors := make(map[string]Tensor)
	for _, t := range p.Inputs {
		tensors[t.Name] = t
	}
	for _, t := range p.Temps {
		tensors[t.Name] = t
	}
	if p.Output.Name != "" {
		tensors[p.Output.Name] = p.Output
	}

	usedNames := map[string]bool{
		name:                      true,
		"launch_" + name + "_pdl": true,
	}
	uniqueStageName := func(base string, idx int) string {
		cand := base + "_pdl"
		if !cIdent(cand) || usedNames[cand] {
			cand = fmt.Sprintf("%s_pdl_stage%d", name, idx)
		}
		usedNames[cand] = true
		return cand
	}

	stages := make([]PDLStageDesc, len(p.Kernels))
	var kernBuf strings.Builder
	seenHelpers := make(map[string]bool)

	for i, pk := range p.Kernels {
		var (
			stName   string
			grid     [3]int
			block    [3]int
			smem     int
			outNames []string
		)
		switch {
		case pk.Attention != nil:
			return nil, fmt.Errorf("pdl %s: stage %d: paged attention stage not supported", name, i)
		case pk.MMA != nil:
			return nil, fmt.Errorf("pdl %s: stage %d: skinny mma stage not supported", name, i)

		case pk.GEMM != nil:
			g := *pk.GEMM
			stName = uniqueStageName(g.Name, i)
			g.Name = stName
			if g.MaxM < pk.M && pk.M <= 16 {
				g.MaxM = pk.M
			}
			normed, err := g.norm()
			if err != nil {
				return nil, fmt.Errorf("pdl %s: stage %d: %w", name, i, err)
			}
			if pk.M < 1 || pk.M > normed.MaxM {
				return nil, fmt.Errorf("pdl %s: stage %d: M %d outside 1..%d", name, i, pk.M, normed.MaxM)
			}
			grid, block, err = normed.Grid()
			if err != nil {
				return nil, fmt.Errorf("pdl %s: stage %d: %w", name, i, err)
			}
			pk.GEMM = &normed
			outIdx := len(pk.Args) - 1
			if normed.SplitK > 1 {
				outIdx--
			}
			if outIdx < 0 || outIdx >= len(pk.Args) {
				return nil, fmt.Errorf("pdl %s: stage %d: bad skinny gemm args %v", name, i, pk.Args)
			}
			outNames = pk.Args[outIdx:]

		case pk.Tile != nil:
			g := *pk.Tile
			stName = uniqueStageName(g.Name, i)
			g.Name = stName
			normed, err := g.norm()
			if err != nil {
				return nil, fmt.Errorf("pdl %s: stage %d: %w", name, i, err)
			}
			if normed.Grouped {
				rows := pk.MaxRows
				if rows <= 0 {
					rows = pk.M
				}
				if rows <= 0 || pk.Experts <= 0 {
					return nil, fmt.Errorf("pdl %s: stage %d: grouped tile gemm needs positive MaxRows and Experts", name, i)
				}
				grid, block = normed.Grid(rows, pk.Experts)
			} else {
				if pk.M <= 0 {
					return nil, fmt.Errorf("pdl %s: stage %d: tile gemm M %d <= 0", name, i, pk.M)
				}
				grid, block = normed.Grid(pk.M, 1)
			}
			smem = normed.SharedBytes()
			pk.Tile = &normed
			end := len(pk.Args)
			nOut := 1
			if normed.SumSq {
				nOut = 2
			}
			if normed.StreamK > 0 {
				nOut++
			}
			if end < nOut {
				return nil, fmt.Errorf("pdl %s: stage %d: bad tile gemm args %v", name, i, pk.Args)
			}
			outNames = pk.Args[end-nOut : end]

		case pk.Row != nil:
			rk := *pk.Row
			if rk.Cluster {
				return nil, fmt.Errorf("pdl %s: stage %d: cluster row kernel not supported", name, i)
			}
			stName = uniqueStageName(rk.Name, i)
			rk.Name = stName
			if _, err := rk.check(); err != nil {
				return nil, fmt.Errorf("pdl %s: stage %d: %w", name, i, err)
			}
			if pk.Rows <= 0 {
				return nil, fmt.Errorf("pdl %s: stage %d: row kernel Rows %d <= 0", name, i, pk.Rows)
			}
			grid, block = rk.Grid(pk.Rows)
			if len(pk.Args) == 0 {
				for _, p := range rk.params()[1:] {
					f := strings.Fields(p)
					pk.Args = append(pk.Args, f[len(f)-1])
				}
			}
			pk.Row = &rk
			for _, o := range rk.Outputs {
				outNames = append(outNames, o.Name)
			}
			if rk.Slots > 0 && !seenHelpers["cluster_prelude"] {
				seenHelpers["cluster_prelude"] = true
				kernBuf.WriteString("\n")
				kernBuf.WriteString(ClusterPrelude)
			}

		case pk.ExpertSum != nil:
			es := *pk.ExpertSum
			stName = uniqueStageName(es.Name, i)
			es.Name = stName
			normed, err := es.norm()
			if err != nil {
				return nil, fmt.Errorf("pdl %s: stage %d: %w", name, i, err)
			}
			if pk.Rows <= 0 {
				return nil, fmt.Errorf("pdl %s: stage %d: expert sum Rows %d <= 0", name, i, pk.Rows)
			}
			grid, block = normed.Grid(pk.Rows)
			pk.ExpertSum = &normed
			if len(pk.Args) < 2 {
				return nil, fmt.Errorf("pdl %s: stage %d: bad expert sum args %v", name, i, pk.Args)
			}
			outNames = pk.Args[len(pk.Args)-1:]

		case pk.TopK != nil:
			tk := *pk.TopK
			stName = uniqueStageName(tk.Name, i)
			tk.Name = stName
			normed, err := tk.norm()
			if err != nil {
				return nil, fmt.Errorf("pdl %s: stage %d: %w", name, i, err)
			}
			if pk.Rows <= 0 {
				return nil, fmt.Errorf("pdl %s: stage %d: topk Rows %d <= 0", name, i, pk.Rows)
			}
			grid, block = normed.Grid(pk.Rows)
			pk.TopK = &normed
			if len(pk.Args) < 3 {
				return nil, fmt.Errorf("pdl %s: stage %d: bad topk args %v", name, i, pk.Args)
			}
			outNames = pk.Args[len(pk.Args)-2:]

		case pk.Align != nil:
			al := *pk.Align
			stName = uniqueStageName(al.Name, i)
			al.Name = stName
			normed, err := al.norm()
			if err != nil {
				return nil, fmt.Errorf("pdl %s: stage %d: %w", name, i, err)
			}
			if pk.Rows <= 0 {
				return nil, fmt.Errorf("pdl %s: stage %d: align Rows %d <= 0", name, i, pk.Rows)
			}
			grid, block = [3]int{1, 1, 1}, [3]int{32 * normed.Warps, 1, 1}
			pk.Align = &normed
			if len(pk.Args) != 5 {
				return nil, fmt.Errorf("pdl %s: stage %d: bad align args %v", name, i, pk.Args)
			}
			outNames = pk.Args[2:]

		default:
			return nil, fmt.Errorf("pdl %s: stage %d: empty kernel", name, i)
		}

		var outs []Tensor
		outSet := make(map[string]bool, len(outNames))
		for _, on := range outNames {
			outSet[on] = true
			if t, ok := tensors[on]; ok {
				outs = append(outs, t)
			}
		}

		var waitStage, trigStage uint32
		if i > 0 {
			waitStage = uint32(i)
		}
		if i < len(p.Kernels)-1 {
			trigStage = uint32(i + 1)
		}

		stDesc := PDLStageDesc{
			Name:         stName,
			Kernel:       pk,
			Grid:         grid,
			Block:        block,
			SharedBytes:  smem,
			WaitStage:    waitStage,
			TriggerStage: trigStage,
			Args:         slices.Clone(pk.Args),
			Outputs:      outs,
		}
		stages[i] = stDesc

		dsrc, err := pdlStageDeviceCUDA(stDesc, seenHelpers)
		if err != nil {
			return nil, err
		}
		kernBuf.WriteString("\n")
		kernBuf.WriteString(dsrc)

		// Emit the __global__ PDL wrapper kernel for step i.
		var stageParams []string
		for _, argName := range pk.Args {
			t, ok := tensors[argName]
			ctype := "float"
			if ok {
				ctype = t.DType.cType()
			}
			if outSet[argName] {
				stageParams = append(stageParams, ctype+"* __restrict__ "+argName)
			} else {
				stageParams = append(stageParams, "const "+ctype+"* __restrict__ "+argName)
			}
		}

		fmt.Fprintf(&kernBuf, "\n__global__ void __launch_bounds__(%d) %s(\n    %s) {\n",
			block[0], stName, strings.Join(stageParams, ",\n    "))
		if waitStage > 0 {
			fmt.Fprintf(&kernBuf, "    eg_pdl_wait(%du);\n", waitStage)
		}
		switch {
		case pk.GEMM != nil:
			fmt.Fprintf(&kernBuf, "    %s_dev<%d>(blockIdx, gridDim, %s);\n",
				stName, pk.M, strings.Join(pk.Args, ", "))
		case pk.Tile != nil:
			fmt.Fprintf(&kernBuf, "    %s_dev(blockIdx, gridDim, %d, %s);\n",
				stName, pk.M, strings.Join(pk.Args, ", "))
		default:
			fmt.Fprintf(&kernBuf, "    %s_dev(blockIdx, gridDim, %d, %s);\n",
				stName, pk.Rows, strings.Join(pk.Args, ", "))
		}
		if trigStage > 0 {
			kernBuf.WriteString("    __syncthreads();\n")
			fmt.Fprintf(&kernBuf, "    eg_pdl_trigger(%du);\n", trigStage)
		}
		kernBuf.WriteString("}\n")
	}

	// Emit the host launcher helper launch_<name>_pdl(...).
	var planParams []string
	var planArgNames []string
	for _, t := range p.Inputs {
		planParams = append(planParams, "const "+t.DType.cType()+"* "+t.Name)
		planArgNames = append(planArgNames, t.Name)
	}
	for _, t := range p.Temps {
		planParams = append(planParams, t.DType.cType()+"* "+t.Name)
		planArgNames = append(planArgNames, t.Name)
	}
	if p.Output.Name != "" {
		planParams = append(planParams, p.Output.DType.cType()+"* "+p.Output.Name)
		planArgNames = append(planArgNames, p.Output.Name)
	}

	launcherName := "launch_" + name + "_pdl"
	var lbuf strings.Builder
	fmt.Fprintf(&lbuf, "inline void %s(\n    %s,\n    cudaStream_t stream = nullptr) {\n",
		launcherName, strings.Join(planParams, ",\n    "))

	// Under host emulation, automatically register all intermediate buffers for PDL hazard tracking.
	hasStageOutputs := false
	for _, st := range stages {
		if st.TriggerStage > 0 && len(st.Outputs) > 0 {
			hasStageOutputs = true
			break
		}
	}
	if hasStageOutputs {
		lbuf.WriteString("#if defined(EG_HOST_EMU) || defined(EG_EMU) || defined(CUDAHOSTEMU)\n")
		for _, st := range stages {
			if st.TriggerStage == 0 {
				continue
			}
			for _, out := range st.Outputs {
				nbytes := size(out.Shape) * out.DType.Bits() / 8
				if nbytes > 0 {
					fmt.Fprintf(&lbuf, "    eg_emu_pdl_register_buffer(%s, %d, %du);\n",
						out.Name, nbytes, st.TriggerStage)
				}
			}
		}
		lbuf.WriteString("#endif\n")
	}

	lbuf.WriteString("    cudaLaunchAttribute attr{};\n")
	lbuf.WriteString("    attr.id = cudaLaunchAttributeProgrammaticStreamSerialization;\n")
	lbuf.WriteString("    attr.val.programmaticStreamSerializationAllowed = 1;\n")
	for i, st := range stages {
		lbuf.WriteString("    {\n")
		if st.SharedBytes > 0 {
			fmt.Fprintf(&lbuf, "        static const cudaError_t attr_smem_%d = cudaFuncSetAttribute(%s, cudaFuncAttributeMaxDynamicSharedMemorySize, %d);\n        (void)attr_smem_%d;\n",
				i, st.Name, st.SharedBytes, i)
		}
		fmt.Fprintf(&lbuf, "        cudaLaunchConfig_t cfg{};\n")
		fmt.Fprintf(&lbuf, "        cfg.gridDim = dim3(%d, %d, %d);\n", st.Grid[0], st.Grid[1], st.Grid[2])
		fmt.Fprintf(&lbuf, "        cfg.blockDim = dim3(%d, %d, %d);\n", st.Block[0], st.Block[1], st.Block[2])
		fmt.Fprintf(&lbuf, "        cfg.dynamicSmemBytes = %d;\n", st.SharedBytes)
		lbuf.WriteString("        cfg.stream = stream;\n")
		lbuf.WriteString("        cfg.attrs = &attr;\n")
		lbuf.WriteString("        cfg.numAttrs = 1;\n")
		fmt.Fprintf(&lbuf, "        (void)cudaLaunchKernelEx(&cfg, %s, %s);\n",
			st.Name, strings.Join(st.Args, ", "))
		lbuf.WriteString("    }\n")
	}
	lbuf.WriteString("}\n")

	fmt.Fprintf(&lbuf, "\ninline void %s_run(%s, cudaStream_t stream = nullptr) {\n    %s(%s, stream);\n}\n",
		name, strings.Join(planParams, ", "), launcherName, strings.Join(planArgNames, ", "))

	helpers := PDLHelpers()
	kernels := kernBuf.String()
	launcher := lbuf.String()
	fullSource := Prelude + "\n" + helpers + "\n" + kernels + "\n" + launcher

	return &PDLPlanSource{
		Name:     name,
		Source:   fullSource,
		Helpers:  helpers,
		Kernels:  kernels,
		Launcher: launcher,
		Inputs:   slices.Clone(p.Inputs),
		Temps:    slices.Clone(p.Temps),
		Output:   p.Output,
		Stages:   stages,
	}, nil
}

func pdlStageDeviceCUDA(st PDLStageDesc, seenHelpers map[string]bool) (string, error) {
	var raw string
	var err error
	k := st.Kernel
	switch {
	case k.GEMM != nil:
		raw, err = k.GEMM.CUDA()
	case k.Tile != nil:
		raw, err = k.Tile.CUDA()
	case k.Row != nil:
		raw, err = k.Row.CUDA()
	case k.ExpertSum != nil:
		raw, err = k.ExpertSum.CUDA()
	case k.TopK != nil:
		raw, err = k.TopK.CUDA()
	case k.Align != nil:
		raw, err = k.Align.CUDA()
	default:
		return "", fmt.Errorf("pdl stage %s: empty kernel", st.Name)
	}
	if err != nil {
		return "", err
	}
	marker := fmt.Sprintf("__global__ void __launch_bounds__(%d) %s(\n    ", st.Block[0], st.Name)
	idx := strings.Index(raw, marker)
	if idx < 0 {
		return "", fmt.Errorf("pdl stage %s: unexpected kernel header in generated CUDA", st.Name)
	}
	prefix := raw[:idx]
	body := raw[idx+len(marker):]
	body = strings.ReplaceAll(body, "blockIdx.", "eg_bidx.")
	body = strings.ReplaceAll(body, "gridDim.", "eg_gdim.")

	var commentLines []string
	var helperLines []string
	for _, line := range strings.Split(strings.TrimSuffix(prefix, "\n"), "\n") {
		if strings.HasPrefix(line, "//") || strings.HasPrefix(line, "template ") {
			commentLines = append(commentLines, line)
		} else {
			helperLines = append(helperLines, line)
		}
	}
	var b strings.Builder
	if helperBlock := strings.TrimSpace(strings.Join(helperLines, "\n")); helperBlock != "" && !seenHelpers[helperBlock] {
		seenHelpers[helperBlock] = true
		b.WriteString(helperBlock)
		b.WriteString("\n\n")
	}
	for _, line := range commentLines {
		b.WriteString(line)
		b.WriteString("\n")
	}
	fmt.Fprintf(&b, "EG_DEV void %s_dev(\n    dim3 eg_bidx,\n    dim3 eg_gdim,\n    %s", st.Name, body)
	return b.String(), nil
}
