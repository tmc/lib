package hopper

import (
	"fmt"
	"slices"
	"strings"
)

// A Megakernel fuses the sequential stages of a [Plan] into a single
// persistent co-resident kernel launch synchronized with epoch-tagged ready
// counters ([ReadyPrelude]) instead of host stream launches.
//
// Each stage owns a contiguous range of CTAs in the megakernel's 1D grid.
// Producer CTAs compute their stage, synchronize with __syncthreads(), and
// signal their stage's ready counter slot for the current epoch from thread 0.
// Consumer CTAs wait on the upstream stage's ready counter slot for the current
// epoch from thread 0, synchronize with __syncthreads(), and read the
// intermediate tensor directly from L2/global memory.
//
// Because each counter word holds (uint64_t(epoch) << 32) | count and advances
// monotonically via atomicMax + atomicAdd, the ready buffer is allocated and
// zeroed once at initialization and never needs cudaMemsetAsync between
// launches or CUDA graph replays: each call simply increments epoch (1, 2, 3, ...).
type Megakernel struct {
	Name   string
	Inputs []Tensor
	Temps  []Tensor
	Output Tensor
	Stages []MegaStage
}

// A MegaStage describes one stage inside a [Megakernel].
type MegaStage struct {
	Kernel     PlanKernel
	Name       string   // device helper base name
	Grid       [3]int   // stage logical grid (gx, gy, gz)
	BlockBase  int      // starting blockIdx.x in the megakernel grid
	Blocks     int      // number of CTAs assigned to this stage (gx * gy * gz)
	Shared     int      // dynamic shared memory bytes needed by this stage
	WaitSlots  []int    // ready counter slots this stage waits on before running
	SignalSlot int      // ready counter slot this stage signals (-1 if none)
	Outputs    []Tensor // intermediate tensors published on SignalSlot
}

// Megakernel lowers p into a persistent multi-stage [Megakernel] named name
// (or p.Name+"_mega" when name is empty).
func (p *Plan) Megakernel(name string) (Megakernel, error) {
	if p == nil || len(p.Kernels) == 0 {
		return Megakernel{}, fmt.Errorf("megakernel: empty plan")
	}
	if name == "" && p.Name != "" {
		name = p.Name + "_mega"
	}
	if !cIdent(name) {
		return Megakernel{}, fmt.Errorf("megakernel: bad name %q", name)
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

	usedNames := map[string]bool{name: true}
	uniqueStageName := func(base string, idx int) string {
		if !cIdent(base) || usedNames[base] {
			base = fmt.Sprintf("%s_stage%d", name, idx)
		}
		usedNames[base] = true
		return base
	}

	producerOf := make(map[string]int)
	stages := make([]MegaStage, len(p.Kernels))
	blockBase := 0

	for i, pk := range p.Kernels {
		var (
			stName  string
			grid    [3]int
			block   [3]int
			smem    int
			inNames []string
			outName []string
		)
		switch {
		case pk.Attention != nil:
			return Megakernel{}, fmt.Errorf("megakernel %s: stage %d: paged attention stage not supported", name, i)
		case pk.MMA != nil:
			return Megakernel{}, fmt.Errorf("megakernel %s: stage %d: skinny mma stage not supported", name, i)

		case pk.GEMM != nil:
			g := *pk.GEMM
			stName = uniqueStageName(g.Name, i)
			g.Name = stName
			normed, err := g.norm()
			if err != nil {
				return Megakernel{}, fmt.Errorf("megakernel %s: stage %d: %w", name, i, err)
			}
			if pk.M < 1 || pk.M > normed.MaxM {
				return Megakernel{}, fmt.Errorf("megakernel %s: stage %d: M %d outside 1..%d", name, i, pk.M, normed.MaxM)
			}
			grid, block, err = normed.Grid()
			if err != nil {
				return Megakernel{}, fmt.Errorf("megakernel %s: stage %d: %w", name, i, err)
			}
			if block[0] != 256 {
				return Megakernel{}, fmt.Errorf("megakernel %s: stage %d: skinny gemm block size %d != 256", name, i, block[0])
			}
			pk.GEMM = &normed
			if len(pk.Args) != len(normed.params()) {
				return Megakernel{}, fmt.Errorf("megakernel %s: stage %d: bad skinny gemm args %v", name, i, pk.Args)
			}
			// out, then a tail's outputs and counter, are written; the
			// split-K workspace is the stage's own.
			for j, p := range normed.params() {
				switch a := pk.Args[j]; {
				case strings.HasPrefix(p, "const "):
					inNames = append(inNames, a)
				case paramName(p) != "ws":
					outName = append(outName, a)
				}
			}

		case pk.Tile != nil:
			g := *pk.Tile
			stName = uniqueStageName(g.Name, i)
			g.Name = stName
			normed, err := g.norm()
			if err != nil {
				return Megakernel{}, fmt.Errorf("megakernel %s: stage %d: %w", name, i, err)
			}
			if normed.WarpSpecialized {
				return Megakernel{}, fmt.Errorf("megakernel %s: stage %d: warp-specialized tile gemm not supported", name, i)
			}
			if normed.Grouped {
				rows := pk.MaxRows
				if rows <= 0 {
					rows = pk.M
				}
				if rows <= 0 || pk.Experts <= 0 {
					return Megakernel{}, fmt.Errorf("megakernel %s: stage %d: grouped tile gemm needs positive MaxRows and Experts", name, i)
				}
				grid, block = normed.Grid(rows, pk.Experts)
			} else {
				if pk.M <= 0 {
					return Megakernel{}, fmt.Errorf("megakernel %s: stage %d: tile gemm M %d <= 0", name, i, pk.M)
				}
				grid, block = normed.Grid(pk.M, 1)
			}
			if block[0] != 256 {
				return Megakernel{}, fmt.Errorf("megakernel %s: stage %d: tile gemm block size %d != 256", name, i, block[0])
			}
			smem = normed.SharedBytes()
			pk.Tile = &normed
			end := len(pk.Args)
			if normed.StreamK > 0 {
				end--
			}
			nOut := 1
			if normed.SumSq {
				nOut = 2
			}
			if end < nOut {
				return Megakernel{}, fmt.Errorf("megakernel %s: stage %d: bad tile gemm args %v", name, i, pk.Args)
			}
			inNames = pk.Args[:end-nOut]
			outName = pk.Args[end-nOut : end]

		case pk.Row != nil:
			rk := *pk.Row
			if rk.Cluster {
				return Megakernel{}, fmt.Errorf("megakernel %s: stage %d: cluster row kernel not supported", name, i)
			}
			stName = uniqueStageName(rk.Name, i)
			rk.Name = stName
			if _, err := rk.check(); err != nil {
				return Megakernel{}, fmt.Errorf("megakernel %s: stage %d: %w", name, i, err)
			}
			if pk.Rows <= 0 {
				return Megakernel{}, fmt.Errorf("megakernel %s: stage %d: row kernel Rows %d <= 0", name, i, pk.Rows)
			}
			grid, block = rk.Grid(pk.Rows)
			if block[0] != 256 {
				return Megakernel{}, fmt.Errorf("megakernel %s: stage %d: row kernel block size %d != 256", name, i, block[0])
			}
			pk.Row = &rk
			for _, in := range rk.Inputs {
				inNames = append(inNames, in.buffer())
			}
			for _, o := range rk.Outputs {
				outName = append(outName, o.Name)
				if o.Index != "" {
					inNames = append(inNames, o.Index)
				}
			}

		case pk.ExpertSum != nil:
			es := *pk.ExpertSum
			stName = uniqueStageName(es.Name, i)
			es.Name = stName
			normed, err := es.norm()
			if err != nil {
				return Megakernel{}, fmt.Errorf("megakernel %s: stage %d: %w", name, i, err)
			}
			if pk.Rows <= 0 {
				return Megakernel{}, fmt.Errorf("megakernel %s: stage %d: expert sum Rows %d <= 0", name, i, pk.Rows)
			}
			grid, block = normed.Grid(pk.Rows)
			pk.ExpertSum = &normed
			if len(pk.Args) < 2 {
				return Megakernel{}, fmt.Errorf("megakernel %s: stage %d: bad expert sum args %v", name, i, pk.Args)
			}
			inNames = pk.Args[:len(pk.Args)-1]
			outName = []string{pk.Args[len(pk.Args)-1]}

		case pk.TopK != nil:
			tk := *pk.TopK
			stName = uniqueStageName(tk.Name, i)
			tk.Name = stName
			normed, err := tk.norm()
			if err != nil {
				return Megakernel{}, fmt.Errorf("megakernel %s: stage %d: %w", name, i, err)
			}
			if pk.Rows <= 0 {
				return Megakernel{}, fmt.Errorf("megakernel %s: stage %d: topk Rows %d <= 0", name, i, pk.Rows)
			}
			grid, block = normed.Grid(pk.Rows)
			pk.TopK = &normed
			if len(pk.Args) < 3 {
				return Megakernel{}, fmt.Errorf("megakernel %s: stage %d: bad topk args %v", name, i, pk.Args)
			}
			inNames = pk.Args[:len(pk.Args)-2]
			outName = pk.Args[len(pk.Args)-2:]

		case pk.Align != nil:
			al := *pk.Align
			stName = uniqueStageName(al.Name, i)
			al.Name = stName
			al.Warps = 8 // 256 threads in the megakernel block
			normed, err := al.norm()
			if err != nil {
				return Megakernel{}, fmt.Errorf("megakernel %s: stage %d: %w", name, i, err)
			}
			if pk.Rows <= 0 {
				return Megakernel{}, fmt.Errorf("megakernel %s: stage %d: align Rows %d <= 0", name, i, pk.Rows)
			}
			grid, block = [3]int{1, 1, 1}, [3]int{256, 1, 1}
			pk.Align = &normed
			if len(pk.Args) != 5 {
				return Megakernel{}, fmt.Errorf("megakernel %s: stage %d: bad align args %v", name, i, pk.Args)
			}
			inNames = pk.Args[:2]
			outName = pk.Args[2:]

		default:
			return Megakernel{}, fmt.Errorf("megakernel %s: stage %d: empty kernel", name, i)
		}

		blocks := grid[0] * grid[1] * grid[2]
		if blocks <= 0 {
			return Megakernel{}, fmt.Errorf("megakernel %s: stage %d: empty grid %v", name, i, grid)
		}

		var waitSlots []int
		for _, in := range inNames {
			if pIdx, ok := producerOf[in]; ok && !slices.Contains(waitSlots, pIdx) {
				waitSlots = append(waitSlots, pIdx)
			}
		}
		if i > 0 && len(waitSlots) == 0 {
			waitSlots = []int{i - 1}
		}
		slices.Sort(waitSlots)

		var outs []Tensor
		for _, on := range outName {
			producerOf[on] = i
			if t, ok := tensors[on]; ok {
				outs = append(outs, t)
			}
		}

		sigSlot := -1
		if i < len(p.Kernels)-1 {
			sigSlot = i
		}

		stages[i] = MegaStage{
			Kernel:     pk,
			Name:       stName,
			Grid:       grid,
			BlockBase:  blockBase,
			Blocks:     blocks,
			Shared:     smem,
			WaitSlots:  waitSlots,
			SignalSlot: sigSlot,
			Outputs:    outs,
		}
		blockBase += blocks
	}

	return Megakernel{
		Name:   name,
		Inputs: slices.Clone(p.Inputs),
		Temps:  slices.Clone(p.Temps),
		Output: p.Output,
		Stages: stages,
	}, nil
}

// Grid returns the persistent megakernel's 1D launch grid and 256-thread block.
func (m Megakernel) Grid() (grid, block [3]int) {
	total := 0
	for _, st := range m.Stages {
		total += st.Blocks
	}
	return [3]int{total, 1, 1}, [3]int{256, 1, 1}
}

// SharedBytes returns the dynamic shared memory per block required by the
// megakernel (the maximum across all stages).
func (m Megakernel) SharedBytes() int {
	smem := 0
	for _, st := range m.Stages {
		smem = max(smem, st.Shared)
	}
	return smem
}

// ReadyWords returns the number of uint64 words needed in the ready counter
// buffer (at least 1).
func (m Megakernel) ReadyWords() int {
	return max(len(m.Stages)-1, 1)
}

// ReadyBytes returns the byte size of the ready counter buffer.
func (m Megakernel) ReadyBytes() int {
	return 8 * m.ReadyWords()
}

func (m Megakernel) params() []string {
	var params []string
	for _, t := range m.Inputs {
		params = append(params, "const "+t.DType.cType()+"* __restrict__ "+t.Name)
	}
	for _, t := range m.Temps {
		params = append(params, t.DType.cType()+"* __restrict__ "+t.Name)
	}
	if m.Output.Name != "" {
		params = append(params, m.Output.DType.cType()+"* __restrict__ "+m.Output.Name)
	}
	params = append(params,
		"unsigned long long* __restrict__ ready",
		"unsigned int epoch",
	)
	return params
}

// stageDeviceCUDA converts a stage kernel's __global__ CUDA source into a
// __device__ function taking (dim3 eg_bidx, dim3 eg_gdim, ...).
func stageDeviceCUDA(st MegaStage, seenHelpers map[string]bool) (string, error) {
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
		return "", fmt.Errorf("stage %s: empty kernel", st.Name)
	}
	if err != nil {
		return "", err
	}
	marker := "__global__ void __launch_bounds__(256) " + st.Name + "(\n    "
	idx := strings.Index(raw, marker)
	if idx < 0 {
		return "", fmt.Errorf("stage %s: unexpected kernel header in generated CUDA", st.Name)
	}
	prefix := raw[:idx]
	body := raw[idx+len(marker):]
	body = strings.ReplaceAll(body, "blockIdx.", "eg_bidx.")
	body = strings.ReplaceAll(body, "gridDim.", "eg_gdim.")

	// Separate any helper blocks in prefix from the trailing comment/template lines.
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

// CUDA returns the megakernel device source (with [ReadyPrelude], without [Prelude]).
func (m Megakernel) CUDA() (string, error) {
	if len(m.Stages) == 0 || !cIdent(m.Name) {
		return "", fmt.Errorf("megakernel: invalid megakernel %q", m.Name)
	}
	var b strings.Builder
	b.WriteString(ReadyPrelude)
	seenHelpers := make(map[string]bool)
	for _, st := range m.Stages {
		if st.Kernel.Row != nil && st.Kernel.Row.Slots > 0 && !seenHelpers["cluster_prelude"] {
			seenHelpers["cluster_prelude"] = true
			b.WriteString("\n")
			b.WriteString(ClusterPrelude)
		}
		dsrc, err := stageDeviceCUDA(st, seenHelpers)
		if err != nil {
			return "", err
		}
		b.WriteString("\n")
		b.WriteString(dsrc)
	}

	grid, _ := m.Grid()
	fmt.Fprintf(&b, "\n// %s: persistent %d-stage megakernel (%d blocks, %d ready counter words).\n",
		m.Name, len(m.Stages), grid[0], m.ReadyWords())
	fmt.Fprintf(&b, "__global__ void __launch_bounds__(256) %s(\n    %s) {\n",
		m.Name, strings.Join(m.params(), ",\n    "))
	b.WriteString("    const int eg_bid = blockIdx.x;\n")

	for _, st := range m.Stages {
		gx, gy, gz := st.Grid[0], st.Grid[1], st.Grid[2]
		fmt.Fprintf(&b, "    if (eg_bid < %d) {\n", st.BlockBase+st.Blocks)
		fmt.Fprintf(&b, "        const int eg_local = eg_bid - %d;\n", st.BlockBase)
		fmt.Fprintf(&b, "        const dim3 eg_bidx(eg_local %% %d, (eg_local / %d) %% %d, eg_local / %d);\n",
			gx, gx, gy, gx*gy)
		fmt.Fprintf(&b, "        const dim3 eg_gdim(%d, %d, %d);\n", gx, gy, gz)

		if len(st.WaitSlots) > 0 {
			b.WriteString("        if (threadIdx.x == 0) {\n")
			for _, ws := range st.WaitSlots {
				need := m.Stages[ws].Blocks
				fmt.Fprintf(&b, "            eg_ready_wait(&ready[%d], epoch, %d);\n", ws, need)
			}
			b.WriteString("        }\n        __syncthreads();\n")
		}

		k := st.Kernel
		switch {
		case k.GEMM != nil:
			fmt.Fprintf(&b, "        %s_dev<%d>(eg_bidx, eg_gdim, %s);\n",
				st.Name, k.M, strings.Join(k.Args, ", "))
		case k.Tile != nil:
			fmt.Fprintf(&b, "        %s_dev(eg_bidx, eg_gdim, %d, %s);\n",
				st.Name, k.M, strings.Join(k.Args, ", "))
		default:
			fmt.Fprintf(&b, "        %s_dev(eg_bidx, eg_gdim, %d, %s);\n",
				st.Name, k.Rows, strings.Join(k.Args, ", "))
		}

		if st.SignalSlot >= 0 {
			b.WriteString("        __syncthreads();\n        if (threadIdx.x == 0) {\n")
			for _, out := range st.Outputs {
				nbytes := size(out.Shape) * out.DType.Bits() / 8
				if nbytes > 0 {
					fmt.Fprintf(&b, "            EG_READY_STAGE(&ready[%d], epoch, %d, %s, %d);\n",
						st.SignalSlot, st.Blocks, out.Name, nbytes)
				}
			}
			fmt.Fprintf(&b, "            eg_ready_signal(&ready[%d], epoch, 1);\n", st.SignalSlot)
			b.WriteString("        }\n")
		}
		b.WriteString("        return;\n    }\n")
	}
	b.WriteString("}\n")
	return b.String(), nil
}

// Launcher returns the host functions
//
//	void Name_launch(<inputs...>, <temps...>, <output>, unsigned long long* ready, unsigned int epoch, cudaStream_t stream)
//	void Name_run(<inputs...>, <temps...>, <output>, unsigned long long* ready, unsigned int epoch, cudaStream_t stream)
//
// that launch the persistent megakernel for the given epoch.
func (m Megakernel) Launcher() (string, error) {
	if len(m.Stages) == 0 || !cIdent(m.Name) {
		return "", fmt.Errorf("megakernel: invalid megakernel %q", m.Name)
	}
	params := m.params()
	names := paramNames(params)
	grid, block := m.Grid()
	smem := m.SharedBytes()

	var b strings.Builder
	fmt.Fprintf(&b, "inline void %s_launch(%s, cudaStream_t stream) {\n",
		m.Name, strings.Join(params, ", "))
	if smem > 0 {
		fmt.Fprintf(&b, "    constexpr int smem = %d;\n", smem)
		fmt.Fprintf(&b, "    static const cudaError_t attr = cudaFuncSetAttribute(%s, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);\n    (void)attr;\n", m.Name)
		fmt.Fprintf(&b, "    const dim3 grid(%d), block(%d);\n", grid[0], block[0])
		fmt.Fprintf(&b, "    EG_LAUNCH(%s, grid, block, smem, stream, %s);\n}\n",
			m.Name, strings.Join(names, ", "))
	} else {
		fmt.Fprintf(&b, "    const dim3 grid(%d), block(%d);\n", grid[0], block[0])
		fmt.Fprintf(&b, "    EG_LAUNCH(%s, grid, block, 0, stream, %s);\n}\n",
			m.Name, strings.Join(names, ", "))
	}
	fmt.Fprintf(&b, "\ninline void %s_run(%s, cudaStream_t stream) {\n    %s_launch(%s, stream);\n}\n",
		m.Name, strings.Join(params, ", "), m.Name, strings.Join(names, ", "))
	return b.String(), nil
}

// Source returns a complete translation unit with [Prelude], [ReadyPrelude],
// the stage device functions, the persistent megakernel, and its host
// launchers.
func (m Megakernel) Source() (string, error) {
	src, err := m.CUDA()
	if err != nil {
		return "", err
	}
	l, err := m.Launcher()
	if err != nil {
		return "", err
	}
	return Prelude + "\n" + src + "\n" + l, nil
}
