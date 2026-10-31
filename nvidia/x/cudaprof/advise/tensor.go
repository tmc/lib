package advise

import (
	"fmt"
	"sort"
	"strconv"
	"strings"

	"github.com/tmc/lib/nvidia/x/cudadevicetable"
	"github.com/tmc/lib/nvidia/x/cudaptx"
	"github.com/tmc/lib/nvidia/x/cudasass"
)

// The tensor-core instruction families, as SASS spells them.
//
//	warp     HMMA, IMMA, QMMA, OMMA, DMMA      mma.sync, every architecture since sm_70
//	warpgroup HGMMA, QGMMA, IGMMA, BGMMA       wgmma, sm_90a
//	tcgen05  UTCHMMA, UTCQMMA, UTCOMMA, UTCIMMA tcgen05.mma, sm_100a/sm_103a/sm_110a
//
// The spellings were read from cuobjdump -sass of kernels compiled with CUDA
// 13.4 for each target (testdata/tensor); UTCOMMA is the block-scaled FP4
// MMA, and a ".2CTA" modifier marks a cta_group::2 MMA.
var tensorFamilies = map[string]string{
	"HMMA": "warp", "IMMA": "warp", "QMMA": "warp", "OMMA": "warp", "DMMA": "warp",
	"HGMMA": "warpgroup", "QGMMA": "warpgroup", "IGMMA": "warpgroup", "BGMMA": "warpgroup",
	"UTCHMMA": "tcgen05", "UTCQMMA": "tcgen05", "UTCOMMA": "tcgen05", "UTCIMMA": "tcgen05",
}

// tensorCensus counts one function's tensor-path instructions.
type tensorCensus struct {
	ops        map[string]int // tensor-core base opcodes
	family     map[string]int // by tensorFamilies
	pairs      int            // cta_group::2 MMAs
	tmaLoads   int            // UTMALDG
	tmaStores  int            // UTMASTG
	mbarWaits  int            // SYNCS.PHASECHK ... TRYWAIT, an mbarrier wait
	tmemMoves  int            // LDTM, STTM, UTCCP
	permutes   int            // PRMT, the table lookup of a narrow-format decode
	first      cudasass.Instruction
	firstFound bool
}

func censusTensor(fn *cudasass.Function) tensorCensus {
	c := tensorCensus{ops: map[string]int{}, family: map[string]int{}}
	for _, inst := range fn.Instructions {
		op := strings.ToUpper(inst.Opcode)
		base := cudasass.BaseOpcode(op)
		if fam, ok := tensorFamilies[base]; ok {
			c.ops[base]++
			c.family[fam]++
			if strings.Contains(op, ".2CTA") {
				c.pairs++
			}
			if !c.firstFound {
				c.first, c.firstFound = inst, true
			}
			continue
		}
		switch {
		case base == "UTMALDG":
			c.tmaLoads++
		case base == "UTMASTG":
			c.tmaStores++
		case base == "SYNCS" && strings.Contains(op, "PHASECHK") && strings.Contains(op, "WAIT"):
			c.mbarWaits++
		case base == "LDTM", base == "STTM", base == "UTCCP":
			c.tmemMoves++
		case base == "PRMT":
			c.permutes++
		}
	}
	return c
}

func (c tensorCensus) describe() string {
	var parts []string
	ops := make([]string, 0, len(c.ops))
	for op := range c.ops {
		ops = append(ops, op)
	}
	sort.Strings(ops)
	for _, op := range ops {
		parts = append(parts, fmt.Sprintf("%s ×%d", op, c.ops[op]))
	}
	if c.pairs > 0 {
		parts = append(parts, fmt.Sprintf("%d cta_group::2", c.pairs))
	}
	if c.tmaLoads > 0 {
		parts = append(parts, fmt.Sprintf("UTMALDG ×%d", c.tmaLoads))
	}
	if c.tmaStores > 0 {
		parts = append(parts, fmt.Sprintf("UTMASTG ×%d", c.tmaStores))
	}
	if c.mbarWaits > 0 {
		parts = append(parts, fmt.Sprintf("mbarrier waits ×%d", c.mbarWaits))
	}
	if c.tmemMoves > 0 {
		parts = append(parts, fmt.Sprintf("TMEM moves ×%d", c.tmemMoves))
	}
	return strings.Join(parts, ", ")
}

// targetSupports reports whether ptxas accepts feature for the
// feature-suffixed target of arch's family, as cudadevicetable recorded it.
// Asking of the suffixed target is deliberate: a cubin built for sm_100
// cannot hold tcgen05, and the question this answers is whether the
// hardware the kernel targets has the instruction at all.
func targetSupports(arch, feature string) bool {
	base := cudadevicetable.Normalize(arch)
	if base == "" {
		return false
	}
	t, ok := cudadevicetable.LookupTarget(base + "a")
	if !ok {
		return false
	}
	ok, err := t.Supports(feature)
	return err == nil && ok
}

// minWarpMMA is the number of warp-level MMAs below which a kernel is not
// treated as a GEMM: a handful is a reduction or an epilogue trick.
const minWarpMMA = 16

// auditTensorPath reports, per SASS function, which tensor-core instruction
// family a GEMM-like kernel uses against what its architecture offers.
//
// The findings rest on opcodes the capture holds, so they need no device and
// no timing: they are what a compile-only cubin can say about a target this
// project has no hardware for. They say what instruction was emitted, not
// that the other one would be faster for this kernel's shapes.
func auditTensorPath(arch string, disasm *cudasass.Disassembly, modules map[*cudasass.Function]string, report *Report) {
	if disasm == nil {
		return
	}
	tcgen05 := targetSupports(arch, "tcgen05")
	wgmma := targetSupports(arch, "wgmma")
	mxf4 := targetSupports(arch, "mmasyncmxf4")
	for _, fn := range disasm.Functions {
		c := censusTensor(fn)
		if len(c.ops) == 0 {
			continue
		}
		loc := Location{Kernel: fn.Name, Module: modules[fn], PC: c.first.Address, File: c.first.SourceFile, Line: c.first.SourceLine}
		warp := c.family["warp"]
		add := func(rule, finding, rec string) {
			if base := cudadevicetable.Normalize(arch); base != "" && !strings.HasSuffix(strings.ToLower(arch), "a") {
				// A cubin's header does not say whether it was built for
				// the suffixed target, so this is stated, not inferred.
				rec += fmt.Sprintf(" These instructions need the architecture-specific target %sa; a build for %s cannot use them.", base, base)
			}
			report.Items = append(report.Items, Item{
				rule:                     rule,
				Severity:                 SeverityCost,
				Category:                 CategoryArchitectureCompute,
				EvidenceLevel:            EvidenceSuspected,
				Location:                 loc,
				Finding:                  finding,
				ActionableRecommendation: rec,
				NextMeasurement:          "A/B the kernel against a tcgen05 or wgmma variant on the target, or compare SASS instruction counts with cudaprof diff",
			})
		}
		fp4 := warp > 0 && c.permutes >= 2*warp && c.permutes >= 2*minWarpMMA
		switch {
		case fp4 && tcgen05 && c.ops["UTCOMMA"] == 0:
			add("fp4-dequant", fmt.Sprintf("Kernel %q on %s decodes a narrow format in registers before warp-level MMA (PRMT ×%d beside %s), where tcgen05.mma kind::mxf4nvf4.block_scale (UTCOMMA) consumes NVFP4 directly", fn.Name, arch, c.permutes, c.describe()),
				"Feed the packed FP4 operands and their block scales to tcgen05.mma.kind::mxf4nvf4.block_scale from shared memory (TMA-loaded, scales staged to TMEM with tcgen05.cp) instead of decoding them to BF16 in registers.")
		case fp4 && mxf4 && c.ops["OMMA"] == 0:
			add("fp4-dequant", fmt.Sprintf("Kernel %q on %s decodes a narrow format in registers before warp-level MMA (PRMT ×%d beside %s), where mma.sync kind::mxf4nvf4.block_scale consumes NVFP4 directly", fn.Name, arch, c.permutes, c.describe()),
				"Use the block-scaled warp MMA (mma.sync.aligned.m16n8k64.kind::mxf4nvf4.block_scale) so the FP4 operands and their scales go to the tensor core without a register decode.")
		case warp >= minWarpMMA && tcgen05 && c.family["tcgen05"] == 0:
			add("warp-mma-tcgen05", fmt.Sprintf("GEMM-like kernel %q on %s issues warp-level MMA (%s) where tcgen05.mma applies: the warp MMA path is the sm_80 one, with operands in registers", fn.Name, arch, c.describe()),
				"Emit tcgen05.mma (UTCHMMA/UTCQMMA/UTCOMMA) with TMA-loaded shared-memory operands and accumulators in Tensor Memory; consider cta_group::2 across a CTA pair for large tiles. Its occupancy is then bounded by TMEM columns (512 per SM), which advise reports when the PTX records tcgen05.alloc.")
		case warp >= minWarpMMA && wgmma && c.family["warpgroup"] == 0:
			add("warp-mma-wgmma", fmt.Sprintf("GEMM-like kernel %q on %s issues warp-level MMA (%s) where wgmma applies", fn.Name, arch, c.describe()),
				"Emit wgmma.mma_async (HGMMA/QGMMA) over a warpgroup with shared-memory operand descriptors, fed by TMA, so the MMA does not wait on register operand loads.")
		}
		report.Items = append(report.Items, Item{
			rule:                     "tensor-path",
			Severity:                 SeverityContext,
			Category:                 CategoryArchitectureCompute,
			EvidenceLevel:            EvidenceObserved,
			Location:                 loc,
			Finding:                  fmt.Sprintf("Tensor path of kernel %q: %s", fn.Name, c.describe()),
			ActionableRecommendation: "None by itself: this records which tensor-core, TMA and mbarrier instructions the kernel's SASS holds, so findings about it can be checked against the code.",
		})
	}
}

// tensorMemory is what a kernel's PTX records about its Tensor Memory and
// cluster: the facts the cubin's .nv.info does not.
type tensorMemory struct {
	columns int  // largest tcgen05.alloc, in columns
	pair    bool // a cta_group::2 tcgen05 instruction
	cluster int  // CTAs per cluster from .reqnctapercluster, 0 if unset
}

// ptxTensorMemory reads tcgen05.alloc column counts, cta_group::2 and
// cluster dimensions from each PTX kernel.
func ptxTensorMemory(mod *cudaptx.Module) map[string]tensorMemory {
	if mod == nil {
		return nil
	}
	out := map[string]tensorMemory{}
	for _, k := range mod.Kernels {
		var tm tensorMemory
		if c := k.Tuning.ReqNCTAPerCluster; c[0] > 0 {
			tm.cluster = c[0] * max(1, c[1]) * max(1, c[2])
		}
		// nvcc moves the column count into a register first, so a
		// register operand is resolved through the constant moved into it.
		consts := map[string]int{}
		for _, inst := range k.Instructions {
			if inst.Mnemonic == "mov" {
				ops := strings.Split(inst.Operands, ",")
				if len(ops) == 2 {
					if n, err := strconv.Atoi(strings.TrimSpace(ops[1])); err == nil {
						consts[strings.TrimSpace(ops[0])] = n
					}
				}
				continue
			}
			if !strings.HasPrefix(inst.Opcode, "tcgen05.") {
				continue
			}
			if strings.Contains(inst.Opcode, "cta_group::2") {
				tm.pair = true
			}
			if strings.HasPrefix(inst.Opcode, "tcgen05.alloc") {
				ops := strings.Split(inst.Operands, ",")
				last := strings.TrimSpace(ops[len(ops)-1])
				n, err := strconv.Atoi(last)
				if err != nil {
					n = consts[last]
				}
				tm.columns = max(tm.columns, n)
			}
		}
		if tm != (tensorMemory{}) {
			out[k.Name] = tm
		}
	}
	return out
}
