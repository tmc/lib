package advise

import (
	"fmt"
	"strings"

	"github.com/tmc/lib/nvidia/x/cudacapture"
	"github.com/tmc/lib/nvidia/x/cudaptx"
	"github.com/tmc/lib/nvidia/x/cudasass"
)

// floatMath is what a kernel's code says about how its FP32 math was
// compiled: the precise operations --use_fast_math would replace, and the
// local memory the kernel uses.
//
// Each count is a marker the compiler emits only for precise math, taken
// from nvcc and NVRTC output for sm_121 (CUDA 13.0 and 13.4) with and without
// --use_fast_math:
//
//   - A precise sinf, cosf or tanf reduces its argument with a Payne-Hanek
//     slow path for |x| >= 105615, which multiplies by the bits of 2/pi into
//     a local array of seven words. The guard is "setp.ltu.f32 %p, %f, 0f47CE4780" in PTX and
//     "FSETP.GE.AND P0, PT, |R|, 105615, PT" in SASS. Fast math calls
//     sin.approx.ftz.f32 (MUFU.SIN) and has no guard.
//   - A precise division is div.rn.f32 in PTX and FCHK plus a call to
//     __cuda_sm3x_div_rn_noftz_f32_slowpath in SASS; fast math is
//     div.approx.ftz.f32 and a bare MUFU.RCP. A precise reciprocal is
//     rcp.rn.f32, calling __cuda_sm20_rcp_rn_f32_slowpath.
//   - A precise square root is sqrt.rn.f32, calling
//     __cuda_sm20_sqrt_rn_f32_slowpath; fast math is sqrt.approx.ftz.f32.
//   - Without -ftz=true, FP32 add, multiply and fma carry no .ftz (.FTZ).
//
// MUFU.EX2 and MUFU.RCP counts are not markers: precise expf uses MUFU.EX2
// too, and integer division uses MUFU.RCP.
type floatMath struct {
	tier string // "PTX" or "SASS"

	reductions int // guards of precise sinf/cosf/tanf argument reductions
	divisions  int // IEEE divisions and reciprocals
	roots      int // IEEE square roots

	fp32 int // FP32 add, multiply and fma in the kernel's own code
	ftz  int // those of fp32 that flush denormals

	local int // SASS local-memory loads and stores

	at Location // the first guard
}

// costs reports whether the kernel's precise math costs it local memory,
// given the kernel's stack frame in bytes.
//
// Only the argument reduction is charged. Its slow path owns a local array,
// and compiling ds4 with --use_fast_math took the stack frame to zero in all
// 13 kernels that had both. The 3 kernels with a stack frame and no
// reduction kept theirs with the same count of local-memory instructions,
// although 2 of them had precise divisions: their local memory was not the
// math's.
func (m floatMath) costs(frame int) bool {
	return m.reductions > 0 && (frame > 0 || m.local > 0)
}

// sassFloatMath counts the float-math markers in fn.
func sassFloatMath(fn *cudasass.Function) floatMath {
	m := floatMath{tier: "SASS"}
	// Subroutines the kernel calls, such as the division slow path, follow
	// its own code in the same section. They are compiled the same way
	// whatever the kernel's options, so .FTZ is counted in the kernel only.
	end := ^uint64(0)
	if len(fn.Symbols) > 0 {
		end = fn.Symbols[0].Address
	}
	for _, inst := range fn.Instructions {
		base := cudasass.BaseOpcode(inst.Opcode)
		switch base {
		case "FSETP":
			if hasOperand(inst.Operands, "105615") {
				if m.reductions == 0 {
					m.at = Location{Kernel: fn.Name, PC: inst.Address, File: inst.SourceFile, Line: inst.SourceLine}
				}
				m.reductions++
			}
		case "FCHK":
			m.divisions++
		case "CALL":
			switch {
			case strings.Contains(inst.Operands, "_rcp_rn_f32_slowpath"):
				m.divisions++
			case strings.Contains(inst.Operands, "_sqrt_rn_f32_slowpath"):
				m.roots++
			}
		case "FADD", "FMUL", "FFMA":
			if inst.Address < end {
				m.fp32++
				if strings.Contains(inst.Opcode, ".FTZ") {
					m.ftz++
				}
			}
		case "LDL", "STL":
			m.local++
		}
	}
	return m
}

// ptxFloatMath counts the float-math markers in k.
func ptxFloatMath(k *cudaptx.Kernel, file string) floatMath {
	m := floatMath{tier: "PTX"}
	for _, inst := range k.Instructions {
		op := inst.Opcode
		f32 := strings.HasSuffix(op, ".f32")
		switch inst.Mnemonic {
		case "setp":
			if f32 && hasOperand(inst.Operands, "0f47CE4780") {
				if m.reductions == 0 {
					m.at = Location{Kernel: k.Name, File: inst.Loc.File, Line: inst.Loc.Line}
					if m.at.File == "" {
						m.at.File, m.at.Line = file, inst.Line
					}
				}
				m.reductions++
			}
		case "div", "rcp":
			if f32 && strings.Contains(op, ".rn.") {
				m.divisions++
			}
		case "sqrt":
			if f32 && strings.Contains(op, ".rn.") {
				m.roots++
			}
		case "add", "sub", "mul", "fma":
			if f32 {
				m.fp32++
				if strings.Contains(op, ".ftz") {
					m.ftz++
				}
			}
		}
	}
	return m
}

// hasOperand reports whether the comma-separated operand list holds tok.
func hasOperand(operands, tok string) bool {
	for _, f := range strings.Split(operands, ",") {
		if strings.TrimSpace(f) == tok {
			return true
		}
	}
	return false
}

// floatMathByKernel returns the float-math markers of each kernel, read from
// the PTX when the capture carries the kernel's PTX, which names the rounding
// of each operation, and from its SASS otherwise. It also returns the kernel
// names in code order: SASS functions, then kernels only the PTX defines.
//
// Local memory is counted in SASS only. PTX declares the reduction's local
// array in every kernel that has one, and ptxas keeps it in registers in
// some: ds4_rope_attention_raw has 20 ld/st.local in PTX, and no LDL, no STL
// and no stack frame in SASS.
func floatMathByKernel(disasm *cudasass.Disassembly, mod *cudaptx.Module, ptxFile string) ([]string, map[string]floatMath) {
	var names []string
	math := make(map[string]floatMath)
	if disasm != nil {
		for _, fn := range disasm.Functions {
			if _, ok := math[fn.Name]; !ok {
				names = append(names, fn.Name)
			}
			math[fn.Name] = sassFloatMath(fn)
		}
	}
	if mod != nil {
		for _, k := range mod.Kernels {
			if !k.Entry || !k.Body {
				continue
			}
			sass, ok := math[k.Name]
			if !ok {
				names = append(names, k.Name)
			}
			m := ptxFloatMath(k, ptxFile)
			m.local = sass.local
			math[k.Name] = m
		}
	}
	return names, math
}

// fastMathLead is appended to a local-memory finding for a kernel whose
// precise math costs it that memory, so that the finding does not send the
// reader to restructure code the compiler added.
const fastMathLead = " This kernel's precise sinf/cosf argument reduction keeps a local array (see its precise float math finding): --use_fast_math or __sinf/__cosf may remove the local memory without touching register lifetimes."

// auditPreciseFloatMath reports each kernel whose precise FP32 math costs it
// local memory. kernels lists the kernels in report order and frames holds
// their stack frames in bytes.
func auditPreciseFloatMath(cap *cudacapture.Capture, report *Report, kernels []string, math map[string]floatMath, frames map[string]int) {
	evidence := EvidenceSuspected
	var recorded string
	if cap != nil && cap.Identity != nil && cap.Identity.CompileOpts != "" && !strings.Contains(cap.Identity.CompileOpts, "use_fast_math") {
		evidence = EvidenceObserved
		recorded = fmt.Sprintf("; the capture records compile options %q", cap.Identity.CompileOpts)
	}
	for _, name := range kernels {
		m, frame := math[name], frames[name]
		if !m.costs(frame) {
			continue
		}
		parts := []string{fmt.Sprintf("%d sinf/cosf argument-reduction guards (|x| >= 105615 takes a slow path that keeps a local array)", m.reductions)}
		if m.divisions > 0 {
			parts = append(parts, fmt.Sprintf("%d IEEE divisions or reciprocals", m.divisions))
		}
		if m.roots > 0 {
			parts = append(parts, fmt.Sprintf("%d IEEE square roots", m.roots))
		}
		if m.fp32 > 0 {
			parts = append(parts, fmt.Sprintf("%d of %d FP32 add/mul/fma flush denormals", m.ftz, m.fp32))
		}
		var costs []string
		if frame > 0 {
			costs = append(costs, fmt.Sprintf("a %d-byte stack frame", frame))
		}
		if m.local > 0 {
			costs = append(costs, fmt.Sprintf("%d local-memory instructions", m.local))
		}

		// NVRTC 13.0 applied -ftz, -prec-div and -prec-sqrt to its PTX and
		// returned the unflagged CUBIN (see compileNVRTCVariant in
		// x/cmd/cudaprof), so the narrow flags are offered for nvcc only.
		intrinsics := []string{"__sinf/__cosf/__sincosf for the argument reductions"}
		var flags []string
		if m.divisions > 0 {
			intrinsics = append(intrinsics, "__fdividef for the divisions")
			flags = append(flags, "-prec-div=false")
		}
		if m.roots > 0 {
			flags = append(flags, "-prec-sqrt=false")
		}
		if m.ftz < m.fp32 {
			flags = append(flags, "-ftz=true")
		}
		narrow := "or narrower, in source: " + strings.Join(intrinsics, ", ") + "."
		if len(flags) > 0 {
			narrow += " Under nvcc " + strings.Join(flags, ", ") + " also apply; NVRTC 13.0 changed only the PTX for them and returned the same CUBIN."
		}

		report.Items = append(report.Items, Item{
			Severity:      SeverityCost,
			Category:      CategoryArchitectureCompute,
			EvidenceLevel: evidence,
			Location:      m.at,
			Finding: fmt.Sprintf("Kernel %q is compiled with precise float math that costs it local memory (read from %s): %s, paying for %s%s",
				name, m.tier, strings.Join(parts, ", "), strings.Join(costs, " and "), recorded),
			ActionableRecommendation: "Try --use_fast_math (an nvcc and NVRTC option), " + narrow + " " +
				"This changes numerics: sinf/cosf lose accuracy for large arguments, divisions and square roots round differently, and denormals flush to zero. Rerun the output-agreement gate (for example KL against a reference) before keeping it.",
			NextMeasurement: "cuobjdump -res-usage before and after (STACK and the LDL/STL count should fall to zero) and this kernel's device time before and after, then the output-agreement gate",
			rule:            "fastmath",
		})
	}
}
