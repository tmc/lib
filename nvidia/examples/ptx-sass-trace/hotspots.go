package main

import (
	"context"
	"strings"

	"github.com/tmc/lib/nvidia/x/cudacapture"
	"github.com/tmc/lib/nvidia/x/cudaprof"
)

// instructionCost returns the estimated relative latency cost weight and primary
// stall metric for a given SASS/PTX opcode.
func instructionCost(op string) (int64, string) {
	upper := strings.ToUpper(strings.TrimSpace(op))
	switch {
	case strings.HasPrefix(upper, "LDG"), strings.HasPrefix(upper, "STG"), strings.HasPrefix(upper, "RED"):
		return 60, "stall_scoreboard"
	case strings.HasPrefix(upper, "LDS"), strings.HasPrefix(upper, "STS"):
		return 20, "stall_scoreboard"
	case strings.HasPrefix(upper, "SHFL"), strings.HasPrefix(upper, "BAR"), strings.HasPrefix(upper, "DEPBAR"):
		return 25, "stall_wait"
	case strings.HasPrefix(upper, "IDP4A"), strings.HasPrefix(upper, "HMMA"), strings.HasPrefix(upper, "MMA"):
		return 16, "stall_math"
	case strings.HasPrefix(upper, "MUFU"), strings.HasPrefix(upper, "RSQRT"), strings.HasPrefix(upper, "DIV"):
		return 12, "stall_math"
	case strings.HasPrefix(upper, "FFMA"), strings.HasPrefix(upper, "FMUL"), strings.HasPrefix(upper, "FADD"),
		strings.HasPrefix(upper, "IMAD"), strings.HasPrefix(upper, "IADD3"), strings.HasPrefix(upper, "LOP3"):
		return 4, "stall_math"
	case strings.HasPrefix(upper, "BRA"), strings.HasPrefix(upper, "RET"), strings.HasPrefix(upper, "EXIT"):
		return 2, "stall_branch"
	default:
		return 1, "stall_math"
	}
}

// instructionOpcode extracts the base mnemonic / opcode from an instruction text
// (e.g. "/*0010*/ @!P0 LDG.E.CONSTANT R2, [R0];" -> "LDG.E.CONSTANT").
func instructionOpcode(text string) string {
	s := strings.TrimSpace(text)
	// Strip comment like /*0010*/
	if strings.HasPrefix(s, "/*") {
		if idx := strings.Index(s, "*/"); idx != -1 {
			s = strings.TrimSpace(s[idx+2:])
		}
	}
	// Strip predicate like @!P0 or @P1
	if strings.HasPrefix(s, "@") {
		fields := strings.Fields(s)
		if len(fields) > 1 {
			s = fields[1]
		}
	} else {
		fields := strings.Fields(s)
		if len(fields) > 0 {
			s = fields[0]
		}
	}
	return strings.TrimSuffix(s, ";")
}

// recordInstructionHotspots records a static opcode cost for each of the
// kernel's disassembled instructions, as modeled_cost and as nothing else.
//
// The result is a model, not a measurement: the weights come from the opcode
// name alone, so they are identical for every launch, block size and input
// size. It is useful for finding which instructions a kernel is made of, and
// useless for deciding which one is slow. -pcsampling answers that by sampling
// the program counter, and the two modes are mutually exclusive so that no
// profile mixes the two kinds of number.
//
// These rows carry no samples and no gpu_time. An earlier version divided the
// launch's measured duration among the instructions by weight, which put the
// cost table, rescaled, under a type name that means a measured CUDA event
// interval in every other mode -- and it was the profile's default view, so a
// -hotspots capture opened on it. The launch's real duration is recorded once
// by the caller, on its own row.
func recordInstructionHotspots(
	ctx context.Context,
	rec *launchRecorder,
	k *cudacapture.Kernel,
	tileIRLine int,
	stack []cudaprof.Frame,
) int {
	if rec == nil || rec.prof == nil || k == nil || len(k.Instructions) == 0 {
		return 0
	}

	type instWeight struct {
		inst   cudacapture.Instruction
		weight int64
	}

	// The stall class the cost table assigns is deliberately dropped. Stall
	// attribution is a measurement -pcsampling makes; asserting one from an
	// opcode name would put a guess in the same counters the hardware fills.
	weights := make([]instWeight, len(k.Instructions))
	for i, inst := range k.Instructions {
		op := instructionOpcode(inst.Text)
		w, _ := instructionCost(op)
		weights[i] = instWeight{inst: inst, weight: w}
	}

	attributed := 0
	for _, iw := range weights {
		// The weight is the opcode table's, which is a model, so it goes to
		// ModeledCost: it is not a PC sample, not an instruction count, and not
		// a stall counter, and naming it after any of those would make a table
		// lookup indistinguishable from a device measurement in the shared
		// profile.
		vals := cudaprof.SampleValues{ModeledCost: iw.weight}

		ptxLine := iw.inst.PTXLine
		if ptxLine <= 0 {
			ptxLine = k.PTXLine
		}
		srcLine := int64(iw.inst.SourceLine)
		if srcLine <= 0 {
			srcLine = int64(k.SourceLine)
		}

		frames := []cudaprof.Frame{
			{
				FunctionName: k.Name + ".SASS",
				FileName:     "kernels.sass",
				Line:         int64(iw.inst.SASSLine),
				// Instruction.Offset is relative to the kernel's own .text
				// section, and cudacapture enforces that the first is zero, so
				// the kernel's own offset has to be added to reach the CUBIN
				// file offset the profile's addresses are defined as. Without
				// it every kernel's entry is 0x7f000000 and their ranges alias
				// each other -- invisible in the line-oriented views, wrong in
				// exactly the address-keyed ones a SASS profile is for.
				Address: cudaprof.DefaultMappingStart + k.Offset + iw.inst.Offset,
			},
			{
				FunctionName: k.Name + ".PTX",
				FileName:     "kernels.ptx",
				Line:         int64(ptxLine),
				Address:      ptxAddress(rec.ptxLines, ptxLine),
			},
		}
		if tileIRLine > 0 {
			frames = append(frames, cudaprof.Frame{
				FunctionName: k.Name + ".TileIR",
				FileName:     "kernels.tileir",
				Line:         int64(tileIRLine),
			})
		}
		frames = append(frames, cudaprof.Frame{
			FunctionName: k.Name,
			FileName:     "kernels.cu",
			Line:         srcLine,
		})

		rec.prof.RecordStackSample(ctx, vals, append(frames, stack...))
		attributed++
	}

	return attributed
}
