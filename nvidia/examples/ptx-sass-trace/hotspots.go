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

// recordInstructionHotspots distributes the measured launch duration across the
// kernel's disassembled instructions according to an opcode cost model, turning
// navigation anchors into genuine instruction-level hotspots with stall attribution.
func recordInstructionHotspots(
	ctx context.Context,
	rec *launchRecorder,
	k *cudacapture.Kernel,
	tileIRLine int,
	elapsedNanos int64,
	stack []cudaprof.Frame,
) int {
	if rec == nil || rec.prof == nil || k == nil || len(k.Instructions) == 0 {
		return 0
	}

	type instWeight struct {
		inst   cudacapture.Instruction
		weight int64
		stall  string
	}

	weights := make([]instWeight, len(k.Instructions))
	var totalWeight int64
	for i, inst := range k.Instructions {
		op := instructionOpcode(inst.Text)
		w, stall := instructionCost(op)
		weights[i] = instWeight{inst: inst, weight: w, stall: stall}
		totalWeight += w
	}
	if totalWeight <= 0 {
		totalWeight = 1
	}

	attributed := 0
	for _, iw := range weights {
		instNanos := (elapsedNanos * iw.weight) / totalWeight
		if instNanos <= 0 {
			instNanos = 1
		}

		vals := cudaprof.SampleValues{
			Samples:      iw.weight,
			PCSamples:    iw.weight,
			InstExecuted: iw.weight,
			GPUNanos:     instNanos,
		}

		switch iw.stall {
		case "stall_scoreboard":
			vals.StallScoreboard = iw.weight
			vals.StallMem = iw.weight / 2
		case "stall_wait":
			vals.StallWait = iw.weight
		case "stall_math":
			vals.StallMath = iw.weight
		case "stall_branch":
			vals.StallBranch = iw.weight
		}

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
				Address:      cudaprof.DefaultMappingStart + iw.inst.Offset,
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
