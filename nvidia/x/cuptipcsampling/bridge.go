package cuptipcsampling

import (
	"context"
	"errors"
	"strings"

	"github.com/tmc/lib/nvidia/x/cudaprof"
)

// CodeLocation identifies the source and assembly location for a kernel PC.
type CodeLocation struct {
	KernelName string
	SourceFile string
	SourceLine int64
	PTXFile    string
	PTXLine    int64
	SASSFile   string
	SASSLine   int64
	Offset     uint64 // Absolute offset within the CUBIN
}

// CodeResolver resolves a kernel function and PC offset to its source coordinates.
type CodeResolver interface {
	ResolvePC(kernelName string, pcOffset uint64) CodeLocation
}

// ProfileBridge routes native PC sampling batches into a cudaprof session.
type ProfileBridge struct {
	Session   *cudaprof.Session
	Inventory ReasonInventory
	Resolver  CodeResolver // optional
}

// RecordBatch records all valid PC samples and stall reasons in batch into the profile session.
func (b *ProfileBridge) RecordBatch(ctx context.Context, batch Batch) error {
	if b == nil || b.Session == nil {
		return errors.New("cuptipcsampling: nil profile session")
	}
	if !batch.Valid {
		return errors.New("cuptipcsampling: invalid sampling batch")
	}

	for _, pc := range batch.PCs {
		if pc.Name == "" {
			continue
		}

		var vals cudaprof.SampleValues
		var totalSamples int64

		for _, r := range pc.Reasons {
			count := int64(r.Samples)
			if count == 0 {
				continue
			}
			totalSamples += count
			name := b.Inventory.Lookup(r.Index)
			classifyStall(name, count, &vals)
		}

		if totalSamples == 0 {
			continue
		}
		vals.Samples = totalSamples
		vals.InstExecuted = totalSamples

		var loc CodeLocation
		if b.Resolver != nil {
			loc = b.Resolver.ResolvePC(pc.Name, pc.Offset)
		}
		if loc.KernelName == "" {
			loc.KernelName = pc.Name
		}
		if loc.Offset == 0 {
			loc.Offset = pc.Offset
		}

		var frames []cudaprof.Frame

		// Leaf: SASS instruction
		sassFile := loc.SASSFile
		if sassFile == "" {
			sassFile = "kernels.sass"
		}
		frames = append(frames, cudaprof.Frame{
			FunctionName: loc.KernelName + ".SASS",
			FileName:     sassFile,
			Line:         loc.SASSLine,
			Address:      cudaprof.DefaultMappingStart + loc.Offset,
		})

		// Intermediate: PTX virtual ISA (if resolved)
		if loc.PTXLine > 0 {
			ptxFile := loc.PTXFile
			if ptxFile == "" {
				ptxFile = "kernels.ptx"
			}
			frames = append(frames, cudaprof.Frame{
				FunctionName: loc.KernelName + ".PTX",
				FileName:     ptxFile,
				Line:         loc.PTXLine,
			})
		}

		// Top device frame: CUDA source line (if resolved)
		if loc.SourceLine > 0 {
			srcFile := loc.SourceFile
			if srcFile == "" {
				srcFile = "kernels.cu"
			}
			frames = append(frames, cudaprof.Frame{
				FunctionName: loc.KernelName,
				FileName:     srcFile,
				Line:         loc.SourceLine,
			})
		}

		b.Session.RecordStackSample(ctx, vals, frames)
	}

	return nil
}

func classifyStall(reasonName string, count int64, vals *cudaprof.SampleValues) {
	name := strings.ToLower(reasonName)
	switch {
	case strings.Contains(name, "mem") || strings.Contains(name, "constant") || strings.Contains(name, "texture"):
		vals.StallMem += count
	case strings.Contains(name, "math") || strings.Contains(name, "pipe_busy") || strings.Contains(name, "pipe_throttle"):
		vals.StallMath += count
	case strings.Contains(name, "sync") || strings.Contains(name, "barrier") || strings.Contains(name, "wait"):
		vals.StallWait += count
	case strings.Contains(name, "scoreboard") || strings.Contains(name, "dependency") || strings.Contains(name, "exec"):
		vals.StallScoreboard += count
	}
}
