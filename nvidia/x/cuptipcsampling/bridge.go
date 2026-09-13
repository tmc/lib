package cuptipcsampling

import (
	"context"
	"errors"
	"strings"

	"github.com/tmc/lib/nvidia/x/cudaprof"
)

// ModuleID identifies a loaded CUDA module or binary image generation.
type ModuleID struct {
	CRC uint64
}

// CodeLocation identifies the source, assembly, and CUBIN coordinates for a kernel PC.
type CodeLocation struct {
	KernelName  string
	SourceFile  string
	SourceLine  int64
	PTXFile     string
	PTXLine     int64
	SASSFile    string
	SASSLine    int64
	CUBINOffset uint64 // Absolute byte offset within the CUBIN ELF image.
	Resolved    bool   // True if the kernel and instruction were successfully located.
}

// CodeResolver resolves a kernel function and function-relative PC offset to its source
// coordinates and CUBIN-absolute offset.
type CodeResolver interface {
	ResolvePC(mod ModuleID, kernelName string, pcOffset uint64) CodeLocation
}

// ProfileBridge routes native PC sampling batches into a cudaprof session.
type ProfileBridge struct {
	Session        *cudaprof.Session
	Inventory      ReasonInventory
	Resolver       CodeResolver // optional
	UnmappedStalls map[string]int64
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
		var selectedSamples int64

		for _, r := range pc.Reasons {
			count := int64(r.Samples)
			if count == 0 {
				continue
			}
			totalSamples += count
			name := b.Inventory.Lookup(r.Index)
			if name == "selected" {
				selectedSamples += count
				continue
			}
			if !classifyStall(name, count, &vals) {
				if b.UnmappedStalls == nil {
					b.UnmappedStalls = make(map[string]int64)
				}
				b.UnmappedStalls[name] += count
			}
		}

		if totalSamples == 0 {
			continue
		}
		vals.Samples = totalSamples
		if selectedSamples > 0 {
			vals.InstExecuted = selectedSamples
		}

		var loc CodeLocation
		if b.Resolver != nil {
			loc = b.Resolver.ResolvePC(ModuleID{CRC: pc.CRC}, pc.Name, pc.Offset)
		}
		if !loc.Resolved {
			loc.KernelName = pc.Name
		}

		var frames []cudaprof.Frame

		// Leaf: SASS instruction
		sassFile := loc.SASSFile
		if sassFile == "" {
			sassFile = "kernels.sass"
		}
		var leafAddr uint64
		if loc.Resolved {
			leafAddr = cudaprof.DefaultMappingStart + loc.CUBINOffset
		}
		frames = append(frames, cudaprof.Frame{
			FunctionName: loc.KernelName + ".SASS",
			FileName:     sassFile,
			Line:         loc.SASSLine,
			Address:      leafAddr,
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

// classifyStall maps a native stall reason name to a cudaprof stall metric.
// Order matters: barrier and membar checks must precede memory throttle checks.
func classifyStall(reasonName string, count int64, vals *cudaprof.SampleValues) bool {
	name := strings.ToLower(reasonName)
	switch {
	// Synchronization and memory barriers (check before general memory throttle)
	case strings.Contains(name, "barrier") || strings.Contains(name, "membar") || strings.Contains(name, "sync") || strings.Contains(name, "wait"):
		vals.StallWait += count
		return true

	// Register scoreboard dependencies
	case strings.Contains(name, "scoreboard"):
		vals.StallScoreboard += count
		return true

	// Arithmetic and math pipeline throttle
	case strings.Contains(name, "math") || strings.Contains(name, "pipe_busy") || strings.Contains(name, "pipe_throttle"):
		vals.StallMath += count
		return true

	// Memory sub-system stalls and throttles
	case strings.Contains(name, "mem") || strings.Contains(name, "lg_throttle") || strings.Contains(name, "mio_throttle") ||
		strings.Contains(name, "constant") || strings.Contains(name, "texture") || strings.Contains(name, "tex_throttle") ||
		strings.Contains(name, "imc") || strings.Contains(name, "dram"):
		vals.StallMem += count
		return true

	default:
		return false
	}
}
