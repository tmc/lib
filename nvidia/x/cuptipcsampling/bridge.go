package cuptipcsampling

import (
	"context"
	"errors"
	"runtime/pprof"
	"strings"

	"github.com/tmc/lib/nvidia/x/cudaprof"
	"github.com/tmc/lib/nvidia/x/cudaptx"
)

// ModuleID identifies a loaded CUDA module image by its native CUBIN CRC.
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
	InlinedAt   []cudaptx.Location
}

// CodeResolver resolves a kernel function and function-relative PC offset to its source
// coordinates and CUBIN-absolute offset.
type CodeResolver interface {
	ResolvePC(mod ModuleID, kernelName string, pcOffset uint64) CodeLocation
}

// ProfileBridge routes native PC sampling batches into a cudaprof session.
type ProfileBridge struct {
	Session   *cudaprof.Session
	Inventory ReasonInventory
	Resolver  CodeResolver // optional
	// PTXOffsets indexes the byte offset of each one-based PTX line, as
	// [cudaprof.PTXLineOffsets] returns it. Without it a PTX frame carries no
	// address, and pprof's address granularity is blind to the samples that
	// have real instruction addresses. Optional.
	PTXOffsets []uint64
	// HostFrames provides host-side callstack frames (e.g. Go caller functions)
	// attached above the device frames on recorded PC samples. Optional.
	HostFrames []cudaprof.Frame
	// LabelFunc supplies additional context labels for a resolved location.
	// Optional.
	LabelFunc func(loc CodeLocation) map[string]string
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
		// PC samples count occurrences at an address, which is a different
		// thing from the launch counts the samples type carries; summing the
		// two would make neither readable.
		vals.PCSamples = totalSamples
		if selectedSamples > 0 {
			vals.InstExecuted = selectedSamples
		}

		var loc CodeLocation
		if b.Resolver != nil {
			loc = b.Resolver.ResolvePC(ModuleID{CRC: pc.CRC}, pc.Name, pc.Offset)
		}
		if loc.KernelName == "" {
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
			var ptxAddr uint64
			if int(loc.PTXLine) < len(b.PTXOffsets) {
				ptxAddr = cudaprof.PTXMappingStart + b.PTXOffsets[loc.PTXLine]
			}
			frames = append(frames, cudaprof.Frame{
				FunctionName: loc.KernelName + ".PTX",
				FileName:     ptxFile,
				Line:         loc.PTXLine,
				Address:      ptxAddr,
			})
		}

		// Inlined frames (innermost first)
		for _, inl := range loc.InlinedAt {
			if inl.Line > 0 {
				file := inl.File
				if file == "" {
					file = "kernels.cu"
				}
				frames = append(frames, cudaprof.Frame{
					FunctionName: loc.KernelName + ".inlined",
					FileName:     file,
					Line:         int64(inl.Line),
				})
			}
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

		if len(b.HostFrames) > 0 {
			frames = append(frames, b.HostFrames...)
		}

		sampleCtx := ctx
		if loc.KernelName != "" || b.LabelFunc != nil {
			var pairs []string
			if loc.KernelName != "" {
				pairs = append(pairs, "kernel", loc.KernelName)
			}
			if b.LabelFunc != nil {
				for k, v := range b.LabelFunc(loc) {
					pairs = append(pairs, k, v)
				}
			}
			sampleCtx = pprof.WithLabels(ctx, pprof.Labels(pairs...))
		}

		b.Session.RecordStackSample(sampleCtx, vals, frames)
	}

	return nil
}

// classifyStall maps a native stall reason name to a cudaprof stall metric.
// Order matters: barrier and membar checks must precede memory throttle checks.
func classifyStall(reasonName string, count int64, vals *cudaprof.SampleValues) bool {
	name := strings.ToLower(reasonName)
	// CUPTI reports each reason twice: smsp__pcsamp_warps_issue_stalled_<r>
	// and the _not_issued companion counting the subset of those cycles where
	// the scheduler issued nothing. Adding both counts the same samples twice,
	// which is what made the stall totals run about 1.9x the collected sample
	// count. NVIDIA's own pc_sampling_start_stop output shows the pairing
	// plainly: long_scoreboard 23 beside long_scoreboard_not_issued 23. The
	// companion is mapped, so it stays out of the unmapped tally, but only the
	// base reason is added.
	if base, ok := strings.CutSuffix(name, "_not_issued"); ok {
		// Classify the base reason to decide, but discard the count: a reason
		// this function does not model stays visible as unmapped under its own
		// full name rather than vanishing.
		var discard cudaprof.SampleValues
		return classifyStall(base, count, &discard)
	}
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
