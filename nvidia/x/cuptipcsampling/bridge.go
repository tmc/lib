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
	// HostFrames reports the host-side frames to attach above the device frames
	// for a resolved location, outermost last. It is called once per PC, so a
	// caller can return the stack that launched that kernel; returning a single
	// stack for every location makes every row filter alike and attributes the
	// device work to whatever host code happened to be running. Optional.
	HostFrames func(loc CodeLocation) []cudaprof.Frame
	// LabelFunc supplies additional context labels for a resolved location.
	// Optional.
	LabelFunc func(loc CodeLocation) map[string]string
	// UnmappedStalls accumulates, per native reason name, the samples of
	// reasons this package does not model. They reach no sample type, so a
	// nonempty map means the stall breakdown is not the whole measurement.
	UnmappedStalls map[string]int64
	// ReasonSamples accumulates every reason's samples under its native name,
	// modelled or not. Routing says where a reason would be counted;
	// ReasonSamples says how much was, which is what makes the routing
	// falsifiable against the same capture. It is also the only place the
	// long_scoreboard and short_scoreboard split survives, and that split is
	// the difference between "waiting on memory data" and a register
	// dependency, which stall_scoreboard alone cannot tell apart.
	ReasonSamples map[string]int64
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
			name := b.Inventory.Lookup(r.Index)
			if b.ReasonSamples == nil {
				b.ReasonSamples = make(map[string]int64)
			}
			b.ReasonSamples[name] += count
			// A _not_issued count is a companion over a subset of the cycles
			// its base reason already counted, so only base reasons add to the
			// sample total. Measured on a GB10 capture: base reasons sum to
			// 2938, exactly CUPTI's own total_samples, while adding the
			// companions gives 5527.
			if !isCompanion(name) {
				totalSamples += count
			}
			// The selected reason's own companion has to be recognised here
			// too. classifyStall has no selected case for its recursion to
			// land on, so a name that reaches it falls through to the
			// unmapped tally under a reason this package does model.
			if isSelected(strings.TrimSuffix(strings.ToLower(name), "_not_issued")) {
				if !isCompanion(name) {
					selectedSamples += count
				}
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

		if b.HostFrames != nil {
			frames = append(frames, b.HostFrames(loc)...)
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

// isCompanion reports whether name is the _not_issued companion of another
// reason rather than a reason of its own.
func isCompanion(name string) bool {
	return strings.HasSuffix(strings.ToLower(name), "_not_issued")
}

// isSelected reports whether name is the reason counting warps the scheduler
// selected, which is issued work rather than a stall. CUPTI names reasons in
// full, as smsp__pcsamp_warps_issue_stalled_selected, so an equality test
// against the short name matches nothing a native inventory reports; the
// suffix has to carry the stalled_ qualifier to keep not_selected out.
func isSelected(name string) bool {
	lower := strings.ToLower(name)
	return lower == "selected" || strings.HasSuffix(lower, "stalled_selected")
}

// Routing reports, for every reason the inventory names, where that reason's
// count goes in the profile: a sample type name, "companion" for a _not_issued
// reason whose count is deliberately discarded as a subset of its base reason,
// or "unmapped" for a reason this package does not model and reports only in
// the unmapped tally.
//
// It exists because the reconciliation identity cannot see a count in the wrong
// bucket: the identity sums every stall type, so routing a fetch stall to
// stall_mem leaves it closed. Recording this map beside a capture makes the
// classifier's decisions auditable from the artifact, including for reasons the
// hardware never sampled on that run. Routing is a pure function of the reason
// name, so one entry per name is total coverage.
func (inv ReasonInventory) Routing() map[string]string {
	routing := make(map[string]string, len(inv.Reasons))
	for _, r := range inv.Reasons {
		routing[r.Name] = reasonRouting(r.Name)
	}
	return routing
}

func reasonRouting(name string) string {
	if isSelected(strings.TrimSuffix(strings.ToLower(name), "_not_issued")) {
		if isCompanion(name) {
			return "companion"
		}
		return "inst_executed"
	}
	if isCompanion(name) {
		if reasonRouting(strings.TrimSuffix(strings.ToLower(name), "_not_issued")) == "unmapped" {
			return "unmapped"
		}
		return "companion"
	}
	var vals cudaprof.SampleValues
	if !classifyStall(name, 1, &vals) {
		return "unmapped"
	}
	switch {
	case vals.StallScoreboard != 0:
		return "stall_scoreboard"
	case vals.StallWait != 0:
		return "stall_wait"
	case vals.StallMath != 0:
		return "stall_math"
	case vals.StallMem != 0:
		return "stall_mem"
	case vals.StallBranch != 0:
		return "stall_branch"
	case vals.StallFetch != 0:
		return "stall_fetch"
	case vals.StallDrain != 0:
		return "stall_drain"
	case vals.StallDispatch != 0:
		return "stall_dispatch"
	case vals.StallNotSelected != 0:
		return "stall_not_selected"
	}
	return "unmapped"
}

// classifyStall maps a native stall reason name to a cudaprof stall metric. It
// covers two naming schemes: the modern smsp__pcsamp_warps_issue_stalled_*
// reasons the PC sampling API reports, and the legacy stall enum of the
// activity API, whose names are the only witnesses several arms below have.
//
// Order matters, because the names overlap by substring. Taking both
// vocabularies together, three names match more than one arm and so depend on
// the order: membar (wait before the bare mem of the memory arm),
// memory_dependency (scoreboard before memory) and constant_memory_dependency
// (constant before both). Each is pinned by a row in the routing test, because
// reordering the switch moves their counts between buckets with every other
// test still green. imc_miss also matches two arms, but both say stall_mem.
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
	// Waits: synchronization, memory barriers, and dependencies on a result
	// whose latency is fixed. Checked before the memory throttle arm below,
	// which the bare mem of membar would otherwise reach.
	//
	// exec_dependency is the legacy activity-API name for a warp waiting on a
	// non-memory input, a union of what the PC sampling API splits into wait and
	// short_scoreboard. A union can only be routed to its dominant member, and
	// the measurement says wait: a dependent FMA chain on a GB10 reports wait
	// 71.88% against short_scoreboard 0.06%, because a fixed-latency result is
	// resolved with compiler-inserted stall cycles rather than a scoreboard slot.
	case strings.Contains(name, "barrier") || strings.Contains(name, "membar") || strings.Contains(name, "sync") ||
		strings.Contains(name, "wait") || strings.Contains(name, "exec_dependency"):
		vals.StallWait += count
		return true

	// Constant-cache misses, before the dependency arm below: the legacy name
	// is constant_memory_dependency, which contains memory_dependency, and the
	// modern equivalent imc_miss is a memory stall rather than a wait on a
	// result the SM has already asked for.
	case strings.Contains(name, "constant") || strings.Contains(name, "imc"):
		vals.StallMem += count
		return true

	// Register scoreboard dependencies: a result whose latency is not known at
	// compile time, so the warp waits on a scoreboard slot. memory_dependency
	// is the legacy activity-API name for the same wait long_scoreboard
	// reports, and is matched here rather than by the memory arm below on
	// purpose -- filing a wait on load data under stall_mem would make the
	// bucket depend on which API produced the capture rather than on what the
	// warp was doing.
	case strings.Contains(name, "scoreboard") || strings.Contains(name, "memory_dependency"):
		vals.StallScoreboard += count
		return true

	// Arithmetic and math pipeline throttle
	case strings.Contains(name, "math") || strings.Contains(name, "pipe_busy") || strings.Contains(name, "pipe_throttle"):
		vals.StallMath += count
		return true

	// Memory sub-system stalls and throttles
	case strings.Contains(name, "mem") || strings.Contains(name, "lg_throttle") || strings.Contains(name, "mio_throttle") ||
		strings.Contains(name, "texture") || strings.Contains(name, "tex_throttle") || strings.Contains(name, "dram"):
		vals.StallMem += count
		return true

	// Front-end: the warp has no instruction to issue yet. The instruction
	// cache miss behind no_instructions is a fetch stall, not a data stall, so
	// it gets its own type rather than joining stall_mem.
	case strings.Contains(name, "no_instruction") || strings.Contains(name, "inst_fetch"):
		vals.StallFetch += count
		return true

	case strings.Contains(name, "branch"):
		vals.StallBranch += count
		return true

	case strings.Contains(name, "drain"):
		vals.StallDrain += count
		return true

	case strings.Contains(name, "dispatch"):
		vals.StallDispatch += count
		return true

	// Not a stall in the usual sense: the warp was ready and the scheduler
	// issued a different one. It is reported among the stall reasons, so it is
	// counted here, but it measures issue contention rather than latency.
	case strings.Contains(name, "not_selected"):
		vals.StallNotSelected += count
		return true

	default:
		return false
	}
}
