package cuptipcsampling

import (
	"fmt"
	"strings"

	"github.com/tmc/lib/nvidia/x/cudaprof"
)

// StallCategory groups raw pipeline stall reasons into diagnostic categories.
type StallCategory string

const (
	StallCategoryMemory           StallCategory = "memory"
	StallCategorySynchronization  StallCategory = "synchronization"
	StallCategoryExecution        StallCategory = "execution"
	StallCategoryInstructionFetch StallCategory = "instruction_fetch"
	StallCategoryScheduler        StallCategory = "scheduler"
	StallCategoryBranch           StallCategory = "branch"
	StallCategoryDrain            StallCategory = "drain"
	StallCategoryDispatch         StallCategory = "dispatch"
	StallCategoryIssued           StallCategory = "issued"
	StallCategoryOther            StallCategory = "other"
)

// PipelineStall describes a classified pipeline stall reason with its sample
// count, human-readable description, and cost percentage.
type PipelineStall struct {
	Name        string        // Canonical stall name (e.g. "long_scoreboard", "stall_mem")
	RawName     string        // Verbatim CUPTI reason name
	Category    StallCategory // Diagnostic category
	Description string        // Human-readable diagnostic description
	Samples     uint64        // Sample count (warp cycles)
	Percent     float64       // Cost percentage of total stall samples (0.0 to 100.0)
	Companion   uint64        // Samples observed under the _not_issued companion, if tracked
}

// SASSMetric represents high-level decoded SASS instruction metrics at a PC offset.
type SASSMetric struct {
	PC           uint64          // Instruction PC offset
	KernelName   string          // Mangled or symbol kernel name
	CRC          uint64          // Cubin CRC
	TotalSamples uint64          // Total warp execution samples at this PC
	Selected     uint64          // Warps selected and issued for execution
	SelectedPct  float64         // Issued percentage of total samples (0.0 to 100.0)
	StallSamples uint64          // Total stall samples (excluding selected)
	Stalls       []PipelineStall // Detailed breakdown of active stall reasons
	Dominant     *PipelineStall  // Primary pipeline bottleneck (stall reason with highest count, excluding selected)
}

// DominantReason returns the canonical name of the dominant stall reason,
// or "none" if there are no stalls.
func (m SASSMetric) DominantReason() string {
	if m.Dominant != nil {
		return m.Dominant.Name
	}
	return "none"
}

// StallCost returns the cost percentage (0.0 to 100.0) attributed to the given stall reason.
func (m SASSMetric) StallCost(reason string) float64 {
	target := strings.ToLower(reason)
	for i := range m.Stalls {
		s := &m.Stalls[i]
		if strings.ToLower(s.Name) == target || strings.ToLower(s.RawName) == target {
			return s.Percent
		}
	}
	return 0.0
}

// CategoryCost returns the cumulative cost percentage (0.0 to 100.0) of all stalls
// belonging to the specified category.
func (m SASSMetric) CategoryCost(cat StallCategory) float64 {
	var total float64
	for i := range m.Stalls {
		if m.Stalls[i].Category == cat {
			total += m.Stalls[i].Percent
		}
	}
	return total
}

// ToCudaprof converts the SASSMetric into a cudaprof.SASSMetric suitable for
// recording into a cudaprof session.
func (m SASSMetric) ToCudaprof() cudaprof.SASSMetric {
	var dominantName string
	var dominantWarps int64
	if m.Dominant != nil {
		dominantName = m.Dominant.Name
		dominantWarps = int64(m.Dominant.Samples)
	}
	return cudaprof.SASSMetric{
		InstructionPC: m.PC,
		StallReason:   dominantName,
		StallWarps:    dominantWarps,
		IssuedWarps:   int64(m.Selected),
		ExecutedWarps: int64(m.Selected),
	}
}

// FlightRecordLabels returns contextual attributes describing this SASS metric
// for use in diagnostic flight recorders.
func (m SASSMetric) FlightRecordLabels() map[string]string {
	labels := make(map[string]string, 6)
	labels["sass.pc"] = fmt.Sprintf("0x%x", m.PC)
	labels["sass.total_samples"] = fmt.Sprintf("%d", m.TotalSamples)
	labels["sass.selected"] = fmt.Sprintf("%d", m.Selected)
	labels["sass.selected_pct"] = fmt.Sprintf("%.2f", m.SelectedPct)
	if m.Dominant != nil {
		labels["sass.dominant_stall"] = m.Dominant.Name
		labels["sass.dominant_pct"] = fmt.Sprintf("%.2f", m.Dominant.Percent)
		labels["sass.dominant_category"] = string(m.Dominant.Category)
	}
	return labels
}

// ClassifyPipelineStall maps a native or short stall reason name to its
// canonical name, category, human-readable description, and whether it is a companion.
func ClassifyPipelineStall(reasonName string) (canonical string, cat StallCategory, desc string, isCompanion bool) {
	name := strings.ToLower(reasonName)
	if base, ok := strings.CutSuffix(name, "_not_issued"); ok {
		c, cat, d, _ := ClassifyPipelineStall(base)
		return c, cat, d, true
	}

	// 1. Issued work
	if isSelected(name) {
		return "selected", StallCategoryIssued, "Warp instructions selected and issued to execution pipeline", false
	}

	// 2. Synchronization and barriers
	if strings.Contains(name, "barrier") || strings.Contains(name, "membar") {
		return "stall_barrier", StallCategorySynchronization, "Waiting on warp barrier, cluster barrier, or memory fence", false
	}
	if strings.Contains(name, "sync") {
		return "stall_sync", StallCategorySynchronization, "Waiting at thread barrier or synchronization primitive (__syncthreads)", false
	}
	if strings.Contains(name, "wait") || strings.Contains(name, "exec_dependency") {
		return "stall_wait", StallCategorySynchronization, "Fixed-latency instruction dependency or explicit stall cycles inserted by compiler", false
	}

	// 3. Constant cache and immediate memory
	if strings.Contains(name, "constant") || strings.Contains(name, "imc") {
		return "stall_constant_mem", StallCategoryMemory, "Waiting on constant cache miss or immediate constant memory fetch", false
	}

	// 4. Register scoreboard dependencies
	if strings.Contains(name, "long_scoreboard") {
		return "long_scoreboard", StallCategoryMemory, "Waiting on scoreboard for global memory, local memory, or texture load data", false
	}
	if strings.Contains(name, "short_scoreboard") {
		return "short_scoreboard", StallCategoryExecution, "Waiting on scoreboard for fixed-latency arithmetic or shared memory instruction", false
	}
	if strings.Contains(name, "scoreboard") || strings.Contains(name, "memory_dependency") {
		return "stall_scoreboard", StallCategoryMemory, "Waiting on register scoreboard dependency from pending memory or arithmetic", false
	}

	// 5. Arithmetic pipeline throttles
	if strings.Contains(name, "math") || strings.Contains(name, "pipe_busy") || strings.Contains(name, "pipe_throttle") {
		return "stall_math", StallCategoryExecution, "Arithmetic execution pipeline congested or busy", false
	}

	// 6. Memory sub-system throttles
	if strings.Contains(name, "lg_throttle") {
		return "stall_lg_throttle", StallCategoryMemory, "Local/global memory pipeline instruction queue full", false
	}
	if strings.Contains(name, "mio_throttle") {
		return "stall_mio_throttle", StallCategoryMemory, "Memory input/output (MIO) pipeline throttle", false
	}
	if strings.Contains(name, "tex_throttle") || strings.Contains(name, "texture") {
		return "stall_tex_throttle", StallCategoryMemory, "Texture sub-system pipeline throttle", false
	}
	if strings.Contains(name, "mem") || strings.Contains(name, "dram") {
		return "stall_mem", StallCategoryMemory, "Memory pipeline throttled or execution stalled due to congested memory sub-system", false
	}

	// 7. Front-end / instruction fetch
	if strings.Contains(name, "no_instruction") || strings.Contains(name, "inst_fetch") {
		return "stall_no_instruction", StallCategoryInstructionFetch, "Instruction cache miss or instruction fetch latency", false
	}

	// 8. Control flow / branch
	if strings.Contains(name, "branch") {
		return "stall_branch", StallCategoryBranch, "Waiting for branch target address or condition resolution", false
	}

	// 9. Pipeline drain
	if strings.Contains(name, "drain") {
		return "stall_drain", StallCategoryDrain, "Pipeline draining execution state", false
	}

	// 10. Dispatch
	if strings.Contains(name, "dispatch") {
		return "stall_dispatch", StallCategoryDispatch, "Waiting on dispatch unit resources", false
	}

	// 11. Scheduler issue contention
	if strings.Contains(name, "not_selected") {
		return "not_selected", StallCategoryScheduler, "Warp ready and eligible to issue, but not chosen by warp scheduler", false
	}

	// 12. Fallback
	return name, StallCategoryOther, "Unclassified pipeline stall", false
}

// DecodeSASSMetric decodes a PC sample record into a typed SASSMetric with human-readable
// descriptions and cost percentages.
func (inv ReasonInventory) DecodeSASSMetric(pc PC) SASSMetric {
	m := SASSMetric{
		PC:         pc.Offset,
		KernelName: pc.Name,
		CRC:        pc.CRC,
	}

	type stallAccum struct {
		canonical string
		rawName   string
		cat       StallCategory
		desc      string
		samples   uint64
		companion uint64
	}
	accumMap := make(map[string]*stallAccum)
	var order []string

	for _, r := range pc.Reasons {
		if r.Samples == 0 {
			continue
		}
		raw := inv.Lookup(r.Index)
		if raw == "" {
			raw = fmt.Sprintf("reason_%d", r.Index)
		}
		count := uint64(r.Samples)

		canonical, cat, desc, companion := ClassifyPipelineStall(raw)
		if canonical == "selected" {
			if !companion {
				m.Selected += count
				m.TotalSamples += count
			}
			continue
		}

		if companion {
			acc := accumMap[canonical]
			if acc != nil {
				acc.companion += count
			}
			continue
		}

		m.TotalSamples += count
		m.StallSamples += count

		acc := accumMap[canonical]
		if acc == nil {
			acc = &stallAccum{
				canonical: canonical,
				rawName:   raw,
				cat:       cat,
				desc:      desc,
			}
			accumMap[canonical] = acc
			order = append(order, canonical)
		}
		acc.samples += count
	}

	if m.TotalSamples > 0 && m.Selected > 0 {
		m.SelectedPct = float64(m.Selected) / float64(m.TotalSamples) * 100.0
	}

	var dominantIdx = -1
	var maxStallSamples uint64

	m.Stalls = make([]PipelineStall, 0, len(order))
	for _, k := range order {
		acc := accumMap[k]
		var pct float64
		if m.StallSamples > 0 {
			pct = float64(acc.samples) / float64(m.StallSamples) * 100.0
		}
		stall := PipelineStall{
			Name:        acc.canonical,
			RawName:     acc.rawName,
			Category:    acc.cat,
			Description: acc.desc,
			Samples:     acc.samples,
			Percent:     pct,
			Companion:   acc.companion,
		}
		if acc.samples > maxStallSamples {
			maxStallSamples = acc.samples
			dominantIdx = len(m.Stalls)
		}
		m.Stalls = append(m.Stalls, stall)
	}

	if dominantIdx >= 0 {
		m.Dominant = &m.Stalls[dominantIdx]
	}

	return m
}

// SASSMetrics decodes all valid PC records in the batch into typed SASSMetric structs.
func (b Batch) SASSMetrics(inv ReasonInventory) []SASSMetric {
	if !b.Valid || len(b.PCs) == 0 {
		return nil
	}
	metrics := make([]SASSMetric, 0, len(b.PCs))
	for _, pc := range b.PCs {
		metrics = append(metrics, inv.DecodeSASSMetric(pc))
	}
	return metrics
}
