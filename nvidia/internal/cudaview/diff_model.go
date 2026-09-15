package cudaview

import (
	"encoding/json"
	"github.com/tmc/lib/nvidia/x/cudasass"
	"html/template"
)

// DiffRowType specifies the relationship between baseline and candidate instructions in a diff row.
type DiffRowType string

const (
	DiffRowMatch  DiffRowType = "match"
	DiffRowAdd    DiffRowType = "add"
	DiffRowDelete DiffRowType = "delete"
	DiffRowModify DiffRowType = "modify"
)

// DiffInstruction represents an annotated SASS instruction in the differential viewer.
type DiffInstruction struct {
	Address         uint64             `json:"address,omitempty"`
	Offset          string             `json:"offset,omitempty"`
	Predicate       string             `json:"predicate,omitempty"`
	Opcode          string             `json:"opcode,omitempty"`
	Operands        string             `json:"operands,omitempty"`
	Raw             string             `json:"raw,omitempty"`
	Text            string             `json:"text,omitempty"`
	SourceLine      int                `json:"source_line,omitempty"`
	IsLocalMemory   bool               `json:"is_local_memory,omitempty"`
	LocalMemoryType string             `json:"local_memory_type,omitempty"`
	Pressure        string             `json:"pressure,omitempty"`
	PressureTooltip string             `json:"pressure_tooltip,omitempty"`
	LiveGPR         int                `json:"live_gpr,omitempty"`
	LivePred        int                `json:"live_pred,omitempty"`
	LiveUGPR        int                `json:"live_ugpr,omitempty"`
	LiveUPred       int                `json:"live_upred,omitempty"`
	StallPct        float64            `json:"stall_pct,omitempty"`
	StallBreakdown  map[string]float64 `json:"stall_breakdown,omitempty"`
	StallTooltip    string             `json:"stall_tooltip,omitempty"`
	SemanticTag     string             `json:"semantic_tag,omitempty"`
}

// DiffRow represents a single aligned comparison row between baseline and candidate.
type DiffRow struct {
	Num            int              `json:"num"`
	Type           DiffRowType      `json:"type"`
	Base           *DiffInstruction `json:"base,omitempty"`
	Candidate      *DiffInstruction `json:"candidate,omitempty"`
	StallDelta     float64          `json:"stall_delta,omitempty"`
	StallReason    string           `json:"stall_reason,omitempty"`
	StallSummary   string           `json:"stall_summary,omitempty"`
	HasLocalMemory bool             `json:"has_local_memory,omitempty"`
}

// DiffSummary contains high-level differential statistics between two captures.
type DiffSummary struct {
	BaseInstructions   int                           `json:"base_instructions"`
	TargetInstructions int                           `json:"target_instructions"`
	InstructionDelta   int                           `json:"instruction_delta"`
	BaseSizeBytes      int                           `json:"base_size_bytes"`
	TargetSizeBytes    int                           `json:"target_size_bytes"`
	TotalSizeDelta     int                           `json:"total_size_delta"`
	BaseBranches       int                           `json:"base_branches"`
	TargetBranches     int                           `json:"target_branches"`
	BranchDelta        int                           `json:"branch_delta"`
	LocalMemoryAdded   int                           `json:"local_memory_added"`
	LocalMemoryRemoved int                           `json:"local_memory_removed"`
	LocalMemoryDelta   int                           `json:"local_memory_delta"`
	BaseRuntimeUs      float64                       `json:"base_runtime_us,omitempty"`
	TargetRuntimeUs    float64                       `json:"target_runtime_us,omitempty"`
	DurationDeltaUs    float64                       `json:"duration_delta_us,omitempty"`
	Speedup            float64                       `json:"speedup,omitempty"`
	Substitutions      []cudasass.OpcodeSubstitution `json:"substitutions,omitempty"`
	TopStallShifts     []StallShiftSummary           `json:"top_stall_shifts,omitempty"`
}

// StallShiftSummary records the aggregate shift of a PC stall reason across two captures.
type StallShiftSummary struct {
	Reason      string  `json:"reason"`
	BaseCount   int64   `json:"base_count"`
	TargetCount int64   `json:"target_count"`
	Delta       int64   `json:"delta"`
	ShiftPct    float64 `json:"shift_pct"`
	ShiftStr    string  `json:"shift_str"`
}

// DiffViewData is the data model passed to the differential web template and HTTP endpoints.
type DiffViewData struct {
	Title           string                 `json:"title"`
	BaseCapture     string                 `json:"base_capture"`
	TargetCapture   string                 `json:"target_capture"`
	Kernels         []string               `json:"kernels"`
	ActiveKernel    string                 `json:"active_kernel"`
	KernelRows      map[string][]DiffRow   `json:"kernel_rows"`
	ActiveRows      []DiffRow              `json:"active_rows"`
	Summary         DiffSummary            `json:"summary"`
	KernelSummaries map[string]DiffSummary `json:"kernel_summaries"`
}

// ToJSON serializes DiffViewData to JavaScript-safe JSON for offline viewing.
func (d *DiffViewData) ToJSON() template.JS {
	snapshot := *d
	if _, ok := snapshot.KernelRows[snapshot.ActiveKernel]; ok {
		snapshot.ActiveRows = nil
	}
	b, err := json.Marshal(&snapshot)
	if err != nil {
		return template.JS("{}")
	}
	return template.JS(b)
}
