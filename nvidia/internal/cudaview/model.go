package cudaview

// PipelineTier represents one stage in the Go -> CUDA C++ -> Tile IR -> PTX -> SASS compilation hierarchy.
type PipelineTier struct {
	Name     string         `json:"name"`
	FileName string         `json:"file_name"`
	Language string         `json:"language"`
	Lines    []PipelineLine `json:"lines"`
}

// PipelineLine represents an annotated line in a pipeline tier.
type PipelineLine struct {
	Kernel          string             `json:"kernel,omitempty"`
	Predicate       string             `json:"predicate,omitempty"`
	Operands        string             `json:"operands,omitempty"`
	Encoding        string             `json:"encoding,omitempty"`
	Num             int                `json:"num"`
	Text            string             `json:"text"`
	Opcode          string             `json:"opcode,omitempty"`
	Category        string             `json:"category,omitempty"`
	Offset          string             `json:"offset,omitempty"`
	StallPct        float64            `json:"stall_pct,omitempty"`
	StallBreakdown  map[string]float64 `json:"stall_breakdown,omitempty"`
	StallTooltip    string             `json:"stall_tooltip,omitempty"`
	HoverDoc        string             `json:"hover_doc,omitempty"`
	RemarkBadge     string             `json:"remark_badge,omitempty"`
	RemarkTooltip   string             `json:"remark_tooltip,omitempty"`
	LaunchKernel    string             `json:"launch_kernel,omitempty"`
	SemanticTag     string             `json:"semantic_tag,omitempty"`
	SourceFile      string             `json:"source_file,omitempty"`
	SourceLine      int                `json:"source_line,omitempty"`
	Pressure        string             `json:"pressure,omitempty"`
	PressureTooltip string             `json:"pressure_tooltip,omitempty"`
}

// PipelineViewData is the data model passed to the web template and API endpoints.
type PipelineViewData struct {
	Diagnostics   []string                  `json:"diagnostics,omitempty"`
	Modules       []PipelineModule          `json:"-"`
	Title         string                    `json:"title"`
	Device        string                    `json:"device"`
	Architecture  string                    `json:"architecture"`
	CaptureFile   string                    `json:"capture_file"`
	Kernels       []string                  `json:"kernels"`
	ActiveKernel  string                    `json:"active_kernel"`
	Tiers         []PipelineTier            `json:"tiers"`
	KernelTiers   map[string][]PipelineTier `json:"kernel_tiers"`
	Remarks       string                    `json:"remarks,omitempty"`
	RemarksCount  int                       `json:"remarks_count,omitempty"`
	TileIRAudit   string                    `json:"tileir_audit,omitempty"`
	TileIROpcodes map[string]int            `json:"tileir_opcodes,omitempty"`
	CFGDot        string                    `json:"cfg_dot,omitempty"`
	// Recompile carries one entry per module the NVRTC panel can recompile.
	// It is empty when the capture retained no CUDA C++ source.
	Recompile []*RecompileSource `json:"recompile,omitempty"`
}

// A RecompileSource is one module's CUDA C++ source, together with the
// artifacts the capture retained for it. The viewer recompiles the source
// under caller-chosen NVRTC options and compares the result against those
// artifacts, so a capture taken without --use_fast_math can be read beside the
// same source compiled with it.
//
// A capture holds one of these per module, because a program compiles modules
// separately and need not pass NVRTC the same options for each: ds4-min
// compiles its math module without --use_fast_math and its Q8_0 quantizer with
// it, and its probe modules target a virtual architecture rather than a
// physical one. One option list per capture would describe at most one module
// and silently misreport the rest.
type RecompileSource struct {
	ID       string   `json:"id"`
	Name     string   `json:"name"`
	File     string   `json:"file"`
	Arch     string   `json:"arch,omitempty"`
	Options  []string `json:"options,omitempty"`
	Recorded bool     `json:"recorded"`
	// Suggested is the option list the panel offers for the second side:
	// Options plus --use_fast_math.
	Suggested []string `json:"suggested,omitempty"`

	Source string `json:"-"`
	PTX    string `json:"-"`
	SASS   string `json:"-"`
}

// PipelineModule identifies one captured artifact generation.
type PipelineModule struct {
	ID   string
	Name string
	Data *PipelineViewData
}
