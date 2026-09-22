package cudaview

import (
	"encoding/json"
	"io"
	"sort"
)

// snapshot interns source text and annotated lines. Kernel selections refer to
// the same data as full listings; no source text is repeated in an export.
type snapshot struct {
	Diagnostics  []string                  `json:"diagnostics,omitempty"`
	Modules      []moduleSnapshot          `json:"modules,omitempty"`
	Sources      []string                  `json:"sources"`
	Lines        []snapshotLine            `json:"lines"`
	Tiers        []snapshotTier            `json:"tiers"`
	KernelTiers  map[string][]snapshotTier `json:"kernel_tiers"`
	ActiveKernel string                    `json:"active_kernel"`
}

// snapshotLine is a PipelineLine as the page reads it. Every string the
// listings repeat is an index into snapshot.Sources, where index 0 is the
// empty string, so an omitted index means an empty field. It is comparable,
// which lets identical lines share one index without encoding them to find
// out.
type snapshotLine struct {
	Kernel       int     `json:"kernel_ref,omitempty"`
	Predicate    string  `json:"predicate,omitempty"`
	Operands     string  `json:"operands,omitempty"`
	Encoding     string  `json:"encoding,omitempty"`
	Num          int     `json:"num"`
	Source       int     `json:"source,omitempty"`
	Opcode       string  `json:"opcode,omitempty"`
	Category     string  `json:"category,omitempty"`
	Offset       string  `json:"offset,omitempty"`
	StallPct     float64 `json:"stall_pct,omitempty"`
	StallDoc     int     `json:"stall_doc,omitempty"`
	Doc          int     `json:"doc,omitempty"`
	RemarkBadge  string  `json:"remark_badge,omitempty"`
	RemarkDoc    int     `json:"remark_doc,omitempty"`
	LaunchKernel string  `json:"launch_kernel,omitempty"`
	SemanticTag  string  `json:"semantic_tag,omitempty"`
	SourceFile   int     `json:"file_ref,omitempty"`
	SourceLine   int     `json:"source_line,omitempty"`
	Pressure     string  `json:"pressure,omitempty"`
	PressureDoc  int     `json:"pressure_doc,omitempty"`
}
type snapshotTier struct {
	Name     string   `json:"name"`
	FileName string   `json:"file_name"`
	Language string   `json:"language"`
	Ranges   [][2]int `json:"ranges"`
}

func makeSnapshot(data *PipelineViewData) snapshot {
	s := snapshot{Diagnostics: data.Diagnostics, ActiveKernel: data.ActiveKernel, KernelTiers: make(map[string][]snapshotTier)}
	texts := make(map[string]int)
	intern := func(text string) int {
		if n, ok := texts[text]; ok {
			return n
		}
		n := len(s.Sources)
		texts[text] = n
		s.Sources = append(s.Sources, text)
		return n
	}
	intern("")
	lines := make(map[snapshotLine]int)
	convert := func(tiers []PipelineTier) []snapshotTier {
		out := make([]snapshotTier, len(tiers))
		for i, t := range tiers {
			v := snapshotTier{Name: t.Name, FileName: t.FileName, Language: t.Language}
			for _, l := range t.Lines {
				line := snapshotLine{
					Kernel:       intern(l.Kernel),
					Predicate:    l.Predicate,
					Operands:     l.Operands,
					Encoding:     l.Encoding,
					Num:          l.Num,
					Source:       intern(l.Text),
					Opcode:       l.Opcode,
					Category:     l.Category,
					Offset:       l.Offset,
					StallPct:     l.StallPct,
					StallDoc:     intern(l.StallTooltip),
					Doc:          intern(l.HoverDoc),
					RemarkBadge:  l.RemarkBadge,
					RemarkDoc:    intern(l.RemarkTooltip),
					LaunchKernel: l.LaunchKernel,
					SemanticTag:  l.SemanticTag,
					SourceFile:   intern(l.SourceFile),
					SourceLine:   l.SourceLine,
					Pressure:     l.Pressure,
					PressureDoc:  intern(l.PressureTooltip),
				}
				index, ok := lines[line]
				if !ok {
					index = len(s.Lines)
					lines[line] = index
					s.Lines = append(s.Lines, line)
				}
				if len(v.Ranges) > 0 && v.Ranges[len(v.Ranges)-1][1] == index {
					v.Ranges[len(v.Ranges)-1][1]++
				} else {
					v.Ranges = append(v.Ranges, [2]int{index, index + 1})
				}
			}
			out[i] = v
		}
		return out
	}
	s.Tiers = convert(data.Tiers)
	names := make([]string, 0, len(data.KernelTiers))
	for name := range data.KernelTiers {
		names = append(names, name)
	}
	sort.Strings(names)
	for _, name := range names {
		s.KernelTiers[name] = convert(data.KernelTiers[name])
	}

	for i, mod := range data.Modules {
		m := moduleSnapshot{ID: mod.ID, Name: mod.Name}
		if i > 0 {
			d := makeSnapshot(mod.Data)
			m.Data = &d
		}
		s.Modules = append(s.Modules, m)
	}
	return s
}

type moduleSnapshot struct {
	ID   string    `json:"id"`
	Name string    `json:"name"`
	Data *snapshot `json:"data,omitempty"`
}

// WriteSnapshot writes compact source data and indexed kernel selections.
func WriteSnapshot(w io.Writer, data *PipelineViewData) error {
	return json.NewEncoder(w).Encode(makeSnapshot(data))
}
