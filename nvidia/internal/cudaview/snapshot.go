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
type snapshotLine struct {
	PipelineLine
	Source      int `json:"source"`
	Doc         int `json:"doc"`
	PressureDoc int `json:"pressure_doc"`
	StallDoc    int `json:"stall_doc"`
	RemarkDoc   int `json:"remark_doc"`
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
	lines := make(map[string]int)
	convert := func(tiers []PipelineTier) []snapshotTier {
		out := make([]snapshotTier, len(tiers))
		for i, t := range tiers {
			v := snapshotTier{Name: t.Name, FileName: t.FileName, Language: t.Language}
			for _, l := range t.Lines {
				n, ok := texts[l.Text]
				if !ok {
					n = len(s.Sources)
					texts[l.Text] = n
					s.Sources = append(s.Sources, l.Text)
				}
				l.Text = ""
				line := snapshotLine{PipelineLine: l, Source: n, Doc: intern(l.HoverDoc), PressureDoc: intern(l.PressureTooltip), StallDoc: intern(l.StallTooltip), RemarkDoc: intern(l.RemarkTooltip)}
				line.HoverDoc = ""
				line.PressureTooltip = ""
				line.StallTooltip = ""
				line.RemarkTooltip = ""
				key, _ := json.Marshal(line)
				index, ok := lines[string(key)]
				if !ok {
					index = len(s.Lines)
					lines[string(key)] = index
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
