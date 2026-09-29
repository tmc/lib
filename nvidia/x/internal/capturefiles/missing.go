package capturefiles

import (
	"encoding/json"
	"fmt"
	"path/filepath"
	"sort"
	"strings"

	"github.com/tmc/lib/nvidia/x/cudacapture"
	"github.com/tmc/lib/nvidia/x/cudaprof/timeline"
)

// MissingCode is a module image a capture recorded loading and did not keep
// because it did not fit a budget.
type MissingCode struct {
	Name   string // the image's file name, when it had one
	Module uint32 // the driver's id of the module, or 0
	Bytes  int64
	Reason string
}

// MissingModuleCode returns the module images cap did not keep for a budget.
// The agent drops an image over its per-module limit or past its total, and
// records it in the status's modules list with stored false; the capture drops
// a CUBIN over its own file or total budget, and records it in omitted_files.
func MissingModuleCode(cap *cudacapture.Capture) []MissingCode {
	if cap == nil {
		return nil
	}
	var status struct {
		Modules []struct {
			ID     uint32 `json:"module_id"`
			Size   int64  `json:"size"`
			File   string `json:"file"`
			Stored bool   `json:"stored"`
			Reason string `json:"reason"`
		} `json:"modules"`
		OmittedFiles []struct {
			Name   string `json:"name"`
			Bytes  int64  `json:"bytes"`
			Reason string `json:"reason"`
		} `json:"omitted_files"`
	}
	data := cap.Files["capture-status.json"]
	var wrapped struct {
		Status *json.RawMessage `json:"status"`
	}
	if json.Unmarshal(data, &wrapped) == nil && wrapped.Status != nil {
		data = *wrapped.Status
	}
	if json.Unmarshal(data, &status) != nil {
		return nil
	}
	var missing []MissingCode
	for _, m := range status.Modules {
		if !m.Stored && strings.Contains(m.Reason, "budget") {
			missing = append(missing, MissingCode{Name: m.File, Module: m.ID, Bytes: m.Size, Reason: m.Reason})
		}
	}
	for _, f := range status.OmittedFiles {
		if strings.EqualFold(filepath.Ext(f.Name), ".cubin") && strings.Contains(f.Reason, "budget") {
			missing = append(missing, MissingCode{Name: f.Name, Bytes: f.Bytes, Reason: f.Reason})
		}
	}
	return missing
}

// DescribeMissingCode summarizes missing by reason, largest group first:
// "2419 loaded modules, 321.6 MB: 2418 over total module budget (200.7 MB),
// 1 over per-module budget (121.0 MB)". It is "" when missing is empty.
//
// A loaded module is not a used one: a PyTorch process loads a module per
// compiled Triton kernel and per autotuning candidate, and most never run.
// [DescribeMissingLaunches] says which of them ran.
func DescribeMissingCode(missing []MissingCode) string {
	if len(missing) == 0 {
		return ""
	}
	type group struct {
		reason string
		n      int
		bytes  int64
	}
	byReason := make(map[string]*group)
	var groups []*group
	var total int64
	for _, m := range missing {
		g := byReason[m.Reason]
		if g == nil {
			g = &group{reason: m.Reason}
			byReason[m.Reason] = g
			groups = append(groups, g)
		}
		g.n++
		g.bytes += m.Bytes
		total += m.Bytes
	}
	sort.SliceStable(groups, func(i, j int) bool { return groups[i].n > groups[j].n })
	var parts []string
	for _, g := range groups {
		parts = append(parts, fmt.Sprintf("%d %s (%s)", g.n, g.reason, megabytes(g.bytes)))
	}
	noun := "loaded modules"
	if len(missing) == 1 {
		noun = "loaded module"
	}
	return fmt.Sprintf("%d %s, %s: %s", len(missing), noun, megabytes(total), strings.Join(parts, ", "))
}

func megabytes(n int64) string {
	return fmt.Sprintf("%.1f MB", float64(n)/1e6)
}

// ModuleLaunches counts a capture's kernel launch records by the module each
// names.
type ModuleLaunches struct {
	Total    int
	ByModule map[uint32]int

	// Unnamed counts launches whose record names no module, and Graph those
	// of them that ran as CUDA graph nodes. Such a launch may have run from
	// any module, stored or not.
	Unnamed, Graph int
}

// CountModuleLaunches decodes cap's journals and counts its kernel launches
// by module.
func CountModuleLaunches(cap *cudacapture.Capture) (*ModuleLaunches, error) {
	journals, err := timeline.Journals(cap)
	if err != nil {
		return nil, err
	}
	records, err := timeline.Decode(journals, timeline.MaxRecords, cudacapture.MaxCaptureInputBytes)
	if err != nil {
		return nil, err
	}
	l := &ModuleLaunches{ByModule: make(map[uint32]int)}
	for _, a := range records.Activities {
		if a.Subtype != "kernel" {
			continue
		}
		l.Total++
		if a.Module != 0 {
			l.ByModule[a.Module]++
			continue
		}
		l.Unnamed++
		if a.GraphID != 0 {
			l.Graph++
		}
	}
	return l, nil
}

// DescribeMissingLaunches says which of the missing modules ran captured
// kernel launches: "3 of those modules (23.3 MB) ran captured kernel
// launches; 3739 of 3904 launches (3495 in CUDA graphs) name no module and
// may have run from the others". It is "" when missing is empty or launches
// is nil.
func DescribeMissingLaunches(missing []MissingCode, launches *ModuleLaunches) string {
	if len(missing) == 0 || launches == nil || launches.Total == 0 {
		return ""
	}
	var ran int
	var bytes int64
	for _, m := range missing {
		if m.Module != 0 && launches.ByModule[m.Module] > 0 {
			ran++
			bytes += m.Bytes
		}
	}
	var s string
	switch {
	case len(missing) == 1 && ran == 0:
		s = "no launch names it"
	case len(missing) == 1:
		s = "it ran captured kernel launches"
	case ran == 0:
		s = "no launch names any of those modules"
	default:
		s = fmt.Sprintf("%d of those modules (%s) ran captured kernel launches", ran, megabytes(bytes))
	}
	if launches.Unnamed == 0 {
		return s
	}
	graph := ""
	if launches.Graph > 0 {
		graph = fmt.Sprintf(" (%d in CUDA graphs)", launches.Graph)
	}
	which := "the others"
	switch {
	case len(missing) == 1:
		which = "it"
	case ran == 0:
		which = "any of them"
	}
	return fmt.Sprintf("%s; %d of %d launches%s name no module and may have run from %s", s, launches.Unnamed, launches.Total, graph, which)
}
