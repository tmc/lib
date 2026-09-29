package capturefiles

import (
	"encoding/json"
	"fmt"
	"path/filepath"
	"sort"
	"strings"

	"github.com/tmc/lib/nvidia/x/cudacapture"
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
// "2419 modules, 1.2 GB: 1 over per-module budget (121.0 MB), 2418 over
// total module budget (1.1 GB)". It is "" when missing is empty.
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
	noun := "modules"
	if len(missing) == 1 {
		noun = "module"
	}
	return fmt.Sprintf("%d %s, %s: %s", len(missing), noun, megabytes(total), strings.Join(parts, ", "))
}

func megabytes(n int64) string {
	return fmt.Sprintf("%.1f MB", float64(n)/1e6)
}
