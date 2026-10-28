package kernelagent

import (
	"fmt"
	"math"

	"github.com/tmc/lib/nvidia/x/cuptipcsampling"
)

// DiagnosticInput binds an already collected PC sampling batch to its caller's
// incumbent, workload and target. Batch itself does not attest those identities.
// Selected lists the queried reason indices actually selected for this capture.
// No collector is started. This is caller-bound evidence, not native attestation.
type DiagnosticInput struct {
	IncumbentID, WorkloadKey, Target, Source string
	Inventory                                cuptipcsampling.ReasonInventory
	Selected                                 []uint32
	Batch                                    cuptipcsampling.Batch
}

// StallCount distinguishes an unsupported/unselected counter from measured zero.
type StallCount struct {
	Name      string
	Available bool
	Samples   uint64 `json:",omitempty"`
	Reason    string `json:",omitempty"`
}

// Summary is bounded diagnostic data. Classification is always an inference
// from stall counts, never a measured causal bottleneck or speedup percentage.
type Summary struct {
	Available                          bool
	Source, Reason, Classification     string
	Inferred                           bool
	Total, Dropped, NonUser, Remaining uint64
	Counters                           []StallCount
}

// Summarize reads at most 256 PCs, 64 inventory/selected reasons and 64 requested
// counters. Unsupported and unselected counters are absent, not measured zero.
// Partial/lossy captures retain counts and quality fields but make no inference.
func Summarize(in *DiagnosticInput, incumbent, workload, target string, wanted []string) (Summary, error) {
	if len(wanted) > 64 {
		return Summary{}, fmt.Errorf("kernelagent: too many diagnostic counters")
	}
	requestedNames := make(map[string]bool)
	for _, name := range wanted {
		if name == "" || len(name) > 256 || requestedNames[name] {
			return Summary{}, fmt.Errorf("kernelagent: invalid requested counter")
		}
		requestedNames[name] = true
	}
	out := Summary{Inferred: true, Classification: "unknown"}
	if in == nil {
		out.Reason = "PC sampling unavailable"
		for _, name := range wanted {
			out.Counters = append(out.Counters, StallCount{Name: name, Reason: out.Reason})
		}
		return out, nil
	}
	if len(in.Batch.PCs) > 256 || len(in.Inventory.Reasons) > 64 || len(in.Selected) > 64 {
		return out, fmt.Errorf("kernelagent: diagnostic input exceeds bounds")
	}
	if in.IncumbentID != incumbent || in.WorkloadKey != workload || in.Target != target || in.Source == "" || len(in.Source) > 1024 || !in.Batch.Valid || in.Batch.CUPTIVersion != in.Inventory.CUPTIVersion {
		return out, fmt.Errorf("kernelagent: stale, invalid or unattributed diagnostic input")
	}
	names := make(map[uint32]string)
	byName := make(map[string]uint32)
	for _, r := range in.Inventory.Reasons {
		if r.Name == "" || len(r.Name) > 256 || names[r.Index] != "" {
			return out, fmt.Errorf("kernelagent: invalid reason inventory")
		}
		if _, ok := byName[r.Name]; ok {
			return out, fmt.Errorf("kernelagent: duplicate reason name")
		}
		names[r.Index], byName[r.Name] = r.Name, r.Index
	}
	selected := make(map[uint32]bool)
	for _, index := range in.Selected {
		if names[index] == "" || selected[index] {
			return out, fmt.Errorf("kernelagent: invalid selected reason")
		}
		selected[index] = true
	}
	counts := make(map[uint32]uint64)
	for _, pc := range in.Batch.PCs {
		if len(pc.Reasons) > 64 {
			return out, fmt.Errorf("kernelagent: PC reason limit exceeded")
		}
		seen := make(map[uint32]bool)
		for _, count := range pc.Reasons {
			if !selected[count.Index] || seen[count.Index] || uint64(count.Samples) > math.MaxUint64-counts[count.Index] {
				return out, fmt.Errorf("kernelagent: invalid PC counts")
			}
			seen[count.Index] = true
			counts[count.Index] += uint64(count.Samples)
		}
	}
	out.Available, out.Source = true, in.Source
	out.Total, out.Dropped, out.NonUser, out.Remaining = in.Batch.TotalSamples, in.Batch.DroppedSamples, in.Batch.NonUserSamples, in.Batch.RemainingPCs
	complete := out.Dropped == 0 && out.Remaining == 0 && in.Batch.HardwareBufferFull == 0
	var largest uint64
	var dominant string
	tied := false
	requested := make(map[string]bool)
	for _, name := range wanted {
		if name == "" || len(name) > 256 || requested[name] {
			return out, fmt.Errorf("kernelagent: invalid requested counter")
		}
		requested[name] = true
		index, ok := byName[name]
		c := StallCount{Name: name}
		if !ok {
			c.Reason = "unsupported on target"
		} else if !selected[index] {
			c.Reason = "not selected for capture"
		} else {
			c.Available, c.Samples = true, counts[index]
			if c.Samples > largest {
				largest, dominant, tied = c.Samples, name, false
			} else if c.Samples == largest && largest > 0 {
				tied = true
			}
		}
		out.Counters = append(out.Counters, c)
	}
	if !complete {
		out.Reason = "partial or lossy capture"
	} else if largest > 0 && !tied {
		out.Classification = "largest requested stall count: " + dominant
	}
	return out, nil
}
