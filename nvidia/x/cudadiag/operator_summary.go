package cudadiag

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"reflect"
	"sort"

	"github.com/tmc/lib/nvidia/cupti"
	"github.com/tmc/lib/nvidia/x/cudaworkload"
	"github.com/tmc/lib/nvidia/x/nvtelemetry"
)

// OperatorPopulation counts one explicitly named observed population. Activity
// and member-edge counts have separate denominators; neither measures offered
// producer work or proves complete collection.
type OperatorPopulation struct {
	Total      uint64 `json:"total"`
	Exact      uint64 `json:"exact"`
	Associated uint64 `json:"associated"`
	Ambiguous  uint64 `json:"ambiguous"`
	Missing    uint64 `json:"missing"`
}

// OperatorEdge associates one activity with one supplied/expected member.
// Nil ActivityIndex means metadata without an observed kernel counterpart.
// Compiler metadata is at most associated; edges never allocate time/counters.
type OperatorEdge struct {
	ActivityIndex *int                          `json:"activity_index,omitempty"`
	GroupID       string                        `json:"group_id"`
	MemberID      string                        `json:"member_id"`
	Source        nvtelemetry.OperatorReference `json:"source"`
	Outcome       cudaworkload.Outcome          `json:"outcome"`
	Basis         string                        `json:"basis"`
	Reason        string                        `json:"reason"`
	ExpectedOnly  bool                          `json:"expected_only,omitempty"`
}

// OperatorMembership reports completeness relative to a separately retained
// inventory. Unknown is not an empty inventory. Agreement is not authenticity.
type OperatorMembership struct {
	GroupID    string `json:"group_id"`
	Status     string `json:"status"`
	Reason     string `json:"reason"`
	CodeStatus string `json:"code_status"`
}

// OperatorSummary retains activity findings once and compiler member edges
// separately. Context omits raw artifact payloads; ContextSHA256 identifies the
// original context file containing them. Retain that file for drilldown.
type OperatorSummary struct {
	Version                   int                         `json:"version"`
	ContextSHA256             string                      `json:"context_sha256"`
	Source                    nvtelemetry.WorkloadSource  `json:"source"`
	Context                   nvtelemetry.OperatorContext `json:"context"`
	RawArtifactSHA256         []string                    `json:"raw_artifact_sha256"`
	Activities                ActivityAnalysis            `json:"activities"`
	KernelPopulation          OperatorPopulation          `json:"observed_kernel_population"`
	MemberEdgePopulation      OperatorPopulation          `json:"member_edge_population"`
	Membership                []OperatorMembership        `json:"membership"`
	Edges                     []OperatorEdge              `json:"edges"`
	UnattributedKernelIndices []int                       `json:"unattributed_kernel_indices"`
	OrphanGroups              []string                    `json:"orphan_groups"`
	PopulationLimits          []string                    `json:"population_limits"`
}

// AnalyzeOperatorJSONL reads versioned application metadata, invokes the
// existing byte-bound activity analysis, and attaches supplied compiler members.
// Source must identify application operator-context-v1. Cancellation is checked
// between reads/records; it cannot interrupt a caller-owned blocking Reader.
// No framework SDK, compiler, device or artifact-reference path is invoked.
func AnalyzeOperatorJSONL(ctx context.Context, activities, metadata io.Reader, source nvtelemetry.WorkloadSource, limits nvtelemetry.WorkloadLimits) (OperatorSummary, error) {
	var out OperatorSummary
	if ctx == nil || activities == nil || metadata == nil {
		return out, fmt.Errorf("operator summary: missing context or reader")
	}
	if err := ctx.Err(); err != nil {
		return out, err
	}
	export, c, err := nvtelemetry.ReadOperatorContext(telemetryReader{ctx, metadata}, source, limits)
	if err != nil {
		return out, fmt.Errorf("operator summary context: %w", err)
	}
	edgeLimit := limits.MaxMetrics
	if edgeLimit == 0 {
		edgeLimit = 10000 // WorkloadLimits' documented default.
	}
	batch := c.Batch
	if batch.MaxRecords == 0 {
		batch.MaxRecords = 4096 // ActivityBatch's documented default.
	}
	if batch.MaxRecords > edgeLimit {
		batch.MaxRecords = edgeLimit
	}
	analysis, err := AnalyzeActivityJSONL(telemetryReader{ctx, activities}, batch, c.Annotations, c.Receipt)
	if err != nil {
		return out, fmt.Errorf("operator summary activities: %w", err)
	}
	if c.Batch.EvidenceRef != c.Receipt.InputSHA256 {
		return out, fmt.Errorf("operator summary: batch evidence differs from raw export")
	}
	out = OperatorSummary{Version: 1, ContextSHA256: export.RawSHA256, Source: export.Source, Context: c, Activities: analysis,
		PopulationLimits: []string{"application-supplied metadata; producer authenticity and framework/native correctness unverified", "activity identity, compiler membership and population completeness are separate", "member edges count supplied/expected slots; unknown member cardinality is not a complete producer denominator", "member edges allocate no kernel time or counters"}}
	out.Context.Artifacts = nil
	for hash := range c.Artifacts {
		out.RawArtifactSHA256 = append(out.RawArtifactSHA256, hash)
	}
	sort.Strings(out.RawArtifactSHA256)
	for _, family := range []string{c.ActivityFamily, c.CorrelationFamily, c.MetadataFamily} {
		loss, ok := c.Receipt.LossFor(family)
		if family == "" || !ok || loss.Unknown || loss.Dropped != 0 {
			out.PopulationLimits = append(out.PopulationLimits, fmt.Sprintf("unqualified population loss for family %q", family))
		}
	}
	if c.Receipt.Partial {
		out.PopulationLimits = append(out.PopulationLimits, "partial capture receipt")
	}
	if source.ClockDomain == "" || source.ClockDomain != c.Receipt.ClockDomain || source.ClockEpochID == "" || source.ClockEpochID != c.Batch.ClockEpoch || c.Receipt.ClockProvenance == "" || c.Receipt.ClockUncertaintyNS == nil {
		out.PopulationLimits = append(out.PopulationLimits, "clock mapping or uncertainty unavailable/contradictory; no operator wall-time claim")
	}
	edgeByteLimit := limits.MaxBytes
	if edgeByteLimit == 0 {
		edgeByteLimit = 4 << 20
	}
	var edgeBytes int64
	groupStatus := make([]OperatorMembership, len(c.Groups))
	groupsByOp := make(map[cudaworkload.OpID][]int)
	for i, group := range c.Groups {
		groupsByOp[group.Binding.OpID] = append(groupsByOp[group.Binding.OpID], i)
		groupStatus[i] = operatorMembership(group)
		groupStatus[i].CodeStatus = "unavailable code/module/pass metadata"
		if code := group.Binding.Code; code != nil && group.Binding.ModuleID != "" && group.Binding.ModuleGeneration != 0 && group.Binding.Pass != "" && group.Binding.Replay != "" {
			if _, ok := c.Artifacts[code.ArtifactDigest]; ok {
				groupStatus[i].CodeStatus = "declared binding and retained image; native/pass association unverified"
			}
		}
		out.Membership = append(out.Membership, groupStatus[i])
	}
	emit := func(index *int, g nvtelemetry.OperatorGroup, base cudaworkload.Outcome, reason string) error {
		if len(g.Members) > edgeLimit-len(out.Edges) {
			return fmt.Errorf("operator summary: member edge limit exceeded")
		}
		members := append([]nvtelemetry.OperatorMember(nil), g.Members...)
		seen := map[string]bool{}
		for _, m := range members {
			seen[operatorMemberKey(m)] = true
		}
		if g.Inventory != nil && separateOperatorInventory(g) {
			for _, m := range g.Inventory.Members {
				if !seen[operatorMemberKey(m)] {
					if len(members) >= edgeLimit-len(out.Edges) {
						return fmt.Errorf("operator summary: member edge limit exceeded")
					}
					members = append(members, m)
				}
			}
		}
		if len(members) > edgeLimit-len(out.Edges) {
			return fmt.Errorf("operator summary: member edge limit exceeded")
		}
		for _, m := range members {
			if err := ctx.Err(); err != nil {
				return err
			}
			edge := OperatorEdge{ActivityIndex: index, GroupID: g.ID, MemberID: m.ID, Source: m.Source, Outcome: base, Basis: "supplied_compiler_metadata", Reason: reason}
			if !seen[operatorMemberKey(m)] {
				edge.Outcome, edge.Reason, edge.ExpectedOnly = cudaworkload.OutcomeMissing, "expected member metadata missing", true
			} else if _, ok := c.Artifacts[m.Source.SHA256]; !ok {
				edge.Outcome, edge.Reason = cudaworkload.OutcomeMissing, "member raw metadata unavailable"
			} else if m.Schema == "" || g.Binding.CompilerBuild == "" {
				edge.Outcome, edge.Reason = cudaworkload.OutcomeMissing, "operator schema or compiler build unavailable"
			}
			encoded, err := json.Marshal(edge)
			if err != nil {
				return err
			}
			if int64(len(encoded)) > edgeByteLimit-edgeBytes {
				return fmt.Errorf("operator summary: member edge byte limit exceeded")
			}
			edgeBytes += int64(len(encoded))
			out.Edges = append(out.Edges, edge)
			countOperatorOutcome(&out.MemberEdgePopulation, edge.Outcome)
		}
		return nil
	}
	used := make([]bool, len(c.Groups))
	for i, join := range analysis.Joins {
		if err := ctx.Err(); err != nil {
			return OperatorSummary{}, err
		}
		if join.Record.Kind != cupti.ActivityKindKernel && join.Record.Kind != cupti.ActivityKindConcurrentKernel {
			continue
		}
		countOperatorOutcome(&out.KernelPopulation, join.Join.Outcome)
		candidates := groupsByOp[join.Join.OpID]
		if len(candidates) == 0 {
			out.UnattributedKernelIndices = append(out.UnattributedKernelIndices, i)
		}
		for _, j := range candidates {
			g := c.Groups[j]
			used[j] = true
			outcome, reason := cudaworkload.OutcomeAssociated, "supplied compiler membership; not native member correlation"
			if len(candidates) != 1 || groupStatus[j].Status == "contradictory" || !operatorActivityBinding(g.Binding, join) || join.Join.Outcome == cudaworkload.OutcomeAmbiguous {
				outcome, reason = cudaworkload.OutcomeAmbiguous, "conflicting group, execution or retained inventory binding"
			} else if join.Join.Outcome == cudaworkload.OutcomeMissing {
				outcome, reason = cudaworkload.OutcomeMissing, "submission activity correlation missing"
			}
			if err := emit(&i, g, outcome, reason); err != nil {
				return OperatorSummary{}, err
			}
		}
	}
	for i, group := range c.Groups {
		if !used[i] {
			out.OrphanGroups = append(out.OrphanGroups, group.ID)
			if err := emit(nil, group, cudaworkload.OutcomeMissing, "no correlated kernel counterpart"); err != nil {
				return OperatorSummary{}, err
			}
		}
	}
	if err := ctx.Err(); err != nil {
		return OperatorSummary{}, err
	}
	return out, nil
}

func operatorMemberKey(m nvtelemetry.OperatorMember) string { return m.Source.SHA256 + ":" + m.ID }

func separateOperatorInventory(g nvtelemetry.OperatorGroup) bool {
	if g.InventorySource == nil {
		return false
	}
	for _, m := range g.Members {
		if m.Source.SHA256 == g.InventorySource.SHA256 {
			return false
		}
	}
	return true
}

func operatorMembership(g nvtelemetry.OperatorGroup) OperatorMembership {
	out := OperatorMembership{GroupID: g.ID, Status: "unknown", Reason: g.InventoryReason}
	if g.Inventory == nil {
		return out
	}
	if !separateOperatorInventory(g) {
		out.Reason = "inventory shares supplied member metadata; not a separate expectation"
		return out
	}
	if !reflect.DeepEqual(g.Inventory.Binding, g.Binding) {
		out.Status, out.Reason = "contradictory", "retained inventory execution/compiler binding differs"
		return out
	}
	expected := map[string]bool{}
	for _, m := range g.Inventory.Members {
		expected[operatorMemberKey(m)] = true
	}
	for _, m := range g.Members {
		if !expected[operatorMemberKey(m)] {
			out.Status, out.Reason = "contradictory", "supplied member absent from retained inventory"
			return out
		}
	}
	if len(g.Members) != len(expected) {
		out.Status, out.Reason = "incomplete", "expected member metadata missing"
		return out
	}
	out.Status, out.Reason = "complete_relative_to_inventory", "member set agrees with separately retained expectation; authenticity unverified"
	return out
}

func operatorActivityBinding(b nvtelemetry.OperatorBinding, j cudaworkload.ActivityJoin) bool {
	return b.Process == j.Join.RawIDs.ProcessGenID && b.Context == j.Join.RawIDs.ContextStream && b.GraphReplay.IsZero() && b.GraphNodeID == 0 && j.Record.GraphID == 0 && j.Record.GraphNodeID == 0
}

func countOperatorOutcome(p *OperatorPopulation, outcome cudaworkload.Outcome) {
	p.Total++
	switch outcome {
	case cudaworkload.OutcomeExact:
		p.Exact++
	case cudaworkload.OutcomeAssociated:
		p.Associated++
	case cudaworkload.OutcomeAmbiguous:
		p.Ambiguous++
	default:
		p.Missing++
	}
}
