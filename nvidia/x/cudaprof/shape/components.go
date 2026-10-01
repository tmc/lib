package shape

import (
	"encoding/json"
	"fmt"
	"reflect"
	"sort"
	"strings"

	"github.com/tmc/lib/nvidia/x/cudaprof/marker"
	"github.com/tmc/lib/nvidia/x/cudaprof/timeline"
)

// ComponentConfiguration retains kernel configuration, including dynamic shared
// memory. Stream, context and module IDs are capture-local scheduling identities,
// not cross-capture equivalence keys. Individual repeated launches are not paired.
type ComponentConfiguration struct {
	Symbol       string `json:"raw_symbol"`
	Device       uint32 `json:"device_ordinal"`
	Grid         [3]int `json:"grid"`
	Block        [3]int `json:"block"`
	SharedMemory uint32 `json:"shared_memory"`
}

// ComponentRoot retains submissions observed under one explicit root scope.
// Declared bank identity alone cannot produce a root here. Scope and Activities
// index the retained ranges and activities in the associated arm. Configuration
// map keys are complete JSON-encoded ComponentConfiguration values.
type ComponentRoot struct {
	Operation string `json:"operation"`

	Ordinal        int            `json:"ordinal"`
	Scope          int            `json:"source_range"`
	Kernels        int            `json:"kernels"`
	Configurations map[string]int `json:"configuration_multiset"`
	Activities     []int          `json:"source_activities"`
}

// ComponentBatch separates lifecycle observations from observed root coverage.
// Wall is application lifecycle wall on one journal clock, not GPU completion.
// StartSource and EndSource index retained instantaneous markers; -1 is missing.
// Nil counters or Wall indicate unavailable metrics. SubmissionKernels counts
// exact batch correlations, independently of the per-root Kernels count.
type ComponentBatch struct {
	SubmissionKernels    int   `json:"batch_submission_correlated_kernels"`
	SubmissionActivities []int `json:"batch_submission_source_activities"`

	Journal          string                   `json:"journal"`
	Process          uint32                   `json:"process"`
	RequestID        string                   `json:"request_id"`
	Role             string                   `json:"role"`
	Pair             *int                     `json:"request_pair"`
	Invocation       int                      `json:"invocation"`
	Ordinal          int                      `json:"batch"`
	Contract         marker.ComponentContract `json:"declared_contract"`
	RootCount        int                      `json:"declared_root_count"`
	StartCount       int                      `json:"batch_start_count"`
	EndCount         int                      `json:"batch_end_count"`
	StartSource      int                      `json:"start_source_marker"`
	EndSource        int                      `json:"end_source_marker"`
	Start            int64                    `json:"start_ns"`
	End              int64                    `json:"end_ns"`
	Wall             *int64                   `json:"application_wall_ns"`
	Before           *int                     `json:"successful_batches_before"`
	After            *int                     `json:"successful_batches_after"`
	SubmissionScopes []int                    `json:"batch_submission_ranges"`
	Roots            []ComponentRoot          `json:"observed_roots"`
	Kernels          int                      `json:"kernels"`
	Failures         []string                 `json:"failures,omitempty"`
}

// ComponentArm preserves counts even when observed execution cannot qualify.
// ExplicitIterations counts valid application lifecycle intervals; it does not
// qualify root attribution or stationarity.
type ComponentArm struct {
	MalformedMarkers int `json:"malformed_component_markers"`

	Batches             []ComponentBatch `json:"batches"`
	ExplicitIterations  int              `json:"explicit_iterations"`
	Kernels             int              `json:"retained_kernels"`
	UnattributedKernels int              `json:"unattributed_kernels"`
	MalformedScopes     int              `json:"malformed_component_scopes"`
	UnpairedMarkers     int              `json:"unpaired_markers"`
	Failures            []string         `json:"failures,omitempty"`
}

// ComponentReport compares declared contracts and exactly observed per-root
// configuration multisets. Pass does not establish correctness, stationarity,
// capture completeness, native completion, isolation or causal timing.
type ComponentReport struct {
	Version        int               `json:"version"`
	Pass           bool              `json:"pass"`
	RequestPairs   []RequestPair     `json:"request_pairs"`
	Arms           [2]ComponentArm   `json:"arms"`
	MatchedRoots   int               `json:"matched_roots"`
	Failures       []string          `json:"failures,omitempty"`
	Qualifications map[string]string `json:"independent_qualifications"`
	Limits         []string          `json:"limits"`
}

type componentKey struct {
	journal           string
	process           uint32
	id, role          string
	invocation, batch int
}

// CompareComponents compares explicitly paired component executions. It uses
// exact submission correlations, never construction proximity or GPU timestamps.
func CompareComponents(base, target timeline.Records, domain string, pairs []RequestPair) ComponentReport {
	r := ComponentReport{Version: 1, RequestPairs: append([]RequestPair(nil), pairs...), Qualifications: map[string]string{"collector": "caller must independently verify", "correctness": "unverified", "stationarity": "unassessed", "native_completion": "unknown", "isolation": "unchecked"}, Limits: []string{"Bank and root-set hashes declare source inputs; they do not prove execution or outputs.", "Repeated kernel configurations compare as exact per-root multisets; individual occurrence pairing and independent timing samples are unavailable.", "Application batch completion and successful counters do not establish native GPU completion.", "Stream, context and module IDs remain capture-local; geometry equality alone cannot establish application equivalence."}}
	mappings := [2]map[requestKey]int{{}, {}}
	if domain != "driver" && domain != "runtime" {
		r.Failures = append(r.Failures, "unsupported submission domain")
		return r
	}
	if len(pairs) == 0 {
		r.Failures = append(r.Failures, "explicit request mappings required")
	}
	for i, p := range pairs {
		b, t := requestKey{p.BaseJournal, p.BaseProcess, p.BaseID, p.Role}, requestKey{p.TargetJournal, p.TargetProcess, p.TargetID, p.Role}
		_, bd := mappings[0][b]
		_, td := mappings[1][t]
		if b.journal == "" || t.journal == "" || b.process == 0 || t.process == 0 || b.id == "" || t.id == "" || bd || td || (p.Role != "measured" && p.Role != "warmup" && p.Role != "control") {
			r.Failures = append(r.Failures, "invalid or duplicate request mapping")
			return r
		}
		mappings[0][b], mappings[1][t] = i, i
	}
	r.Arms[0] = componentArm(base, domain, mappings[0])
	r.Arms[1] = componentArm(target, domain, mappings[1])
	keys := [2]map[string]*ComponentBatch{{}, {}}
	for arm := range r.Arms {
		for i := range r.Arms[arm].Batches {
			b := &r.Arms[arm].Batches[i]
			if b.Pair == nil {
				continue
			}
			key := fmt.Sprintf("%d/%d/%d", *b.Pair, b.Invocation, b.Ordinal)
			keys[arm][key] = b
		}
	}
	for i := range pairs {
		for arm := range r.Arms {
			seen := false
			for _, b := range r.Arms[arm].Batches {
				if b.Pair != nil && *b.Pair == i && len(b.Failures) == 0 {
					seen = true
				}
			}
			if !seen {
				r.Failures = append(r.Failures, "declared request mapping lacks qualified batches in both arms")
			}
		}
	}
	for key, b := range keys[0] {
		t := keys[1][key]
		if t == nil {
			r.Failures = append(r.Failures, "unmatched explicit batch")
			continue
		}
		if !reflect.DeepEqual(b.Contract, t.Contract) || b.RootCount != t.RootCount || len(b.Failures) > 0 || len(t.Failures) > 0 {
			r.Failures = append(r.Failures, "component contract or root coverage differs")
			continue
		}
		tm := map[int]ComponentRoot{}
		for _, root := range t.Roots {
			tm[root.Ordinal] = root
		}
		for _, root := range b.Roots {
			other, ok := tm[root.Ordinal]
			if !ok || root.Operation != other.Operation || !reflect.DeepEqual(root.Configurations, other.Configurations) {
				r.Failures = append(r.Failures, "per-root configuration multiset differs")
			} else {
				r.MatchedRoots++
			}
		}
	}
	if len(keys[0]) != len(keys[1]) {
		r.Failures = append(r.Failures, "explicit batch sets differ")
	}
	for _, arm := range r.Arms {
		if len(arm.Failures) > 0 {
			r.Failures = append(r.Failures, arm.Failures...)
		}
		for _, b := range arm.Batches {
			if len(b.Failures) > 0 {
				r.Failures = append(r.Failures, "invalid explicit batch")
			}
		}
	}
	r.Failures = componentUnique(r.Failures)
	r.Pass = len(r.Failures) == 0 && r.MatchedRoots > 0 && len(keys[0]) > 0
	return r
}

func componentArm(records timeline.Records, domain string, mappings map[requestKey]int) ComponentArm {
	arm := ComponentArm{UnpairedMarkers: records.UnpairedMarkers}
	batches := map[componentKey]*ComponentBatch{}
	roots := map[componentKey]map[int]*ComponentRoot{}
	get := func(journal string, pid uint32, m marker.Metadata) *ComponentBatch {
		key := componentKey{journal, pid, m.RequestID, m.Role, *m.Invocation, *m.Batch}
		b := batches[key]
		if b == nil {
			b = &ComponentBatch{Journal: journal, Process: pid, RequestID: m.RequestID, Role: m.Role, Invocation: *m.Invocation, Ordinal: *m.Batch, Contract: *m.Component, RootCount: *m.RootCount, StartSource: -1, EndSource: -1}
			batches[key] = b
			roots[key] = map[int]*ComponentRoot{}
			if pair, ok := mappings[requestKey{journal, pid, m.RequestID, m.Role}]; ok {
				b.Pair = &pair
			} else {
				b.Failures = append(b.Failures, "explicit request mapping unavailable")
			}
		}
		if !reflect.DeepEqual(b.Contract, *m.Component) || b.RootCount != *m.RootCount {
			b.Failures = append(b.Failures, "inconsistent batch contract")
		}
		return b
	}
	for i, e := range records.Markers {
		m, err := marker.ParseRange(e.Name)
		if err != nil && strings.Contains(e.Name, "cudaprof:v3 ") {
			arm.MalformedMarkers++
		}
		if err != nil || m.Version != 3 || e.Flags&timeline.MarkerInstantaneous == 0 {
			continue
		}
		b := get(e.Journal, e.Process, m)
		if m.Event == "batch_start" {
			b.StartCount++
			b.Start = e.Time
			b.StartSource = i
			b.Before = m.Executions
		}
		if m.Event == "batch_end" {
			b.EndCount++
			b.End = e.Time
			b.EndSource = i
			b.After = m.Executions
		}
	}
	for i, scope := range records.Ranges {
		m, err := marker.ParseRange(scope.Name)
		if err != nil {
			if strings.Contains(scope.Name, "cudaprof:v3 ") {
				arm.MalformedScopes++
			}
			continue
		}
		if m.Version != 3 {
			continue
		}
		b := get(scope.Journal, scope.Process, m)
		if m.Event != "" || scope.Start < 0 || scope.End <= scope.Start || scope.Process == 0 || scope.Thread == 0 || scope.Journal == "" {
			b.Failures = append(b.Failures, "invalid component submission range")
			continue
		}
		if (m.Phase != "submit" && m.Phase != "evaluate") || m.Boundary != "submission" {
			continue
		}
		if m.Executions == nil {
			b.Failures = append(b.Failures, "successful submission counter missing")
		}
		if m.Root == nil {
			b.SubmissionScopes = append(b.SubmissionScopes, i)
			continue
		}
		key := componentKey{scope.Journal, scope.Process, m.RequestID, m.Role, *m.Invocation, *m.Batch}
		if _, dup := roots[key][*m.Root]; dup {
			b.Failures = append(b.Failures, "duplicate observed root scope")
			continue
		}
		roots[key][*m.Root] = &ComponentRoot{Ordinal: *m.Root, Operation: m.Operation, Scope: i, Configurations: map[string]int{}}
	}
	links := timeline.Correlations(records, domain)
	for i, a := range records.Activities {
		if a.Subtype != "" && a.Subtype != "kernel" {
			continue
		}
		arm.Kernels++
		link := links[i]
		m := link.Metadata
		if link.APIStatus != "matched" || link.MetadataStatus != "matched" || m == nil || m.Version != 3 || m.Boundary != "submission" || (m.Phase != "submit" && m.Phase != "evaluate") || a.GraphID != 0 {
			arm.UnattributedKernels++
			continue
		}
		call := records.Calls[link.Call]
		b := get(a.Journal, call.Process, *m)
		b.SubmissionKernels++
		b.SubmissionActivities = append(b.SubmissionActivities, i)
		if m.Root == nil {
			arm.UnattributedKernels++
			continue
		}
		key := componentKey{a.Journal, call.Process, m.RequestID, m.Role, *m.Invocation, *m.Batch}
		root := roots[key][*m.Root]
		if root == nil || root.Scope != link.Range {
			arm.UnattributedKernels++
			continue
		}
		valid := a.End > a.Start && a.Start >= 0 && a.SharedMemory != nil && a.Name != ""
		for _, n := range a.Grid {
			valid = valid && n > 0
		}
		for _, n := range a.Block {
			valid = valid && n > 0
		}
		if !valid {
			b.Failures = append(b.Failures, "launch duration, geometry or shared memory unavailable")
			continue
		}
		config := ComponentConfiguration{a.Name, a.Device, a.Grid, a.Block, *a.SharedMemory}
		data, _ := json.Marshal(config)
		root.Configurations[string(data)]++
		root.Kernels++
		root.Activities = append(root.Activities, i)
		b.Kernels++
	}
	for key, b := range batches {
		if b.StartCount != 1 || b.EndCount != 1 || b.Start < 0 || b.End <= b.Start || b.Before == nil || b.After == nil || *b.After-*b.Before != 1 {
			b.Failures = append(b.Failures, "batch lifecycle or successful counter unavailable")
		} else {
			wall := b.End - b.Start
			b.Wall = &wall
			arm.ExplicitIterations++
		}
		if len(b.SubmissionScopes) != 1 {
			b.Failures = append(b.Failures, "batch requires one explicit submission scope")
		}
		for _, scopeIndex := range b.SubmissionScopes {
			m, _ := marker.ParseRange(records.Ranges[scopeIndex].Name)
			if m.Executions == nil || b.Before == nil || *m.Executions != *b.Before {
				b.Failures = append(b.Failures, "submission counter differs from batch start")
			}
		}
		if len(roots[key]) != b.RootCount {
			b.Failures = append(b.Failures, "observed root coverage incomplete")
		}
		for _, root := range roots[key] {
			m, _ := marker.ParseRange(records.Ranges[root.Scope].Name)
			if m.Executions == nil || b.Before == nil || *m.Executions != *b.Before {
				b.Failures = append(b.Failures, "root counter differs from batch start")
			}
			if root.Kernels == 0 {
				b.Failures = append(b.Failures, "declared root has no observed kernels")
			}
			if len(b.SubmissionScopes) == 1 {
				batchScope := records.Ranges[b.SubmissionScopes[0]]
				rootScope := records.Ranges[root.Scope]
				if batchScope.Start < b.Start || batchScope.End > b.End || rootScope.Start < batchScope.Start || rootScope.End > batchScope.End {
					b.Failures = append(b.Failures, "submission scopes outside explicit batch lifecycle")
				}
			}
			b.Roots = append(b.Roots, *root)
		}
		sort.Slice(b.Roots, func(i, j int) bool { return b.Roots[i].Ordinal < b.Roots[j].Ordinal })
		b.Failures = componentUnique(b.Failures)
		arm.Batches = append(arm.Batches, *b)
	}
	sort.Slice(arm.Batches, func(i, j int) bool {
		a, b := arm.Batches[i], arm.Batches[j]
		return fmt.Sprintf("%s/%010d/%s/%s/%020d/%020d", a.Journal, a.Process, a.RequestID, a.Role, a.Invocation, a.Ordinal) < fmt.Sprintf("%s/%010d/%s/%s/%020d/%020d", b.Journal, b.Process, b.RequestID, b.Role, b.Invocation, b.Ordinal)
	})
	if arm.Kernels == 0 {
		arm.Failures = append(arm.Failures, "no retained component kernels")
	}
	if arm.UnattributedKernels > 0 {
		arm.Failures = append(arm.Failures, "kernel root attribution unavailable")
	}
	if arm.MalformedScopes > 0 || arm.MalformedMarkers > 0 || arm.UnpairedMarkers > 0 || len(records.Graphs) > 0 {
		arm.Failures = append(arm.Failures, "malformed/unpaired component scopes or unsupported graph replay")
	}
	// Validate cumulative counters across batches of each capture-local invocation.
	previous := map[string]*ComponentBatch{}
	for i := range arm.Batches {
		b := &arm.Batches[i]
		group := fmt.Sprintf("%q/%d/%q/%q/%d", b.Journal, b.Process, b.RequestID, b.Role, b.Invocation)
		if p := previous[group]; p != nil && p.After != nil && b.Before != nil && (*b.Before != *p.After || b.Start < p.End || b.Ordinal != p.Ordinal+1) {
			b.Failures = componentUnique(append(b.Failures, "nonconsecutive batches, lifecycle overlap or counter discontinuity"))
		}
		previous[group] = b
	}
	return arm
}
func componentUnique(ss []string) []string {
	seen := map[string]bool{}
	out := []string{}
	for _, s := range ss {
		if !seen[s] {
			seen[s] = true
			out = append(out, s)
		}
	}
	sort.Strings(out)
	return out
}
