package main

import (
	"bytes"
	"encoding/json"
	"io"
	"maps"
	"reflect"
	"slices"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/internal/cudainject/wire"
	"github.com/tmc/lib/nvidia/x/cudacapture"
)

// rec is one device record of a synthetic compare capture.
type rec struct {
	subtype, name string
	start, end    uint64
	stream        uint32
	bytes         uint64
	copyKind      uint8
	graphID       uint32
	config        *LaunchConfig // nil: the record carries no configuration
}

func compareJournal(recs []rec) []byte {
	b := newJournalBuilder()
	for i, r := range recs {
		payload := map[string]any{
			"kind": "activity", "subtype": r.subtype, "name": r.name, "valid": true,
			"start": r.start, "end": r.end, "stream": r.stream, "bytes": r.bytes,
			"copy_kind": r.copyKind, "graph_id": r.graphID, "correlation": i + 1,
		}
		if c := r.config; c != nil {
			payload["grid_x"], payload["grid_y"], payload["grid_z"] = c.Grid[0], c.Grid[1], c.Grid[2]
			payload["block_x"], payload["block_y"], payload["block_z"] = c.Block[0], c.Block[1], c.Block[2]
			payload["dynamic_shared_memory"] = c.SharedMemory
		}
		b.append(wire.KindActivity, payload)
	}
	return b.finalize()
}

// phaseStatus is a complete status whose phase receipt is [start, stop).
func phaseStatus(start, stop uint64) map[string]any {
	s := completeStatus()
	s["phase_receipt"] = map[string]any{"status": "MEASURED", "api": "cuProfilerStart", "start_timestamp": start, "stop_timestamp": stop}
	return s
}

func compareCapture(recs []rec, status map[string]any) *cudacapture.Capture {
	return testCapture(map[string][]byte{"capture.journal": compareJournal(recs)}, status)
}

func mustRules(t *testing.T, text string) opRules {
	t.Helper()
	rules, err := parseOpRules(strings.NewReader(text))
	if err != nil {
		t.Fatal(err)
	}
	return rules
}

func TestCompareSidePartition(t *testing.T) {
	tests := []struct {
		name   string
		recs   []rec
		status map[string]any
		want   CompareSide
		ops    map[string]opTotals
	}{
		{
			// Two kernels overlap across streams: the overlap goes to the
			// one that started first, so exclusive sums to the union while
			// summed counts the overlap twice.
			name: "overlap across streams",
			recs: []rec{
				{subtype: "kernel", name: "a", start: 1100, end: 1500, stream: 1},
				{subtype: "kernel", name: "b", start: 1300, end: 1700, stream: 2},
			},
			status: phaseStatus(1000, 2000),
			want: CompareSide{Window: "phase", WindowNanos: 1000, LeadInNanos: 100, TailNanos: 300,
				BusyNanos: 600, IdleNanos: 400, KernelNanos: 600, KernelCount: 2},
			ops: map[string]opTotals{"a": {1, 400, 400}, "b": {1, 200, 400}},
		},
		{
			// A kernel outranks a memcpy, which outranks a memset.
			name: "memcpy overlapping a kernel",
			recs: []rec{
				{subtype: "kernel", name: "k", start: 1100, end: 1400},
				{subtype: "memcpy", start: 1300, end: 1600, bytes: 4096, copyKind: 1},
				{subtype: "memset", start: 1550, end: 1650},
			},
			status: phaseStatus(1000, 2000),
			want: CompareSide{Window: "phase", WindowNanos: 1000, LeadInNanos: 100, TailNanos: 350,
				BusyNanos: 550, IdleNanos: 450, KernelNanos: 300, MemcpyNanos: 200, MemsetNanos: 50,
				KernelCount: 1, MemcpyCount: 1, MemcpyBytes: 4096, MemcpyKinds: map[string]int{"Htod": 1}, MemsetCount: 1},
			ops: map[string]opTotals{"k": {1, 300, 300}},
		},
		{
			// A later kernel of higher class takes an instant from an
			// earlier memcpy; a later kernel of equal class does not.
			name: "later kernel outranks earlier memcpy",
			recs: []rec{
				{subtype: "memcpy", start: 1000, end: 1400},
				{subtype: "kernel", name: "k", start: 1200, end: 1300},
			},
			status: phaseStatus(1000, 1400),
			want: CompareSide{Window: "phase", WindowNanos: 400, BusyNanos: 400,
				KernelNanos: 100, MemcpyNanos: 300, KernelCount: 1, MemcpyCount: 1, MemcpyKinds: map[string]int{"Unknown": 1}},
			ops: map[string]opTotals{"k": {1, 100, 100}},
		},
		{
			name: "record straddling a phase edge is outside",
			recs: []rec{
				{subtype: "kernel", name: "edge", start: 900, end: 1100},
				{subtype: "kernel", name: "in", start: 1200, end: 1300},
				{subtype: "kernel", name: "after", start: 3000, end: 3100},
			},
			status: phaseStatus(1000, 2000),
			want: CompareSide{Window: "phase", WindowNanos: 1000, LeadInNanos: 200, TailNanos: 700,
				BusyNanos: 100, IdleNanos: 900, KernelNanos: 100, KernelCount: 1, EdgeStraddlers: 1},
			ops: map[string]opTotals{"in": {1, 100, 100}},
		},
		{
			// The envelope is clipped to the window and only the part no
			// resolved record covers is unresolved.
			name: "graph envelope is unresolved, not idle",
			recs: []rec{
				{subtype: "graph_trace", start: 800, end: 1500, graphID: 7},
				{subtype: "kernel", name: "node", start: 1100, end: 1200, graphID: 7},
			},
			status: phaseStatus(1000, 2000),
			want: CompareSide{Window: "phase", WindowNanos: 1000, LeadInNanos: 100, TailNanos: 800,
				BusyNanos: 100, IdleNanos: 500, UnresolvedNanos: 400, KernelNanos: 100, KernelCount: 1},
			ops: map[string]opTotals{"node": {1, 100, 100}},
		},
		{
			name: "no receipt falls back to the capture span",
			recs: []rec{
				{subtype: "kernel", name: "a", start: 100, end: 200},
				{subtype: "kernel", name: "a", start: 300, end: 400},
			},
			status: completeStatus(),
			want: CompareSide{Window: "capture span", PhaseIndex: -1, WindowNanos: 300,
				BusyNanos: 200, IdleNanos: 100, KernelNanos: 200, KernelCount: 2},
			ops: map[string]opTotals{"a": {2, 200, 200}},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := measureCompareSide(compareCapture(tt.recs, tt.status), nil, nil, compareOptions{phase: -1, allowUnresolved: true})
			if err != nil {
				t.Fatal(err)
			}
			ops := make(map[string]opTotals)
			for k, v := range got.ops {
				ops[k] = *v
			}
			if !maps.Equal(ops, tt.ops) {
				t.Errorf("ops = %v, want %v", ops, tt.ops)
			}
			// The records carry no launch configuration.
			if got.UnrecordedConfigs != got.KernelCount {
				t.Errorf("unrecorded configs = %d, want every kernel, %d", got.UnrecordedConfigs, got.KernelCount)
			}
			got.ops, got.launches, got.Clocks, got.UnrecordedConfigs = nil, nil, nil, 0
			got.WindowNote, got.StartTimestamp, got.StopTimestamp = "", 0, 0
			if tt.want.Window == "phase" {
				tt.want.PhaseIndex = 0
			}
			if !reflect.DeepEqual(*got, tt.want) {
				t.Errorf("side =\n%+v\nwant\n%+v", *got, tt.want)
			}
		})
	}
}

func TestCompareRefuses(t *testing.T) {
	good := []rec{{subtype: "kernel", name: "k", start: 1100, end: 1200}}
	with := func(edit func(map[string]any)) map[string]any {
		s := phaseStatus(1000, 2000)
		edit(s)
		return s
	}
	provenance := func(device, cc string) map[string][]byte {
		data, _ := json.Marshal(map[string]string{"device": device, "compute_capability": cc})
		return map[string][]byte{"capture.journal": compareJournal(good), "capture-provenance.json": data}
	}
	twoPhases := with(func(s map[string]any) {
		s["phase_receipt"] = map[string]any{"status": "UNREAD", "reason": "multiple_phases", "phases": []any{
			map[string]any{"status": "MEASURED", "start_timestamp": 1000, "stop_timestamp": 2000},
			map[string]any{"status": "MEASURED", "start_timestamp": 3000, "stop_timestamp": 4000},
		}}
	})
	tests := []struct {
		name         string
		base, target *cudacapture.Capture
		opts         compareOptions
		want         string
	}{
		{"partial run", compareCapture(good, with(func(s map[string]any) { s["run_completeness"] = "partial" })), nil, compareOptions{phase: -1}, `run_completeness is "partial"`},
		{"target died", compareCapture(good, with(func(s map[string]any) { s["target_signal"] = "segmentation fault" })), nil, compareOptions{phase: -1}, "target died by SIGSEGV"},
		// A capture written before target-failed existed still says
		// capture_success true; its exit status is what gives it away.
		{"target exited nonzero", compareCapture(good, with(func(s map[string]any) { s["target_exit_code"] = 2 })), nil, compareOptions{phase: -1}, "target exited with status 2"},
		{"dropped records", compareCapture(good, with(func(s map[string]any) { s["dropped_records"] = 3 })), nil, compareOptions{phase: -1}, "dropped 3 record(s)"},
		{"record loss", compareCapture(good, with(func(s map[string]any) {
			s["record_loss"] = map[string]any{"offered_records": 10, "retained_records": 9}
		})), nil, compareOptions{phase: -1}, "10 offered, 9 retained"},
		{"unresolved", compareCapture(append([]rec{{subtype: "graph_trace", start: 1000, end: 1500}}, good...), phaseStatus(1000, 2000)), nil, compareOptions{phase: -1}, "-allow-unresolved"},
		{"device mismatch", testCapture(provenance("A", "12.1"), phaseStatus(1000, 2000)), testCapture(provenance("B", "12.1"), phaseStatus(1000, 2000)), compareOptions{phase: -1}, "different devices"},
		{"capability mismatch", testCapture(provenance("A", "12.1"), phaseStatus(1000, 2000)), testCapture(provenance("A", "9.0"), phaseStatus(1000, 2000)), compareOptions{phase: -1}, "different compute capabilities"},
		{"phase out of range", compareCapture(good, phaseStatus(1000, 2000)), nil, compareOptions{phase: 1}, "-phase 1 out of range"},
		{"phase without receipt", compareCapture(good, completeStatus()), nil, compareOptions{phase: 0}, "no qualified phase"},
		{"phases need -phase", compareCapture(good, twoPhases), nil, compareOptions{phase: -1}, "select one with -phase"},
		{"phases out of range", compareCapture(good, twoPhases), nil, compareOptions{phase: 2}, "-phase 2 out of range: capture has 2 phases"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			target := tt.target
			if target == nil {
				target = compareCapture(good, phaseStatus(1000, 2000))
			}
			c, err := compareCaptures(tt.base, target, nil, nil, tt.opts)
			if err == nil || !strings.Contains(err.Error(), tt.want) {
				t.Fatalf("err = %v, want it to contain %q", err, tt.want)
			}
			if c != nil {
				t.Errorf("refusal returned a comparison")
			}
		})
	}
}

func TestComparePhaseSelectsWindow(t *testing.T) {
	s := phaseStatus(0, 0)
	s["phase_receipt"] = map[string]any{"status": "UNREAD", "reason": "multiple_phases", "phases": []any{
		map[string]any{"status": "MEASURED", "start_timestamp": 1000, "stop_timestamp": 2000},
		map[string]any{"status": "MEASURED", "start_timestamp": 3000, "stop_timestamp": 5000},
	}}
	cap := compareCapture([]rec{
		{subtype: "kernel", name: "first", start: 1100, end: 1200},
		{subtype: "kernel", name: "second", start: 3500, end: 3700},
	}, s)
	side, err := measureCompareSide(cap, nil, nil, compareOptions{phase: 1})
	if err != nil {
		t.Fatal(err)
	}
	if side.PhaseIndex != 1 || side.WindowNanos != 2000 || side.BusyNanos != 200 || side.ops["second"] == nil || side.ops["first"] != nil {
		t.Errorf("phase 1: index %d window %d busy %d ops %v", side.PhaseIndex, side.WindowNanos, side.BusyNanos, side.ops)
	}
}

// TestCompareMaxRecords checks that -max-records bounds the journal decode:
// a capture written with a larger capture --max-records is refused under a
// smaller limit and read under its own.
func TestCompareMaxRecords(t *testing.T) {
	cap := compareCapture([]rec{
		{subtype: "kernel", name: "a", start: 1100, end: 1200},
		{subtype: "kernel", name: "b", start: 1300, end: 1400},
		{subtype: "kernel", name: "c", start: 1500, end: 1600},
	}, phaseStatus(1000, 2000))
	if _, err := measureCompareSide(cap, nil, nil, compareOptions{phase: -1, maxRecords: 1}); err == nil || !strings.Contains(err.Error(), "raise -max-records past 1") {
		t.Errorf("maxRecords 1: err = %v, want a limit error", err)
	}
	side, err := measureCompareSide(cap, nil, nil, compareOptions{phase: -1, maxRecords: 1 << 23})
	if err != nil {
		t.Fatal(err)
	}
	if side.BusyNanos != 300 {
		t.Errorf("maxRecords 1<<23: busy %d, want 300", side.BusyNanos)
	}
}

func TestCompareAllowUnresolved(t *testing.T) {
	base := compareCapture([]rec{{subtype: "kernel", name: "k", start: 1100, end: 1200}}, phaseStatus(1000, 2000))
	target := compareCapture([]rec{
		{subtype: "graph_trace", start: 1000, end: 1800, graphID: 1},
		{subtype: "kernel", name: "k", start: 1100, end: 1300, graphID: 1},
	}, phaseStatus(1000, 2000))
	c, err := compareCaptures(base, target, nil, nil, compareOptions{phase: -1, allowUnresolved: true})
	if err != nil {
		t.Fatal(err)
	}
	if c.OpStatus != "UNREAD" || c.Ops != nil {
		t.Errorf("op status %q with %d rows, want UNREAD and none", c.OpStatus, len(c.Ops))
	}
	if u := c.term("unresolved"); u.Base != 0 || u.Target != 600 {
		t.Errorf("unresolved = %+v, want 0 -> 600", u)
	}
	var out bytes.Buffer
	if err := c.write(&out); err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{"per-op: UNREAD", "unresolved", "Δkernel"} {
		if !strings.Contains(out.String(), want) {
			t.Errorf("output lacks %q:\n%s", want, out.String())
		}
	}
}

func TestParseOpRules(t *testing.T) {
	tests := []struct {
		name    string
		text    string
		want    []string // op of each rule
		wantErr string
	}{
		{"comments and blanks", "# c\n\n  gateup\tgate\n^go_rms$   rms norm\n", []string{"gate", "rms norm"}, ""},
		{"missing op", "gateup\n", nil, "line 1"},
		{"bad regex", "\nga(te op\n", nil, "line 2"},
		{"reserved op", "x (unmatched)\n", nil, "reserved"},
		{"quoted pattern", "\"rms norm\" rms\n`a\\s+b c`\tab\n", []string{"rms", "ab"}, ""},
		{"quoted no op", "\"rms norm\"\n", nil, "line 1"},
		{"quoted no space", "\"rms\"op\n", nil, "line 1"},
		{"bad quote", "\"rms op\n", nil, "line 1"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			rules, err := parseOpRules(strings.NewReader(tt.text))
			if tt.wantErr != "" {
				if err == nil || !strings.Contains(err.Error(), tt.wantErr) {
					t.Fatalf("err = %v, want %q", err, tt.wantErr)
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			var ops []string
			for _, r := range rules {
				ops = append(ops, r.Op)
			}
			if !reflect.DeepEqual(ops, tt.want) {
				t.Errorf("ops = %q, want %q", ops, tt.want)
			}
		})
	}
}

func TestKernelOpFirstMatch(t *testing.T) {
	rules := mustRules(t, "gateup first\ngate second\n^hc_expand_kernel$ expand\n")
	tests := []struct {
		kernel string
		op     string
		rule   int
	}{
		{"go_gateup_iq2", "first", 0},
		{"gate_only", "second", 1},
		{"_Z16hc_expand_kernelPfPKfS1_S1_S1_S1_S1_jjjjjii", "expand", 2}, // matched demangled
		{"other", unmatchedOp, -1},
	}
	for _, tt := range tests {
		op, rule := rules.kernelOp(opKey(tt.kernel, false))
		if op != tt.op || rule != tt.rule {
			t.Errorf("kernelOp(%q) = %q, %d; want %q, %d", tt.kernel, op, rule, tt.op, tt.rule)
		}
	}
	if op, _ := opRules(nil).kernelOp(opKey("_Z16hc_expand_kernelPfPKfS1_S1_S1_S1_S1_jjjjjii", false)); op != "hc_expand_kernel" {
		t.Errorf("without rules op = %q, want the demangled name", op)
	}
}

func TestOpKey(t *testing.T) {
	const fixup8 = "_Z24mul_mat_q_stream_k_fixupIL9ggml_type8ELi128ELb0EEvPKiS2_PfS3_5uint3iiiS4_iS4_iS4_"
	tests := []struct {
		name      string
		templates bool
		want      string
	}{
		{fixup8, false, "mul_mat_q_stream_k_fixup"},
		{fixup8, true, "mul_mat_q_stream_k_fixup<(ggml_type)8, 128, false>"},
		{"_Z16hc_expand_kernelPfPKfS1_S1_S1_S1_S1_jjjjjii", true, "hc_expand_kernel"},
		{"foo<int>(int*)", false, "foo"},
		{"foo<int>(int*)", true, "foo<int>"},
		{"go_rms", true, "go_rms"},
	}
	for _, tt := range tests {
		if got := opKey(tt.name, tt.templates); got != tt.want {
			t.Errorf("opKey(%q, %v) = %q, want %q", tt.name, tt.templates, got, tt.want)
		}
	}
}

// TestCompareTemplatesSplitsInstantiations checks that -templates gives each
// instantiation of a template kernel its own op, where the default joins them.
func TestCompareTemplatesSplitsInstantiations(t *testing.T) {
	const (
		q8 = "_Z24mul_mat_q_stream_k_fixupIL9ggml_type8ELi128ELb0EEvPKiS2_PfS3_5uint3iiiS4_iS4_iS4_"
		q2 = "_Z24mul_mat_q_stream_k_fixupIL9ggml_type2ELi128ELb0EEvPKiS2_PfS3_5uint3iiiS4_iS4_iS4_"
	)
	recs := []rec{
		{subtype: "kernel", name: q8, start: 1100, end: 1200},
		{subtype: "kernel", name: q2, start: 1300, end: 1600},
	}
	for _, tt := range []struct {
		templates bool
		want      []string
	}{
		{false, []string{"mul_mat_q_stream_k_fixup"}},
		{true, []string{"mul_mat_q_stream_k_fixup<(ggml_type)2, 128, false>", "mul_mat_q_stream_k_fixup<(ggml_type)8, 128, false>"}},
	} {
		c, err := compareCaptures(compareCapture(recs, phaseStatus(1000, 2000)), compareCapture(recs, phaseStatus(1000, 2000)), nil, nil, compareOptions{phase: -1, templates: tt.templates})
		if err != nil {
			t.Fatal(err)
		}
		var ops []string
		for _, o := range c.Ops {
			ops = append(ops, o.Op)
		}
		slices.Sort(ops)
		if !reflect.DeepEqual(ops, tt.want) {
			t.Errorf("templates=%v: ops = %q, want %q", tt.templates, ops, tt.want)
		}
	}
}

// TestCompareJoinsOpsAcrossNames compares two captures whose kernels differ
// in name and checks the ops join, the unmatched row, the unused rule report
// and the identity as printed.
func TestCompareJoinsOpsAcrossNames(t *testing.T) {
	rules := mustRules(t, "gateup gateup\nrms rms\n^go_repack_ repack\nnever_matches ghost\n")
	base := compareCapture([]rec{
		{subtype: "kernel", name: "_ZN12_GLOBAL__N_117gateup_d2r_kernelEv", start: 1100, end: 1400},
		{subtype: "kernel", name: "rms_norm_plain", start: 1400, end: 1500},
		{subtype: "kernel", name: "mystery", start: 1600, end: 1650},
		{subtype: "memset", start: 1700, end: 1720},
	}, phaseStatus(1000, 2000))
	target := compareCapture([]rec{
		{subtype: "kernel", name: "go_repack_q2", start: 1000, end: 1200},
		{subtype: "kernel", name: "go_gateup", start: 1200, end: 1600, stream: 1},
		{subtype: "kernel", name: "go_rms", start: 1500, end: 1700, stream: 2},
		{subtype: "memcpy", start: 1650, end: 1800, bytes: 64, copyKind: 2},
	}, phaseStatus(1000, 2500))
	c, err := compareCaptures(base, target, rules, nil, compareOptions{phase: -1})
	if err != nil {
		t.Fatal(err)
	}
	rows := make(map[string]CompareOp)
	for _, o := range c.Ops {
		rows[o.Op] = o
	}
	checks := []struct {
		op                   string
		baseExcl, targetExcl int64
		targetSummed         int64
	}{
		{"gateup", 300, 400, 400},
		{"rms", 100, 100, 200}, // 1500-1600 belongs to gateup, which started first
		{"repack", 0, 200, 200},
		{unmatchedOp, 50, 0, 0},
	}
	for _, ck := range checks {
		r := rows[ck.op]
		if r.BaseExclusiveNanos != ck.baseExcl || r.TargetExclusiveNanos != ck.targetExcl || r.TargetSummedNanos != ck.targetSummed {
			t.Errorf("%s = %+v, want exclusive %d -> %d, target summed %d", ck.op, r, ck.baseExcl, ck.targetExcl, ck.targetSummed)
		}
	}
	if c.Ops[0].Op != "repack" {
		t.Errorf("first row %q, want repack (largest |Δ exclusive|)", c.Ops[0].Op)
	}
	for _, s := range []*CompareSide{c.Base, c.Target} {
		if len(s.UnusedRules) == 0 || s.UnusedRules[len(s.UnusedRules)-1].Op != "ghost" {
			t.Errorf("unused rules %v, want ghost reported", s.UnusedRules)
		}
	}
	if len(c.Base.UnusedRules) != 2 || c.Base.UnusedRules[0].Op != "repack" {
		t.Errorf("base unused rules %v, want repack and ghost", c.Base.UnusedRules)
	}
	// Window 1000 -> 1500; memcpy 0 -> 100 (1700-1800); memset 20 -> 0.
	if w, idle := c.term("window"), c.term("idle"); w.Delta != 500 || idle.Base != 530 || idle.Target != 700 {
		t.Errorf("window %+v idle %+v", w, idle)
	}
	var out bytes.Buffer
	if err := c.write(&out); err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{
		"+500 = +300 -50 +100 -20 +170 +0 (exact)",
		`ops line 4 "never_matches" (ghost) matched no kernel in base`,
		`ops line 3 "^go_repack_" (repack) matched no kernel in base`,
	} {
		if !strings.Contains(out.String(), want) {
			t.Errorf("output lacks %q:\n%s", want, out.String())
		}
	}
}

// TestCheckIdentityRejectsABrokenSum corrupts a valid comparison one term at
// a time. Each must be caught, or a comparison that does not add up could be
// printed.
func TestCheckIdentityRejectsABrokenSum(t *testing.T) {
	build := func() *Comparison {
		base := compareCapture([]rec{{subtype: "kernel", name: "a", start: 1100, end: 1200}}, phaseStatus(1000, 2000))
		target := compareCapture([]rec{
			{subtype: "kernel", name: "a", start: 1100, end: 1300},
			{subtype: "memcpy", start: 1300, end: 1350},
		}, phaseStatus(1000, 3000))
		c, err := compareCaptures(base, target, nil, nil, compareOptions{phase: -1})
		if err != nil {
			t.Fatal(err)
		}
		return c
	}
	if err := checkIdentity(build()); err != nil {
		t.Fatalf("valid comparison: %v", err)
	}
	tests := []struct {
		name    string
		corrupt func(*Comparison)
	}{
		{"idle", func(c *Comparison) { c.Target.IdleNanos++ }},
		{"memcpy", func(c *Comparison) { c.Target.MemcpyNanos-- }},
		{"window", func(c *Comparison) { c.Base.WindowNanos++ }},
		{"op", func(c *Comparison) { c.Base.ops["a"].exclusive++ }},
		{"row dropped", func(c *Comparison) { c.Ops = c.Ops[1:] }},
		{"delta", func(c *Comparison) { c.Terms[0].Delta++ }},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			c := build()
			tt.corrupt(c)
			if err := checkIdentity(c); err == nil {
				t.Errorf("identity held after corrupting %s", tt.name)
			}
		})
	}
}

func TestParseCompareArgs(t *testing.T) {
	tests := []struct {
		name     string
		args     []string
		want     compareOptions
		paths    []string
		wantErr  string
		wantHelp bool
	}{
		{"defaults", []string{"a.pprof", "b.pprof"}, compareOptions{phase: -1, maxRecords: controllerMaxJournalRecords, journalBytes: cudacapture.MaxCaptureInputBytes}, []string{"a.pprof", "b.pprof"}, "", false},
		{"all flags", []string{"-phase", "2", "-ops", "c-go.ops", "-json", "-allow-unresolved", "a", "b"},
			compareOptions{phase: 2, opsPath: "c-go.ops", json: true, allowUnresolved: true, maxRecords: controllerMaxJournalRecords, journalBytes: cudacapture.MaxCaptureInputBytes}, []string{"a", "b"}, "", false},
		{"one capture", []string{"a.pprof"}, compareOptions{}, nil, "usage: cudaprof compare", false},
		{"unknown flag", []string{"-bogus", "a", "b"}, compareOptions{}, nil, "-bogus", false},
		{"bad phase", []string{"-phase", "x", "a", "b"}, compareOptions{}, nil, "invalid value", false},
		{"stage entries", []string{"-stages", "s.txt", "-stage-entries", "a", "b"}, compareOptions{phase: -1, stagesPath: "s.txt", stageEntries: true, maxRecords: controllerMaxJournalRecords, journalBytes: cudacapture.MaxCaptureInputBytes}, []string{"a", "b"}, "", false},
		{"max records", []string{"-max-records", "4194304", "a", "b"}, compareOptions{phase: -1, maxRecords: 4194304, journalBytes: cudacapture.MaxCaptureInputBytes}, []string{"a", "b"}, "", false},
		{"read limits", []string{"--capture-file-bytes=1073741824", "--capture-total-bytes=2147483648", "a", "b"},
			compareOptions{phase: -1, maxRecords: controllerMaxJournalRecords, journalBytes: cudacapture.MaxCaptureInputBytes, limits: cudacapture.Limits{FileBytes: 1 << 30, TotalBytes: 2 << 30, InputBytes: 2 << 30}}, []string{"a", "b"}, "", false},
		{"bad read limit", []string{"--capture-file-bytes=0", "a", "b"}, compareOptions{}, nil, "invalid --capture-file-bytes", false},
		{"journal bytes", []string{"-journal-bytes", "1073741824", "a", "b"}, compareOptions{phase: -1, maxRecords: controllerMaxJournalRecords, journalBytes: 1 << 30}, []string{"a", "b"}, "", false},
		{"zero journal bytes", []string{"-journal-bytes", "0", "a", "b"}, compareOptions{}, nil, "invalid -journal-bytes 0", false},
		{"zero max records", []string{"-max-records", "0", "a", "b"}, compareOptions{}, nil, "invalid -max-records 0", false},
		{"stage entries without stages", []string{"-stage-entries", "a", "b"}, compareOptions{}, nil, "-stage-entries needs -stages", false},
		{"help", []string{"-h"}, compareOptions{}, nil, "", true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var out bytes.Buffer
			opts, paths, err := parseCompareArgs(tt.args, &out)
			if tt.wantErr != "" {
				if err == nil || !strings.Contains(err.Error(), tt.wantErr) {
					t.Fatalf("err = %v, want %q", err, tt.wantErr)
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			if tt.wantHelp {
				if paths != nil || !strings.HasPrefix(out.String(), compareUsage) || !strings.Contains(out.String(), "-allow-unresolved") {
					t.Errorf("help: paths %v, output %q", paths, out.String())
				}
				return
			}
			if opts != tt.want || !reflect.DeepEqual(paths, tt.paths) {
				t.Errorf("got %+v %q, want %+v %q", opts, paths, tt.want, tt.paths)
			}
		})
	}
}

func TestCompareJSON(t *testing.T) {
	cap := compareCapture([]rec{{subtype: "kernel", name: "k", start: 1100, end: 1200}}, phaseStatus(1000, 2000))
	c, err := compareCaptures(cap, cap, nil, nil, compareOptions{phase: -1})
	if err != nil {
		t.Fatal(err)
	}
	data, err := json.Marshal(c)
	if err != nil {
		t.Fatal(err)
	}
	var m map[string]any
	if err := json.Unmarshal(data, &m); err != nil {
		t.Fatal(err)
	}
	base := m["base"].(map[string]any)
	if base["window_ns"] != 1000.0 || base["busy_ns"] != 100.0 || base["idle_ns"] != 900.0 || m["op_status"] != "MEASURED" {
		t.Errorf("json = %s", data)
	}
	if ops := m["ops"].([]any); ops[0].(map[string]any)["base_exclusive_ns"] != 100.0 {
		t.Errorf("ops = %v", ops)
	}
	var stamped struct {
		Cudaprof *buildStamp `json:"cudaprof"`
	}
	if err := json.Unmarshal(data, &stamped); err != nil {
		t.Fatal(err)
	}
	if stamped.Cudaprof == nil || *stamped.Cudaprof != currentBuild() {
		t.Errorf("cudaprof = %+v, want %+v", stamped.Cudaprof, currentBuild())
	}
}

// TestCompareGraphTraceMode checks that the mode comes from the capability
// record and is UNREAD, not assumed, when a capture predates the key.
func TestCompareGraphTraceMode(t *testing.T) {
	recs := []rec{{subtype: "kernel", name: "k", start: 1100, end: 1200}}
	tests := []struct {
		name       string
		capability map[string]any
		want       string
	}{
		{"node", map[string]any{"capture_range": "cudaProfilerApi", "graph_trace_mode": "node"}, "node"},
		{"graph", map[string]any{"capture_range": "cudaProfilerApi", "graph_trace_mode": "graph"}, "graph"},
		{"absent", map[string]any{"capture_range": "cudaProfilerApi"}, "UNREAD"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			status := phaseStatus(1000, 2000)
			status["capability"] = tt.capability
			side, err := measureCompareSide(compareCapture(recs, status), nil, nil, compareOptions{phase: -1})
			if err != nil {
				t.Fatal(err)
			}
			if got := side.graphTraceModeString(); got != tt.want {
				t.Errorf("graph trace mode = %q, want %q", got, tt.want)
			}
		})
	}
}

// TestCompareNotesUncollectedMemsets checks that a side whose agent did not
// collect memsets says so, rather than reporting its memset time as zero.
func TestCompareNotesUncollectedMemsets(t *testing.T) {
	recs := []rec{{subtype: "kernel", name: "k", start: 1100, end: 1200}}
	const note = "memsets not collected"
	tests := []struct {
		name       string
		capability map[string]any
		want       bool
	}{
		{"off", map[string]any{"memsets": false}, true},
		{"on", map[string]any{"memsets": true}, false},
		{"absent", map[string]any{}, false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			status := phaseStatus(1000, 2000)
			status["capability"] = tt.capability
			c, err := compareCaptures(compareCapture(recs, phaseStatus(1000, 2000)), compareCapture(recs, status), nil, nil, compareOptions{phase: -1})
			if err != nil {
				t.Fatal(err)
			}
			notes := strings.Join(c.Notes, "\n")
			if got := strings.Contains(notes, "target: "+note); got != tt.want {
				t.Errorf("notes %q: target %q = %v, want %v", c.Notes, note, got, tt.want)
			}
			if strings.Contains(notes, "base: "+note) {
				t.Errorf("notes %q: base has no capability record and must not be reported", c.Notes)
			}
		})
	}
}

// sequence returns back-to-back kernels of the given names and durations,
// starting at 1000.
func sequence(kernels ...any) []rec {
	var recs []rec
	at := uint64(1000)
	for i := 0; i < len(kernels); i += 2 {
		d := uint64(kernels[i+1].(int))
		recs = append(recs, rec{subtype: "kernel", name: kernels[i].(string), start: at, end: at + d})
		at += d
	}
	return recs
}

// TestCompareStages checks that stages join work the two sides split
// differently, that a repeated anchor names different stages by position,
// and that a misaligned boundary is reported rather than silently absorbed.
func TestCompareStages(t *testing.T) {
	// Two layers. Base fuses the attention output into one kernel; target
	// runs it as two. expand runs twice per layer on both sides.
	const stages = "expand pre\nattn attn\n. out\nexpand ffn\n"
	base := sequence(
		"embed", 5,
		"expand", 10, "c_norm", 20, "c_attn", 30, "c_out_fused", 40, "expand", 10, "c_ffn", 50,
		"expand", 10, "c_norm", 20, "c_attn", 30, "c_out_fused", 40, "expand", 10, "c_ffn", 50,
	)
	target := sequence(
		"embed", 5,
		"expand", 10, "go_norm", 20, "go_attn", 35, "go_out_a", 15, "go_out_b", 15, "expand", 10, "go_ffn", 60,
		"expand", 10, "go_norm", 20, "go_attn", 35, "go_out_a", 15, "go_out_b", 15, "expand", 10, "go_ffn", 60,
	)
	tests := []struct {
		name   string
		target []rec
		want   map[string][3]int64 // stage: base entries, base exclusive, target exclusive
		note   string
	}{
		{
			name:   "aligned",
			target: target,
			want: map[string][3]int64{
				stageUnstarted: {1, 5, 5},
				"pre":          {2, 60, 60},
				"attn":         {2, 60, 70},
				"out":          {2, 80, 60},
				"ffn":          {2, 120, 140},
			},
		},
		{
			// The second layer's attention is missing, so the machine keeps
			// waiting for it and charges the rest of the capture to pre.
			name:   "misaligned",
			target: append(append([]rec(nil), target[:8]...), target[9:]...),
			note:   "stage attn began 2 times in base and 1 in target",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			b := compareCapture(base, phaseStatus(1000, 1500))
			g := compareCapture(tt.target, phaseStatus(1000, 1500))
			c, err := compareCaptures(b, g, nil, stageBoundaries(mustRules(t, stages)), compareOptions{phase: -1})
			if err != nil {
				t.Fatal(err)
			}
			if tt.note != "" {
				if !strings.Contains(strings.Join(c.Notes, "\n"), tt.note) {
					t.Errorf("notes %q, want %q", c.Notes, tt.note)
				}
				return
			}
			for _, n := range c.Notes {
				if strings.Contains(n, "did not align") {
					t.Errorf("aligned stages reported as misaligned: %s", n)
				}
			}
			if len(c.Stages) != len(tt.want) {
				t.Fatalf("stages = %+v, want %d rows", c.Stages, len(tt.want))
			}
			for _, st := range c.Stages {
				w, ok := tt.want[st.Stage]
				if !ok {
					t.Errorf("unexpected stage %q", st.Stage)
					continue
				}
				if int64(st.BaseEntries) != w[0] || st.TargetEntries != st.BaseEntries || st.BaseExclusiveNanos != w[1] || st.TargetExclusiveNanos != w[2] {
					t.Errorf("%s = %+v, want entries %d, exclusive %d -> %d", st.Stage, st, w[0], w[1], w[2])
				}
			}
			if c.Stages[0].Stage != stageUnstarted {
				t.Errorf("first row %q, want %q", c.Stages[0].Stage, stageUnstarted)
			}
			var out bytes.Buffer
			if err := c.write(&out); err != nil {
				t.Fatal(err)
			}
			if !strings.Contains(out.String(), "per stage") {
				t.Errorf("text output has no stage table:\n%s", out.String())
			}
		})
	}
}

// TestCompareStageEntries checks that -stage-entries reports each layer on
// its own row, that a stage's entries sum to its per-stage row, and that
// entry counts that differ are refused, not paired out of step.
func TestCompareStageEntries(t *testing.T) {
	const stages = "expand pre\nattn attn\nffn ffn\n"
	base := sequence(
		"expand", 10, "attn", 30, "ffn", 50,
		"expand", 10, "attn", 30, "ffn", 50,
		"expand", 10, "attn", 30, "ffn", 50,
	)
	// Target's attention slows with depth; its prologue is one kernel base
	// does not run.
	target := sequence(
		"embed", 5,
		"expand", 10, "attn", 35, "ffn", 50,
		"expand", 10, "attn", 45, "ffn", 50,
		"expand", 10, "attn", 55, "ffn", 40,
	)
	type row struct {
		stage        string
		entry        int
		base, target int64
	}
	tests := []struct {
		name    string
		target  []rec
		want    []row
		wantErr string
	}{
		{
			name:   "three layers",
			target: target,
			want: []row{
				{stageUnstarted, 0, 0, 5},
				{"pre", 0, 10, 10}, {"attn", 0, 30, 35}, {"ffn", 0, 50, 50},
				{"pre", 1, 10, 10}, {"attn", 1, 30, 45}, {"ffn", 1, 50, 50},
				{"pre", 2, 10, 10}, {"attn", 2, 30, 55}, {"ffn", 2, 50, 40},
			},
		},
		{
			name:    "a layer short",
			target:  target[:7],
			wantErr: "stage pre began 3 times in base and 2 in target",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			b := compareCapture(base, phaseStatus(1000, 1500))
			g := compareCapture(tt.target, phaseStatus(1000, 1500))
			c, err := compareCaptures(b, g, nil, stageBoundaries(mustRules(t, stages)), compareOptions{phase: -1, stageEntries: true})
			if tt.wantErr != "" {
				if err == nil || !strings.Contains(err.Error(), tt.wantErr) {
					t.Fatalf("err = %v, want %q", err, tt.wantErr)
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			var got []row
			sums := make(map[string][2]int64)
			for _, e := range c.StageEntries {
				got = append(got, row{e.Stage, e.Entry, e.BaseExclusiveNanos, e.TargetExclusiveNanos})
				s := sums[e.Stage]
				sums[e.Stage] = [2]int64{s[0] + e.BaseExclusiveNanos, s[1] + e.TargetExclusiveNanos}
				if e.DeltaExclusiveNanos != e.TargetExclusiveNanos-e.BaseExclusiveNanos {
					t.Errorf("%s %d delta %d, want %d", e.Stage, e.Entry, e.DeltaExclusiveNanos, e.TargetExclusiveNanos-e.BaseExclusiveNanos)
				}
			}
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("entries =\n%v\nwant\n%v", got, tt.want)
			}
			for _, st := range c.Stages {
				if s := sums[st.Stage]; s != [2]int64{st.BaseExclusiveNanos, st.TargetExclusiveNanos} {
					t.Errorf("%s entries sum to %v, per-stage row is %d, %d", st.Stage, s, st.BaseExclusiveNanos, st.TargetExclusiveNanos)
				}
			}
			var out bytes.Buffer
			if err := c.write(&out); err != nil {
				t.Fatal(err)
			}
			hasRow := slices.ContainsFunc(strings.Split(out.String(), "\n"), func(line string) bool {
				return strings.Join(strings.Fields(line), " ") == "1 1 30 55 +25 30 55 2 attn"
			})
			if !strings.Contains(out.String(), "per stage entry") || !hasRow {
				t.Errorf("text output lacks the entry table:\n%s", out.String())
			}
		})
	}
}

// TestCompareOpsByStage checks that with -ops and -stages each stage row and
// each entry row carries its ops, that the cells sum exactly to both
// margins, and that checkIdentity catches a cell that breaks either.
func TestCompareOpsByStage(t *testing.T) {
	const stages = "expand pre\nattn attn\n"
	// Stage attn holds attn and ffn; the fused kernel on the target side
	// is an op of its own. embed is unmatched and before the first boundary.
	base := sequence(
		"expand", 10, "attn", 30, "ffn", 50,
		"expand", 10, "attn", 30, "ffn", 50,
	)
	target := sequence(
		"embed", 5,
		"expand", 10, "attn", 35, "ffn", 50,
		"expand", 10, "attn_ffn_fused", 70,
	)
	rules := mustRules(t, "fused fused\n^attn attn\n^ffn ffn\n^expand expand\n")
	build := func(t *testing.T) *Comparison {
		c, err := compareCaptures(compareCapture(base, phaseStatus(1000, 1500)), compareCapture(target, phaseStatus(1000, 1500)),
			rules, stageBoundaries(mustRules(t, stages)), compareOptions{phase: -1, stageEntries: true})
		if err != nil {
			t.Fatal(err)
		}
		return c
	}
	c := build(t)

	// cell is an op's base and target exclusive ns within a row.
	type cell struct{ base, target int64 }
	cells := func(ops []CompareOp) map[string]cell {
		m := make(map[string]cell)
		for _, o := range ops {
			m[o.Op] = cell{o.BaseExclusiveNanos, o.TargetExclusiveNanos}
		}
		return m
	}
	stageWant := map[string]map[string]cell{
		stageUnstarted: {unmatchedOp: {0, 5}},
		"pre":          {"expand": {20, 20}},
		"attn":         {"attn": {60, 35}, "ffn": {100, 50}, "fused": {0, 70}},
	}
	for _, st := range c.Stages {
		if got := cells(st.Ops); !maps.Equal(got, stageWant[st.Stage]) {
			t.Errorf("stage %s ops = %v, want %v", st.Stage, got, stageWant[st.Stage])
		}
	}
	entryWant := []map[string]cell{
		{unmatchedOp: {0, 5}},
		{"expand": {10, 10}},
		{"attn": {30, 35}, "ffn": {50, 50}},
		{"expand": {10, 10}},
		{"attn": {30, 0}, "ffn": {50, 0}, "fused": {0, 70}},
	}
	if len(c.StageEntries) != len(entryWant) {
		t.Fatalf("got %d entries, want %d", len(c.StageEntries), len(entryWant))
	}
	for i, e := range c.StageEntries {
		if got := cells(e.Ops); !maps.Equal(got, entryWant[i]) {
			t.Errorf("%s entry %d ops = %v, want %v", e.Stage, e.Entry, got, entryWant[i])
		}
	}

	// The margins, summed here independently of checkCells.
	for _, rows := range [][]cellRow{
		func() (rs []cellRow) {
			for _, st := range c.Stages {
				rs = append(rs, cellRow{st.Stage, st.BaseExclusiveNanos, st.TargetExclusiveNanos, st.Ops})
			}
			return rs
		}(),
		func() (rs []cellRow) {
			for _, e := range c.StageEntries {
				rs = append(rs, cellRow{e.Stage, e.BaseExclusiveNanos, e.TargetExclusiveNanos, e.Ops})
			}
			return rs
		}(),
	} {
		byOp := make(map[string]cell)
		for _, r := range rows {
			var sum cell
			for _, o := range r.ops {
				sum.base += o.BaseExclusiveNanos
				sum.target += o.TargetExclusiveNanos
				s := byOp[o.Op]
				byOp[o.Op] = cell{s.base + o.BaseExclusiveNanos, s.target + o.TargetExclusiveNanos}
			}
			if sum != (cell{r.base, r.target}) {
				t.Errorf("row %s: cells sum to %v, row is %d, %d", r.name, sum, r.base, r.target)
			}
		}
		if want := cells(c.Ops); !maps.Equal(byOp, want) {
			t.Errorf("cells by op = %v, op rows %v", byOp, want)
		}
	}

	tests := []struct {
		name    string
		corrupt func(*Comparison)
	}{
		{"stage cell", func(c *Comparison) { c.Stages[2].Ops[0].TargetExclusiveNanos++ }},
		{"stage cell dropped", func(c *Comparison) { c.Stages[2].Ops = c.Stages[2].Ops[1:] }},
		{"entry cell", func(c *Comparison) { c.StageEntries[4].Ops[0].BaseExclusiveNanos-- }},
		{"moved between ops", func(c *Comparison) {
			// The row still sums; the op margins do not.
			ops := c.Stages[2].Ops
			ops[0].TargetExclusiveNanos += 5
			ops[1].TargetExclusiveNanos -= 5
		}},
		{"moved between entries", func(c *Comparison) {
			// attn still sums over the entries; the rows do not.
			attn := func(ops []CompareOp) *CompareOp {
				return &ops[slices.IndexFunc(ops, func(o CompareOp) bool { return o.Op == "attn" })]
			}
			attn(c.StageEntries[2].Ops).BaseExclusiveNanos += 5
			attn(c.StageEntries[4].Ops).BaseExclusiveNanos -= 5
		}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			c := build(t)
			tt.corrupt(c)
			if err := checkIdentity(c); err == nil {
				t.Errorf("identity held after corrupting %s", tt.name)
			}
		})
	}
}

func TestCompareWarmupDropsPrologue(t *testing.T) {
	// A prologue -- two weight loads and a copy -- then the anchor kernel
	// and the work, inside one phase [1000, 10000).
	cap := compareCapture([]rec{
		{subtype: "kernel", name: "load", start: 1100, end: 1300},
		{subtype: "kernel", name: "load", start: 1400, end: 1500},
		{subtype: "memcpy", start: 1600, end: 1700, bytes: 64},
		{subtype: "kernel", name: "anchor", start: 2000, end: 2100},
		{subtype: "kernel", name: "work", start: 2200, end: 2500},
	}, phaseStatus(1000, 10000))
	stages := stageBoundaries(mustRules(t, "anchor layer\n"))
	tests := []struct {
		name    string
		warmup  *gapOptions
		window  int64
		cut     int64
		kernel  int64
		ops     []string
		stages  []string
		wantErr string
	}{
		{name: "none", window: 9000, kernel: 700, ops: []string{"anchor", "load", "work"}, stages: []string{stageUnstarted, "layer"}},
		{name: "until", warmup: &gapOptions{warmupUntil: "^anch"}, window: 8000, cut: 1000, kernel: 400, ops: []string{"anchor", "work"}, stages: []string{"layer"}},
		{name: "ms", warmup: &gapOptions{warmupMS: 0.0005}, window: 8000, cut: 1000, kernel: 400, ops: []string{"anchor", "work"}, stages: []string{"layer"}},
		{name: "unmatched", warmup: &gapOptions{warmupUntil: "no_such_kernel"}, wantErr: "warmup"},
		{name: "longer than the window", warmup: &gapOptions{warmupMS: 1}, wantErr: "never ended"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			side, err := measureCompareSide(cap, nil, stages, compareOptions{phase: -1, warmup: tt.warmup})
			if tt.wantErr != "" {
				if err == nil || !strings.Contains(err.Error(), tt.wantErr) {
					t.Fatalf("err = %v, want %q", err, tt.wantErr)
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			if side.WindowNanos != tt.window || side.WarmupNanos != tt.cut || side.KernelNanos != tt.kernel {
				t.Errorf("window %d cut %d kernel %d, want %d %d %d", side.WindowNanos, side.WarmupNanos, side.KernelNanos, tt.window, tt.cut, tt.kernel)
			}
			if ops := slices.Sorted(maps.Keys(side.ops)); !reflect.DeepEqual(ops, tt.ops) {
				t.Errorf("ops = %q, want %q", ops, tt.ops)
			}
			if st := slices.Sorted(maps.Keys(side.stages)); !reflect.DeepEqual(st, tt.stages) {
				t.Errorf("stages = %q, want %q", st, tt.stages)
			}
			wantCopies := 1
			if tt.warmup != nil {
				wantCopies = 0
			}
			if side.MemcpyCount != wantCopies {
				t.Errorf("memcpy count = %d, want %d", side.MemcpyCount, wantCopies)
			}
		})
	}
}

func TestParseCompareWarmup(t *testing.T) {
	tests := []struct {
		name    string
		args    []string
		want    *gapOptions
		wantErr string
	}{
		{"none", nil, nil, ""},
		{"until", []string{"-warmup-until=hc_expand"}, &gapOptions{warmupUntil: "hc_expand"}, ""},
		{"ms zero", []string{"-warmup-ms=0"}, &gapOptions{}, ""},
		{"both", []string{"-warmup-ms=1", "-warmup-until=k"}, nil, "exclusive"},
		{"bad pattern", []string{"-warmup-until=("}, nil, "warmup-until"},
		{"negative", []string{"-warmup-ms=-1"}, nil, "invalid"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			opts, _, err := parseCompareArgs(append(tt.args, "a.pprof", "b.pprof"), io.Discard)
			if tt.wantErr != "" {
				if err == nil || !strings.Contains(err.Error(), tt.wantErr) {
					t.Fatalf("err = %v, want %q", err, tt.wantErr)
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(opts.warmup, tt.want) {
				t.Errorf("warmup = %+v, want %+v", opts.warmup, tt.want)
			}
		})
	}
}

// TestCompareLaunchConfigs checks that an op's launch configurations are
// compared as sets, that a change is flagged with both sides' sets, and that
// a launch whose record carries no configuration is never read as SAME.
func TestCompareLaunchConfigs(t *testing.T) {
	attn := &LaunchConfig{Grid: [3]uint32{512, 8, 1}, Block: [3]uint32{512, 1, 1}, SharedMemory: 88576}
	narrow := &LaunchConfig{Grid: [3]uint32{256, 8, 1}, Block: [3]uint32{512, 1, 1}, SharedMemory: 88576}
	launches := func(cfgs ...*LaunchConfig) []rec {
		var recs []rec
		for i, c := range cfgs {
			at := uint64(1100 + 100*i)
			recs = append(recs, rec{subtype: "kernel", name: "attn", start: at, end: at + 50, config: c})
		}
		return recs
	}
	tests := []struct {
		name          string
		base, target  []rec
		status        string
		baseConfigs   []LaunchConfigCount
		targetConfigs []LaunchConfigCount
		text, noText  []string
		wantNote      string
	}{
		{
			name:        "identical",
			base:        launches(attn, attn),
			target:      launches(attn, attn),
			status:      configSame,
			baseConfigs: []LaunchConfigCount{{*attn, 2}}, targetConfigs: []LaunchConfigCount{{*attn, 2}},
			text:   []string{"1 ops SAME, 0 CHANGED"},
			noText: []string{"CHANGED attn"},
		},
		{
			// Sets, not counts: a third launch of the same shape is a
			// launch count matter.
			name:        "same set, other count",
			base:        launches(attn, attn),
			target:      launches(attn, attn, attn),
			status:      configSame,
			baseConfigs: []LaunchConfigCount{{*attn, 2}}, targetConfigs: []LaunchConfigCount{{*attn, 3}},
		},
		{
			name:        "grid differs",
			base:        launches(attn, attn),
			target:      launches(narrow, attn),
			status:      configChanged,
			baseConfigs: []LaunchConfigCount{{*attn, 2}}, targetConfigs: []LaunchConfigCount{{*narrow, 1}, {*attn, 1}},
			text: []string{
				"1 CHANGED",
				"CHANGED attn\n  base    grid 512x8x1 block 512x1x1 smem 88576  launches 2\n  target  grid 256x8x1 block 512x1x1 smem 88576  launches 1\n  target  grid 512x8x1 block 512x1x1 smem 88576  launches 1\n",
			},
		},
		{
			name:          "not recorded on one side",
			base:          launches(nil, nil),
			target:        launches(attn, attn),
			status:        configUnread,
			targetConfigs: []LaunchConfigCount{{*attn, 2}},
			text:          []string{"0 ops SAME, 0 CHANGED, 1 UNREAD (not recorded)"},
			noText:        []string{"1 ops SAME"},
			wantNote:      "base: launch configuration (grid, block, dynamic shared memory) not recorded",
		},
		{
			name:        "not recorded for some launches",
			base:        launches(attn, nil),
			target:      launches(attn, attn),
			status:      configUnread,
			baseConfigs: []LaunchConfigCount{{*attn, 1}}, targetConfigs: []LaunchConfigCount{{*attn, 2}},
			wantNote: "base: launch configuration not recorded for 1 of 2 kernel launches",
		},
		{
			name:        "one side",
			base:        launches(attn),
			target:      []rec{{subtype: "kernel", name: "other", start: 1100, end: 1150, config: attn}},
			status:      configOneSided,
			baseConfigs: []LaunchConfigCount{{*attn, 1}},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			c, err := compareCaptures(compareCapture(tt.base, phaseStatus(1000, 2000)), compareCapture(tt.target, phaseStatus(1000, 2000)), nil, nil, compareOptions{phase: -1})
			if err != nil {
				t.Fatal(err)
			}
			var o CompareOp
			for _, row := range c.Ops {
				if row.Op == "attn" {
					o = row
				}
			}
			if o.ConfigStatus != tt.status {
				t.Errorf("config status = %q, want %q", o.ConfigStatus, tt.status)
			}
			if !reflect.DeepEqual(o.BaseConfigs, tt.baseConfigs) && len(o.BaseConfigs)+len(tt.baseConfigs) > 0 {
				t.Errorf("base configs = %v, want %v", o.BaseConfigs, tt.baseConfigs)
			}
			if !reflect.DeepEqual(o.TargetConfigs, tt.targetConfigs) && len(o.TargetConfigs)+len(tt.targetConfigs) > 0 {
				t.Errorf("target configs = %v, want %v", o.TargetConfigs, tt.targetConfigs)
			}
			notes := strings.Join(c.Notes, "\n")
			if tt.wantNote != "" && !strings.Contains(notes, tt.wantNote) {
				t.Errorf("notes %q, want %q", notes, tt.wantNote)
			}
			if tt.wantNote == "" && strings.Contains(notes, "not recorded") {
				t.Errorf("notes %q report a configuration every record carries as not recorded", notes)
			}
			var out bytes.Buffer
			if err := c.write(&out); err != nil {
				t.Fatal(err)
			}
			for _, want := range tt.text {
				if !strings.Contains(squash(out.String()), squash(want)) {
					t.Errorf("output lacks %q:\n%s", want, out.String())
				}
			}
			for _, bad := range tt.noText {
				if strings.Contains(squash(out.String()), squash(bad)) {
					t.Errorf("output has %q:\n%s", bad, out.String())
				}
			}
		})
	}
}

func TestPercentile(t *testing.T) {
	tests := []struct {
		name string
		ns   []int64
		want *Spread
	}{
		{"empty", nil, nil},
		{"one", []int64{7}, &Spread{7, 7, 7}},
		{"three", []int64{30, 10, 20}, &Spread{10, 20, 30}},
		{"ten", []int64{10, 9, 8, 7, 6, 5, 4, 3, 2, 1}, &Spread{1, 5, 9}},
		{"eleven", []int64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}, &Spread{2, 6, 10}},
		{"outlier", []int64{100, 100, 100, 100, 100, 100, 100, 100, 100, 900}, &Spread{100, 100, 100}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := spreadOf(tt.ns); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("spreadOf(%v) = %+v, want %+v", tt.ns, got, tt.want)
			}
		})
	}
}

// TestCompareSpread checks the spread is of per-launch exclusive time, not
// raw duration, and that the text lists it.
func TestCompareSpread(t *testing.T) {
	base := compareCapture([]rec{
		{subtype: "kernel", name: "k", start: 1100, end: 1200},
		{subtype: "kernel", name: "k", start: 1300, end: 1500},
		{subtype: "kernel", name: "k", start: 1600, end: 1900},
	}, phaseStatus(1000, 2000))
	target := compareCapture([]rec{
		{subtype: "kernel", name: "k", start: 1100, end: 1200},
		{subtype: "kernel", name: "k", start: 1150, end: 1400, stream: 1}, // owns 1200-1400
	}, phaseStatus(1000, 2000))
	c, err := compareCaptures(base, target, nil, nil, compareOptions{phase: -1})
	if err != nil {
		t.Fatal(err)
	}
	o := c.Ops[0]
	if *o.BaseSpread != (Spread{100, 200, 300}) || *o.TargetSpread != (Spread{100, 100, 200}) {
		t.Errorf("spread %+v -> %+v, want {100 200 300} -> {100 100 200}", *o.BaseSpread, *o.TargetSpread)
	}
	var out bytes.Buffer
	if err := c.write(&out); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(out.String(), "per-launch exclusive ns") || !strings.Contains(squash(out.String()), "100 200 300 100 100 200 k") {
		t.Errorf("output lacks the spread:\n%s", out.String())
	}
}

func TestLaunchCountChanged(t *testing.T) {
	tests := []struct {
		base, target int
		want         bool
	}{
		{10, 10, false},
		{100, 149, false},
		{100, 150, true},
		{150, 100, true},
		{149, 100, false},
		{2, 3, true},
		{3, 2, true},
		{0, 1, true},
		{1, 0, true},
		{0, 0, false},
	}
	for _, tt := range tests {
		if got := launchCountChanged(tt.base, tt.target); got != tt.want {
			t.Errorf("launchCountChanged(%d, %d) = %v, want %v", tt.base, tt.target, got, tt.want)
		}
	}
}

func TestCompareLaunchCountSection(t *testing.T) {
	var base, target []rec
	for i := range uint64(4) {
		base = append(base, rec{subtype: "kernel", name: "steady", start: 1000 + 10*i, end: 1005 + 10*i})
		target = append(target, rec{subtype: "kernel", name: "steady", start: 1000 + 10*i, end: 1005 + 10*i})
	}
	base = append(base, rec{subtype: "kernel", name: "split", start: 1100, end: 1110}, rec{subtype: "kernel", name: "gone", start: 1200, end: 1210})
	for i := range uint64(3) {
		target = append(target, rec{subtype: "kernel", name: "split", start: 1100 + 10*i, end: 1105 + 10*i})
	}
	c, err := compareCaptures(compareCapture(base, phaseStatus(1000, 2000)), compareCapture(target, phaseStatus(1000, 2000)), nil, nil, compareOptions{phase: -1})
	if err != nil {
		t.Fatal(err)
	}
	var out bytes.Buffer
	if err := c.write(&out); err != nil {
		t.Fatal(err)
	}
	_, section, ok := strings.Cut(out.String(), "launch count changes")
	section, _, _ = strings.Cut(section, "\n\n")
	if !ok {
		t.Fatalf("no launch count section:\n%s", out.String())
	}
	for _, want := range []string{"1 3 3.00x split", "1 0 - gone"} {
		if !strings.Contains(squash(section), want) {
			t.Errorf("section lacks %q:\n%s", want, section)
		}
	}
	if strings.Contains(section, "steady") {
		t.Errorf("section lists an op whose count did not move:\n%s", section)
	}
}

// TestCompareClocks checks that each side's edge readings come from the
// edges that bound its window, that a difference in SM clock is noted, and
// that a capture without readings says so.
func TestCompareClocks(t *testing.T) {
	recs := []rec{{subtype: "kernel", name: "k", start: 1100, end: 1200}}
	sample := func(sm, mem, temp, power uint32) map[string]any {
		return map[string]any{"sampled_at": 1, "lag_nanos": 4000, "devices": []any{
			map[string]any{"index": 0, "sm_clock_mhz": sm, "mem_clock_mhz": mem, "temperature_c": temp, "power_mw": power},
		}}
	}
	edge := func(seq, ts int, kind string, gpu map[string]any) map[string]any {
		e := map[string]any{"sequence": seq, "timestamp": ts, "edge": kind, "api": "cuProfilerStart"}
		if gpu != nil {
			e["gpu"] = gpu
		}
		return e
	}
	withEdges := func(start, stop uint64, edges ...any) map[string]any {
		s := phaseStatus(start, stop)
		s["phase_receipt"].(map[string]any)["edges"] = edges
		return s
	}
	twoPhases := phaseStatus(0, 0)
	twoPhases["phase_receipt"] = map[string]any{"status": "UNREAD", "reason": "multiple_phases",
		"edges": []any{
			edge(1, 1000, "start", sample(900, 8000, 40, 20000)),
			edge(2, 2000, "stop", sample(900, 8000, 41, 21000)),
			edge(3, 3000, "start", sample(2522, 8533, 55, 90000)),
			edge(4, 4000, "stop", sample(2522, 8533, 60, 95000)),
		},
		"phases": []any{
			map[string]any{"status": "MEASURED", "start_timestamp": 1000, "stop_timestamp": 2000},
			map[string]any{"status": "MEASURED", "start_timestamp": 3000, "stop_timestamp": 4000},
		}}
	recorded := withEdges(1000, 2000, edge(1, 1000, "start", sample(2522, 8533, 50, 80000)), edge(2, 2000, "stop", sample(2522, 8533, 52, 85000)))
	tests := []struct {
		name         string
		base, target map[string]any
		phase        int
		want, not    []string
	}{
		{
			name: "same clocks", base: recorded, target: recorded, phase: -1,
			want: []string{"sm clock MHz 2522 2522 2522 2522", "power mW 80000 85000 80000 85000", "read lag ns"},
			not:  []string{"not recorded:", "SM clock at the"},
		},
		{
			// Phase 0 of the target reads the edges at 1000 and 2000, not
			// those of phase 1.
			name: "phase edges", base: recorded, target: twoPhases, phase: 0,
			want: []string{"sm clock MHz 2522 2522 900 900", "SM clock at the start edge was 2522 MHz in base and 900 MHz in target"},
		},
		{
			name: "no edges", base: recorded, target: phaseStatus(1000, 2000), phase: -1,
			want: []string{"target: not recorded: the phase receipt holds no edges for the window", "sm clock MHz 2522 2522 - -"},
		},
		{
			name: "edges without readings", base: withEdges(1000, 2000, edge(1, 1000, "start", nil), edge(2, 2000, "stop", nil)), target: completeStatus(), phase: -1,
			want: []string{"base: not recorded: the capture predates GPU readings", "target: not recorded: the window is not a capture range"},
			not:  []string{"sm clock MHz"},
		},
		{
			name: "unavailable", base: withEdges(1000, 2000, edge(1, 1000, "start", map[string]any{"unavailable": "nvml init: no library"}), edge(2, 2000, "stop", sample(1, 2, 3, 4))), target: recorded, phase: -1,
			want: []string{"base start: not recorded: nvml init: no library"},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			c, err := compareCaptures(compareCapture(recs, tt.base), compareCapture(recs, tt.target), nil, nil, compareOptions{phase: tt.phase})
			if err != nil {
				t.Fatal(err)
			}
			var out bytes.Buffer
			if err := c.write(&out); err != nil {
				t.Fatal(err)
			}
			for _, want := range tt.want {
				if !strings.Contains(squash(out.String()), squash(want)) {
					t.Errorf("output lacks %q:\n%s", want, out.String())
				}
			}
			for _, bad := range tt.not {
				if strings.Contains(squash(out.String()), squash(bad)) {
					t.Errorf("output has %q:\n%s", bad, out.String())
				}
			}
			data, err := json.Marshal(c)
			if err != nil {
				t.Fatal(err)
			}
			if !strings.Contains(string(data), `"clocks":{`) {
				t.Errorf("json has no clocks: %s", data)
			}
		})
	}
}

// squash collapses runs of white space, so a check does not depend on
// tabwriter's padding.
func squash(s string) string { return strings.Join(strings.Fields(s), " ") }
