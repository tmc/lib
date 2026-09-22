package main

import (
	"bytes"
	"encoding/json"
	"maps"
	"reflect"
	"strings"
	"testing"

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
}

func compareJournal(recs []rec) []byte {
	b := newJournalBuilder()
	for i, r := range recs {
		b.append(recordActivity, map[string]any{
			"kind": "activity", "subtype": r.subtype, "name": r.name, "valid": true,
			"start": r.start, "end": r.end, "stream": r.stream, "bytes": r.bytes,
			"copy_kind": r.copyKind, "graph_id": r.graphID, "correlation": i + 1,
		})
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
			got, err := measureCompareSide(compareCapture(tt.recs, tt.status), nil, compareOptions{phase: -1, allowUnresolved: true})
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
			got.ops, got.WindowNote, got.StartTimestamp, got.StopTimestamp = nil, "", 0, 0
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
			c, err := compareCaptures(tt.base, target, nil, tt.opts)
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
	side, err := measureCompareSide(cap, nil, compareOptions{phase: 1})
	if err != nil {
		t.Fatal(err)
	}
	if side.PhaseIndex != 1 || side.WindowNanos != 2000 || side.BusyNanos != 200 || side.ops["second"] == nil || side.ops["first"] != nil {
		t.Errorf("phase 1: index %d window %d busy %d ops %v", side.PhaseIndex, side.WindowNanos, side.BusyNanos, side.ops)
	}
}

func TestCompareAllowUnresolved(t *testing.T) {
	base := compareCapture([]rec{{subtype: "kernel", name: "k", start: 1100, end: 1200}}, phaseStatus(1000, 2000))
	target := compareCapture([]rec{
		{subtype: "graph_trace", start: 1000, end: 1800, graphID: 1},
		{subtype: "kernel", name: "k", start: 1100, end: 1300, graphID: 1},
	}, phaseStatus(1000, 2000))
	c, err := compareCaptures(base, target, nil, compareOptions{phase: -1, allowUnresolved: true})
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
		op, rule := rules.kernelOp(tt.kernel)
		if op != tt.op || rule != tt.rule {
			t.Errorf("kernelOp(%q) = %q, %d; want %q, %d", tt.kernel, op, rule, tt.op, tt.rule)
		}
	}
	if op, _ := opRules(nil).kernelOp("_Z16hc_expand_kernelPfPKfS1_S1_S1_S1_S1_jjjjjii"); op != "hc_expand_kernel" {
		t.Errorf("without rules op = %q, want the demangled name", op)
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
	c, err := compareCaptures(base, target, rules, compareOptions{phase: -1})
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
		c, err := compareCaptures(base, target, nil, compareOptions{phase: -1})
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
		{"defaults", []string{"a.pprof", "b.pprof"}, compareOptions{phase: -1}, []string{"a.pprof", "b.pprof"}, "", false},
		{"all flags", []string{"-phase", "2", "-ops", "c-go.ops", "-json", "-allow-unresolved", "a", "b"},
			compareOptions{phase: 2, opsPath: "c-go.ops", json: true, allowUnresolved: true}, []string{"a", "b"}, "", false},
		{"one capture", []string{"a.pprof"}, compareOptions{}, nil, "usage: cudaprof compare", false},
		{"unknown flag", []string{"-bogus", "a", "b"}, compareOptions{}, nil, "-bogus", false},
		{"bad phase", []string{"-phase", "x", "a", "b"}, compareOptions{}, nil, "invalid value", false},
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
	c, err := compareCaptures(cap, cap, nil, compareOptions{phase: -1})
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
			side, err := measureCompareSide(compareCapture(recs, status), nil, compareOptions{phase: -1})
			if err != nil {
				t.Fatal(err)
			}
			if got := side.graphTraceModeString(); got != tt.want {
				t.Errorf("graph trace mode = %q, want %q", got, tt.want)
			}
		})
	}
}
