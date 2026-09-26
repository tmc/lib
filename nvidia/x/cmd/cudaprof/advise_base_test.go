package main

import (
	"reflect"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudaprof/advise"
)

const ms = 1_000_000

// testComparison is a gap of +540 ms: idle +500, memcpy +30, op gemm +20,
// op rms -40 and (unmatched) +30.
func testComparison(t *testing.T) (*Comparison, opRules) {
	t.Helper()
	rules, err := parseOpRules(strings.NewReader("^gemm gemm\n^rms rms\n"))
	if err != nil {
		t.Fatal(err)
	}
	c := &Comparison{
		OpStatus: "MEASURED",
		Terms: []GapIntDelta{
			intDelta("window", 1000*ms, 1540*ms),
			intDelta("kernel", 800*ms, 810*ms),
			intDelta("memcpy", 0, 30*ms),
			intDelta("memset", 0, 0),
			intDelta("idle", 200*ms, 700*ms),
			intDelta("unresolved", 0, 0),
		},
		Ops: []CompareOp{
			{Op: "gemm", DeltaExclusiveNanos: 20 * ms},
			{Op: "rms", DeltaExclusiveNanos: -40 * ms},
			{Op: unmatchedOp, DeltaExclusiveNanos: 30 * ms},
		},
	}
	return c, rules
}

// TestAdviseBaselineGapBound is advise's TestGapBound through a Comparison:
// the terms and ops a finding moves are the comparison's, and a kernel's op
// is the one the ops rules name.
func TestAdviseBaselineGapBound(t *testing.T) {
	c, rules := testComparison(t)
	tests := []struct {
		name string
		item advise.Item
		want *advise.GapBound
	}{
		{
			name: "dispatch moves idle, capped at its host time",
			item: advise.Item{Category: advise.CategoryDispatch, HostTimeNanos: 320 * ms},
			want: &advise.GapBound{Term: "idle", TermDeltaNanos: 500 * ms, BoundNanos: 320 * ms},
		},
		{
			name: "dispatch naming a kernel still moves idle",
			item: advise.Item{Category: advise.CategoryArchitectureDispatch, Location: advise.Location{Kernel: "gemm_a"}, DeviceTimeNanos: 900 * ms},
			want: &advise.GapBound{Term: "idle", TermDeltaNanos: 500 * ms, BoundNanos: 500 * ms},
		},
		{
			name: "kernel finding moves its op, capped at the kernel's time",
			item: advise.Item{Category: advise.CategoryRegisterPressure, Location: advise.Location{Kernel: "gemm_a"}, DeviceTimeNanos: 5 * ms},
			want: &advise.GapBound{Term: "op gemm", TermDeltaNanos: 20 * ms, BoundNanos: 5 * ms},
		},
		{
			name: "kernel whose op improved explains none of the gap",
			item: advise.Item{Category: advise.CategoryKernel, Location: advise.Location{Kernel: "rms_norm"}, DeviceTimeNanos: 900 * ms},
			want: &advise.GapBound{Term: "op rms", TermDeltaNanos: -40 * ms, BoundNanos: -40 * ms},
		},
		{
			name: "pair moves each distinct op once",
			item: advise.Item{Category: advise.CategoryMemoryCompute, Location: advise.Location{Kernel: "rms_norm -> gemm_a"}, DeviceTimeNanos: 900 * ms},
			want: &advise.GapBound{Term: "op rms + op gemm", TermDeltaNanos: -20 * ms, BoundNanos: -20 * ms},
		},
		{
			name: "kernel no rule names moves (unmatched)",
			item: advise.Item{Category: advise.CategoryCompute, Location: advise.Location{Kernel: "softmax"}},
			want: &advise.GapBound{Term: "op (unmatched)", TermDeltaNanos: 30 * ms, BoundNanos: 30 * ms},
		},
		{
			name: "no kernel and not dispatch moves no term",
			item: advise.Item{Category: advise.CategoryArchitecture},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			report := &advise.Report{Items: []advise.Item{tt.item}}
			report.RankAgainst(newAdviseBaseline(c, rules))
			if got := report.Items[0].Gap; !reflect.DeepEqual(got, tt.want) {
				t.Errorf("gap = %+v, want %+v", got, tt.want)
			}
		})
	}
	t.Run("kernel finding moves no term when ops are unread", func(t *testing.T) {
		c, rules := testComparison(t)
		c.OpStatus = "UNREAD"
		report := &advise.Report{Items: []advise.Item{
			{Category: advise.CategoryRegisterPressure, Location: advise.Location{Kernel: "gemm_a"}, DeviceTimeNanos: 5 * ms},
		}}
		report.RankAgainst(newAdviseBaseline(c, rules))
		if got := report.Items[0].Gap; got != nil {
			t.Errorf("gap = %+v, want nil", got)
		}
	})
}

func TestAdviseBaseMarkdown(t *testing.T) {
	c, rules := testComparison(t)
	c.Base = &CompareSide{Capture: "base.pprof", PhaseIndex: 0}
	c.Target = &CompareSide{Capture: "target.pprof", PhaseIndex: 0}
	report := &advise.Report{Items: []advise.Item{
		{Finding: "improved", Severity: advise.SeverityCost, Location: advise.Location{Kernel: "rms_norm"}, DeviceTimeNanos: 900 * ms},
		{Finding: "alloc", Severity: advise.SeverityCost, Category: advise.CategoryDispatch, HostTimeNanos: 300 * ms},
	}}
	report.RankAgainst(newAdviseBaseline(c, rules))
	md := report.MarkdownWithLimit(0)
	for _, want := range []string{
		"identity: Δwindow = ΣΔop",
		"not an estimate of what fixing it saves",
		"## Findings that could explain the gap, by bound\n\n### 1. [Cost]",
		"explains at most 300.00 ms of the gap (term idle, Δ +500.00 ms)",
		"## Findings that do not explain the gap: their term did not grow\n\n### 2.",
		"explains none of the gap (term op rms, Δ -40.00 ms)",
	} {
		if !strings.Contains(md, want) {
			t.Errorf("markdown lacks %q:\n%s", want, md)
		}
	}
	if n := strings.Count(md, "not an estimate"); n != 1 {
		t.Errorf("bound caveat printed %d times, want once", n)
	}
}

func TestParseAdviseArgs(t *testing.T) {
	tests := []struct {
		name    string
		args    []string
		want    adviseOptions
		wantErr string
	}{
		{
			name: "plain",
			args: []string{"go.pprof"},
			want: adviseOptions{limit: advise.DefaultLimit, phase: -1, skipIterations: 1, capture: "go.pprof"},
		},
		{
			name: "base and ops as separate values",
			args: []string{"-base", "c.pprof", "-ops", "ds4.ops", "go.pprof"},
			want: adviseOptions{limit: advise.DefaultLimit, phase: -1, skipIterations: 1, capture: "go.pprof", base: "c.pprof", ops: "ds4.ops"},
		},
		{
			name: "base, ops and phase with =",
			args: []string{"--base=c.pprof", "-ops=ds4.ops", "-phase=1", "-e", "--format=json", "go.pprof"},
			want: adviseOptions{format: "json", limit: 0, phase: 1, skipIterations: 1, capture: "go.pprof", base: "c.pprof", ops: "ds4.ops"},
		},
		{
			name: "limit",
			args: []string{"-limit", "3", "go.pprof"},
			want: adviseOptions{limit: 3, phase: -1, skipIterations: 1, capture: "go.pprof"},
		},
		{
			name: "iterations",
			args: []string{"-iteration-marker", "^Embed$", "-nvtx", "-skip-iterations=2", "-expect-markers", "11", "go.pprof"},
			want: adviseOptions{limit: advise.DefaultLimit, phase: -1, capture: "go.pprof", iterationMarker: "^Embed$", nvtx: true, skipIterations: 2, expectMarkers: 11},
		},
		{name: "skip without marker", args: []string{"-skip-iterations=2", "go.pprof"}, wantErr: "need -iteration-marker"},
		{name: "negative skip", args: []string{"-iteration-marker=x", "-skip-iterations=-1", "go.pprof"}, wantErr: "invalid -skip-iterations -1"},
		{name: "ops without base", args: []string{"-ops", "ds4.ops", "go.pprof"}, wantErr: "need -base"},
		{name: "phase without base", args: []string{"-phase", "0", "go.pprof"}, wantErr: "need -base"},
		{name: "base without value", args: []string{"go.pprof", "-base"}, wantErr: "flag -base requires an argument"},
		{name: "bad phase", args: []string{"-base", "c.pprof", "-phase", "x", "go.pprof"}, wantErr: "invalid -phase value: x"},
		{name: "no capture", args: []string{"-base", "c.pprof"}, wantErr: "usage: cudaprof advise"},
		{name: "two captures", args: []string{"a.pprof", "b.pprof"}, wantErr: "multiple capture files"},
		{name: "unknown", args: []string{"-bogus", "go.pprof"}, wantErr: "unknown flag: -bogus"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := parseAdviseArgs(tt.args)
			if tt.wantErr != "" {
				if err == nil || !strings.Contains(err.Error(), tt.wantErr) {
					t.Fatalf("err = %v, want %q", err, tt.wantErr)
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			if got != tt.want {
				t.Errorf("options = %+v, want %+v", got, tt.want)
			}
		})
	}
}
