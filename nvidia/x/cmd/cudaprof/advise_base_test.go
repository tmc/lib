package main

import (
	"bytes"
	"encoding/binary"
	"reflect"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudacapture"
	"github.com/tmc/lib/nvidia/x/cudaoccupancy"
)

// findings returns r's items with each kernel's patterns in place of the item
// that holds them.
func findings(r *AdviceReport) []AdviceItem {
	var out []AdviceItem
	for _, item := range r.Items {
		if len(item.Patterns) == 0 {
			out = append(out, item)
		}
		out = append(out, item.Patterns...)
	}
	return out
}

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

func TestGapBound(t *testing.T) {
	c, rules := testComparison(t)
	tests := []struct {
		name string
		item AdviceItem
		want *GapBound
	}{
		{
			name: "dispatch moves idle, capped at its host time",
			item: AdviceItem{Category: CategoryDispatch, HostTimeNanos: 320 * ms},
			want: &GapBound{Term: "idle", TermDeltaNanos: 500 * ms, BoundNanos: 320 * ms},
		},
		{
			name: "dispatch naming a kernel still moves idle",
			item: AdviceItem{Category: CategoryArchitectureDispatch, Location: Location{Kernel: "gemm_a"}, DeviceTimeNanos: 900 * ms},
			want: &GapBound{Term: "idle", TermDeltaNanos: 500 * ms, BoundNanos: 500 * ms},
		},
		{
			name: "synchronous copy moves memcpy and idle",
			item: AdviceItem{Category: CategoryDispatch, HostTimeNanos: 2000 * ms, terms: []string{"memcpy", "idle"}},
			want: &GapBound{Term: "memcpy + idle", TermDeltaNanos: 530 * ms, BoundNanos: 530 * ms},
		},
		{
			name: "kernel finding moves its op, capped at the kernel's time",
			item: AdviceItem{Category: CategoryRegisterPressure, Location: Location{Kernel: "gemm_a"}, DeviceTimeNanos: 5 * ms},
			want: &GapBound{Term: "op gemm", TermDeltaNanos: 20 * ms, BoundNanos: 5 * ms},
		},
		{
			name: "kernel whose op improved explains none of the gap",
			item: AdviceItem{Category: CategoryKernel, Location: Location{Kernel: "rms_norm"}, DeviceTimeNanos: 900 * ms},
			want: &GapBound{Term: "op rms", TermDeltaNanos: -40 * ms, BoundNanos: -40 * ms},
		},
		{
			name: "pair moves each distinct op once",
			item: AdviceItem{Category: CategoryMemoryCompute, Location: Location{Kernel: "rms_norm -> gemm_a"}, DeviceTimeNanos: 900 * ms},
			want: &GapBound{Term: "op rms + op gemm", TermDeltaNanos: -20 * ms, BoundNanos: -20 * ms},
		},
		{
			name: "kernel no rule names moves (unmatched)",
			item: AdviceItem{Category: CategoryCompute, Location: Location{Kernel: "softmax"}},
			want: &GapBound{Term: "op (unmatched)", TermDeltaNanos: 30 * ms, BoundNanos: 30 * ms},
		},
		{
			name: "no kernel and not dispatch moves no term",
			item: AdviceItem{Category: CategoryArchitecture},
		},
	}
	ops := map[string]int64{}
	for _, o := range c.Ops {
		ops[o.Op] = o.DeltaExclusiveNanos
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := gapBound(tt.item, c, ops, rules)
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("gapBound = %+v, want %+v", got, tt.want)
			}
		})
	}
}

func TestRankAgainstBase(t *testing.T) {
	c, rules := testComparison(t)
	tests := []struct {
		name  string
		items []AdviceItem
		want  []string // findings in order
	}{
		{
			name: "ranked by bound, not by own time",
			items: []AdviceItem{
				{Finding: "big kernel", Severity: SeverityCost, Location: Location{Kernel: "gemm_a"}, DeviceTimeNanos: 900 * ms},
				{Finding: "alloc", Severity: SeverityCost, Category: CategoryDispatch, HostTimeNanos: 300 * ms},
				{Finding: "copy", Severity: SeverityCost, Category: CategoryDispatch, HostTimeNanos: 2000 * ms, terms: []string{"memcpy", "idle"}},
			},
			want: []string{"copy", "alloc", "big kernel"},
		},
		{
			name: "a finding whose term improved goes last, however large",
			items: []AdviceItem{
				{Finding: "largest kernel", Severity: SeverityCost, Location: Location{Kernel: "rms_norm"}, DeviceTimeNanos: 900 * ms},
				{Finding: "no term", Severity: SeverityContext, Category: CategoryArchitecture},
				{Finding: "small kernel", Severity: SeverityContext, Location: Location{Kernel: "gemm_a"}, DeviceTimeNanos: 1 * ms},
			},
			want: []string{"small kernel", "no term", "largest kernel"},
		},
		{
			name: "read first stays first",
			items: []AdviceItem{
				{Finding: "alloc", Severity: SeverityCost, Category: CategoryDispatch, HostTimeNanos: 300 * ms},
				{Finding: "unusable", Severity: SeverityReadFirst, severityByKind: true, Category: CategoryArchitecture},
			},
			want: []string{"unusable", "alloc"},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			report := &AdviceReport{Items: tt.items}
			rankAgainstBase(report, c, rules)
			var got []string
			for _, item := range report.Items {
				got = append(got, item.Finding)
			}
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("order = %q, want %q", got, tt.want)
			}
		})
	}
}

func TestAdviseBaseMarkdown(t *testing.T) {
	c, rules := testComparison(t)
	c.Base = &CompareSide{Capture: "base.pprof", PhaseIndex: 0}
	c.Target = &CompareSide{Capture: "target.pprof", PhaseIndex: 0}
	report := &AdviceReport{Items: []AdviceItem{
		{Finding: "improved", Severity: SeverityCost, Location: Location{Kernel: "rms_norm"}, DeviceTimeNanos: 900 * ms},
		{Finding: "alloc", Severity: SeverityCost, Category: CategoryDispatch, HostTimeNanos: 300 * ms},
	}}
	rankAgainstBase(report, c, rules)
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

func TestFoldKernelPatterns(t *testing.T) {
	pattern := func(kernel, finding string) AdviceItem {
		return AdviceItem{Severity: SeverityCost, Category: CategoryArchitecture, Location: Location{Kernel: kernel}, Finding: finding}
	}
	tests := []struct {
		name         string
		items        []AdviceItem
		wantItems    int
		wantPatterns map[string]int // kernel -> patterns held by its item
	}{
		{
			name:         "three patterns in one kernel fold into one",
			items:        []AdviceItem{pattern("k", "imad"), pattern("k", "fchk"), pattern("k", "occupancy")},
			wantItems:    1,
			wantPatterns: map[string]int{"k": 3},
		},
		{
			name:         "one pattern stays as it is",
			items:        []AdviceItem{pattern("k", "imad"), pattern("j", "fchk")},
			wantItems:    2,
			wantPatterns: map[string]int{},
		},
		{
			name: "measured, dispatch and pair findings do not fold",
			items: []AdviceItem{
				pattern("k", "imad"), pattern("k", "fchk"),
				{Severity: SeverityCost, Location: Location{Kernel: "k"}, DeviceTimeNanos: 3 * ms, Finding: "family"},
				{Severity: SeverityCost, Category: CategoryArchitectureDispatch, Location: Location{Kernel: "k"}, Finding: "loop"},
				{Severity: SeverityCost, Location: Location{Kernel: "k -> j"}, Finding: "pair"},
			},
			wantItems:    4,
			wantPatterns: map[string]int{"k": 2},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			report := &AdviceReport{Items: tt.items}
			finalizeAndRankFindings(report, map[string]int64{"k": 10 * ms, "j": 1 * ms}, 20*ms)
			if len(report.Items) != tt.wantItems {
				t.Fatalf("items = %d, want %d: %+v", len(report.Items), tt.wantItems, report.Items)
			}
			var charged int64
			got := map[string]int{}
			for _, item := range report.Items {
				if len(item.Patterns) > 0 {
					got[item.Location.Kernel] = len(item.Patterns)
					if item.DeviceTimeNanos != 10*ms {
						t.Errorf("kernel item time = %d, want the kernel's %d", item.DeviceTimeNanos, 10*ms)
					}
				}
				for _, p := range item.Patterns {
					charged += p.DeviceTimeNanos
				}
			}
			if charged != 0 {
				t.Errorf("patterns carry %d ns of their own; the kernel's time is counted once, on its item", charged)
			}
			if !reflect.DeepEqual(got, tt.wantPatterns) {
				t.Errorf("patterns = %v, want %v", got, tt.wantPatterns)
			}
		})
	}
}

// moduleCUBIN returns a CUBIN defining each of kernels.
func moduleCUBIN(kernels ...string) []byte {
	sections := map[string][]byte{}
	for _, k := range kernels {
		var nv bytes.Buffer
		nv.Write([]byte{0x04, cudaoccupancy.EIATTR_REGCOUNT, 0x04, 0x00})
		binary.Write(&nv, binary.LittleEndian, uint32(32))
		sections[".text."+k] = make([]byte, 32)
		sections[".nv.info."+k] = nv.Bytes()
	}
	return synthesizeTestELFForAdvise(sections)
}

// TestUnreadSource pins the ds4_rms case: three modules of one process define
// the kernel, and a source location read from any but the one it ran from is
// another compilation's line table.
func TestUnreadSource(t *testing.T) {
	modules := []cudacapture.Module{
		{ID: "probe", DriverID: 1, Code: cudacapture.CodeFiles{CUBIN: "probe.cubin"}},
		{ID: "model", DriverID: 2, Code: cudacapture.CodeFiles{CUBIN: "model.cubin"}},
		{ID: "math", DriverID: 3, Code: cudacapture.CodeFiles{CUBIN: "math.cubin"}},
	}
	files := map[string][]byte{
		"probe.cubin": moduleCUBIN("ds4_rms", "ds4_probe"),
		"model.cubin": moduleCUBIN("ds4_rms"),
		"math.cubin":  moduleCUBIN("ds4_rms", "ds4_gemm"),
	}
	ran := func(modules ...uint32) []byte {
		b := newJournalBuilder()
		for i, m := range modules {
			b.append(recordActivity, map[string]any{
				"kind": "activity", "subtype": "kernel", "valid": true, "name": "ds4_rms",
				"start": 1000 * (i + 1), "end": 1000*(i+1) + 500, "device": 0, "correlation": i + 1, "module_id": m,
			})
		}
		return b.finalize()
	}
	tests := []struct {
		name      string
		journal   []byte
		fusion    string
		wantImage string // the image that answers for ds4_rms, or ""
		wantRead  bool
		wantWhy   string
	}{
		{name: "journal names the module", journal: ran(3), wantImage: "math.cubin", wantRead: true},
		{name: "no journal, three modules define it", wantImage: "probe.cubin", wantWhy: "defined in 3 captured modules"},
		{name: "ran from two modules", journal: ran(1, 3), wantImage: "probe.cubin", wantWhy: "ran from 2 modules"},
		{name: "fusion clears every kernel", journal: ran(3), fusion: "shifted", wantImage: "math.cubin", wantWhy: "not from one compilation"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			all := map[string][]byte{}
			for k, v := range files {
				all[k] = v
			}
			if tt.journal != nil {
				all["capture.journal"] = tt.journal
			}
			cap := testCapture(all, completeStatus())
			cap.Modules = modules
			images, unread := kernelImages(cap)
			var image string
			for _, img := range images {
				if img.answers("ds4_rms") {
					if image != "" {
						t.Fatalf("ds4_rms answered by %s and %s", image, img.file)
					}
					image = img.file
				}
			}
			if image != tt.wantImage {
				t.Errorf("ds4_rms read from %q, want %q", image, tt.wantImage)
			}
			report := &AdviceReport{SourceFusion: tt.fusion, Items: []AdviceItem{
				{Location: Location{Kernel: "ds4_rms", File: "/home/user/ds4_probe.cu", Line: 80}},
			}}
			markUnreadSources(report, unread)
			loc := report.Items[0].Location
			if tt.wantRead {
				if loc.File == "" || loc.Source != "" {
					t.Errorf("location = %s, want the source kept", loc)
				}
				return
			}
			if loc.File != "" || loc.Line != 0 || !strings.HasPrefix(loc.Source, "UNREAD: ") || !strings.Contains(loc.Source, tt.wantWhy) {
				t.Errorf("location = %s, want file cleared and UNREAD naming %q", loc, tt.wantWhy)
			}
		})
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
			want: adviseOptions{limit: defaultAdviceLimit, phase: -1, capture: "go.pprof"},
		},
		{
			name: "base and ops as separate values",
			args: []string{"-base", "c.pprof", "-ops", "ds4.ops", "go.pprof"},
			want: adviseOptions{limit: defaultAdviceLimit, phase: -1, capture: "go.pprof", base: "c.pprof", ops: "ds4.ops"},
		},
		{
			name: "base, ops and phase with =",
			args: []string{"--base=c.pprof", "-ops=ds4.ops", "-phase=1", "-e", "--format=json", "go.pprof"},
			want: adviseOptions{format: "json", limit: 0, phase: 1, capture: "go.pprof", base: "c.pprof", ops: "ds4.ops"},
		},
		{
			name: "limit",
			args: []string{"-limit", "3", "go.pprof"},
			want: adviseOptions{limit: 3, phase: -1, capture: "go.pprof"},
		},
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
