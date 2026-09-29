package advise

import (
	"bytes"
	"encoding/binary"
	"reflect"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/internal/cudainject/wire"
	"github.com/tmc/lib/nvidia/x/cudacapture"
	"github.com/tmc/lib/nvidia/x/cudaoccupancy"
	"github.com/tmc/lib/nvidia/x/cudaprof/internal/journaltest"
)

// findings returns r's items with each kernel's patterns in place of the item
// that holds them.
func findings(r *Report) []Item {
	var out []Item
	for _, item := range r.Items {
		if len(item.Patterns) == 0 {
			out = append(out, item)
		}
		out = append(out, item.Patterns...)
	}
	return out
}

const ms = 1_000_000

// testBaseline is a gap of +540 ms: idle +500, memcpy +30, op gemm +20,
// op rms -40 and (unmatched) +30. Kernels named gemm* count under gemm and
// rms* under rms; any other kernel is unmatched. It is cudaprof's
// testComparison with ops rules "^gemm gemm" and "^rms rms".
type testBaseline struct {
	unmeasured bool // ops were not measured
}

func (testBaseline) String() string {
	return "identity: Δwindow = ΣΔop + Δunmatched + Δmemcpy + Δmemset + Δidle + Δunresolved\n"
}

func (testBaseline) TermDelta(term string) int64 {
	return map[string]int64{
		"window": 540 * ms,
		"kernel": 10 * ms,
		"memcpy": 30 * ms,
		"idle":   500 * ms,
	}[term]
}

func (b testBaseline) OpDelta(kernel string) (string, int64, bool) {
	switch {
	case b.unmeasured:
		return "", 0, false
	case strings.HasPrefix(kernel, "gemm"):
		return "gemm", 20 * ms, true
	case strings.HasPrefix(kernel, "rms"):
		return "rms", -40 * ms, true
	}
	return "(unmatched)", 30 * ms, true
}

func TestGapBound(t *testing.T) {
	tests := []struct {
		name string
		item Item
		want *GapBound
	}{
		{
			name: "dispatch moves idle, capped at its host time",
			item: Item{Category: CategoryDispatch, HostTimeNanos: 320 * ms},
			want: &GapBound{Term: "idle", TermDeltaNanos: 500 * ms, BoundNanos: 320 * ms},
		},
		{
			name: "dispatch naming a kernel still moves idle",
			item: Item{Category: CategoryArchitectureDispatch, Location: Location{Kernel: "gemm_a"}, DeviceTimeNanos: 900 * ms},
			want: &GapBound{Term: "idle", TermDeltaNanos: 500 * ms, BoundNanos: 500 * ms},
		},
		{
			name: "synchronous copy moves memcpy and idle",
			item: Item{Category: CategoryDispatch, HostTimeNanos: 2000 * ms, terms: []string{"memcpy", "idle"}},
			want: &GapBound{Term: "memcpy + idle", TermDeltaNanos: 530 * ms, BoundNanos: 530 * ms},
		},
		{
			name: "kernel finding moves its op, capped at the kernel's time",
			item: Item{Category: CategoryRegisterPressure, Location: Location{Kernel: "gemm_a"}, DeviceTimeNanos: 5 * ms},
			want: &GapBound{Term: "op gemm", TermDeltaNanos: 20 * ms, BoundNanos: 5 * ms},
		},
		{
			name: "kernel whose op improved explains none of the gap",
			item: Item{Category: CategoryKernel, Location: Location{Kernel: "rms_norm"}, DeviceTimeNanos: 900 * ms},
			want: &GapBound{Term: "op rms", TermDeltaNanos: -40 * ms, BoundNanos: -40 * ms},
		},
		{
			name: "pair moves each distinct op once",
			item: Item{Category: CategoryMemoryCompute, Location: Location{Kernel: "rms_norm -> gemm_a"}, DeviceTimeNanos: 900 * ms},
			want: &GapBound{Term: "op rms + op gemm", TermDeltaNanos: -20 * ms, BoundNanos: -20 * ms},
		},
		{
			name: "kernel no rule names moves (unmatched)",
			item: Item{Category: CategoryCompute, Location: Location{Kernel: "softmax"}},
			want: &GapBound{Term: "op (unmatched)", TermDeltaNanos: 30 * ms, BoundNanos: 30 * ms},
		},
		{
			name: "no kernel and not dispatch moves no term",
			item: Item{Category: CategoryArchitecture},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := gapBound(tt.item, testBaseline{})
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("gapBound = %+v, want %+v", got, tt.want)
			}
		})
	}
	t.Run("kernel finding moves no term when ops were not measured", func(t *testing.T) {
		item := Item{Category: CategoryRegisterPressure, Location: Location{Kernel: "gemm_a"}, DeviceTimeNanos: 5 * ms}
		if got := gapBound(item, testBaseline{unmeasured: true}); got != nil {
			t.Errorf("gapBound = %+v, want nil", got)
		}
	})
}

func TestRankAgainst(t *testing.T) {
	tests := []struct {
		name  string
		items []Item
		want  []string // findings in order
	}{
		{
			name: "ranked by bound, not by own time",
			items: []Item{
				{Finding: "big kernel", Severity: SeverityCost, Location: Location{Kernel: "gemm_a"}, DeviceTimeNanos: 900 * ms},
				{Finding: "alloc", Severity: SeverityCost, Category: CategoryDispatch, HostTimeNanos: 300 * ms},
				{Finding: "copy", Severity: SeverityCost, Category: CategoryDispatch, HostTimeNanos: 2000 * ms, terms: []string{"memcpy", "idle"}},
			},
			want: []string{"copy", "alloc", "big kernel"},
		},
		{
			name: "a finding whose term improved goes last, however large",
			items: []Item{
				{Finding: "largest kernel", Severity: SeverityCost, Location: Location{Kernel: "rms_norm"}, DeviceTimeNanos: 900 * ms},
				{Finding: "no term", Severity: SeverityContext, Category: CategoryArchitecture},
				{Finding: "small kernel", Severity: SeverityContext, Location: Location{Kernel: "gemm_a"}, DeviceTimeNanos: 1 * ms},
			},
			want: []string{"small kernel", "no term", "largest kernel"},
		},
		{
			name: "read first stays first",
			items: []Item{
				{Finding: "alloc", Severity: SeverityCost, Category: CategoryDispatch, HostTimeNanos: 300 * ms},
				{Finding: "unusable", Severity: SeverityReadFirst, severityByKind: true, Category: CategoryArchitecture},
			},
			want: []string{"unusable", "alloc"},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			report := &Report{Items: tt.items}
			report.RankAgainst(testBaseline{})
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

func TestBaseMarkdown(t *testing.T) {
	report := &Report{Items: []Item{
		{Finding: "improved", Severity: SeverityCost, Location: Location{Kernel: "rms_norm"}, DeviceTimeNanos: 900 * ms},
		{Finding: "alloc", Severity: SeverityCost, Category: CategoryDispatch, HostTimeNanos: 300 * ms},
	}}
	report.RankAgainst(testBaseline{})
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
	pattern := func(kernel, finding string) Item {
		return Item{Severity: SeverityCost, Category: CategoryArchitecture, Location: Location{Kernel: kernel}, Finding: finding}
	}
	tests := []struct {
		name         string
		items        []Item
		wantItems    int
		wantPatterns map[string]int // kernel -> patterns held by its item
	}{
		{
			name:         "three patterns in one kernel fold into one",
			items:        []Item{pattern("k", "imad"), pattern("k", "fchk"), pattern("k", "occupancy")},
			wantItems:    1,
			wantPatterns: map[string]int{"k": 3},
		},
		{
			name:         "one pattern stays as it is",
			items:        []Item{pattern("k", "imad"), pattern("j", "fchk")},
			wantItems:    2,
			wantPatterns: map[string]int{},
		},
		{
			name: "measured, dispatch and pair findings do not fold",
			items: []Item{
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
			report := &Report{Items: tt.items}
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
		b := journaltest.New()
		for i, m := range modules {
			b.Append(wire.KindActivity, map[string]any{
				"kind": "activity", "subtype": "kernel", "valid": true, "name": "ds4_rms",
				"start": 1000 * (i + 1), "end": 1000*(i+1) + 500, "device": 0, "correlation": i + 1, "module_id": m,
			})
		}
		return b.Finalize()
	}
	tests := []struct {
		name      string
		journal   []byte
		fusion    string
		wantImage string // the images that answer for ds4_rms, comma-separated
		wantRead  bool
		wantWhy   string
	}{
		{name: "journal names the module", journal: ran(3), wantImage: "math.cubin", wantRead: true},
		{name: "no journal, three modules define it", wantImage: "probe.cubin", wantWhy: "defined in 3 captured modules"},
		{name: "ran from two modules", journal: ran(1, 3), wantImage: "probe.cubin,math.cubin", wantRead: true},
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
			cap := testCapture(all, journaltest.CompleteStatus())
			cap.Modules = modules
			images, unread := kernelImages(cap)
			var answer []string
			for _, img := range images {
				if img.answers("ds4_rms") {
					answer = append(answer, img.file)
				}
			}
			if image := strings.Join(answer, ","); image != tt.wantImage {
				t.Errorf("ds4_rms read from %q, want %q", image, tt.wantImage)
			}
			report := &Report{SourceFusion: tt.fusion, Items: []Item{
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

// TestModuleCharge pins per-module accounting: a kernel name that ran from
// two modules is two kernels, each charged the share of the name's time the
// journal records from its module, and their findings are not folded into
// one kernel's.
func TestModuleCharge(t *testing.T) {
	modules := []cudacapture.Module{
		{ID: "a", DriverID: 1, Code: cudacapture.CodeFiles{CUBIN: "a.cubin"}},
		{ID: "b", DriverID: 2, Code: cudacapture.CodeFiles{CUBIN: "b.cubin"}},
	}
	files := map[string][]byte{
		"a.cubin": moduleCUBIN("gemm", "only_a"),
		"b.cubin": moduleCUBIN("gemm"),
	}
	type launch struct {
		module uint32
		dur    int
	}
	tests := []struct {
		name     string
		launches []launch
		want     map[string]int64 // kernel key -> time charged
	}{
		{
			name:     "one module",
			launches: []launch{{1, 300}, {1, 100}},
			want:     map[string]int64{"gemm": 4e6},
		},
		{
			name:     "split by launch time",
			launches: []launch{{1, 300}, {2, 100}},
			want:     map[string]int64{"gemm": 4e6, "gemm\x00a.cubin": 3e6, "gemm\x00b.cubin": 1e6},
		},
		{
			name:     "no durations split evenly",
			launches: []launch{{1, 0}, {2, 0}},
			want:     map[string]int64{"gemm": 4e6, "gemm\x00a.cubin": 2e6, "gemm\x00b.cubin": 2e6},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			b := journaltest.New()
			start := 1000
			for i, l := range tt.launches {
				b.Append(wire.KindActivity, map[string]any{
					"kind": "activity", "subtype": "kernel", "valid": true, "name": "gemm",
					"start": start, "end": start + l.dur, "device": 0, "correlation": i + 1, "module_id": l.module,
				})
				start += 1000
			}
			all := map[string][]byte{"capture.journal": b.Finalize()}
			for k, v := range files {
				all[k] = v
			}
			cap := testCapture(all, journaltest.CompleteStatus())
			cap.Modules = modules
			images, unread := kernelImages(cap)
			if len(unread) != 0 {
				t.Errorf("unread = %v, want none", unread)
			}
			times := map[string]int64{"gemm": 4e6}
			chargeModules(times, images)
			if !reflect.DeepEqual(times, tt.want) {
				t.Errorf("times = %v, want %v", times, tt.want)
			}

			var items []Item
			for _, r := range kernelResources(images) {
				if r.Name != "gemm" {
					continue
				}
				for _, f := range []string{"spill", "pressure"} {
					items = append(items, Item{Severity: SeverityCost, Category: CategoryKernel, Finding: f,
						Location: Location{Kernel: r.Name, Module: r.module}})
				}
			}
			report := &Report{Items: items}
			finalizeAndRankFindings(report, times, 8e6)
			var charged int64
			for _, item := range report.Items {
				charged += item.DeviceTimeNanos
				if len(item.Patterns) != 2 {
					t.Errorf("%s holds %d patterns, want 2", item.Location, len(item.Patterns))
				}
			}
			if charged != 4e6 {
				t.Errorf("charged %d ns across %d kernels, want 4e6", charged, len(report.Items))
			}
		})
	}
}

// TestMissingModuleCode pins the notice for module images the capture did not
// keep: it names them by reason and charges it the share of kernel time that
// ran from them, which code-based findings cannot see.
func TestMissingModuleCode(t *testing.T) {
	status := journaltest.CompleteStatus()
	status["modules"] = []any{
		map[string]any{"module_id": 1, "size": 10, "file": "a.cubin", "stored": true},
		map[string]any{"module_id": 2, "size": 121e6, "stored": false, "reason": "over per-module budget"},
	}
	type launch struct {
		name   string
		module uint32
		dur    int
	}
	tests := []struct {
		name     string
		launches []launch
		want     string
		wantTime int64
	}{
		{
			name:     "no kernel ran from it",
			launches: []launch{{"a", 1, 100}},
			want:     "Code not captured for 1 module, 121.0 MB: 1 over per-module budget (121.0 MB)",
		},
		{
			name:     "kernels ran from it",
			launches: []launch{{"a", 1, 100}, {"b", 2, 100}, {"c", 1, 300}, {"c", 2, 100}},
			want:     "; 2 kernels that ran from them hold 30.0% of kernel time",
			wantTime: 3e6,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			b := journaltest.New()
			start := 1000
			for i, l := range tt.launches {
				b.Append(wire.KindActivity, map[string]any{
					"kind": "activity", "subtype": "kernel", "valid": true, "name": l.name,
					"start": start, "end": start + l.dur, "device": 0, "correlation": i + 1, "module_id": l.module,
				})
				start += 1000
			}
			cap := testCapture(map[string][]byte{"capture.journal": b.Finalize()}, status)
			times := map[string]int64{"a": 2e6, "b": 2e6, "c": 4e6}
			var report Report
			auditMissingModuleCode(cap, &report, times, 10e6, true)
			if len(report.Items) != 1 {
				t.Fatalf("got %d items, want 1", len(report.Items))
			}
			item := report.Items[0]
			if !strings.Contains(item.Finding, tt.want) {
				t.Errorf("finding = %q, want it to contain %q", item.Finding, tt.want)
			}
			if item.DeviceTimeNanos != tt.wantTime {
				t.Errorf("device time = %d, want %d", item.DeviceTimeNanos, tt.wantTime)
			}
		})
	}
}
