package cudalineinfo

import (
	"bytes"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cuptipcsampling"
	"github.com/tmc/lib/nvidia/x/cuptisass"
)

// siluOffset is an FFMA on the silu line of the gate fixture and gateOffset
// a load on the x input line; prologueOffset is a generated-line instruction
// and slowpathOffset is inside the division subroutine.
const (
	siluOffset     = 0x1f0
	gateOffset     = 0x0b0
	prologueOffset = 0x000
	slowpathOffset = 0x320
)

func TestResolvePC(t *testing.T) {
	a := loadFixture(t, "fused_gate", "")
	r, err := NewResolver(a, nil)
	if err != nil {
		t.Fatal(err)
	}
	for _, tt := range []struct {
		offset   uint64
		resolved bool
		line     int64
	}{
		{siluOffset, true, 14},
		{gateOffset, true, 12},
		{prologueOffset, true, 0},
		{slowpathOffset, true, 0},
		{0xfff0, false, 0},
	} {
		loc := r.ResolvePC(cuptipcsampling.ModuleID{}, "fused_gate", tt.offset)
		if loc.Resolved != tt.resolved {
			t.Errorf("0x%x resolved = %v, want %v", tt.offset, loc.Resolved, tt.resolved)
		}
		if tt.line > 0 && (loc.SourceLine != tt.line || !strings.HasSuffix(loc.SourceFile, "kernels.go")) {
			t.Errorf("0x%x = %s:%d, want kernels.go:%d", tt.offset, loc.SourceFile, loc.SourceLine, tt.line)
		}
		if tt.line == 0 && loc.Resolved && strings.HasSuffix(loc.SourceFile, "kernels.go") {
			t.Errorf("0x%x names kernels.go:%d, want no program line", tt.offset, loc.SourceLine)
		}
		if loc.KernelName != "fused_gate" {
			t.Errorf("0x%x kernel = %q", tt.offset, loc.KernelName)
		}
	}
	loc := r.ResolvePC(cuptipcsampling.ModuleID{}, "fused_gate", siluOffset)
	if loc.PTXLine == 0 || loc.PTXFile != "fused_gate.ptx" {
		t.Errorf("silu PTX = %s:%d, want a line of fused_gate.ptx", loc.PTXFile, loc.PTXLine)
	}
	if loc.SASSLine == 0 || loc.SASSFile != "fused_gate.sass" {
		t.Errorf("silu SASS = %s:%d, want an ordinal in fused_gate.sass", loc.SASSFile, loc.SASSLine)
	}
	if loc.CUBINOffset != siluOffset {
		t.Errorf("CUBINOffset without a cubin = 0x%x, want the function-relative 0x%x", loc.CUBINOffset, siluOffset)
	}
	if other := r.ResolvePC(cuptipcsampling.ModuleID{}, "other_kernel", siluOffset); other.Resolved {
		t.Error("resolved a PC in a kernel the attribution does not have")
	}

	r.CRC = 7
	if got := r.ResolvePC(cuptipcsampling.ModuleID{CRC: 8}, "fused_gate", siluOffset); got.Resolved {
		t.Error("resolved a PC from a module with the wrong CRC")
	}
	if got := r.ResolvePC(cuptipcsampling.ModuleID{CRC: 7}, "fused_gate", siluOffset); !got.Resolved {
		t.Error("did not resolve a PC from the module with the right CRC")
	}
}

// TestHostFrames walks the stack the compiler recorded for the line a PC
// falls on. The layered kernel is the one whose lines were built inside
// helpers, so its stack is deeper than the call that launched the kernel.
func TestHostFrames(t *testing.T) {
	a := loadFixture(t, "fused_layered", "")
	r, _ := NewResolver(a, nil)
	frames := a.HostFrames(r.ResolvePC(cuptipcsampling.ModuleID{}, "fused_layered", siluOffset))
	want := []struct {
		function, file string
		line           int64
	}{
		{"main.activate", kernelsFile, 45},
		{"main.layered", kernelsFile, 40},
		{"main.attribute", mainFile, 113},
		{"main.run", mainFile, 79},
		{"main.main", mainFile, 70},
	}
	if len(frames) != len(want) {
		t.Fatalf("got %d frames, want %d: %+v", len(frames), len(want), frames)
	}
	for i, w := range want {
		f := frames[i]
		if f.FunctionName != w.function || f.FileName != w.file || f.Line != w.line {
			t.Errorf("frame %d = %+v, want %s %s:%d", i, f, w.function, w.file, w.line)
		}
	}
	if got := a.HostFrames(r.ResolvePC(cuptipcsampling.ModuleID{}, "fused_layered", prologueOffset)); got != nil {
		t.Errorf("prologue has host frames %+v", got)
	}
}

func TestCountSamples(t *testing.T) {
	a := loadFixture(t, "fused_gate", "")
	inv := cuptipcsampling.ReasonInventory{Reasons: []cuptipcsampling.StallReason{
		{Index: 1, Name: "smsp__pcsamp_warps_issue_stalled_math_pipe_throttle"},
		{Index: 2, Name: "smsp__pcsamp_warps_issue_stalled_math_pipe_throttle_not_issued"},
		{Index: 3, Name: "smsp__pcsamp_warps_issue_stalled_selected"},
	}}
	pc := func(offset uint64, samples ...uint32) cuptipcsampling.PC {
		p := cuptipcsampling.PC{Name: "fused_gate", Offset: offset}
		for i, n := range samples {
			p.Reasons = append(p.Reasons, cuptipcsampling.StallSamples{Index: uint32(i + 1), Samples: n})
		}
		return p
	}
	batch := cuptipcsampling.Batch{Valid: true, PCs: []cuptipcsampling.PC{
		pc(siluOffset, 10, 4, 5), // the companion (index 2) is not counted
		pc(gateOffset, 0, 0, 3),
		pc(prologueOffset, 2),
		pc(slowpathOffset, 7),
		pc(0xfff0, 1),
		{Name: "", Offset: siluOffset, Reasons: []cuptipcsampling.StallSamples{{Index: 1, Samples: 99}}},
	}}
	p := CountSamples(a, inv, batch, cuptipcsampling.Batch{Valid: false, PCs: batch.PCs})
	silu := Site{File: kernelsFile, Line: 14}
	gate := Site{File: kernelsFile, Line: 12}
	if p.Sites[silu] != 15 || p.Sites[gate] != 3 || len(p.Sites) != 2 {
		t.Errorf("sites = %v, want silu 15 and gate 3", p.Sites)
	}
	if p.Other != 9 || p.Unresolved != 1 || p.Total() != 28 {
		t.Errorf("other %d unresolved %d total %d, want 9 1 28", p.Other, p.Unresolved, p.Total())
	}

	var buf bytes.Buffer
	if err := WriteProfile(&buf, a, p); err != nil {
		t.Fatal(err)
	}
	out := buf.String()
	for _, want := range []string{"fused_gate: 28 samples by program line", "kernels.go:14", "53.6%", "elsewhere: 9 at instructions naming no program line, 1 at unresolved PCs"} {
		if !strings.Contains(out, want) {
			t.Errorf("profile lacks %q:\n%s", want, out)
		}
	}
}

func TestCountExecuted(t *testing.T) {
	a := loadFixture(t, "fused_gate", "")
	const metric = 42
	rec := func(offset uint32, vals ...cuptisass.InstanceValue) cuptisass.Record {
		return cuptisass.Record{FunctionName: "fused_gate", PCOffset: offset, Values: vals}
	}
	p := CountExecuted(a, metric,
		rec(siluOffset, cuptisass.InstanceValue{MetricID: metric, Value: 6}, cuptisass.InstanceValue{MetricID: metric, Value: 4}),
		rec(siluOffset, cuptisass.InstanceValue{MetricID: 1, Value: 100}),
		rec(gateOffset, cuptisass.InstanceValue{MetricID: metric, Value: 2}),
		rec(slowpathOffset, cuptisass.InstanceValue{MetricID: metric, Value: 1}),
	)
	if p.Measure != "executed" || p.Sites[Site{File: kernelsFile, Line: 14}] != 10 || p.Sites[Site{File: kernelsFile, Line: 12}] != 2 || p.Other != 1 || p.Unresolved != 0 {
		t.Errorf("profile = %+v", p)
	}
}

func TestSiteAtLine(t *testing.T) {
	a := loadFixture(t, "fused_gate", "")
	fn := a.Disassembly.Function("fused_gate")
	for i, in := range fn.Instructions {
		if in.Address != siluOffset {
			continue
		}
		s, ok := a.SiteAtLine("fused_gate", i+1)
		if !ok || s != (Site{kernelsFile, 14}) {
			t.Errorf("SiteAtLine(fused_gate, %d) = %v, %v; want kernels.go:14", i+1, s, ok)
		}
	}
	if _, ok := a.SiteAtLine("fused_gate", 1); ok {
		t.Error("the prologue names a program line")
	}
	if _, ok := a.SiteAtLine("fused_gate", len(fn.Instructions)+1); ok {
		t.Error("a line past the end resolved")
	}
	if _, ok := a.SiteAtLine("missing", 1); ok {
		t.Error("a missing function resolved")
	}
}

func TestWriteProfileUnattributedSite(t *testing.T) {
	a := loadFixture(t, "fused_gate", "")
	p := Profile{Measure: "pc_samples", Sites: map[Site]int64{{kernelsFile, 14}: 3, {"other.go", 7}: 1}}
	var buf bytes.Buffer
	if err := WriteProfile(&buf, a, p); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(buf.String(), "other.go:7") || !strings.Contains(buf.String(), "25.0%") {
		t.Errorf("profile omits the site the attribution lacks:\n%s", buf.String())
	}
}
