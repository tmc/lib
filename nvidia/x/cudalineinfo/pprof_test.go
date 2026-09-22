package cudalineinfo

import (
	"bytes"
	"context"
	"slices"
	"strconv"
	"strings"
	"testing"

	"github.com/google/pprof/profile"
	"github.com/tmc/lib/nvidia/x/cuptipcsampling"
)

// sampledBatch returns an inventory and one batch of samples against the
// hand-written unit: a memory stall and an issued instruction at the FMUL of
// forward.cu:7, and a reason no classifier models.
func sampledBatch() (cuptipcsampling.ReasonInventory, cuptipcsampling.Batch) {
	inv := cuptipcsampling.ReasonInventory{
		CUPTIVersion: 130001,
		Reasons: []cuptipcsampling.StallReason{
			{Index: 1, Name: "smsp__pcsamp_warps_issue_stalled_lg_throttle"},
			{Index: 2, Name: "smsp__pcsamp_warps_issue_stalled_selected"},
			{Index: 3, Name: "sleeping"},
		},
	}
	batch := cuptipcsampling.Batch{
		Valid:        true,
		TotalSamples: 11,
		PCs: []cuptipcsampling.PC{{
			CRC:    1,
			Name:   "geglu",
			Offset: 0x20,
			Reasons: []cuptipcsampling.StallSamples{
				{Index: 1, Samples: 7},
				{Index: 2, Samples: 3},
				{Index: 3, Samples: 1},
			},
		}},
	}
	return inv, batch
}

// TestWritePprof checks the profile a caller gets without assembling one: the
// columns the bridge fills are all carried, and the sampled PC reaches the
// line of the program rather than an offset.
func TestWritePprof(t *testing.T) {
	a := Attribute(Unit{Name: "forward", Source: handWrittenSource, Kernels: []string{"geglu", "rmsnorm"}}, "", handWrittenDisassembly(t))
	inv, batch := sampledBatch()

	var buf bytes.Buffer
	if err := WritePprof(context.Background(), &buf, a, inv, batch); err != nil {
		t.Fatalf("WritePprof: %v", err)
	}
	p, err := profile.ParseData(buf.Bytes())
	if err != nil {
		t.Fatalf("ParseData: %v", err)
	}

	// Every type the bridge counts into is carried. Read from the bridge's own
	// published set rather than restated, which is the whole point of the
	// helper: a type added upstream appears here without editing this test.
	var carried []string
	for _, st := range p.SampleType {
		carried = append(carried, st.Type)
	}
	for _, want := range append(cuptipcsampling.StallSampleTypes(), "pc_samples", "inst_executed") {
		if !slices.Contains(carried, want) {
			t.Errorf("profile carries %v, missing %q", carried, want)
		}
	}
	if p.DefaultSampleType != "pc_samples" {
		t.Errorf("DefaultSampleType = %q, want pc_samples", p.DefaultSampleType)
	}

	// The measurement: pc_samples is every base reason at the PC, the issued
	// one included, since it is a cycle the sampler observed like any other.
	// inst_executed then counts the issued subset again, on purpose.
	value := func(name string) int64 {
		i := slices.IndexFunc(p.SampleType, func(st *profile.ValueType) bool { return st.Type == name })
		if i < 0 {
			t.Fatalf("profile has no %q", name)
		}
		var sum int64
		for _, s := range p.Sample {
			sum += s.Value[i]
		}
		return sum
	}
	for _, tt := range []struct {
		name string
		want int64
	}{
		{"pc_samples", 11},
		{"stall_mem", 7},
		{"stall_other", 1},
		{"inst_executed", 3},
	} {
		if got := value(tt.name); got != tt.want {
			t.Errorf("%s = %d, want %d", tt.name, got, tt.want)
		}
	}

	if len(p.Sample) != 1 {
		t.Fatalf("profile has %d samples, want 1", len(p.Sample))
	}
	var lines []string
	for _, loc := range p.Sample[0].Location {
		for _, l := range loc.Line {
			lines = append(lines, l.Function.Filename+":"+strconv.FormatInt(l.Line, 10))
		}
	}
	if !slices.Contains(lines, "forward.cu:7") {
		t.Errorf("stack = %v, want a frame at forward.cu:7", lines)
	}
	// A hand-written unit has no line map, so HostFrames reports nothing and
	// the CUDA line is the outermost frame. A Go frame here would be invented.
	for _, l := range lines {
		if strings.HasSuffix(l, ".go") || strings.Contains(l, ".go:") {
			t.Errorf("stack = %v, want no host frames for a hand-written unit", lines)
		}
	}

	comments := strings.Join(p.Comments, "\n")
	if !strings.Contains(comments, "forward") {
		t.Errorf("comments = %q, want the unit named", comments)
	}
	// stall_other says the breakdown is incomplete; the comment is the only
	// place the reason behind it keeps its name.
	if !strings.Contains(comments, "sleeping (1)") {
		t.Errorf("comments = %q, want the unmodelled reason named", comments)
	}
	if strings.Contains(comments, "incomplete: the bridge counted into") {
		t.Errorf("comments = %q, want no dropped types", comments)
	}
}

// TestWritePprofSaysNothingWasAttributed carries R6 into the profile: a join
// that named no line writes samples against a kernel and no program line, and
// nothing in the samples themselves says why.
func TestWritePprofSaysNothingWasAttributed(t *testing.T) {
	a := Attribute(Unit{Name: "forward"}, "", handWrittenDisassembly(t))
	if !a.Unattributed() {
		t.Fatal("bare unit attributed something; the fixture no longer tests this")
	}
	inv, batch := sampledBatch()
	var buf bytes.Buffer
	if err := WritePprof(context.Background(), &buf, a, inv, batch); err != nil {
		t.Fatalf("WritePprof: %v", err)
	}
	p, err := profile.ParseData(buf.Bytes())
	if err != nil {
		t.Fatalf("ParseData: %v", err)
	}
	if got := strings.Join(p.Comments, "\n"); !strings.Contains(got, "nothing attributed") {
		t.Errorf("comments = %q, want the failed join reported", got)
	}
}
