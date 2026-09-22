package cudaview

import (
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"strings"
	"testing"
)

func TestLargeListing(t *testing.T) {
	lines := make([]PipelineLine, 150000)
	for i := range lines {
		lines[i] = PipelineLine{Num: i + 1, Text: fmt.Sprintf("/*%08x*/ MOV R1, R2; // line %d", i*16, i+1), Offset: fmt.Sprintf("%08x", i*16), Opcode: "MOV"}
	}
	lines[len(lines)-1].Text += " OFFSCREEN_SENTINEL </script><script>throw new Error('injection')</script>"
	data := &PipelineViewData{Title: "150,000-line fixture", Tiers: []PipelineTier{{Name: "SASS", FileName: "large.sass", Language: "sass", Lines: lines}}}
	var b bytes.Buffer
	if err := Render(&b, data); err != nil {
		t.Fatal(err)
	}
	if strings.Contains(b.String(), `<div class="code-row"`) {
		t.Fatal("server rendered instruction rows")
	}
	if strings.Contains(b.String(), "</script><script>throw") {
		t.Fatal("unescaped source in script")
	}
	if path := os.Getenv("CUDAVIEW_FIXTURE"); path != "" {
		if err := os.WriteFile(path, b.Bytes(), 0600); err != nil {
			t.Fatal(err)
		}
	}
}

func TestSharedSnapshot(t *testing.T) {
	line := PipelineLine{Num: 1, Text: "unique source", HoverDoc: "shared documentation"}
	tier := PipelineTier{Name: "SASS", Lines: []PipelineLine{line}}
	data := &PipelineViewData{Tiers: []PipelineTier{tier}, KernelTiers: map[string][]PipelineTier{"k": {tier}}}
	s := makeSnapshot(data)
	if len(s.Lines) != 1 {
		t.Fatalf("%d annotated lines, want 1", len(s.Lines))
	}
	if len(s.Tiers[0].Ranges) != 1 || s.Tiers[0].Ranges[0] != [2]int{0, 1} {
		t.Fatalf("ranges: %v", s.Tiers[0].Ranges)
	}
	var b bytes.Buffer
	if err := Render(&b, data); err != nil {
		t.Fatal(err)
	}
	if strings.Count(b.String(), "unique source") != 1 {
		t.Fatal("source repeated")
	}
}

func TestBrowserFixtures(t *testing.T) {
	dir := os.Getenv("CUDAVIEW_BROWSER_FIXTURES")
	if dir == "" {
		return
	}
	makeData := func(label string) *PipelineViewData {
		data := &PipelineViewData{Title: label, KernelTiers: make(map[string][]PipelineTier)}
		for _, name := range []string{"Go", "CUDA C++", "Tile IR", "PTX", "SASS"} {
			tier := PipelineTier{Name: name, FileName: label + ".txt", Language: "sass"}
			for i := range 300 {
				tier.Lines = append(tier.Lines, PipelineLine{Num: i + 1, Text: fmt.Sprintf("%s %s %d", label, name, i+1), Opcode: "MOV"})
			}
			data.Tiers = append(data.Tiers, tier)
		}
		data.Kernels = []string{label}
		data.KernelTiers[label] = data.Tiers
		return data
	}
	a, b := makeData("first"), makeData("second")
	a.Modules = []PipelineModule{{ID: "m0", Name: "first", Data: a}, {ID: "m1", Name: "second", Data: b}}
	a.Recompile = []*RecompileSource{{ID: "m0", Name: "first", File: "first.cu", Options: []string{"--fmad=false"}, Suggested: []string{"--use_fast_math"}}, {ID: "m1", Name: "second", File: "second.cu", Options: []string{"--gpu-architecture=compute_80"}, Suggested: []string{"--gpu-architecture=compute_80"}}}
	var out bytes.Buffer
	if err := Render(&out, a); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(dir+"/live-fixture.html", out.Bytes(), 0600); err != nil {
		t.Fatal(err)
	}
	diff := &DiffViewData{Title: "Large diff", ActiveKernel: "k", Kernels: []string{"k"}, KernelRows: make(map[string][]DiffRow)}
	for i := range 150000 {
		diff.ActiveRows = append(diff.ActiveRows, DiffRow{Num: i + 1, Type: DiffRowMatch, Base: &DiffInstruction{Text: fmt.Sprintf("base %d", i+1)}, Candidate: &DiffInstruction{Text: fmt.Sprintf("target %d", i+1)}})
	}
	diff.KernelRows["k"] = diff.ActiveRows
	out.Reset()
	if err := RenderDiff(&out, diff); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(dir+"/diff-fixture.html", out.Bytes(), 0600); err != nil {
		t.Fatal(err)
	}
}

func TestSnapshotOmitsEmptyFields(t *testing.T) {
	data := &PipelineViewData{Tiers: []PipelineTier{{Name: "SASS", Lines: []PipelineLine{{Num: 1}, {Num: 2, Text: "EXIT ;", Kernel: "k"}}}}}
	var b bytes.Buffer
	if err := WriteSnapshot(&b, data); err != nil {
		t.Fatal(err)
	}
	var s struct {
		Sources []string         `json:"sources"`
		Lines   []map[string]any `json:"lines"`
	}
	if err := json.Unmarshal(b.Bytes(), &s); err != nil {
		t.Fatal(err)
	}
	if len(s.Sources) == 0 || s.Sources[0] != "" {
		t.Fatalf("sources[0] = %q, want the empty string an omitted index stands for", s.Sources)
	}
	if len(s.Lines[0]) != 1 {
		t.Errorf("empty line encoded as %v, want only num", s.Lines[0])
	}
	if got := s.Sources[int(s.Lines[1]["kernel_ref"].(float64))]; got != "k" {
		t.Errorf("kernel = %q, want k", got)
	}
}
