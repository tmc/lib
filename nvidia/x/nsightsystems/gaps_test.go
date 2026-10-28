package nsightsystems

import (
	"encoding/json"
	"slices"
	"strings"
	"testing"
)

func TestAnalyzeLaunchGaps(t *testing.T) {
	tests := []struct {
		name string
		step []Span
		want LaunchGaps
	}{
		{"empty", nil, LaunchGaps{}},
		{
			"chain",
			[]Span{{"a", 0, 10, 7}, {"b", 13, 20, 7}, {"a", 25, 30, 7}},
			LaunchGaps{Launches: 3, SpanNS: 30, BusyNS: 22, IdleNS: 8, MaxGapNS: 5, Kernels: []KernelGaps{
				{"a", 2, 15, 5}, {"b", 1, 7, 3},
			}},
		},
		{
			// b starts early under PDL and waits inside its first CTAs.
			"overlap",
			[]Span{{"a", 0, 10, 7}, {"b", 8, 20, 7}, {"c", 21, 22, 7}},
			LaunchGaps{Launches: 3, SpanNS: 22, BusyNS: 21, IdleNS: 1, MaxGapNS: 1, Kernels: []KernelGaps{
				{"c", 1, 1, 1}, {"a", 1, 10, 0}, {"b", 1, 12, 0},
			}},
		},
		{
			// b ends inside a, so c's gap is measured from a's end.
			"nested",
			[]Span{{"a", 0, 10, 7}, {"b", 2, 4, 8}, {"c", 12, 14, 7}},
			LaunchGaps{Launches: 3, SpanNS: 14, BusyNS: 12, IdleNS: 2, MaxGapNS: 2, Kernels: []KernelGaps{
				{"c", 1, 2, 2}, {"a", 1, 10, 0}, {"b", 1, 2, 0},
			}},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := AnalyzeLaunchGaps(tt.step)
			gj, _ := json.Marshal(got)
			wj, _ := json.Marshal(tt.want)
			if string(gj) != string(wj) {
				t.Errorf("AnalyzeLaunchGaps =\n%s\nwant\n%s", gj, wj)
			}
		})
	}
}

func TestProject(t *testing.T) {
	g := AnalyzeLaunchGaps([]Span{{"a", 0, 1000, 7}, {"b", 1004, 2000, 7}, {"c", 2006, 3000, 7}})
	if got := g.Project(500); got != 510 {
		t.Errorf("Project(500) = %d, want 510", got)
	}
	if got := g.GapPerLaunchNS(); got != 5 {
		t.Errorf("GapPerLaunchNS = %v, want 5", got)
	}
}

func TestSplitSteps(t *testing.T) {
	ks := []Span{{Name: "x"}, {Name: "argmax"}, {Name: "y"}, {Name: "z"}, {Name: "argmax_logits"}, {Name: "tail"}}
	steps := SplitSteps(ks, "argmax")
	var got []int
	for _, s := range steps {
		got = append(got, len(s))
	}
	if !slices.Equal(got, []int{2, 3}) {
		t.Errorf("step sizes = %v, want [2 3]", got)
	}
}

// gpuTraceCSV is the shape nsys 2026.1 writes for cuda_gpu_trace, with the
// notice it can print before the header.
const gpuTraceCSV = `NOTICE: Existing SQLite export found: x.sqlite
Start (ns),Duration (ns),CorrId,GrdX,GrdY,GrdZ,BlkX,BlkY,BlkZ,Reg/Trd,StcSMem (MB),DymSMem (MB),Bytes (MB),Throughput (MB/s),SrcMemKd,DstMemKd,Device,Ctx,GreenCtx,Strm,Name
300,50,62,48,1,1,256,1,1,40,0.000,0.000,,,,,NVIDIA GB10 (0),1,,7,"llama_attn_in_k0<1>(int, const float *)"
100,20,50,,,,,,,,,,1050.673,58837.697,Pageable,Device,NVIDIA GB10 (0),1,,7,[CUDA memcpy Host-to-Device]
200,40,61,1,1,1,256,1,1,16,0.000,0.000,,,,,NVIDIA GB10 (0),1,,7,EmbeddingLookupBF16
`

func TestReadGPUTraceCSV(t *testing.T) {
	got, err := ReadGPUTraceCSV(strings.NewReader(gpuTraceCSV))
	if err != nil {
		t.Fatal(err)
	}
	want := []Span{{"EmbeddingLookupBF16", 200, 240, 7}, {"llama_attn_in_k0<1>(int, const float *)", 300, 350, 7}}
	if !slices.Equal(got, want) {
		t.Errorf("ReadGPUTraceCSV = %v, want %v", got, want)
	}
}

func TestReadTraceCSVErrors(t *testing.T) {
	for _, in := range []string{
		"",
		"Start (ns),Name\n1,a\n",
		"Start (ns),Duration (ns),Name\n1,-2,a\n",
		"Start (ns),Duration (ns),Name\nx,2,a\n",
	} {
		if _, err := ReadAPITraceCSV(strings.NewReader(in)); err == nil {
			t.Errorf("ReadAPITraceCSV(%q) succeeded", in)
		}
	}
}

func TestAPILaunches(t *testing.T) {
	api := []Span{{"cuLaunchKernel", 0, 3, -1}, {"cuLaunchKernelEx", 5, 9, -1}, {"cuGraphLaunch", 10, 30, -1}, {"cuStreamSynchronize", 30, 90, -1}}
	n, ns := APILaunches(api, "cuLaunchKernel")
	if n != 2 || ns != 7 {
		t.Errorf("APILaunches = %d, %d, want 2, 7", n, ns)
	}
}

func TestKernelSpans(t *testing.T) {
	end := int64(30)
	r := &Report{Events: []Event{
		{Table: "CUPTI_ACTIVITY_KIND_RUNTIME", StartNS: 1},
		{Table: "CUPTI_ACTIVITY_KIND_KERNEL", StartNS: 20, EndNS: &end, Name: "k", Raw: json.RawMessage(`{"streamId":13}`)},
	}}
	got, err := KernelSpans(r)
	if err != nil {
		t.Fatal(err)
	}
	if want := []Span{{"k", 20, 30, 13}}; !slices.Equal(got, want) {
		t.Errorf("KernelSpans = %v, want %v", got, want)
	}
}
