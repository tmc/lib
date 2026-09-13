package main

import (
	"context"
	"maps"
	"slices"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudaprof"
)

func TestSampleTypes(t *testing.T) {
	// Without PC sampling there are no stall counters to report, and a type
	// that is always zero is a dead view in every pprof consumer.
	off := sampleTypes(false)
	if slices.Contains(off, "stall_mem") {
		t.Errorf("sampleTypes(false) = %v, want no stall counters", off)
	}
	if !slices.Contains(off, "gpu_time") || !slices.Contains(off, "samples") {
		t.Errorf("sampleTypes(false) = %v, want samples and gpu_time", off)
	}
	// Every type the PC-sampling path writes has to be advertised here:
	// SetSampleTypes is a projection, so a metric classifyStall fills and this
	// list omits is written nowhere and does not show up as unmapped either.
	// pc_samples is separate from samples because samples counts launches.
	on := sampleTypes(true)
	for _, want := range []string{
		"samples", "pc_samples", "inst_executed",
		"stall_mem", "stall_wait", "stall_scoreboard", "stall_math",
		"stall_branch", "stall_fetch", "stall_drain", "stall_dispatch",
		"stall_not_selected",
		"gpu_time",
	} {
		if !slices.Contains(on, want) {
			t.Errorf("sampleTypes(true) = %v, missing %q", on, want)
		}
	}
}

func TestLineOffsetsAndPTXAddress(t *testing.T) {
	// Three lines of two bytes each: offsets are 0, 2, 4, then the sentinel.
	const text = "a\nb\nc\n"
	offsets := cudaprof.PTXLineOffsets(text)
	tests := []struct {
		name string
		line int
		want uint64
	}{
		{name: "first line is the image base", line: 1, want: ptxBase},
		{name: "second line", line: 2, want: ptxBase + 2},
		{name: "third line", line: 3, want: ptxBase + 4},
		{name: "zero line is unmapped", line: 0, want: 0},
		{name: "negative line is unmapped", line: -1, want: 0},
		{name: "past the sentinel is unmapped", line: len(offsets), want: 0},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := ptxAddress(offsets, tt.line); got != tt.want {
				t.Errorf("ptxAddress(offsets, %d) = %#x, want %#x", tt.line, got, tt.want)
			}
		})
	}
}

func TestRewriteSASSSourcePath(t *testing.T) {
	tests := []struct {
		name, in, want string
	}{{
		name: "absolute build path is replaced",
		in:   "\t//## File \"/tmp/mote-2687102363/Volumes/tmc/go/src/x/kernels.cu\", line 12\n",
		want: "\t//## File \"kernels.cu\", line 12\n",
	}, {
		name: "already relative is unchanged",
		in:   "\t//## File \"kernels.cu\", line 3\n",
		want: "\t//## File \"kernels.cu\", line 3\n",
	}, {
		name: "multiple directives across lines",
		in:   "//## File \"/a/b.cu\", line 1\n\tMOV R1, R2 ;\n//## File \"/a/b.cu\", line 2\n",
		want: "//## File \"kernels.cu\", line 1\n\tMOV R1, R2 ;\n//## File \"kernels.cu\", line 2\n",
	}, {
		name: "unterminated quote is left alone",
		in:   "//## File \"/a/b.cu, line 1\n",
		want: "//## File \"/a/b.cu, line 1\n",
	}, {
		name: "instruction lines are untouched",
		in:   "\t/*0000*/ IMAD R2, R3, R4, RZ ;\n",
		want: "\t/*0000*/ IMAD R2, R3, R4, RZ ;\n",
	}, {
		name: "inlined directive rewrites both call site and inlined path",
		in:   "\t//## File \"/home/tmc/kernels.cu\", line 27 inlined at \"/home/tmc/kernels.cu\", line 98\n",
		want: "\t//## File \"kernels.cu\", line 27 inlined at \"kernels.cu\", line 98\n",
	}, {
		name: "ptx directive is preserved",
		in:   "\t//## File \".nv_debug_ptx_txt\", line 1695\n",
		want: "\t//## File \".nv_debug_ptx_txt\", line 1695\n",
	}, {
		name: "no trailing newline",
		in:   "//## File \"/a/b.cu\", line 9",
		want: "//## File \"kernels.cu\", line 9",
	}}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := string(rewriteSASSSourcePath([]byte(tt.in), "kernels.cu")); got != tt.want {
				t.Errorf("rewriteSASSSourcePath(%q) = %q, want %q", tt.in, got, tt.want)
			}
		})
	}
}

// goFrames must report the launch site, not the line that asks for the stack.
// That is the defect it replaces: runtime.Caller(0) named the recorder.
func TestGoFramesReportsTheCaller(t *testing.T) {
	frames := outerForTest()
	if len(frames) == 0 {
		t.Fatal("goFrames returned no frames")
	}
	if got := frames[0].FunctionName; !strings.HasSuffix(got, "outerForTest") {
		t.Errorf("leaf frame = %q, want the caller outerForTest", got)
	}
	for _, f := range frames {
		if strings.Contains(f.FunctionName, "goFrames") {
			t.Errorf("frames include goFrames itself: %+v", frames)
		}
		if strings.ContainsRune(f.FileName, '/') {
			t.Errorf("FileName = %q, want a base name with no directory", f.FileName)
		}
		if f.Line <= 0 {
			t.Errorf("frame %q has line %d, want a positive line", f.FunctionName, f.Line)
		}
	}
}

// outerForTest exists so the test has a known caller frame to find.
func outerForTest() []cudaprof.Frame { return goFrames(0) }

// The label plumbing is three frames deep and appears once per cudaprof.Do on
// the stack, so a nested launch shows it twice. Eliding it the way the runtime
// elides its own leaves only frames the reader wrote.
func TestGoFramesElidesLabelPlumbing(t *testing.T) {
	frames := throughLabelledDo(t)
	if len(frames) == 0 {
		t.Fatal("goFrames returned no frames")
	}
	for _, f := range frames {
		if strings.Contains(f.FunctionName, "cudaprof.Do") || strings.Contains(f.FunctionName, "runtime/pprof.Do") {
			t.Errorf("frames include label plumbing %q: %+v", f.FunctionName, frames)
		}
	}
	var sawInner bool
	for _, f := range frames {
		if strings.HasSuffix(f.FunctionName, "throughLabelledDo") {
			sawInner = true
		}
	}
	if !sawInner {
		t.Errorf("frames lost the caller under Do: %+v", frames)
	}
}

// throughLabelledDo takes the stack from inside nested cudaprof.Do calls, the
// shape capture.go produces: Do for the run-wide labels, Do again per kernel.
func throughLabelledDo(t *testing.T) []cudaprof.Frame {
	t.Helper()
	var frames []cudaprof.Frame
	cudaprof.Do(context.Background(), cudaprof.Labels("phase", "test"), func(ctx context.Context) {
		cudaprof.Do(ctx, cudaprof.Labels("kernel", "test"), func(ctx context.Context) {
			frames = goFrames(0)
		})
	})
	return frames
}

// PC samples are drained after the workload, so the launch stack has to be
// remembered per kernel. A kernel launched from two different places has no
// single launch site, and one of the two is not attribution.
func TestLaunchRecorderStacks(t *testing.T) {
	fromA := []cudaprof.Frame{{FunctionName: "main.runA", FileName: "workload.go", Line: 10}}
	fromB := []cudaprof.Frame{{FunctionName: "main.runB", FileName: "workload.go", Line: 20}}
	tests := []struct {
		name     string
		observed [][]cudaprof.Frame
		want     []cudaprof.Frame
	}{
		{"never launched", nil, nil},
		{"one site", [][]cudaprof.Frame{fromA}, fromA},
		{"same site twice", [][]cudaprof.Frame{fromA, fromA}, fromA},
		{"two sites", [][]cudaprof.Frame{fromA, fromB}, nil},
		{"two sites then the first again", [][]cudaprof.Frame{fromA, fromB, fromA}, nil},
		{"empty stack ignored", [][]cudaprof.Frame{fromA, {}}, fromA},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			r := newLaunchRecorder(nil, nil)
			for _, frames := range tt.observed {
				r.observe("kernel", frames)
			}
			if got := r.launchStack("kernel"); !slices.Equal(got, tt.want) {
				t.Errorf("launchStack = %v, want %v", got, tt.want)
			}
			if got := r.launchStack("other"); got != nil {
				t.Errorf("launchStack(other) = %v, want nil", got)
			}
		})
	}
}

// A PC sample is an aggregate over every launch of its kernel, so only the
// labels that held for all of them describe it. pass and moe.expert vary across
// launches and must not survive; phase holds and must.
func TestLaunchRecorderLabels(t *testing.T) {
	r := newLaunchRecorder(nil, nil)
	launch := func(pairs ...string) {
		cudaprof.Do(context.Background(), cudaprof.Labels(pairs...), func(ctx context.Context) {
			r.observeLabels(ctx, "kernel")
		})
	}
	launch("kernel", "kernel", "phase", "compute", "pass", "0", "moe.expert", "3")
	launch("kernel", "kernel", "phase", "compute", "pass", "1", "moe.expert", "4")
	got := r.launchLabels("kernel")
	want := map[string]string{"phase": "compute"}
	if !maps.Equal(got, want) {
		t.Errorf("launchLabels = %v, want %v", got, want)
	}
	// A kernel nobody launched has no labels, rather than another kernel's.
	if got := r.launchLabels("other"); got != nil {
		t.Errorf("launchLabels(other) = %v, want nil", got)
	}
	// One launch is not a pattern, but it is also not a disagreement: its labels
	// hold for every launch there was.
	single := newLaunchRecorder(nil, nil)
	cudaprof.Do(context.Background(), cudaprof.Labels("phase", "memory", "pass", "7"), func(ctx context.Context) {
		single.observeLabels(ctx, "vector_add")
	})
	if got, want := single.launchLabels("vector_add"), map[string]string{"phase": "memory", "pass": "7"}; !maps.Equal(got, want) {
		t.Errorf("launchLabels after one launch = %v, want %v", got, want)
	}
}
