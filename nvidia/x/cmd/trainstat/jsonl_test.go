package main

import (
	"strings"
	"testing"
)

func TestConvert(t *testing.T) {
	tests := []struct {
		name string
		in   string
		want []string // lines that must appear, in order
		not  []string // substrings that must not appear
		err  string
	}{
		{
			name: "steps",
			in: `{"event":"step","step":1,"loss":2.5,"grad_norm":{"head":0.5},"ms":250,"tokens":100}
{"event":"step","step":2,"loss":2,"ms":150,"tokens":100}
{"event":"done"}`,
			want: []string{
				"BenchmarkTrain/step=1 1 2.5 loss 0.5 grad-norm-head 0.25 sec/step 100 tokens",
				"BenchmarkTrain/step=2 1 2 loss 0.15 sec/step 100 tokens",
				"BenchmarkRun 1 2 steps 200 tokens 0.4 train-sec 1 done",
			},
		},
		{
			name: "final is last eval",
			in: `{"event":"eval","step":1,"split":"val","metrics":{"all":{"accuracy":0.25}}}
{"event":"eval","step":2,"split":"val","metrics":{"all":{"accuracy":0.5}}}
{"event":"eval","step":2,"split":"val","metrics":{"all":{"accuracy":0.5}},"calibrated":{"all":{"accuracy":0.5}}}`,
			want: []string{
				"BenchmarkEval/split=val/group=all/cal=raw/step=1 1 0.25 accuracy",
				"BenchmarkEval/split=val/group=all/cal=raw/step=2 1 0.5 accuracy",
				"BenchmarkEval/split=val/group=all/cal=temp/step=2 1 0.5 accuracy",
				"BenchmarkFinal/split=val/group=all/cal=raw 1 0.5 accuracy",
				"BenchmarkFinal/split=val/group=all/cal=temp 1 0.5 accuracy",
				"BenchmarkRun 1 0 done",
			},
			not: []string{"cal=raw 1 0.25"},
		},
		{
			name: "differing duplicate",
			in: `{"event":"eval","step":2,"metrics":{"all":{"accuracy":0.5}}}
{"event":"eval","step":2,"metrics":{"all":{"accuracy":0.25}}}`,
			err: "x: object 2: Eval/group=all/cal=raw/step=2 differs from an earlier evaluation at the same step",
		},
		{
			name: "stop after done",
			in:   `{"event":"done"}` + "\n" + `{"event":"stop"}`,
			want: []string{"BenchmarkRun 1 0 done"},
		},
		{
			name: "no event",
			in:   `{"split":"test","metrics":{"all":{"accuracy":0.5}},"calibrated":{"all":{"ece":0.125}}}`,
			want: []string{
				"BenchmarkFinal/split=test/group=all/cal=raw 1 0.5 accuracy",
				"BenchmarkFinal/split=test/group=all/cal=temp 1 0.125 ece",
			},
			not: []string{"Run"},
		},
		{
			name: "start config",
			in:   `{"event":"start","mode":"lora","lr":{"head":0.001},"step":0}`,
			want: []string{"mode: lora", "head: 0.001", "BenchmarkRun 1 0 done"},
			not:  []string{"step:"},
		},
		{
			name: "step without x",
			in:   `{"event":"step","loss":1}`,
			err:  "x: object 1: step event without step",
		},
		{
			name: "not an object",
			in:   `[1]`,
			err:  "x: object 1: not a JSON object",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var b strings.Builder
			err := newConverter().convert("x", []byte(tt.in), &b)
			if tt.err != "" {
				if err == nil || err.Error() != tt.err {
					t.Fatalf("convert: %v, want error %q", err, tt.err)
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			out := b.String()
			rest := out
			for _, line := range tt.want {
				i := strings.Index(rest, line+"\n")
				if i < 0 {
					t.Fatalf("missing or out of order: %q\noutput:\n%s", line, out)
				}
				rest = rest[i+len(line):]
			}
			for _, s := range tt.not {
				if strings.Contains(out, s) {
					t.Errorf("output contains %q:\n%s", s, out)
				}
			}
		})
	}
}
