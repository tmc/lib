package main

import (
	"testing"

	"github.com/tmc/lib/nvidia/x/cudasample"
)

func TestSymbolicPlan(t *testing.T) {
	plan := symbolicPlan(DefaultTinyLlamaConfig())
	if plan.TotalBytes == 0 || len(plan.Offsets) == 0 {
		t.Fatalf("empty plan: %#v", plan)
	}
}

func TestSyntheticLogits(t *testing.T) {
	for _, test := range []struct {
		step int
		want int32
	}{
		{step: 0, want: 7},
		{step: 1, want: 24},
		{step: 4, want: 75},
	} {
		logits, want := syntheticLogits(256, test.step)
		if want != test.want {
			t.Errorf("step %d: token = %d, want %d", test.step, want, test.want)
		}
		if got := cudasample.SampleArgmaxCPU(logits); got != want {
			t.Errorf("step %d: argmax = %d, want %d", test.step, got, want)
		}
	}
}

func TestCheckAttention(t *testing.T) {
	if err := checkAttention([]float32{2, 2}, 2); err != nil {
		t.Fatal(err)
	}
	if err := checkAttention([]float32{2, 3}, 2); err == nil {
		t.Fatal("checkAttention accepted a wrong value")
	}
}

func TestByteTokenizer(t *testing.T) {
	tok := NewByteTokenizer()
	prompt := "Hello LLaMA"

	tokens, err := tok.Encode(prompt)
	if err != nil {
		t.Fatalf("Encode failed: %v", err)
	}

	decoded, err := tok.Decode(tokens)
	if err != nil {
		t.Fatalf("Decode failed: %v", err)
	}

	if decoded != prompt {
		t.Errorf("expected %q, got %q", prompt, decoded)
	}
}

