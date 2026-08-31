package main

import (
	"testing"

	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudaattn"
	"github.com/tmc/lib/nvidia/x/cudabench"
	"github.com/tmc/lib/nvidia/x/cudalaunch"
	"github.com/tmc/lib/nvidia/x/cudamem"
	"github.com/tmc/lib/nvidia/x/cudasample"
	"github.com/tmc/lib/nvidia/x/cudatest"
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

func BenchmarkByteTokenizerEncode(b *testing.B) {
	tok := NewByteTokenizer()
	prompt := "The quick brown fox jumps over the lazy dog. Antigravity inference engine in pure Go."
	b.SetBytes(int64(len(prompt)))
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		_, _ = tok.Encode(prompt)
	}
}

func BenchmarkByteTokenizerDecode(b *testing.B) {
	tok := NewByteTokenizer()
	tokens := []int32{84, 104, 101, 32, 113, 117, 105, 99, 107, 32, 98, 114, 111, 119, 110}
	b.SetBytes(int64(len(tokens)))
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		_, _ = tok.Decode(tokens)
	}
}

func BenchmarkLlamaDecodeStep(b *testing.B) {
	dev := cudatest.Require(b)
	cudabench.Run(b, dev, func(cb *cudabench.B) {
		major, minor, err := dev.ComputeCapability()
		if err != nil {
			b.Fatalf("ComputeCapability: %v", err)
		}

		attn, err := cudaattn.Compile(major, minor)
		if err != nil {
			b.Fatalf("Compile attn: %v", err)
		}
		attnMod, err := attn.Load(dev)
		if err != nil {
			b.Fatalf("Load attn: %v", err)
		}
		defer attnMod.Unload()
		attnFn, err := attnMod.GetFunction("CausalFlashAttentionDecode")
		if err != nil {
			b.Fatalf("GetFunction: %v", err)
		}

		cfg := DefaultTinyLlamaConfig()
		q, _ := cudamem.Alloc[float32](dev, cfg.NumHeads*cfg.HeadDim)
		defer q.Free()
		k, _ := cudamem.Alloc[float32](dev, cfg.MaxSeqLen*cfg.NumKVHeads*cfg.HeadDim)
		defer k.Free()
		v, _ := cudamem.Alloc[float32](dev, cfg.MaxSeqLen*cfg.NumKVHeads*cfg.HeadDim)
		defer v.Free()
		out, _ := cudamem.Alloc[float32](dev, cfg.NumHeads*cfg.HeadDim)
		defer out.Free()

		attnCfg := cudaattn.DefaultConfig(cfg.NumHeads, cfg.NumKVHeads, cfg.HeadDim)
		cb.SetBytes(int64((cfg.NumHeads + 2*cfg.NumKVHeads) * cfg.HeadDim * 4))

		cb.Time(func(stream *cuda.Stream) {
			_ = cudalaunch.Kernel(attnFn).
				Grid(uint32(cfg.NumHeads)).
				Block(1).
				SharedMem(uint32(cfg.MaxSeqLen * 4)).
				Launch(q, k, v, out, int32(cfg.MaxSeqLen-1), int32(cfg.MaxSeqLen), int32(cfg.NumHeads), int32(cfg.NumKVHeads), int32(cfg.HeadDim), attnCfg.Scale)
		})
	})
}
