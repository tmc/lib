package llamamodel_test

import (
	"fmt"
	"log"

	"github.com/tmc/lib/examples/llama-infer/internal/llamamodel"
)

func ExampleLoad() {
	arch, _, err := llamamodel.Load("testdata/llama-3.2-1b")
	if err != nil {
		log.Fatal(err)
	}
	l := arch.Layers[0]
	fmt.Println(arch.Family, len(arch.Layers), "layers")
	fmt.Println(l.Heads, "heads,", l.KVHeads, "kv heads,", l.HeadDim, "per head")
	fmt.Println(arch.Norm, arch.NormEps, arch.MLP, arch.Activation)
	fmt.Println("eos", arch.EOS)
	// Output:
	// llama 16 layers
	// 32 heads, 8 kv heads, 64 per head
	// rmsnorm 1e-05 swiglu silu
	// eos [128001 128008 128009]
}
