package metalattn_test

import (
	"fmt"

	"github.com/tmc/lib/apple/x/metal/metalattn"
)

func ExampleDefaultConfig() {
	cfg := metalattn.DefaultConfig(8, 2, 64)
	fmt.Printf("q=%d kv=%d dim=%d scale=%.3f\n", cfg.NumQHeads, cfg.NumKVHeads, cfg.HeadDim, cfg.Scale)
	// Output:
	// q=8 kv=2 dim=64 scale=0.125
}

func ExampleValidatePersistentGQA() {
	cfg := metalattn.DefaultConfig(32, 8, 128)
	fmt.Println(metalattn.ValidatePersistentGQA(cfg, 4096))
	// Output:
	// <nil>
}
