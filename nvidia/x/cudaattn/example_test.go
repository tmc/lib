package cudaattn_test

import (
	"fmt"
	"log"

	"github.com/tmc/lib/nvidia/x/cudaattn"
)

func ExampleDecodeSharedMemoryBytes() {
	cfg := cudaattn.DefaultConfig(32, 8, 128)
	if err := cfg.Validate(); err != nil {
		log.Fatal(err)
	}
	// One block per query head, 128 threads, 4096 cached tokens.
	shared, err := cudaattn.DecodeSharedMemoryBytes(4096, 128)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(cfg.NumQHeads, "blocks,", shared, "bytes of shared memory each")
	// Output: 32 blocks, 16896 bytes of shared memory each
}

// Node 1 of a two-node candidate tree attends to its parent, node 0, and to
// itself; node 0 attends only to itself.
func ExampleTreeAttentionReference() {
	q := []float32{1, 0, 1, 0}
	k := []float32{1, 0, 1, 0}
	v := []float32{2, 4, 6, 8}
	masks := []uint64{0b01, 0b11}
	out, err := cudaattn.TreeAttentionReference(q, k, v, masks, 2, 1, 2, 1)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(out)
	// Output: [2 4 4 6]
}
