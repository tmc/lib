// SPDX-License-Identifier: Apache-2.0

package kvcache_test

import (
	"context"
	"fmt"

	"github.com/tmc/lib/lm/kvcache"
)

func ExampleRadixTree() {
	tree := kvcache.NewRadixTree(100)
	tree.Insert([]int32{1, 2, 3, 4}, []int{10, 11})

	matched, blocks, _ := tree.MatchPrefix([]int32{1, 2, 3, 4, 5})
	fmt.Printf("Matched: %d, Blocks: %v\n", matched, blocks)
	// Output:
	// Matched: 4, Blocks: [10 11]
}

func ExampleStore_Reserve() {
	store, _ := kvcache.NewStore(1)
	lease, _ := store.Allocate(kvcache.Spec{
		Owner:  "request-1",
		Tokens: kvcache.Range{Start: 0, End: 16},
		Format: kvcache.Format{DType: "f16", Layout: "paged", KVHeads: 8, HeadDim: 64},
	}, kvcache.Residency{Backend: "cuda", Tier: "device", Domain: "gpu0"})
	reservation, _ := store.Reserve(lease)
	_ = reservation.Migrate(context.Background(), kvcache.Residency{Backend: "cuda", Tier: "host", Domain: "gpu0"}, func(context.Context, kvcache.Descriptor, kvcache.Residency) error {
		return nil
	})
	desc, _ := store.Describe(lease)
	fmt.Println(desc.Residency.Tier)
	// Output:
	// host
}
