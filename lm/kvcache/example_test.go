// SPDX-License-Identifier: Apache-2.0

package kvcache_test

import (
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
