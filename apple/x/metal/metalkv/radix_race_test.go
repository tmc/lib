// SPDX-License-Identifier: Apache-2.0

package metalkv

import (
	"sync"
	"testing"
)

// TestMatchPrefixConcurrent runs concurrent lookups, which MatchPrefix admits
// under a read lock while it stamps LastAccess on the nodes it walks.
func TestMatchPrefixConcurrent(t *testing.T) {
	tree := NewRadixTree(64)
	tree.Insert([]int32{1, 2, 3, 4}, []int{10, 11})
	tree.Insert([]int32{5, 6, 7, 8}, []int{12, 13})

	var wg sync.WaitGroup
	for i := 0; i < 8; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			tokens := []int32{1, 2, 3, 4}
			if i%2 == 1 {
				tokens = []int32{5, 6, 7, 8}
			}
			for j := 0; j < 200; j++ {
				if n, _, _ := tree.MatchPrefix(tokens); n != 4 {
					t.Errorf("matched %d tokens, want 4", n)
					return
				}
			}
		}(i)
	}
	wg.Wait()
}
