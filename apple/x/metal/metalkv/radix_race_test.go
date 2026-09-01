// SPDX-License-Identifier: Apache-2.0

package metalkv

import (
	"reflect"
	"sync"
	"testing"
)

// TestMatchPrefixConcurrent runs lookups from several goroutines while the tree
// stamps the nodes it walks for the eviction order. Each lookup must return the
// blocks its own sequence was stored with, not another's.
func TestMatchPrefixConcurrent(t *testing.T) {
	tree := NewRadixTree(64, 2)
	if err := tree.Insert([]int32{1, 2, 3, 4}, []int{10, 11}); err != nil {
		t.Fatalf("Insert: %v", err)
	}
	if err := tree.Insert([]int32{5, 6, 7, 8}, []int{12, 13}); err != nil {
		t.Fatalf("Insert: %v", err)
	}

	var wg sync.WaitGroup
	for i := 0; i < 8; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			tokens := []int32{1, 2, 3, 4}
			want := []int{10, 11}
			if i%2 == 1 {
				tokens = []int32{5, 6, 7, 8}
				want = []int{12, 13}
			}
			for j := 0; j < 200; j++ {
				n, blocks, _ := tree.MatchPrefix(tokens)
				if n != 4 || !reflect.DeepEqual(blocks, want) {
					t.Errorf("MatchPrefix(%v) = %d tokens, blocks %v; want 4 tokens, blocks %v",
						tokens, n, blocks, want)
					return
				}
			}
		}(i)
	}
	wg.Wait()
}
