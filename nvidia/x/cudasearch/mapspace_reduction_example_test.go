package cudasearch_test

import (
	"context"
	"fmt"
)

func ExampleMapSpaceSearchResult_TotalDRAM() {
	search := reductionSearch(1, 64)
	search.SpatialTilings = search.SpatialTilings[:1]
	stats, err := search.Search(context.Background())
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(stats.Best.Movement.DRAM.TotalBytes, stats.Best.TotalDRAM().TotalBytes)
	// Output: 6 9
}

func ExampleLinearPrunedSearch_Search_splitK() {
	search := reductionSearch(1, 64)
	search.SplitK = true // explicit tagged partials plus direct-DRAM binary tree
	stats, err := search.Search(context.Background())
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(stats.EvaluatedCandidates, stats.Best.SplitK == nil, stats.Best.TotalDRAM().TotalBytes)
	// Output: 2 true 7
}
