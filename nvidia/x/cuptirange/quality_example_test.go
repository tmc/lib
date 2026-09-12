package cuptirange_test

import (
	"fmt"

	"github.com/tmc/lib/nvidia/x/cuptirange"
)

func ExampleCollectionQuality() {
	var q cuptirange.CollectionQuality
	if !q.Decoded {
		fmt.Println("range loss not observed")
	}
	// Output: range loss not observed
}
