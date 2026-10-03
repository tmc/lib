package advise_test

import (
	"fmt"
	"github.com/tmc/lib/nvidia/x/cudaevidence"
	"github.com/tmc/lib/nvidia/x/cudaprof/advise"
)

func ExampleItem_WithShareEvidence() {
	item, err := (advise.Item{}).WithShareEvidence(cudaevidence.Source{}, false, "")
	fmt.Println(item.ClassifiedShare.Record().Reason, err)
	// Output: profile share unavailable <nil>
}
