package qualify_test

import (
	"crypto/sha256"
	"fmt"

	"github.com/tmc/lib/nvidia/x/egraph/qualify"
	"github.com/tmc/lib/nvidia/x/pairedtiming"
)

func ExampleSelectMeasured() {
	// Synthetic CPU example only. A production verifier must authenticate the
	// exact admitted source/recipe/case/device/boundary and observed blocks.
	id := qualify.MeasuredIdentity{Context: sha256.Sum256([]byte("synthetic context")), Admission: sha256.Sum256([]byte("synthetic admission"))}
	block := pairedtiming.Block{A1: 10, B1: 20, B2: 20, A2: 10}
	pair := qualify.MeasuredPair{Identity: id, Reference: "reference", Candidate: "candidate", Blocks: []pairedtiming.Block{block, block, block}}
	selected, err := qualify.SelectMeasured(id, []qualify.MeasuredPair{pair}, func(qualify.MeasuredPair) error { return nil })
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(selected.Selected.Name, selected.CandidateState, selected.Observed.Ratio)
	// Output: reference MEASURED_LOSS 0.5
}
