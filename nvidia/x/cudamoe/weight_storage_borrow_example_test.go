package cudamoe_test

import (
	"fmt"
	"github.com/tmc/lib/nvidia/x/cudamoe"
)

func ExampleMoEWeightView_Borrow() {
	var view *cudamoe.MoEWeightView
	_, release, err := view.Borrow(nil)
	fmt.Println(release == nil, err != nil)
	// Output: true true
}
