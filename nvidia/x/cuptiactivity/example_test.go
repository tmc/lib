package cuptiactivity_test

import (
	"fmt"

	"github.com/tmc/lib/nvidia/x/cuptiactivity"
)

func ExampleRecord() {
	r := cuptiactivity.Record{Start: 10, End: 25, Complete: true}
	if r.Complete {
		fmt.Println(r.End - r.Start)
	}
	// Output: 15
}
