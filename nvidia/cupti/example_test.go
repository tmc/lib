package cupti_test

import (
	"fmt"

	"github.com/tmc/lib/nvidia/cupti"
)

// Example queries the CUPTI version and enables collection of driver API
// activity records. It returns early when the CUPTI library is not
// available.
func Example() {
	version, err := cupti.GetVersion()
	if err != nil {
		fmt.Println("CUPTI unavailable:", err)
		return
	}
	fmt.Println("CUPTI version:", version)

	if err := cupti.ActivityEnable(cupti.ActivityKindDriver); err != nil {
		fmt.Println("activity enable:", err)
		return
	}
	defer cupti.ActivityDisable(cupti.ActivityKindDriver)
	fmt.Println("driver activity records enabled")
}
