package cuptirange_test

import (
	"fmt"
	"github.com/tmc/lib/nvidia/x/cuptirange"
)

func ExampleReplayMode_String() {
	fmt.Println(cuptirange.UserReplay)
	fmt.Println(cuptirange.KernelReplay)
	// Output:
	// user
	// kernel
}

func ExampleReplayModeSupport() {
	fmt.Println(cuptirange.ReplayModeSupport(130202, cuptirange.KernelReplay))
	// Output: <nil>
}
