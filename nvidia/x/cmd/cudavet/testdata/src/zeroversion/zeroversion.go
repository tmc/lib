package zeroversion

import (
	"fmt"

	"github.com/tmc/lib/nvidia/nvtx"
)

func old() bool {
	return nvtx.GetVersion() < 3 // want `nvtx.GetVersion returns 0 when the library is not loaded; check for 0 before comparing it as a version`
}

func show() {
	fmt.Println("nvtx", nvtx.GetVersion()) // want `nvtx.GetVersion returns 0 when the library is not loaded; a printed 0 is not a version`
}

func checkedFirst() bool {
	v := nvtx.GetVersion()
	if v == 0 {
		return false
	}
	fmt.Println("nvtx", v)
	return v >= 3
}
