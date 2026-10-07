package dcgm_test

import (
	"fmt"

	"github.com/tmc/lib/nvidia/dcgm"
)

func ExampleErrorCode() {
	detail := dcgm.DiagErrorDetail{Code: 101}
	fmt.Println(dcgm.ErrorCode(detail.Code))
	fmt.Println(dcgm.ErrorCode(0xffffffff))
	// Output:
	// DCGM_FR_XID_ERROR
	// ErrorCode(4294967295)
}

func ExampleErrorCode_String() {
	fmt.Println(dcgm.ErrorCode(109).String())
	// Output: DCGM_FR_SXID_ERROR
}
