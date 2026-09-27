package zeroversion

import (
	"fmt"

	"github.com/tmc/lib/nvidia/cutensor"
)

func old() bool {
	v, _ := cutensor.GetVersion() // want `the error of GetVersion is discarded, but v is used; when the call fails it is the zero value, not a version`
	return v < 20000
}

func show() {
	var v uint
	v, _ = cutensor.GetVersion() // want `the error of GetVersion is discarded, but v is used; when the call fails it is the zero value, not a version`
	fmt.Println("cutensor", v)
}

func checked() (bool, error) {
	v, err := cutensor.GetVersion()
	if err != nil {
		return false, err
	}
	fmt.Println("cutensor", v)
	return v >= 20000, nil
}

// probe asks only whether the call succeeds.
func probe() bool {
	_, err := cutensor.GetVersion()
	return err == nil
}
