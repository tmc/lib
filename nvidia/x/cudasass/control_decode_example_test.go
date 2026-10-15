package cudasass_test

import (
	"errors"
	"fmt"

	"github.com/tmc/lib/nvidia/x/cudasass"
)

func ExampleDecodeControl() {
	control, err := cudasass.DecodeControl(cudasass.Encoding{Lo: 1}, cudasass.SM121a)
	fmt.Println(control == (cudasass.ControlCode{}), err == nil)
	_, err = cudasass.DecodeControl(cudasass.Encoding{}, cudasass.SM121a)
	var refusal *cudasass.ControlDecodeError
	if errors.As(err, &refusal) {
		fmt.Println(refusal.Arch, refusal.MissingEncoding)
	}
	// Output:
	// true true
	// sm_121a true
}

func ExampleControlDecodeError() {
	_, err := cudasass.DecodeControl(cudasass.Encoding{Lo: 1}, "sm_90")
	var refusal *cudasass.ControlDecodeError
	if errors.As(err, &refusal) {
		fmt.Println(refusal.Arch, refusal.MissingEncoding)
	}
	// Output:
	// sm_90 false
}
