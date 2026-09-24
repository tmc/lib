package cudaliveness_test

import (
	"errors"
	"fmt"

	"github.com/tmc/lib/nvidia/x/cudaliveness"
)

// A patch is shown to execute by a derived prediction, a changed output
// against the unpatched image, and no stale copy of the text.
func ExampleExecuted() {
	err := cudaliveness.Executed{
		Site:      "intmix@0x10",
		Predicted: "9f2c",
		Why:       "S2R R7, SR_TID.Y replaces SR_TID.X: out[i] changes where tid.y != tid.x",
		Unpatched: "41a7",
		Patched:   "9f2c",
		Ran:       true,
	}.Check()
	fmt.Println(err)
	// Output: <nil>
}

// An image that keeps an unpatched copy of the text fails, even when both
// outputs agree with the prediction.
func ExampleExecuted_mirrored() {
	err := cudaliveness.Executed{
		Site:      "intmix@0x10",
		Predicted: "9f2c",
		Why:       "S2R R7, SR_TID.Y replaces SR_TID.X: out[i] changes where tid.y != tid.x",
		Unpatched: "41a7",
		Patched:   "9f2c",
		Mirrors:   []string{".nv.capmerc.text.intmix"},
		Ran:       true,
	}.Check()
	fmt.Println(errors.Is(err, cudaliveness.ErrNotLive))
	fmt.Println(err)
	// Output:
	// true
	// patch execution at intmix@0x10: the image keeps unpatched copies of the text [".nv.capmerc.text.intmix"], so no run can show which copy a later load will use: cudaliveness: probe was not live
}
