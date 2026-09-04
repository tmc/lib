package metal_test

import (
	"fmt"

	"github.com/tmc/apple/objectivec"
	"github.com/tmc/lib/apple/x/metal"
)

func ExampleDevice_AllocBuffer() {
	dev, err := metal.NewDevice()
	if err != nil {
		fmt.Println(err)
		return
	}
	defer dev.Close()

	buf, err := dev.AllocBuffer(16)
	if err != nil {
		fmt.Println(err)
		return
	}
	defer objectivec.ObjectFromID(buf.GetID()).Release()

	fmt.Println(buf.Length())
	// Output: 16
}
