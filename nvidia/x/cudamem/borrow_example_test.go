package cudamem_test

import (
	"fmt"
	"github.com/tmc/lib/nvidia/x/cudamem"
)

func ExampleSlice_Device() {
	var s cudamem.Slice[float32]
	fmt.Println(s.Device() == nil)
	// Output: true
}

func ExampleSlice_Borrow() {
	var s cudamem.Slice[float32]
	_, err := s.Borrow(nil)
	fmt.Println(err != nil)
	// Output: true
}
