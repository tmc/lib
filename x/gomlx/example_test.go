package gomlx_test

import (
	"fmt"
	"github.com/tmc/lib/x/gomlx"
)

func Example() {
	var b gomlx.Backend
	fmt.Println(b.IsFinalized())
	fmt.Println(b.Close())
	// Output:
	// true
	// <nil>
}

func ExampleNew() {
	_, err := gomlx.New("/no/such/pjrt-plugin.so")
	fmt.Println(err != nil)
	// Output: true
}

func ExampleBackend_Close() {
	var b gomlx.Backend
	fmt.Println(b.Close())
	fmt.Println(b.Close())
	// Output:
	// <nil>
	// <nil>
}

func ExampleBackend_Finalize() {
	var b gomlx.Backend
	b.Finalize()
	fmt.Println(b.IsFinalized())
	// Output: true
}
