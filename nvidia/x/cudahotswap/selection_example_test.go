package cudahotswap

import (
	"context"
	"fmt"
)

func ExampleVersion() {
	v := Version{Number: 7, Current: true}
	fmt.Println(v.Number, v.Current)
	// Output: 7 true
}

func ExampleService_Versions() {
	s := settingsExampleService() // Host fake loader and lease.
	defer s.Close(context.Background())
	versions, err := s.Versions("kernel")
	fmt.Println(len(versions), versions[0].Original, err)
	// Output: 1 true <nil>
}

func ExampleService_Select() {
	s := settingsExampleService()
	defer s.Close(context.Background())
	state, err := s.Select(context.Background(), "kernel", 1, "retained code")
	fmt.Println(state.Version, state.Generation, state.Label, err)
	// Output: 1 2 retained code <nil>
}

func ExampleService_Revert() {
	s := settingsExampleService()
	defer s.Close(context.Background())
	state, err := s.Revert(context.Background(), "kernel", "original")
	fmt.Println(state.Version, state.Generation, state.Label, err)
	// Output: 1 2 original <nil>
}
