package cudahotswap

import (
	"context"
	"fmt"
)

func ExampleChecker() {
	check := Checker(func(ctx context.Context, state State) error {
		// Run the application and compare its outputs here.
		fmt.Println(state.Name, state.Generation)
		return ctx.Err()
	})
	fmt.Println(check(context.Background(), State{Name: "kernel", Generation: 2}))
	// Output:
	// kernel 2
	// <nil>
}

func ExampleCheckConfig() {
	c := CheckConfig{Checker: func(context.Context, State) error { return nil }}
	fmt.Println(c.Fixture == nil, c.KeepOnFailure)
	// Output: true false
}

func ExampleService_SetCheck() {
	// A fake loader keeps this settings example independent of a GPU.
	s, _ := fixtureExampleService()
	defer s.Close(context.Background())
	c := CheckConfig{Checker: func(context.Context, State) error { return nil }}
	state, err := s.SetCheck("kernel", c, "application check")
	fmt.Println(state.Generation, state.ConfigRevision, err)
	// Output: 1 1 <nil>
}
