package cudasearch

import (
	"context"
	"fmt"
)

func Example_ownedOutputs() {
	x := &prefillExecutable{}
	values, owner, err := x.outputsOwnedWith(context.Background(), outputReadOps{
		run:        func(context.Context) error { return nil },
		length:     func(int) (int, error) { return 1, nil },
		readFloats: func(_ int, data []float32) error { data[0] = 42; return nil },
	})
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(values["out"])
	owner.Close() // after comparison; the borrowed executable stays open
	// Output: [42]
}
