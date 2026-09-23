package cudalayer_test

import (
	"fmt"
	"github.com/tmc/lib/nvidia/x/cudalayer"
)

func ExampleProjectionNormConfig() {
	config := cudalayer.ProjectionNormConfig{Dim: 4096, Threads: 256, Epsilon: 1e-5}
	bytes, err := config.SharedMemoryBytes()
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(bytes)
	// Output: 1024
}
