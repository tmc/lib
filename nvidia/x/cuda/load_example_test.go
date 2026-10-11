package cuda_test

import (
	"fmt"

	"github.com/tmc/lib/nvidia/x/cuda"
)

func ExampleLoadModule() {
	// An empty image is refused before acquiring a device context.
	_, err := cuda.LoadModule(nil, nil)
	fmt.Println(err)
	// Output: cuda: empty module image
}
