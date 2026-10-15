package cudakv_test

import (
	"fmt"

	"github.com/tmc/lib/nvidia/x/cudakv"
)

func ExampleNewDeviceCommandRing() {
	ring, err := cudakv.NewDeviceCommandRing(2, 7)
	if err != nil {
		fmt.Println(err)
		return
	}
	if err := ring.Publish(cudakv.Descriptor{SeqID: 42, Generation: 7, PromptToken: 9}); err != nil {
		fmt.Println(err)
		return
	}
	ring.Close()
	desc, ok := ring.Consume()
	fmt.Println(desc.SeqID, desc.Generation, desc.PromptToken, ok, ring.Len())
	// Output: 42 7 9 true 0
}
