package cuda_test

import (
	"context"
	"fmt"
	"time"

	"github.com/tmc/lib/nvidia/x/cuda"
)

func Example() {
	if err := cuda.Init(); err != nil {
		fmt.Printf("CUDA not available: %v\n", err)
		return
	}

	devs, err := cuda.Devices()
	if err != nil || len(devs) == 0 {
		fmt.Println("No CUDA devices found")
		return
	}

	dev := devs[0]
	stream, err := dev.NewStream()
	if err != nil {
		fmt.Printf("Failed to create stream: %v\n", err)
		return
	}
	defer stream.Destroy()

	event, err := dev.NewEvent()
	if err != nil {
		fmt.Printf("Failed to create event: %v\n", err)
		return
	}
	defer event.Destroy()

	_ = event.Record(stream)

	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()

	if err := stream.Synchronize(ctx); err != nil {
		fmt.Printf("Stream sync failed: %v\n", err)
	}
}
