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
	// A stream or event returned with an error can still hold a handle,
	// so close any non-nil result and report a failed close.
	stream, err := dev.NewStream()
	if stream != nil {
		defer func() {
			if err := stream.Close(); err != nil {
				fmt.Printf("Failed to close stream: %v\n", err)
			}
		}()
	}
	if err != nil {
		fmt.Printf("Failed to create stream: %v\n", err)
		return
	}

	event, err := dev.NewEvent()
	if event != nil {
		defer func() {
			if err := event.Close(); err != nil {
				fmt.Printf("Failed to close event: %v\n", err)
			}
		}()
	}
	if err != nil {
		fmt.Printf("Failed to create event: %v\n", err)
		return
	}

	_ = event.Record(stream)

	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()

	if err := stream.Synchronize(ctx); err != nil {
		fmt.Printf("Stream sync failed: %v\n", err)
	}
}

// A device always has a stream, so callers need not special-case unstreamed work.
func ExampleDevice_DefaultStream() {
	devs, err := cuda.Devices()
	if err != nil || len(devs) == 0 {
		fmt.Println("No CUDA devices found")
		return
	}

	stream := devs[0].DefaultStream()
	if err := stream.Synchronize(context.Background()); err != nil {
		fmt.Printf("Stream sync failed: %v\n", err)
	}
}
