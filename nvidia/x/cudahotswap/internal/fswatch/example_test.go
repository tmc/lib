package fswatch_test

import (
	"context"
	"fmt"

	"github.com/tmc/lib/nvidia/x/cudahotswap/internal/fswatch"
)

func ExampleWatch() {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	err := fswatch.Watch(ctx, []string{"."}, fswatch.Config{}, func(change fswatch.Change) error {
		fmt.Println(change.Path, change.Removed)
		return nil
	})
	fmt.Println(err)
	// Output: context canceled
}
