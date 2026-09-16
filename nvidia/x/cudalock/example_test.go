package cudalock_test

import (
	"context"
	"fmt"
	"os"
	"path/filepath"

	"github.com/tmc/lib/nvidia/x/cudalock"
)

func Example() {
	tmpDir, err := os.MkdirTemp("", "cudalock-example-*")
	if err != nil {
		panic(err)
	}
	defer os.RemoveAll(tmpDir)

	lockDir := filepath.Join(tmpDir, "lock")

	// Acquire exclusive lock
	ctx := context.Background()
	handle, err := cudalock.Acquire(ctx, cudalock.WithDir(lockDir), cudalock.WithLabel("example-job"))
	if err != nil {
		panic(err)
	}

	fmt.Printf("Lock acquired: %v\n", handle.Owner().Label)

	// Release lock
	if err := handle.Release(); err != nil {
		panic(err)
	}
	fmt.Println("Lock released")

	// Output:
	// Lock acquired: example-job
	// Lock released
}
