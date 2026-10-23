package cudaplan_test

import (
	"context"
	"os"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudalock"
)

func requireNativePlanDevice(t *testing.T) (*cuda.Device, *workflowNativeOwners) {
	t.Helper()
	owners := &workflowNativeOwners{}
	t.Cleanup(func() {
		if err := closeNativePlanOwners(owners); err != nil {
			t.Error(err)
		}
	})
	var dev *cuda.Device
	err := setupNativePlanOwners(owners,
		func() error { _, err := cudalock.Inherited(os.Getenv); return err },
		func() (func() error, error) {
			ctx, cancel := context.WithTimeout(context.Background(), time.Minute)
			defer cancel()
			lease, err := cudalock.Acquire(ctx, cudalock.WithLabel(t.Name()))
			if err != nil {
				return nil, err
			}
			return lease.Release, nil
		},
		func() (func() error, error) {
			var err error
			dev, err = cuda.NewDevice(0)
			if err != nil {
				return nil, err
			}
			return dev.Close, nil
		},
	)
	if err != nil {
		t.Fatalf("enabled native plan test requires an owned CUDA device: %v", err)
	}
	return dev, owners
}
