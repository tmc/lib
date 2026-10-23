package egraph_test

import (
	"context"
	"testing"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudalock"
	"github.com/tmc/lib/nvidia/x/cudatest"
)

// requireNativeDevice returns CUDA device 0 and the chain that owns it, or
// skips when this machine has no CUDA driver or device.
//
// Only the driver library is loaded before the GPU lease is acquired; cuInit
// and every device call come after. The lease is the bottom entry of the
// chain: it is released only after the device and everything acquired from
// it, and a failed release of any of those keeps it.
func requireNativeDevice(t *testing.T) (*cuda.Device, *nativeOwners) {
	t.Helper()
	return ownedDevice(t,
		func() error { return driver.Load("") },
		func() (func() error, error) {
			h, err := cudalock.Acquire(context.Background(), cudalock.WithLabel(t.Name()))
			if err != nil {
				return nil, err
			}
			return h.Release, nil
		},
		cudatest.RequireDevice,
		(*cuda.Device).Close,
	)
}

// ownedDevice runs the native fixture sequence with the given steps: load
// the driver library or skip, acquire the lease as the bottom of a new
// chain, require the device, and add its release above the lease.
func ownedDevice[D any](t testing.TB, load func() error, acquire func() (func() error, error), require func(testing.TB) D, closeDevice func(D) error) (D, *nativeOwners) {
	t.Helper()
	if err := load(); err != nil {
		t.Skipf("SKIPPED, NOT VERIFIED: the CUDA driver could not be loaded: %v", err)
	}
	owners := leaseOwners(t, acquire)
	dev := require(t)
	owners.add(func() error { return closeDevice(dev) })
	return dev, owners
}

// leaseOwners returns a chain for t whose bottom entry releases the lease
// that acquire returns. An acquisition error that retains cleanup is kept.
func leaseOwners(t testing.TB, acquire func() (release func() error, err error)) *nativeOwners {
	t.Helper()
	owners := newNativeOwners(t)
	release, err := acquire()
	owners.addError(err)
	if err != nil {
		t.Fatalf("acquire GPU lease: %v", err)
	}
	owners.add(release)
	return owners
}
