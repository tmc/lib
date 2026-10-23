package cudatest

import (
	"context"
	"errors"
	"fmt"
	"io"
	"sync"
	"testing"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudalock"
)

// errNotVerified marks an admission outcome that skips the test.
var errNotVerified = errors.New("SKIPPED, NOT VERIFIED")

// operations are the driver and lock calls made by admission and custody.
// Tests replace them to run the real algorithm without a driver.
type operations struct {
	load        func() error
	acquire     func(ctx context.Context, label string) (release func() error, err error)
	init        func() error
	count       func() (int, error)
	open        func(ordinal int) (*cuda.Device, error)
	closeDevice func(*cuda.Device) error
	closeStream func(*cuda.Stream) error
}

var driverOperations = operations{
	load: func() error { return driver.Load("") },
	acquire: func(ctx context.Context, label string) (func() error, error) {
		h, err := cudalock.Acquire(ctx, cudalock.WithLabel(label))
		if err != nil {
			return nil, err
		}
		return h.Release, nil
	},
	init:        cuda.Init,
	count:       cuda.DeviceCount,
	open:        cuda.NewDevice,
	closeDevice: (*cuda.Device).Close,
	closeStream: (*cuda.Stream).Close,
}

// RequireOwned returns a CUDA device satisfying all requirements, together
// with the DeviceOwner that holds it and the GPU lease. If this machine has
// no CUDA driver, has no devices, or has no device satisfying the
// requirements, the test is skipped; a skipped call establishes no evidence.
//
// The device is borrowed from the owner. The caller must not close it or
// release the lease. The owner is closed once by a cleanup registered with t,
// or earlier by an explicit call to [DeviceOwner.Close].
//
// Admission runs in a fixed order. A nil or canceled ctx fails the test
// before the driver library is loaded. Loading the library may run its
// constructors; a library that cannot be loaded skips the test before any
// lease is taken. RequireOwned then always acquires the canonical lease with
// [cudalock.Acquire], labeled with t.Name(), before any other CUDA call; it
// does not honor GPU_LOCK_DISABLED or CUDATEST_NO_LOCK. Only then is the
// driver initialized and are devices opened and filtered.
//
// ctx bounds only the lease wait and the checks between admission stages.
// It cannot interrupt driver calls or requirements, and cleanup never
// depends on it. A ctx found done at any check fails the test, even after a
// device has been selected or every device rejected.
//
// Unlike [Require], RequireOwned fails the test when a device cannot be
// opened, rather than trying the next one. Each rejected device is closed
// before the next is opened; if that close fails, the test fails and the
// device and lease stay with the owner.
//
// If Acquire fails with an error carrying an [io.Closer], the owner keeps
// that closer and calls it on Close. Acquire can lose its handle when closing
// its namespace guard fails; that error carries no closer, and RequireOwned
// cannot recover the lease.
func RequireOwned(ctx context.Context, t testing.TB, reqs ...Requirement) (*cuda.Device, *DeviceOwner) {
	t.Helper()
	dev, o, err := admit(ctx, t.Name(), &driverOperations, reqs)
	o.register(t)
	if errors.Is(err, errNotVerified) {
		t.Skip(err)
	}
	if err != nil {
		t.Fatal(err)
	}
	return dev, o
}

// admit selects a device. It returns the owner on every path, holding
// whatever lease and device admission had acquired when it stopped.
// An error wrapping errNotVerified is a skip; any other error is a failure.
func admit(ctx context.Context, label string, ops *operations, reqs []Requirement) (*cuda.Device, *DeviceOwner, error) {
	o := &DeviceOwner{ops: ops}
	if ctx == nil {
		return nil, o, errors.New("cudatest: nil admission context")
	}
	if err := canceled(ctx, "before driver load"); err != nil {
		return nil, o, err
	}
	if err := ops.load(); err != nil {
		return nil, o, fmt.Errorf("%w: the CUDA driver could not be loaded: %w", errNotVerified, err)
	}
	if err := canceled(ctx, "before GPU lease"); err != nil {
		return nil, o, err
	}
	release, err := ops.acquire(ctx, label)
	if err != nil {
		var c io.Closer
		if errors.As(err, &c) {
			o.lease = c.Close
		}
		// Acquire reports ctx.Err, not its cause.
		return nil, o, fmt.Errorf("cudatest: acquire GPU lease: %w", joinCanceled(ctx, "during GPU lease", err))
	}
	o.lease = release
	if err := canceled(ctx, "after GPU lease"); err != nil {
		return nil, o, err
	}
	if err := ops.init(); err != nil {
		return nil, o, fmt.Errorf("cudatest: initialize loaded CUDA driver: %w", joinCanceled(ctx, "during driver init", err))
	}
	if err := canceled(ctx, "after driver init"); err != nil {
		return nil, o, err
	}
	count, err := ops.count()
	if err != nil {
		return nil, o, fmt.Errorf("cudatest: count CUDA devices: %w", joinCanceled(ctx, "during device count", err))
	}
	if err := canceled(ctx, "after device count"); err != nil {
		return nil, o, err
	}
	if count == 0 {
		return nil, o, fmt.Errorf("%w: the CUDA driver reports no devices", errNotVerified)
	}
	var rejected error
	for i := 0; i < count; i++ {
		if err := canceled(ctx, fmt.Sprintf("before device %d", i)); err != nil {
			return nil, o, err
		}
		dev, err := ops.open(i)
		if dev != nil {
			o.dev = dev
		}
		if err != nil {
			err = joinCanceled(ctx, fmt.Sprintf("during device %d open", i), err)
			return nil, o, fmt.Errorf("cudatest: open device %d: %w", i, err)
		}
		if err := canceled(ctx, fmt.Sprintf("after device %d open", i)); err != nil {
			return nil, o, err
		}
		rejected = satisfies(dev, reqs)
		if rejected == nil {
			if err := canceled(ctx, fmt.Sprintf("after device %d requirements", i)); err != nil {
				return nil, o, err
			}
			return dev, o, nil
		}
		if err := ops.closeDevice(dev); err != nil {
			err = joinCanceled(ctx, fmt.Sprintf("during rejected device %d close", i), err)
			return nil, o, fmt.Errorf("cudatest: close rejected device %d: %w", i, err)
		}
		o.dev = nil
		rejected = fmt.Errorf("device %d: %w", i, rejected)
	}
	if err := canceled(ctx, "after last device"); err != nil {
		return nil, o, err
	}
	return nil, o, fmt.Errorf("%w: no CUDA device satisfied the requirements: %w", errNotVerified, rejected)
}

func satisfies(dev *cuda.Device, reqs []Requirement) error {
	for _, req := range reqs {
		if err := req(dev); err != nil {
			return err
		}
	}
	return nil
}

// canceled reports why ctx is done, keeping both its error and any distinct
// cause, or returns nil.
func canceled(ctx context.Context, stage string) error {
	err := ctx.Err()
	if err == nil {
		return nil
	}
	if cause := context.Cause(ctx); cause != err {
		return fmt.Errorf("cudatest: admission canceled %s: %w: %w", stage, err, cause)
	}
	return fmt.Errorf("cudatest: admission canceled %s: %w", stage, err)
}

// joinCanceled joins err from an admission stage with the reason ctx is done,
// if it is, so that neither is lost.
func joinCanceled(ctx context.Context, stage string, err error) error {
	if cerr := canceled(ctx, stage); cerr != nil {
		return errors.Join(err, cerr)
	}
	return err
}

// A DeviceOwner holds a device returned by [RequireOwned], the GPU lease
// taken for it, and the dependents registered with it.
//
// Close releases every registered stream, then every other dependent, each in
// reverse registration order, then the device, then the lease. Streams are
// released first regardless of when they were registered, so no buffer,
// module or graph is released while a stream may still use it.
//
// Close stops at the first error. Entries released before it are forgotten;
// the failed entry and every entry after it are kept. A later explicit Close
// resumes from the failed entry and attempts each remaining entry once. After
// a successful Close, Close does nothing.
//
// The cleanup registered by RequireOwned makes one attempt. If an explicit
// Close has failed and not since succeeded, that cleanup does not retry: it
// reports the failure. A failed cleanup reports its error. In both cases the
// owner and everything it still holds stay reachable for the life of the
// process. There is no fallback release of the device or the lease.
//
// Only registered releases are owned. An error dropped by an ordinary defer,
// an unregistered cleanup, or a goroutine cannot reach the owner.
// Registration and Close must not run concurrently, and release functions must
// not call methods on the owner.
type DeviceOwner struct {
	ops     *operations
	streams []*cuda.Stream
	deps    []func() error
	dev     *cuda.Device
	lease   func() error
	err     error // last Close error, nil after a successful Close
}

// Defer registers release to run when the owner is closed, after every
// stream and before every dependent registered earlier. A nil release is
// ignored. Register a partial result before checking the error returned
// with it:
//
//	g, err := cudagraph.New(dev)
//	if g != nil {
//		owner.Defer(g.Destroy)
//	}
//	if err != nil {
//		t.Fatal(err)
//	}
func (o *DeviceOwner) Defer(release func() error) {
	if release == nil {
		return
	}
	o.deps = append(o.deps, release)
}

// DeferStream registers stream to be closed before any dependent is released.
// Any non-nil stream should be registered, including one returned with an
// error. A nil stream is ignored.
func (o *DeviceOwner) DeferStream(stream *cuda.Stream) {
	if stream == nil {
		return
	}
	o.streams = append(o.streams, stream)
}

// Close releases what the owner holds, as described on [DeviceOwner].
// The returned error wraps the first failure.
func (o *DeviceOwner) Close() error {
	o.err = o.close()
	return o.err
}

func (o *DeviceOwner) close() error {
	// A stream is forgotten once its Close succeeds, so a retry never
	// synchronizes it again. A retry does close a stream whose Close failed;
	// Stream.Close does not synchronize a stream it has already destroyed.
	for len(o.streams) > 0 {
		i := len(o.streams) - 1
		if err := o.ops.closeStream(o.streams[i]); err != nil {
			return fmt.Errorf("cudatest: close stream: %w", err)
		}
		o.streams[i] = nil
		o.streams = o.streams[:i]
	}
	for len(o.deps) > 0 {
		i := len(o.deps) - 1
		if err := o.deps[i](); err != nil {
			return fmt.Errorf("cudatest: release dependent: %w", err)
		}
		o.deps[i] = nil
		o.deps = o.deps[:i]
	}
	if o.dev != nil {
		if err := o.ops.closeDevice(o.dev); err != nil {
			return fmt.Errorf("cudatest: close device %d: %w", o.dev.Ordinal(), err)
		}
		o.dev = nil
	}
	if o.lease != nil {
		if err := o.lease(); err != nil {
			return fmt.Errorf("cudatest: release GPU lease: %w", err)
		}
		o.lease = nil
	}
	return nil
}

// register arranges for the owner's single automatic cleanup.
func (o *DeviceOwner) register(t testing.TB) {
	t.Cleanup(func() {
		if err := o.cleanup(); err != nil {
			t.Error(err)
		}
	})
}

// cleanup is the automatic cleanup. It does not retry a failed explicit
// Close, and it retains the owner whenever something is left unreleased.
func (o *DeviceOwner) cleanup() error {
	if o.err != nil {
		retain(o)
		return fmt.Errorf("cudatest: device owner retained after failed Close: %w", o.err)
	}
	if err := o.Close(); err != nil {
		retain(o)
		return fmt.Errorf("cudatest: device owner retained after failed cleanup: %w", err)
	}
	return nil
}

// retained keeps owners with unreleased resources reachable for the life of
// the process.
var retained struct {
	mu     sync.Mutex
	owners []*DeviceOwner
}

func retain(o *DeviceOwner) {
	retained.mu.Lock()
	retained.owners = append(retained.owners, o)
	retained.mu.Unlock()
}
