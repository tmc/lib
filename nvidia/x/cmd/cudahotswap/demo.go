package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"io"
	"math"
	"net/http"
	"sync"
	"time"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudahotswap"
	"github.com/tmc/lib/nvidia/x/cudahotswap/httpapi"
	"github.com/tmc/lib/nvidia/x/cudalaunch"
	"github.com/tmc/lib/nvidia/x/cudamem"
	"github.com/tmc/lib/nvidia/x/cudashape"
)

const saxpySource = `extern "C" __global__ void saxpy(const float *x, const float *y, float *out, float a, unsigned n) {
 unsigned i = blockIdx.x*blockDim.x+threadIdx.x;
 if (i<n) out[i] = a*x[i]+y[i];
}`

// demoApplication owns the buffers. The checker and ordinary loop serialize
// launches and host reads; Service owns the launch's GPU lease and completion.
type demoApplication struct {
	owners    *sessionOwners
	mu        sync.Mutex
	pending   *cudahotswap.PendingLaunch
	readLease io.Closer
	n         int
	x, y      []float32
	launch    cudalaunch.Config
}

func newDemo(ctx context.Context, device, n int) (*demoApplication, error) {
	a := &demoApplication{owners: &sessionOwners{acquire: acquireSessionLock}, n: n, x: make([]float32, n), y: make([]float32, n)}
	o := a.owners
	var err error
	o.lease, err = o.acquire(ctx, "cudahotswap: saxpy setup")
	if err != nil {
		return a, err
	}
	o.dev, err = cuda.NewDevice(device)
	if err != nil {
		return a, err
	}
	o.stream, err = o.dev.NewStream()
	if err != nil {
		return a, err
	}
	for i := 0; i < 3; i++ {
		b, e := cudamem.Alloc[byte](o.dev, 4*n)
		if b.Ptr() != 0 {
			o.buffers = append(o.buffers, b)
		}
		if e != nil {
			return a, e
		}
	}
	for i := range a.x {
		a.x[i] = float32(i + 1)
		a.y[i] = float32(2*i + 1)
	}
	for i, values := range [][]float32{a.x, a.y} {
		view, e := cudamem.Reinterpret[float32](o.buffers[i])
		if e != nil {
			return a, e
		}
		if e = view.CopyFromHost(values); e != nil {
			return a, e
		}
	}
	if err = o.lease.Close(); err != nil {
		return a, err
	}
	o.lease = nil
	loader, err := cudahotswap.NewNVRTCLoader(o.dev)
	if err != nil {
		return a, err
	}
	o.service, err = cudahotswap.NewService(loader)
	if o.service != nil {
		o.serviceClose = func() error { return o.service.Close(context.Background()) }
	}
	if err != nil {
		return a, err
	}
	if _, err = o.service.Register(ctx, "saxpy", cudahotswap.Source{Code: saxpySource, Filename: "saxpy.cu", Entry: "saxpy"}, "demo baseline"); err != nil {
		return a, err
	}
	a.launch = cudalaunch.Config{Grid: cudashape.Grid3D{X: uint32((n + 31) / 32), Y: 1, Z: 1}, Block: driver.Dims(32, 1, 1), Stream: o.stream}
	_, err = o.service.SetCheck("saxpy", cudahotswap.CheckConfig{Checker: a.check}, "demo CPU reference")
	return a, err
}

func (a *demoApplication) check(ctx context.Context, _ cudahotswap.State) error {
	a.mu.Lock()
	defer a.mu.Unlock()
	if err := a.releaseRead(); err != nil {
		return err
	}
	if a.pending != nil {
		if err := a.pending.Wait(ctx); err != nil {
			return err
		}
		a.pending = nil
	}
	lease, err := a.owners.acquire(ctx, "cudahotswap: saxpy output prefill")
	if lease != nil {
		a.readLease = lease
	}
	if err != nil {
		return err
	}
	view, err := cudamem.Reinterpret[float32](a.owners.buffers[2])
	if err != nil {
		return errors.Join(err, a.releaseRead())
	}
	prefill := make([]float32, a.n)
	for i := range prefill {
		prefill[i] = math.Float32frombits(0x7fc00001)
	}
	err = errors.Join(view.CopyFromHost(prefill), a.releaseRead())
	if err != nil {
		return err
	}
	p, err := a.owners.service.Launch(ctx, "saxpy", a.launch, a.owners.buffers[0], a.owners.buffers[1], a.owners.buffers[2], float32(2), uint32(a.n))
	if p != nil {
		a.pending = p
	}
	if err != nil {
		return err
	}
	if p == nil {
		return errors.New("demo: launch returned no completion owner")
	}
	if err = p.Wait(ctx); err != nil {
		return err
	}
	a.pending = nil
	// The launch has completed. The host read itself acquires a separate lease.
	lease, err = a.owners.acquire(ctx, "cudahotswap: saxpy readback")
	if lease != nil {
		a.readLease = lease
	}
	if err != nil {
		return err
	}
	view, err = cudamem.Reinterpret[float32](a.owners.buffers[2])
	if err != nil {
		return errors.Join(err, a.releaseRead())
	}
	out := make([]float32, a.n)
	err = view.CopyToHost(out)
	err = errors.Join(err, a.releaseRead())
	if err != nil {
		return err
	}
	return compareSaxpy(a.x, a.y, out)
}

func compareSaxpy(x, y, out []float32) error {
	if len(x) != len(y) || len(x) != len(out) {
		return errors.New("saxpy CPU reference: extent mismatch")
	}
	for i, v := range out {
		if v != 2*x[i]+y[i] {
			return fmt.Errorf("saxpy CPU reference: element %d got %g want %g", i, v, 2*x[i]+y[i])
		}
	}
	return nil
}

func (a *demoApplication) releaseRead() error {
	if a.readLease == nil {
		return nil
	}
	if err := a.readLease.Close(); err != nil {
		return err
	}
	a.readLease = nil
	return nil
}

func (a *demoApplication) close(ctx context.Context) error {
	a.mu.Lock()
	defer a.mu.Unlock()
	if err := a.releaseRead(); err != nil {
		return err
	}
	if a.pending != nil {
		if err := a.pending.Wait(ctx); err != nil {
			return err
		}
		a.pending = nil
	}
	return a.owners.close()
}

func runDemo(ctx context.Context, args []string, stdout, stderr io.Writer) error {
	f := flag.NewFlagSet("cudahotswap demo", flag.ContinueOnError)
	f.SetOutput(stderr)
	addr := f.String("addr", "unix:/var/run/cudahotswap.sock", "unix:path or tcp:host:port")
	n := f.Int("n", 32, "FP32 elements (1..4096)")
	device := f.Int("device", 0, "device ordinal")
	if err := f.Parse(args); err != nil {
		return err
	}
	if f.NArg() != 0 || *n < 1 || *n > 4096 || *device < 0 {
		return errors.New("demo requires n in [1,4096] and a nonnegative device")
	}
	a, err := newDemo(ctx, *device, *n)
	if err != nil {
		if a != nil {
			err = errors.Join(err, a.close(context.Background()))
		}
		return err
	}
	// Demo buffers remain private so external launches cannot overwrite the
	// ordinary loop's CPU-reference inputs or output between launch/readback.
	h, err := httpapi.New(a.owners.service, httpapi.Resources{Checkers: map[string]cudahotswap.Checker{"cpu_reference": a.check}})
	if err != nil {
		return errors.Join(err, a.close(context.Background()))
	}
	listener, err := httpapi.Listen(*addr)
	if err != nil {
		return errors.Join(err, h.Close(context.Background()), a.close(context.Background()))
	}
	server := &http.Server{Handler: h, ReadHeaderTimeout: 5 * time.Second, IdleTimeout: 30 * time.Second}
	result := make(chan error, 1)
	go func() { result <- server.Serve(listener) }()
	fmt.Fprintln(stdout, "saxpy", *addr)
	ticker := time.NewTicker(time.Second)
	defer ticker.Stop()
loop:
	for {
		select {
		case <-ctx.Done():
			err = ctx.Err()
			break loop
		case err = <-result:
			break loop
		case <-ticker.C:
			if e := a.check(ctx, cudahotswap.State{}); e != nil {
				fmt.Fprintln(stderr, e)
				if outcome, _ := cudahotswap.Classify(e); outcome.PoisonsContext() {
					err = e
					break loop
				}
			}
		}
	}
	cleanup, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	if errors.Is(err, http.ErrServerClosed) {
		err = nil
	}
	err = errors.Join(err, server.Shutdown(cleanup), server.Close())
	if e := h.Close(cleanup); e != nil {
		return errors.Join(err, e)
	}
	return errors.Join(err, a.close(cleanup))
}
