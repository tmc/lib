package main

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"os/signal"
	"path/filepath"
	"syscall"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudahotswap"
	"github.com/tmc/lib/nvidia/x/cudalaunch"
	"github.com/tmc/lib/nvidia/x/cudalock"
	"github.com/tmc/lib/nvidia/x/cudamem"
	"github.com/tmc/lib/nvidia/x/cudashape"
)

type includes []string

func (p *includes) String() string { return fmt.Sprint([]string(*p)) }
func (p *includes) Set(path string) error {
	if len(*p) >= 64 || path == "" {
		return errors.New("include paths: empty or more than 64 directories")
	}
	path, err := filepath.Abs(path)
	if err != nil {
		return err
	}
	*p = append(*p, path)
	return nil
}

type config struct {
	path     string
	entry    string
	device   int
	elements int
	poll     bool
	includes includes
}

func main() {
	ctx, cancel := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer cancel()
	invoke := func() error { return run(ctx, os.Args[1:], os.Stdout, os.Stderr, newSession) }
	if isClientCommand(os.Args[1:]) {
		invoke = func() error { return runClient(ctx, os.Args[1:], os.Stdout, os.Stderr) }
	}
	if err := invoke(); err != nil && !onlyCancellation(err) && !errors.Is(err, flag.ErrHelp) {
		fmt.Fprintln(os.Stderr, "cudahotswap:", err)
		os.Exit(4)
	}
}

func onlyCancellation(err error) bool {
	if err == context.Canceled {
		return true
	}
	if joined, ok := err.(interface{ Unwrap() []error }); ok {
		children := joined.Unwrap()
		if len(children) == 0 {
			return false
		}
		for _, child := range children {
			if !onlyCancellation(child) {
				return false
			}
		}
		return true
	}
	if wrapped, ok := err.(interface{ Unwrap() error }); ok {
		return onlyCancellation(wrapped.Unwrap())
	}
	return false
}

func run(ctx context.Context, args []string, stdout, stderr io.Writer, create func(context.Context, config) (*cudahotswap.Service, func() error, error)) error {
	var c config
	f := flag.NewFlagSet("cudahotswap", flag.ContinueOnError)
	f.SetOutput(stderr)
	f.StringVar(&c.path, "watch", "", "CUDA source file; candidate must implement vector addition")
	f.StringVar(&c.entry, "entry", "replay", "kernel entry point")
	f.IntVar(&c.device, "device", 0, "CUDA device ordinal")
	f.IntVar(&c.elements, "n", 32, "number of FP32 elements (1..4096)")
	f.BoolVar(&c.poll, "poll", false, "poll for changes on network mounts")
	f.Var(&c.includes, "I", "disk include directory (repeatable)")
	if err := f.Parse(args); err != nil {
		return err
	}
	if ctx == nil || c.path == "" || c.entry == "" || c.device < 0 || c.elements < 1 || c.elements > 4096 || f.NArg() != 0 {
		return errors.New("-watch, a nonempty entry, and n in [1,4096] are required")
	}
	var err error
	c.path, err = filepath.Abs(c.path)
	if err != nil {
		return err
	}
	s, close, err := create(ctx, c)
	if err != nil {
		if close != nil {
			err = errors.Join(err, close())
		}
		return err
	}
	if s == nil || close == nil {
		return errors.New("create watch session: missing service or cleanup owner")
	}
	if _, err := s.SetWatchPaths("kernel", &cudahotswap.WatchPaths{SourceFile: c.path, Directories: c.includes}, "local watch"); err != nil {
		return errors.Join(err, close())
	}
	enc := json.NewEncoder(stdout)
	err = s.Watch(ctx, cudahotswap.WatchConfig{Poll: c.poll}, func(name string, result *cudahotswap.PushResult, pushErr error) error {
		report := struct {
			Kernel string
			Result *cudahotswap.PushResult
			Error  string `json:",omitempty"`
		}{Kernel: name, Result: result}
		if pushErr != nil {
			report.Error = pushErr.Error()
		}
		return enc.Encode(report)
	})
	return errors.Join(err, close())
}

type sessionOwners struct {
	service      *cudahotswap.Service
	dev          *cuda.Device
	stream       *cuda.Stream
	buffers      []cudamem.Slice[byte]
	lease        io.Closer
	teardown     bool
	acquire      func(context.Context, string) (io.Closer, error)
	serviceClose func() error
	nativeClose  func() error
}

type sessionLock struct{ handle *cudalock.Handle }

func (l sessionLock) Close() error { return l.handle.Release() }

func acquireSessionLock(ctx context.Context, label string) (io.Closer, error) {
	h, err := cudalock.Acquire(ctx, cudalock.WithLabel(label))
	if h == nil {
		return nil, err
	}
	return sessionLock{h}, err
}

func (o *sessionOwners) close() error {
	// Service.Close owns its own per-request lease. Release setup's lease
	// first; all setup work is synchronous, and no Service work was submitted
	// under that external owner.
	if o.lease != nil && !o.teardown {
		if err := o.lease.Close(); err != nil {
			return err
		}
		o.lease = nil
	}
	if o.serviceClose != nil {
		if err := o.serviceClose(); err != nil {
			return err
		}
		o.serviceClose = nil
		o.service = nil
	}
	if o.dev == nil && o.stream == nil && len(o.buffers) == 0 && o.lease == nil && o.nativeClose == nil {
		return nil
	}
	if o.lease == nil {
		var err error
		o.lease, err = o.acquire(context.Background(), "cudahotswap: teardown")
		o.teardown = err == nil
		if err != nil {
			return err
		}
		if o.lease == nil {
			return errors.New("teardown acquisition returned no owner")
		}
	}
	if o.nativeClose != nil {
		if err := o.nativeClose(); err != nil {
			return err
		}
		o.nativeClose = nil
	} else if err := o.closeResources(); err != nil {
		return err
	}
	err := o.lease.Close()
	if err == nil {
		o.lease = nil
	}
	return err
}

func (o *sessionOwners) closeResources() error {
	if o.stream != nil {
		if err := o.stream.Close(); err != nil {
			return err
		}
		o.stream = nil
	}
	for len(o.buffers) > 0 {
		last := len(o.buffers) - 1
		if err := o.buffers[last].Free(); err != nil {
			return err
		}
		o.buffers = o.buffers[:last]
	}
	if o.dev != nil {
		if err := o.dev.Close(); err != nil {
			return err
		}
		o.dev = nil
	}
	return nil
}

func newSession(ctx context.Context, c config) (*cudahotswap.Service, func() error, error) {
	return newSessionWithLoader(ctx, c, func(dev *cuda.Device, paths ...string) (cudahotswap.Loader, error) {
		return cudahotswap.NewNVRTCLoader(dev, paths...)
	})
}

// newSessionWithLoader preserves the session owner lifecycle while allowing a
// test to observe the actual native compiler and loader without replacing setup.
func newSessionWithLoader(ctx context.Context, c config, create func(*cuda.Device, ...string) (cudahotswap.Loader, error)) (*cudahotswap.Service, func() error, error) {
	if create == nil {
		return nil, nil, errors.New("create watch session: missing loader factory")
	}
	o := new(sessionOwners)
	o.acquire = acquireSessionLock
	cleanup := func() error {
		if err := o.close(); err != nil {
			return errors.Join(err, o.close()) // One bounded retry; retain failed owners.
		}
		return nil
	}
	var err error
	o.lease, err = o.acquire(ctx, "cudahotswap: setup")
	if err != nil {
		return nil, cleanup, err
	}
	if o.lease == nil {
		return nil, cleanup, errors.New("setup acquisition returned no owner")
	}
	o.dev, err = cuda.NewDevice(c.device)
	if err != nil {
		return nil, cleanup, err
	}
	o.stream, err = o.dev.NewStream()
	if err != nil {
		return nil, cleanup, err
	}
	for i := 0; i < 3; i++ {
		buffer, err := cudamem.Alloc[byte](o.dev, 4*c.elements)
		if buffer.Ptr() != 0 {
			o.buffers = append(o.buffers, buffer)
		}
		if err != nil {
			return nil, cleanup, err
		}
	}
	for i, buffer := range o.buffers {
		view, err := cudamem.Reinterpret[float32](buffer)
		if err != nil {
			return nil, cleanup, err
		}
		data := make([]float32, c.elements)
		for j := range data {
			data[j] = float32((i + 1) * (j + 1))
		}
		if err := view.CopyFromHost(data); err != nil {
			return nil, cleanup, err
		}
	}
	if err := o.lease.Close(); err != nil {
		return nil, cleanup, err
	}
	o.lease = nil
	loader, err := create(o.dev, append([]string{filepath.Dir(c.path)}, c.includes...)...)
	if err != nil {
		return nil, cleanup, err
	}
	o.service, err = cudahotswap.NewService(loader)
	if o.service != nil {
		o.serviceClose = func() error { return o.service.Close(context.Background()) }
	}
	if err != nil {
		return nil, cleanup, err
	}
	source := fmt.Sprintf("extern \"C\" __global__ void %s(float *x, float *y, const float *w, unsigned n) { unsigned i = blockIdx.x*blockDim.x+threadIdx.x; if (i<n) y[i]=x[i]+w[i]; }", c.entry)
	if _, err := o.service.Register(ctx, "kernel", cudahotswap.Source{Code: source, Filename: c.path, Entry: c.entry}, "retained baseline"); err != nil {
		return nil, cleanup, err
	}
	cfg := cudahotswap.FixtureConfig{Args: []any{nil, nil, nil, uint32(c.elements)}, Outputs: []cudahotswap.FixtureOutput{{Arg: 1, Elem: "f32"}}, Policy: cudahotswap.DefaultReplayPolicy(), Launch: cudalaunch.Config{Grid: cudashape.Grid3D{X: uint32((c.elements + 31) / 32), Y: 1, Z: 1}, Block: driver.Dims(32, 1, 1), Stream: o.stream}}
	for i, buffer := range o.buffers {
		cfg.Buffers = append(cfg.Buffers, cudahotswap.FixtureBuffer{Arg: i, Bytes: 4 * c.elements, Buffer: buffer, Stable: i == 2})
	}
	fixture, err := o.service.RecordFixture(ctx, "kernel", cfg)
	if err != nil {
		return nil, cleanup, err // Service retains even a partial fixture.
	}
	if _, err := o.service.SetCheck("kernel", cudahotswap.CheckConfig{Fixture: fixture}, "replay check"); err != nil {
		return nil, cleanup, err
	}
	return o.service, cleanup, nil
}
