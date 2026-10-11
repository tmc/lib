package cudahotswap

import (
	"context"
	"io"
	"runtime"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudalaunch"
)

const serviceSwapTestPTX = `.version 7.0
.target sm_70
.address_size 64
.visible .entry entry() { ret; }
`

type serviceSwapKernel struct {
	raw           driver.Function
	launchEntered chan struct{}
	launchRelease <-chan struct{}
	events        chan<- string
	launchOnce    sync.Once
	launches      atomic.Int32
	waits         atomic.Int32
	closes        atomic.Int32
}

func (k *serviceSwapKernel) Raw() driver.Function { return k.raw }

func (k *serviceSwapKernel) Launch(cudalaunch.Config, ...any) error {
	k.launches.Add(1)
	if k.launchEntered != nil {
		k.launchOnce.Do(func() { close(k.launchEntered) })
		<-k.launchRelease
	}
	if k.events != nil {
		k.events <- "launch-return"
	}
	return nil
}

func (k *serviceSwapKernel) Wait(ctx context.Context, _ *cuda.Stream) error {
	k.waits.Add(1)
	return ctx.Err()
}

func (k *serviceSwapKernel) Close() error {
	k.closes.Add(1)
	return nil
}

type serviceSwapLoader struct {
	baseline, candidate Kernel
	candidateCompiled   chan struct{}
	candidateLoaded     chan struct{}
	events              chan<- string
}

func (l *serviceSwapLoader) Compile(_ context.Context, src Source) (*Compilation, error) {
	key := "baseline"
	if src.Code == "candidate" {
		key = "candidate"
		close(l.candidateCompiled)
	}
	return &Compilation{PTX: serviceSwapTestPTX, Entry: "entry", Key: key}, nil
}

func (l *serviceSwapLoader) Load(_ context.Context, c *Compilation) (Kernel, error) {
	if c.Key == "candidate" {
		l.events <- "candidate-load"
		close(l.candidateLoaded)
		return l.candidate, nil
	}
	return l.baseline, nil
}

type serviceSwapLease struct{}

func (serviceSwapLease) Close() error { return nil }

// TestServiceLaunchSerializesWithPush checks the public Launch and Push paths.
// The launch blocks inside its kernel while Push compiles; a mutation writer
// must queue before it can load and publish the replacement. The event order
// proves the existing launch reached its call boundary before candidate load,
// and the old generation remains pinned until Wait completes.
func TestServiceLaunchSerializesWithPush(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	events := make(chan string, 4)
	launchRelease := make(chan struct{})
	var releaseOnce sync.Once
	releaseLaunch := func() { releaseOnce.Do(func() { close(launchRelease) }) }
	var service *Service
	t.Cleanup(func() {
		releaseLaunch()
		if service != nil {
			if err := service.Close(context.Background()); err != nil {
				t.Errorf("close service: %v", err)
			}
		}
	})

	baseline := &serviceSwapKernel{
		raw:           1,
		launchEntered: make(chan struct{}),
		launchRelease: launchRelease,
		events:        events,
	}
	candidate := &serviceSwapKernel{raw: 2}
	loader := &serviceSwapLoader{
		baseline:          baseline,
		candidate:         candidate,
		candidateCompiled: make(chan struct{}),
		candidateLoaded:   make(chan struct{}),
		events:            events,
	}
	var err error
	service, err = NewService(loader)
	if err != nil {
		t.Fatal(err)
	}
	service.acquire = func(context.Context, string) (io.Closer, error) {
		return serviceSwapLease{}, nil
	}
	if _, err := service.Register(ctx, "kernel", Source{Code: "baseline", Entry: "entry"}, "baseline"); err != nil {
		t.Fatal(err)
	}
	if _, err := service.SetCheck("kernel", CheckConfig{Checker: func(context.Context, State) error { return nil }}, "test"); err != nil {
		t.Fatal(err)
	}

	type launchResult struct {
		pending *PendingLaunch
		err     error
	}
	launchDone := make(chan launchResult, 1)
	go func() {
		pending, err := service.Launch(ctx, "kernel", cudalaunch.Config{})
		launchDone <- launchResult{pending: pending, err: err}
	}()
	select {
	case <-baseline.launchEntered:
	case <-ctx.Done():
		t.Fatalf("launch did not enter kernel: %v", ctx.Err())
	}

	type pushResult struct {
		result *PushResult
		err    error
	}
	pushDone := make(chan pushResult, 1)
	go func() {
		result, err := service.Push(ctx, "kernel", Source{Code: "candidate", Entry: "entry"}, "candidate")
		pushDone <- pushResult{result: result, err: err}
	}()
	select {
	case <-loader.candidateCompiled:
	case <-ctx.Done():
		releaseLaunch()
		t.Fatalf("candidate did not compile: %v", ctx.Err())
	}

	// TryRLock succeeds while only readers are active and fails once Push has
	// queued its writer. A candidate load or lease acquisition before that
	// barrier means one of the service paths crossed the active launch.
	for {
		select {
		case <-loader.candidateLoaded:
			releaseLaunch()
			t.Fatal("candidate loaded while the prior launch was still in flight")
		default:
		}
		if !service.mutations.TryRLock() {
			break
		}
		service.mutations.RUnlock()
		select {
		case <-ctx.Done():
			releaseLaunch()
			t.Fatalf("Push did not queue behind the active launch: %v", ctx.Err())
		default:
			runtime.Gosched()
		}
	}

	// Release only after observing the pending writer. If publication slips
	// through while Launch is blocked, its event arrives before launch-return.
	releaseLaunch()
	select {
	case got := <-events:
		if got != "launch-return" {
			t.Fatalf("first concurrent event = %q, want launch-return", got)
		}
	case <-ctx.Done():
		t.Fatalf("launch did not return: %v", ctx.Err())
	}
	select {
	case got := <-events:
		if got != "candidate-load" {
			t.Fatalf("second concurrent event = %q, want candidate-load", got)
		}
	case <-ctx.Done():
		t.Fatalf("candidate load did not follow launch: %v", ctx.Err())
	}

	var launched launchResult
	select {
	case launched = <-launchDone:
	case <-ctx.Done():
		t.Fatalf("Launch did not return: %v", ctx.Err())
	}
	if launched.err != nil || launched.pending == nil || launched.pending.kernel != baseline {
		t.Fatalf("Launch result = %+v, want a pending baseline generation", launched)
	}
	var pushed pushResult
	select {
	case pushed = <-pushDone:
	case <-ctx.Done():
		t.Fatalf("Push did not return: %v", ctx.Err())
	}
	if pushed.err != nil || pushed.result == nil || !pushed.result.Published || pushed.result.State.Version != 2 {
		t.Fatalf("Push result = %+v, want published version 2", pushed)
	}
	if baseline.launches.Load() != 1 || baseline.closes.Load() != 0 {
		t.Fatalf("baseline launches=%d closes=%d while launch is pending", baseline.launches.Load(), baseline.closes.Load())
	}
	baselinePin := launched.pending.pin
	if baselinePin == nil || baselinePin.released.Load() || baselinePin.gen.pins.Load() != 1 {
		var pins int64
		var released bool
		if baselinePin != nil {
			pins = baselinePin.gen.pins.Load()
			released = baselinePin.released.Load()
		}
		t.Fatalf("baseline launch pin=%v released=%t generation pins=%d before Wait, want live pin count 1", baselinePin != nil, released, pins)
	}
	if err := launched.pending.Wait(ctx); err != nil {
		t.Fatalf("wait baseline launch: %v", err)
	}
	if baseline.waits.Load() != 1 || !baselinePin.released.Load() || baselinePin.gen.pins.Load() != 0 || launched.pending.pin != nil {
		t.Fatalf("baseline waits=%d released=%t generation pins=%d pending pin=%v after completion", baseline.waits.Load(), baselinePin.released.Load(), baselinePin.gen.pins.Load(), launched.pending.pin != nil)
	}

	current, err := service.Launch(ctx, "kernel", cudalaunch.Config{})
	if err != nil || current == nil || current.kernel != candidate {
		t.Fatalf("launch selected generation 2: pending=%v err=%v", current, err)
	}
	if err := current.Wait(ctx); err != nil {
		t.Fatalf("wait candidate launch: %v", err)
	}
	if candidate.launches.Load() != 1 {
		t.Fatalf("candidate launches=%d, want 1", candidate.launches.Load())
	}
	if err := service.Close(ctx); err != nil {
		t.Fatalf("close service: %v", err)
	}
	if baseline.closes.Load() != 1 || candidate.closes.Load() != 1 {
		t.Fatalf("generation closes baseline=%d candidate=%d, want one each", baseline.closes.Load(), candidate.closes.Load())
	}
}
