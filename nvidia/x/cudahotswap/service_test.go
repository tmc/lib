package cudahotswap

import (
	"context"
	"errors"
	"io"
	"sync"
	"testing"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudalaunch"
)

type serviceCloser struct {
	calls int
	fail  bool
}

func (c *serviceCloser) Close() error {
	c.calls++
	if c.fail {
		return errors.New("injected cleanup failure")
	}
	return nil
}

type serviceCleanupError struct{ *serviceCloser }

func (e *serviceCleanupError) Error() string { return "partial load" }

type serviceKernel struct {
	serviceCloser
	raw       driver.Function
	waits     int
	waitErr   error
	launchErr error
}

func (k *serviceKernel) Raw() driver.Function                   { return k.raw }
func (k *serviceKernel) Launch(cudalaunch.Config, ...any) error { return k.launchErr }
func (k *serviceKernel) Wait(ctx context.Context, _ *cuda.Stream) error {
	k.waits++
	if err := ctx.Err(); err != nil {
		return err
	}
	return k.waitErr
}

type serviceLoader struct {
	compile func(context.Context, Source) (*Compilation, error)
	kernel  Kernel
	loadErr error
	loads   int
}

func (l *serviceLoader) Compile(ctx context.Context, src Source) (*Compilation, error) {
	if l.compile != nil {
		return l.compile(ctx, src)
	}
	return &Compilation{PTX: ".version 7.0\n.target sm_70\n.address_size 64\n.visible .entry entry() { ret; }", Entry: "entry", Key: "compiler"}, nil
}
func (l *serviceLoader) Load(context.Context, *Compilation) (Kernel, error) {
	l.loads++
	return l.kernel, l.loadErr
}

func testService(t *testing.T, l *serviceLoader) (*Service, *[]*serviceCloser) {
	t.Helper()
	s, err := NewService(l)
	if err != nil {
		t.Fatal(err)
	}
	leases := new([]*serviceCloser)
	s.acquire = func(context.Context, string) (io.Closer, error) {
		c := new(serviceCloser)
		*leases = append(*leases, c)
		return c, nil
	}
	return s, leases
}

func registerService(t *testing.T, s *Service) {
	t.Helper()
	if _, err := s.Register(context.Background(), "kernel", Source{Code: "source", Entry: "entry"}, "original"); err != nil {
		t.Fatal(err)
	}
}

func TestServiceNilContextsPreserveOwners(t *testing.T) {
	for _, operation := range []string{"register", "launch", "release", "wait", "close"} {
		t.Run(operation, func(t *testing.T) {
			k := &serviceKernel{raw: 1}
			l := &serviceLoader{kernel: k}
			s, leases := testService(t, l)
			t.Cleanup(func() {
				if err := s.Close(context.Background()); err != nil {
					t.Error(err)
				}
			})
			if operation != "register" {
				registerService(t, s)
			}
			var pin *Pin
			var pending *PendingLaunch
			var err error
			if operation == "release" {
				pin, err = s.Pin("kernel")
				if err != nil {
					t.Fatal(err)
				}
				t.Cleanup(func() {
					if err := pin.Release(context.Background()); err != nil {
						t.Error(err)
					}
				})
			}
			if operation == "wait" {
				pending, err = s.Launch(context.Background(), "kernel", cudalaunch.Config{})
				if err != nil {
					t.Fatal(err)
				}
			}
			beforeLeases, beforeLoads := len(*leases), l.loads
			switch operation {
			case "register":
				_, err = s.Register(nil, "kernel", Source{Code: "source", Entry: "entry"}, "")
			case "launch":
				var result *PendingLaunch
				result, err = s.Launch(nil, "kernel", cudalaunch.Config{})
				if result != nil {
					t.Fatal("nil context submitted a launch")
				}
			case "release":
				err = pin.Release(nil)
				if pin.Function() != k.raw {
					t.Fatal("nil context released a live pin")
				}
			case "wait":
				err = pending.Wait(nil)
				if k.waits != 0 || pending.pin == nil || pending.lease == nil || pending.completed {
					t.Fatal("nil context observed or released pending work")
				}
			case "close":
				err = s.Close(nil)
				if s.closed || k.calls != 0 {
					t.Fatal("nil context closed the service")
				}
			}
			if err == nil {
				t.Fatal("nil context accepted")
			}
			if len(*leases) != beforeLeases || l.loads != beforeLoads || k.calls != 0 {
				t.Fatal("nil context changed GPU ownership")
			}
			if operation == "register" && len(s.List()) != 0 {
				t.Fatal("nil context published a kernel")
			}
		})
	}
}

func TestServiceCompileBeforeLeaseAndIdleClose(t *testing.T) {
	l := &serviceLoader{}
	s, leases := testService(t, l)
	if len(*leases) != 0 {
		t.Fatal("constructor acquired lease")
	}
	l.compile = func(context.Context, Source) (*Compilation, error) {
		if len(*leases) != 0 {
			t.Fatal("compile ran under lease")
		}
		return nil, errors.New("compile failed")
	}
	if _, err := s.Register(context.Background(), "kernel", Source{Entry: "entry"}, ""); err == nil {
		t.Fatal("compile error missing")
	}
	if l.loads != 0 || len(*leases) != 0 {
		t.Fatal("failed compilation touched device")
	}
	if err := s.Close(context.Background()); err != nil {
		t.Fatal(err)
	}
	if len(*leases) != 0 {
		t.Fatal("idle close acquired lease")
	}
}

func TestServicePartialLoadRetainsEveryOwnerAndLease(t *testing.T) {
	for _, withKernel := range []bool{false, true} {
		t.Run(map[bool]string{false: "error-owner", true: "kernel-and-error-owner"}[withKernel], func(t *testing.T) {
			extra := &serviceCleanupError{&serviceCloser{fail: true}}
			k := &serviceKernel{serviceCloser: serviceCloser{fail: true}, raw: 1}
			l := &serviceLoader{loadErr: extra}
			if withKernel {
				l.kernel = k
			}
			s, leases := testService(t, l)
			if _, err := s.Register(context.Background(), "kernel", Source{Code: "source", Entry: "entry"}, ""); !errors.Is(err, extra) {
				t.Fatalf("original cause lost: %v", err)
			}
			if len(s.cleanup) != 1 || (*leases)[0].calls != 0 {
				t.Fatal("failed cleanup released ownership or lease")
			}
			k.fail = false
			if err := s.Close(context.Background()); err == nil {
				t.Fatal("extra owner failure lost")
			}
			if (*leases)[0].calls != 0 {
				t.Fatal("lease released before extra owner")
			}
			extra.fail = false
			if err := s.Close(context.Background()); err != nil {
				t.Fatal(err)
			}
			if (*leases)[0].calls != 1 || len(s.cleanup) != 0 {
				t.Fatal("retry failed to release lease")
			}
			if withKernel && k.calls != 2 {
				t.Fatalf("successful kernel cleanup repeated: %d", k.calls)
			}
		})
	}
}

func TestServiceMissingFunctionRetainsPartialKernel(t *testing.T) {
	k := &serviceKernel{serviceCloser: serviceCloser{fail: true}}
	s, leases := testService(t, &serviceLoader{kernel: k})
	if _, err := s.Register(context.Background(), "kernel", Source{Code: "source", Entry: "entry"}, ""); err == nil {
		t.Fatal("missing function accepted")
	}
	if len(s.cleanup) != 1 || (*leases)[0].calls != 0 {
		t.Fatal("missing function cleanup orphaned")
	}
	k.fail = false
	if err := s.Close(context.Background()); err != nil {
		t.Fatal(err)
	}
}

func retireServiceGeneration(t *testing.T, s *Service) {
	t.Helper()
	s.table.mu.Lock()
	defer s.table.mu.Unlock()
	e := s.table.entries["kernel"]
	e.current.Load().retired = true
	e.current.Store(nil)
}

func TestServicePendingCompletionAndCollectionRetry(t *testing.T) {
	k := &serviceKernel{raw: 1}
	s, leases := testService(t, &serviceLoader{kernel: k})
	registerService(t, s)
	p, err := s.Launch(context.Background(), "kernel", cudalaunch.Config{})
	if err != nil {
		t.Fatal(err)
	}
	canceled, cancel := context.WithCancel(context.Background())
	cancel()
	if err := p.Wait(canceled); !errors.Is(err, context.Canceled) {
		t.Fatal(err)
	}
	if p.pin == nil || (*leases)[1].calls != 0 || k.calls != 0 {
		t.Fatal("cancellation released launch")
	}
	retireServiceGeneration(t, s)
	k.fail = true
	if err := p.Wait(context.Background()); err == nil {
		t.Fatal("collection failure missing")
	}
	if !p.completed || !p.pin.released.Load() || (*leases)[1].calls != 0 {
		t.Fatal("failed collection lost completion or lease")
	}
	oldWaits := k.waits
	k.fail = false
	k.waitErr = errors.New("must not wait twice")
	if err := p.Wait(context.Background()); err != nil {
		t.Fatal(err)
	}
	if k.waits != oldWaits || k.calls != 2 || (*leases)[1].calls != 1 || len(s.pending) != 0 {
		t.Fatal("retry skipped collection or repeated completion")
	}
	if err := s.Close(context.Background()); err != nil {
		t.Fatal(err)
	}
}

func TestServicePinAndCloseRetainFailedCleanupLease(t *testing.T) {
	k := &serviceKernel{raw: 1}
	s, leases := testService(t, &serviceLoader{kernel: k})
	registerService(t, s)
	p, err := s.Pin("kernel")
	if err != nil {
		t.Fatal(err)
	}
	if len(*leases) != 1 {
		t.Fatal("pin acquired lease")
	}
	retireServiceGeneration(t, s)
	k.fail = true
	if err := p.Release(context.Background()); err == nil {
		t.Fatal("release error missing")
	}
	if p.Function() != 0 || (*leases)[1].calls != 0 {
		t.Fatal("released pin still usable or lease lost")
	}
	k.fail = false
	if err := s.Close(context.Background()); err != nil {
		t.Fatal(err)
	}
	if k.calls != 2 || (*leases)[1].calls != 1 {
		t.Fatal("close failed to retry pin cleanup")
	}
	if err := p.Release(context.Background()); err != nil {
		t.Fatal(err)
	}
}

func TestServiceCloseRetriesModuleAndLease(t *testing.T) {
	k := &serviceKernel{raw: 1}
	s, leases := testService(t, &serviceLoader{kernel: k})
	registerService(t, s)
	k.fail = true
	if err := s.Close(context.Background()); err == nil {
		t.Fatal("module failure missing")
	}
	if (*leases)[1].calls != 0 {
		t.Fatal("lease closed under failed module")
	}
	k.fail = false
	(*leases)[1].fail = true
	if err := s.Close(context.Background()); err == nil {
		t.Fatal("lease failure missing")
	}
	if k.calls != 2 {
		t.Fatal("module close retry missing")
	}
	(*leases)[1].fail = false
	if err := s.Close(context.Background()); err != nil {
		t.Fatal(err)
	}
	if k.calls != 2 || len(*leases) != 2 || (*leases)[1].calls != 2 {
		t.Fatal("retry repeated module close or acquired new lease")
	}
}

func TestServiceSubmissionPoisonStillOwnsPending(t *testing.T) {
	k := &serviceKernel{raw: 1, launchErr: driver.ErrIllegalAddress}
	s, leases := testService(t, &serviceLoader{kernel: k})
	registerService(t, s)
	p, err := s.Launch(context.Background(), "kernel", cudalaunch.Config{})
	if p == nil || !errors.Is(err, driver.ErrIllegalAddress) || len(s.pending) != 1 || (*leases)[1].calls != 0 {
		t.Fatal("failed submission owner lost")
	}
	if _, err := s.Pin("kernel"); err == nil {
		t.Fatal("poisoned context accepted request")
	}
	k.waitErr = driver.ErrIllegalAddress
	if err := s.Close(context.Background()); err == nil || (*leases)[1].calls != 0 {
		t.Fatal("failed drain released lease")
	}
	k.waitErr = nil
	if err := s.Close(context.Background()); err != nil {
		t.Fatal(err)
	}
}

func TestServiceInvalidCompilationBeforeLease(t *testing.T) {
	for _, c := range []*Compilation{nil, {Entry: "other"}, {Entry: "entry", PTX: "not PTX"}} {
		l := &serviceLoader{compile: func(context.Context, Source) (*Compilation, error) { return c, nil }}
		s, leases := testService(t, l)
		if _, err := s.Register(context.Background(), "kernel", Source{Entry: "entry"}, ""); err == nil {
			t.Fatal("invalid compilation accepted")
		}
		if l.loads != 0 || len(*leases) != 0 {
			t.Fatal("invalid compilation acquired device ownership")
		}
	}
}

func TestServicePinLeaseFailurePreservesFunction(t *testing.T) {
	k := &serviceKernel{raw: 1}
	s, _ := testService(t, &serviceLoader{kernel: k})
	registerService(t, s)
	p, err := s.Pin("kernel")
	if err != nil {
		t.Fatal(err)
	}
	acquire := s.acquire
	s.acquire = func(context.Context, string) (io.Closer, error) { return nil, errors.New("no lease") }
	if err := p.Release(context.Background()); err == nil || p.Function() != 1 {
		t.Fatal("failed acquisition released pin")
	}
	s.acquire = acquire
	if err := p.Release(context.Background()); err != nil {
		t.Fatal(err)
	}
	if err := s.Close(context.Background()); err != nil {
		t.Fatal(err)
	}
}

func TestServicePendingLeaseCloseRetryDoesNotWaitAgain(t *testing.T) {
	k := &serviceKernel{raw: 1}
	s, leases := testService(t, &serviceLoader{kernel: k})
	registerService(t, s)
	p, err := s.Launch(context.Background(), "kernel", cudalaunch.Config{})
	if err != nil {
		t.Fatal(err)
	}
	(*leases)[1].fail = true
	if err := p.Wait(context.Background()); err == nil || len(s.pending) != 1 {
		t.Fatal("failed lease cleanup owner lost")
	}
	k.waitErr = errors.New("must not repeat completion")
	(*leases)[1].fail = false
	if err := p.Wait(context.Background()); err != nil {
		t.Fatal(err)
	}
	if k.waits != 1 || (*leases)[1].calls != 2 {
		t.Fatal("lease retry repeated GPU completion")
	}
	if err := s.Close(context.Background()); err != nil {
		t.Fatal(err)
	}
}

func TestServiceCompileCleanupOwnerNeedsNoDeviceLease(t *testing.T) {
	owner := &serviceCleanupError{&serviceCloser{fail: true}}
	l := &serviceLoader{compile: func(context.Context, Source) (*Compilation, error) { return nil, owner }}
	s, leases := testService(t, l)
	if _, err := s.Register(context.Background(), "kernel", Source{Entry: "entry"}, ""); !errors.Is(err, owner) {
		t.Fatal("compilation cause lost")
	}
	if len(s.cleanup) != 1 || len(*leases) != 0 || l.loads != 0 {
		t.Fatal("compiler owner lost or acquired device lease")
	}
	owner.fail = false
	if err := s.Close(context.Background()); err != nil {
		t.Fatal(err)
	}
	if owner.calls != 2 || len(s.cleanup) != 0 || len(*leases) != 0 {
		t.Fatal("compiler cleanup retry did not preserve idle device")
	}
}

func TestServicePinReleaseUsesRetainedCloseLease(t *testing.T) {
	k := &serviceKernel{raw: 1}
	s, leases := testService(t, &serviceLoader{kernel: k})
	registerService(t, s)
	p, err := s.Pin("kernel")
	if err != nil {
		t.Fatal(err)
	}
	if err := s.Close(context.Background()); !errors.Is(err, errGenerationPinned) {
		t.Fatal(err)
	}
	if len(*leases) != 2 || (*leases)[1].calls != 0 {
		t.Fatal("close failed to retain cleanup lease")
	}
	s.acquire = func(context.Context, string) (io.Closer, error) {
		t.Fatal("attempted second exclusive lease")
		return nil, nil
	}
	if err := p.Release(context.Background()); err != nil {
		t.Fatal(err)
	}
	if k.calls != 1 || (*leases)[1].calls != 0 {
		t.Fatal("pin release lost retained close lease")
	}
	if err := s.Close(context.Background()); err != nil {
		t.Fatal(err)
	}
	if (*leases)[1].calls != 1 || k.calls != 1 {
		t.Fatal("close did not finish retained request once")
	}
}

func TestServiceCloseDrainsCompilationAndLateCleanupOwner(t *testing.T) {
	entered := make(chan struct{})
	finish := make(chan struct{})
	owner := &serviceCleanupError{&serviceCloser{fail: true}}
	l := &serviceLoader{compile: func(context.Context, Source) (*Compilation, error) {
		close(entered)
		<-finish
		return nil, owner
	}}
	s, leases := testService(t, l)
	registered := make(chan error, 1)
	go func() {
		_, err := s.Register(context.Background(), "kernel", Source{Entry: "entry"}, "")
		registered <- err
	}()
	<-entered
	// A blocked compiler must own the shutdown gate, not the mutation gate or
	// a GPU lease. These checks do not depend on goroutine scheduling delays.
	if s.requests.TryLock() {
		s.requests.Unlock()
		close(finish)
		<-registered
		t.Fatal("in-flight compilation has no shutdown owner")
	}
	if !s.mutations.TryLock() {
		t.Fatal("compilation holds mutation gate")
	}
	s.mutations.Unlock()
	closing := make(chan error, 1)
	go func() { closing <- s.Close(context.Background()) }()
	close(finish)
	if err := <-registered; !errors.Is(err, owner) {
		t.Fatal(err)
	}
	if err := <-closing; err == nil {
		t.Fatal("close returned before late compiler cleanup succeeded")
	}
	if len(s.cleanup) != 1 || len(*leases) != 0 {
		t.Fatal("late compiler owner lost or device lease acquired")
	}
	owner.fail = false
	if err := s.Close(context.Background()); err != nil {
		t.Fatal(err)
	}
	if len(s.cleanup) != 0 || len(*leases) != 0 {
		t.Fatal("retry did not finish compiler owner")
	}
}

func TestServiceConcurrentCloseAndRegistration(t *testing.T) {
	k := &serviceKernel{raw: 1}
	s, _ := testService(t, &serviceLoader{kernel: k})
	registerService(t, s)
	// Multiple close callers serialize cleanup and do not double-close owners.
	var wg sync.WaitGroup
	for i := 0; i < 4; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			if err := s.Close(context.Background()); err != nil {
				t.Error(err)
			}
		}()
	}
	wg.Wait()
	if k.calls != 1 {
		t.Fatalf("module cleanup calls = %d", k.calls)
	}
	if _, err := s.Register(context.Background(), "later", Source{Entry: "entry"}, ""); err == nil {
		t.Fatal("closed service accepted registration")
	}
}

func TestServiceJoinedCancellationDoesNotHidePoison(t *testing.T) {
	for _, observation := range []error{context.Canceled, context.DeadlineExceeded} {
		s, _ := testService(t, &serviceLoader{})
		s.observeError(observation)
		if err := s.available(); err != nil {
			t.Fatalf("standalone observation poisoned context: %v", err)
		}
		fault := errors.Join(observation, driver.ErrInvalidValue, driver.ErrIllegalAddress)
		s.observeError(fault)
		if err := s.available(); !errors.Is(err, driver.ErrIllegalAddress) {
			t.Fatalf("joined fault not terminal: %v", err)
		}
	}
}
