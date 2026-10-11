package cudahotswap

import (
	"context"
	"errors"
	"io"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudalaunch"
)

func TestServiceRegisterTypedNilKernel(t *testing.T) {
	var kernel *serviceKernel
	s, leases := testService(t, &serviceLoader{kernel: kernel})
	if _, err := s.Register(context.Background(), "kernel", Source{Code: "source", Entry: "entry"}, ""); err == nil {
		t.Fatal("typed nil kernel accepted")
	}
	if len(s.List()) != 0 || len(s.cleanup) != 0 || len(*leases) != 1 || (*leases)[0].calls != 1 {
		t.Fatal("invalid kernel retained state or lease")
	}
	if err := s.Close(context.Background()); err != nil {
		t.Fatal(err)
	}
}

func TestServiceLaunchCancellationAdmission(t *testing.T) {
	for _, when := range []string{"before-acquire", "during-acquire"} {
		t.Run(when, func(t *testing.T) {
			s, _ := testService(t, &serviceLoader{kernel: &serviceKernel{raw: 1}})
			registerService(t, s)
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			lease := new(serviceCloser)
			acquires := 0
			s.acquire = func(context.Context, string) (io.Closer, error) {
				acquires++
				cancel() // A borrowed lease need not check the request context.
				return lease, nil
			}
			if when == "before-acquire" {
				cancel()
			}
			pending, err := s.Launch(ctx, "kernel", cudalaunch.Config{})
			if pending != nil || !errors.Is(err, context.Canceled) || len(s.pending) != 0 {
				t.Fatalf("canceled request submitted work: pending=%v err=%v", pending, err)
			}
			if when == "before-acquire" && acquires != 0 || when == "during-acquire" && (acquires != 1 || lease.calls != 1) {
				t.Fatal("cancellation leaked or unnecessarily acquired a lease")
			}
			s.acquire = func(context.Context, string) (io.Closer, error) { return new(serviceCloser), nil }
			if err := s.Close(context.Background()); err != nil {
				t.Fatal(err)
			}
		})
	}
}

type admissionLoader struct {
	serviceLoader
	load func() (Kernel, error)
}

func (l *admissionLoader) Load(context.Context, *Compilation) (Kernel, error) {
	l.loads++
	return l.load()
}

func TestServiceRegisterCancellationAdmission(t *testing.T) {
	for _, when := range []string{"before-compile", "during-compile", "during-acquire", "during-load"} {
		t.Run(when, func(t *testing.T) {
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			kernel := &serviceKernel{raw: 1}
			loader := &admissionLoader{}
			compiles := 0
			loader.compile = func(context.Context, Source) (*Compilation, error) {
				compiles++
				if when == "during-compile" {
					cancel()
				}
				return &Compilation{PTX: ".version 7.0\n.target sm_70\n.address_size 64\n.visible .entry entry() { ret; }", Entry: "entry"}, nil
			}
			loader.load = func() (Kernel, error) {
				if when == "during-load" {
					cancel()
				}
				return kernel, nil
			}
			s, err := NewService(loader)
			if err != nil {
				t.Fatal(err)
			}
			lease := new(serviceCloser)
			acquires := 0
			s.acquire = func(context.Context, string) (io.Closer, error) {
				acquires++
				if when == "during-acquire" {
					cancel()
				}
				return lease, nil
			}
			if when == "before-compile" {
				cancel()
			}
			if _, err := s.Register(ctx, "kernel", Source{Code: "source", Entry: "entry"}, ""); !errors.Is(err, context.Canceled) {
				t.Fatalf("canceled registration: %v", err)
			}
			if len(s.List()) != 0 || len(s.cleanup) != 0 {
				t.Fatal("canceled registration published state")
			}
			if when == "before-compile" && compiles != 0 {
				t.Fatal("canceled request compiled")
			}
			if when == "during-acquire" || when == "during-load" {
				if acquires != 1 || lease.calls != 1 {
					t.Fatal("canceled registration leaked lease")
				}
			} else if acquires != 0 {
				t.Fatal("canceled request acquired lease")
			}
			if when == "during-load" {
				if loader.loads != 1 || kernel.calls != 1 {
					t.Fatal("canceled load leaked kernel")
				}
			} else if loader.loads != 0 {
				t.Fatal("canceled request loaded kernel")
			}
			if err := s.Close(context.Background()); err != nil {
				t.Fatal(err)
			}
		})
	}
}
