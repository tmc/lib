package cudahotswap_test

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudahotswap"
	"github.com/tmc/lib/nvidia/x/cudalaunch"
)

var ownerValueFailure = errors.New("test load failure")
var ownerCloseFailure = errors.New("test close failure")

type ownerValue struct {
	tag   any
	calls *int
	fail  *bool
}

func (k ownerValue) Raw() driver.Function                     { return 1 }
func (k ownerValue) Launch(cudalaunch.Config, ...any) error   { return nil }
func (k ownerValue) Wait(context.Context, *cuda.Stream) error { return nil }
func (k ownerValue) Close() error {
	(*k.calls)++
	if *k.fail {
		return ownerCloseFailure
	}
	return nil
}
func (k ownerValue) Error() string     { return ownerValueFailure.Error() }
func (k ownerValue) Is(err error) bool { return err == ownerValueFailure }

type sliceOwner struct {
	ownerValue
	data []byte
}

type ownerValueLoader struct {
	kernel cudahotswap.Kernel
	err    error
}

func (l ownerValueLoader) Compile(context.Context, cudahotswap.Source) (*cudahotswap.Compilation, error) {
	return &cudahotswap.Compilation{PTX: ".version 7.0\n.target sm_70\n.address_size 64\n.visible .entry entry() { ret; }", Entry: "entry"}, nil
}
func (l ownerValueLoader) Load(context.Context, *cudahotswap.Compilation) (cudahotswap.Kernel, error) {
	return l.kernel, l.err
}

func TestPublicRegisterOwnerIdentityRecovery(t *testing.T) {
	for _, name := range []string{"alias-interface", "alias-slice", "distinct-interface", "distinct-mixed", "shared-pointer", "distinct-pointers", "alias-close-retry"} {
		t.Run(name, func(t *testing.T) {
			dir := filepath.Join(t.TempDir(), "lock")
			t.Setenv("GPU_LOCKDIR", dir)
			t.Setenv("GPU_LOCK_HELD", "")
			first, second := 0, 0
			fail := name == "alias-close-retry"
			a := ownerValue{tag: []byte{1}, calls: &first, fail: &fail}
			b := ownerValue{tag: []byte{2}, calls: &second, fail: &fail}
			l := ownerValueLoader{kernel: a, err: a}
			groups := [][]int{{0, 1}}
			wantSecond := 0
			ambiguous := true
			switch name {
			case "alias-slice":
				v := sliceOwner{ownerValue: a, data: []byte{1}}
				l.kernel, l.err = v, v
			case "distinct-interface", "distinct-mixed":
				if name == "distinct-mixed" {
					b.tag = "comparable value"
				}
				l.err = b
				groups, wantSecond = [][]int{{0}, {1}}, 1
			case "shared-pointer":
				l.kernel, l.err, ambiguous = &a, &a, false
			case "distinct-pointers":
				l.kernel, l.err, ambiguous, wantSecond = &a, &b, false, 1
			}
			s, err := cudahotswap.NewService(l)
			if err != nil {
				t.Fatal(err)
			}
			ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			defer cancel()
			_, err = s.Register(ctx, "kernel", cudahotswap.Source{Code: "host fixture", Entry: "entry"}, "owner identity")
			if !errors.Is(err, ownerValueFailure) {
				t.Fatalf("load cause lost: %v", err)
			}
			var identity *cudahotswap.CleanupIdentityError
			if errors.As(err, &identity) != ambiguous {
				t.Fatalf("identity error = %v, want %t", err, ambiguous)
			}
			if ambiguous {
				if first != 0 || second != 0 {
					t.Fatal("ambiguous cleanup called an owner")
				}
				if _, err := os.Stat(dir); err != nil {
					t.Fatalf("ambiguous cleanup released lease: %v", err)
				}
				owners := identity.Owners()
				if len(owners) != 2 {
					t.Fatalf("retained owners = %d, want 2", len(owners))
				}
				owners[0] = nil
				if identity.Owners()[0] == nil {
					t.Fatal("Owners aliases service storage")
				}
				if err := s.Close(ctx); !errors.As(err, &identity) || first != 0 || second != 0 {
					t.Fatalf("unresolved close = %v; calls %d,%d", err, first, second)
				}
				if err := identity.ResolveAliases(groups...); err != nil {
					t.Fatal(err)
				}
				if first != 0 || second != 0 {
					t.Fatal("resolution closed an owner")
				}
				if err := identity.ResolveAliases(groups...); err == nil {
					t.Fatal("resolved snapshot accepted another resolution")
				}
			}
			if fail {
				if err := s.Close(ctx); !errors.Is(err, ownerCloseFailure) || first != 1 {
					t.Fatalf("failed close = %v, calls %d", err, first)
				}
				if _, err := os.Stat(dir); err != nil {
					t.Fatalf("failed owner close released lease: %v", err)
				}
				fail = false
			}
			if err := s.Close(ctx); err != nil {
				t.Fatal(err)
			}
			wantFirst := 1
			if name == "alias-close-retry" {
				wantFirst = 2
			}
			if first != wantFirst || second != wantSecond {
				t.Fatalf("close counts = %d,%d, want %d,%d", first, second, wantFirst, wantSecond)
			}
			if _, err := os.Stat(dir); !errors.Is(err, os.ErrNotExist) {
				t.Fatalf("successful cleanup kept lease: %v", err)
			}
		})
	}
}

func TestPublicPushOwnerIdentityRecovery(t *testing.T) {
	dir := filepath.Join(t.TempDir(), "lock")
	t.Setenv("GPU_LOCKDIR", dir)
	t.Setenv("GPU_LOCK_HELD", "")
	baseCalls, candidateCalls := 0, 0
	fail := false
	base := &ownerValue{calls: &baseCalls, fail: &fail}
	l := &ownerValueLoader{kernel: base}
	s, err := cudahotswap.NewService(l)
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if _, err := s.Register(ctx, "kernel", cudahotswap.Source{Code: "original", Entry: "entry"}, "original"); err != nil {
		t.Fatal(err)
	}
	if _, err := s.SetCheck("kernel", cudahotswap.CheckConfig{Checker: func(context.Context, cudahotswap.State) error { return nil }}, "check"); err != nil {
		t.Fatal(err)
	}
	candidate := ownerValue{tag: []byte{1}, calls: &candidateCalls, fail: &fail}
	l.kernel, l.err = candidate, candidate
	_, err = s.Push(ctx, "kernel", cudahotswap.Source{Code: "candidate", Entry: "entry"}, "candidate")
	var identity *cudahotswap.CleanupIdentityError
	if !errors.Is(err, ownerValueFailure) || !errors.As(err, &identity) {
		t.Fatalf("Push lost load or identity error: %v", err)
	}
	if candidateCalls != 0 || baseCalls != 0 {
		t.Fatal("ambiguous Push closed a dependent owner")
	}
	if _, err := os.Stat(dir); err != nil {
		t.Fatalf("ambiguous Push released lease: %v", err)
	}
	if err := identity.ResolveAliases([]int{0, 1}); err != nil {
		t.Fatal(err)
	}
	if err := s.Close(ctx); err != nil {
		t.Fatal(err)
	}
	if candidateCalls != 1 || baseCalls != 1 {
		t.Fatalf("Push recovery closes = %d,%d, want 1,1", candidateCalls, baseCalls)
	}
}
