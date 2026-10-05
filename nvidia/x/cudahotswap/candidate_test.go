package cudahotswap

import (
	"context"
	"errors"
	"reflect"
	"testing"
)

func TestPrepareCandidateFlagsAndNoLease(t *testing.T) {
	for _, tc := range []struct {
		name     string
		geometry bool
		set      bool
		settings []string
		source   []string
		want     []string
	}{
		{name: "inherit", want: []string{"--fmad=false"}},
		{name: "geometry inherits", geometry: true, want: []string{"--fmad=false"}},
		{name: "source override", source: []string{"--use_fast_math"}, want: []string{"--use_fast_math"}},
		{name: "settings override", set: true, settings: []string{"--std=c++17"}, source: []string{"--use_fast_math"}, want: []string{"--std=c++17"}},
		{name: "explicit clear", set: true, source: []string{"--use_fast_math"}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			loader := &serviceLoader{kernel: &serviceKernel{raw: 1}}
			s, leases := testService(t, loader)
			if _, err := s.Register(context.Background(), "kernel", Source{Code: "original", Entry: "entry", Flags: []string{"--fmad=false"}}, "original"); err != nil {
				t.Fatal(err)
			}
			if tc.geometry {
				if _, err := s.SetGeometry("kernel", nil, "geometry"); err != nil {
					t.Fatal(err)
				}
			}
			if tc.set {
				if _, err := s.SetCompileFlags("kernel", tc.settings, "flags"); err != nil {
					t.Fatal(err)
				}
			}
			loader.compile = func(_ context.Context, source Source) (*Compilation, error) {
				if !reflect.DeepEqual(source.Flags, tc.want) {
					t.Fatalf("compiler flags = %v, want %v", source.Flags, tc.want)
				}
				if len(*leases) != 1 || (*leases)[0].calls != 1 {
					t.Fatal("compilation acquired or retained a GPU lease")
				}
				return &Compilation{Entry: "entry", PTX: ".version 7.0\n.target sm_70\n.address_size 64\n.visible .entry entry() { ret; }"}, nil
			}
			p, err := s.prepareCandidate(context.Background(), "kernel", Source{Code: "candidate", Entry: "entry", Flags: tc.source})
			if err != nil || p == nil {
				t.Fatalf("prepare: %v", err)
			}
			if len(tc.source) != 0 {
				tc.source[0] = "mutated"
			}
			if !reflect.DeepEqual(p.source.Flags, tc.want) || p.digest == "" || loader.loads != 1 || len(*leases) != 1 {
				t.Fatal("candidate aliases flags, lacks digest or performs GPU work")
			}
			if err := s.Close(context.Background()); err != nil {
				t.Fatal(err)
			}
		})
	}
}

func TestPrepareCandidateRefusesStaleAndABI(t *testing.T) {
	for _, change := range []string{"settings", "selection", "abi", "missing compilation", "wrong entry", "canceled"} {
		t.Run(change, func(t *testing.T) {
			loader := &serviceLoader{kernel: &serviceKernel{raw: 1}}
			s, leases := testService(t, loader)
			registerService(t, s)
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			loader.compile = func(context.Context, Source) (*Compilation, error) {
				c := &Compilation{Entry: "entry", PTX: ".version 7.0\n.target sm_70\n.address_size 64\n.visible .entry entry() { ret; }"}
				switch change {
				case "settings":
					if _, err := s.SetCompileFlags("kernel", nil, "changed"); err != nil {
						t.Fatal(err)
					}
				case "selection":
					s.mutations.Lock()
					stamp, err := s.table.revert("kernel")
					if err != nil {
						t.Fatal(err)
					}
					s.mu.Lock()
					state := s.states["kernel"]
					state.Generation = stamp.revision
					s.states["kernel"] = state
					s.mu.Unlock()
					s.mutations.Unlock()
				case "abi":
					c.PTX = ".version 7.0\n.target sm_70\n.address_size 64\n.visible .entry entry(.param .u64 ptr) { ret; }"
				case "missing compilation":
					return nil, nil
				case "wrong entry":
					c.Entry = "other"
				case "canceled":
					cancel()
				}
				return c, nil
			}
			p, err := s.prepareCandidate(ctx, "kernel", Source{Code: "candidate", Entry: "entry"})
			if p != nil || err == nil || loader.loads != 1 || len(*leases) != 1 {
				t.Fatalf("refusal = %v, %v; loads=%d leases=%d", p, err, loader.loads, len(*leases))
			}
			if change == "abi" {
				var abi *ABIError
				if !errors.As(err, &abi) {
					t.Fatalf("not an ABI error: %v", err)
				}
			}
			if err := s.Close(context.Background()); err != nil {
				t.Fatal(err)
			}
		})
	}
}

func TestPrepareCandidatePublicationRecheck(t *testing.T) {
	loader := &serviceLoader{kernel: &serviceKernel{raw: 1}}
	s, _ := testService(t, loader)
	registerService(t, s)
	p, err := s.prepareCandidate(context.Background(), "kernel", Source{Code: "candidate", Entry: "entry"})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := s.SetCompileFlags("kernel", nil, "later"); err != nil {
		t.Fatal(err)
	}
	s.mutations.Lock()
	err = s.validateCandidate(p)
	s.mutations.Unlock()
	if err == nil {
		t.Fatal("publication accepted a stale settings snapshot")
	}
	if err := s.Close(context.Background()); err != nil {
		t.Fatal(err)
	}
}

func TestPrepareCandidateRetainsCompilerCleanup(t *testing.T) {
	loader := &serviceLoader{kernel: &serviceKernel{raw: 1}}
	s, leases := testService(t, loader)
	registerService(t, s)
	owner := &serviceCloser{fail: true}
	loader.compile = func(context.Context, Source) (*Compilation, error) {
		return nil, &serviceCleanupError{owner}
	}
	if p, err := s.prepareCandidate(context.Background(), "kernel", Source{Code: "candidate", Entry: "entry"}); p != nil || err == nil {
		t.Fatalf("partial compiler cleanup accepted: %v, %v", p, err)
	}
	if len(s.cleanup) != 1 || owner.calls != 1 || len(*leases) != 1 {
		t.Fatal("compiler cleanup orphaned or acquired a lease")
	}
	owner.fail = false
	if err := s.Close(context.Background()); err != nil {
		t.Fatal(err)
	}
	if owner.calls != 2 || len(s.cleanup) != 0 {
		t.Fatal("compiler cleanup not retried exactly once")
	}
}
