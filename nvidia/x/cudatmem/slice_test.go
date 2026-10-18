package cudatmem_test

import (
	"errors"
	"sync"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudagroup"
	"github.com/tmc/lib/nvidia/x/cudalayout"
	"github.com/tmc/lib/nvidia/x/cudatmem"
)

func view(t *testing.T, x *cudatmem.Allocation, row cudatmem.Row, col cudatmem.Column, rows, cols uint32) cudatmem.Slice {
	t.Helper()
	s, err := x.Slice(row, col, rows, cols)
	if err != nil {
		t.Fatal(err)
	}
	return s
}

func accessInputs(t *testing.T, s cudatmem.Slice, warp uint32, cols int) (cudalayout.Tile, []cudagroup.Issuer, []cudatmem.Address) {
	t.Helper()
	layout, err := cudalayout.NewTile(32, cols, 4)
	if err != nil {
		t.Fatal(err)
	}
	base, err := s.Address(0, 0)
	if err != nil {
		t.Fatal(err)
	}
	bases := make([]cudatmem.Address, 32)
	for i := range bases {
		bases[i] = base
	}
	return layout, issuingWarp(warp), bases
}

func TestCheckedSliceBounds(t *testing.T) {
	a, ids, p := checkedAllocator(t, 128, 0)
	x := checkedAllocation(t, a, p, ids, 32, 128)
	for _, d := range [][4]uint32{{0, 0, 0, 1}, {0, 0, 1, 0}, {128, 0, 1, 1}, {0, 32, 1, 1}, {127, 31, 2, 1}, {0, 0, ^uint32(0), 1}, {^uint32(0), 0, 2, 1}, {0, ^uint32(0), 1, 2}} {
		if _, err := x.Slice(cudatmem.Row(d[0]), cudatmem.Column(d[1]), d[2], d[3]); !errors.Is(err, cudatmem.ErrInvalidUnit) {
			t.Fatalf("bounds %v: %v", d, err)
		}
	}
	s := view(t, x, 32, 3, 64, 10)
	sub, err := s.Sub(32, 2, 32, 2)
	if err != nil {
		t.Fatal(err)
	}
	got, err := sub.Address(31, 1)
	if err != nil || got != (cudatmem.Address{Row: 95, Column: 6}) {
		t.Fatalf("nested coordinate: %v %v", got, err)
	}
	for _, d := range [][4]uint32{{64, 0, 1, 1}, {0, 10, 1, 1}, {63, 0, 2, 1}, {0, 0, 0, 1}, {^uint32(0), 0, 2, 1}} {
		if _, err := s.Sub(cudatmem.Row(d[0]), cudatmem.Column(d[1]), d[2], d[3]); !errors.Is(err, cudatmem.ErrInvalidUnit) {
			t.Fatalf("sub bounds %v: %v", d, err)
		}
	}
	if _, err := sub.Address(32, 0); !errors.Is(err, cudatmem.ErrInvalidUnit) {
		t.Fatalf("row bound: %v", err)
	}
	if _, err := sub.Address(0, 2); !errors.Is(err, cudatmem.ErrInvalidUnit) {
		t.Fatalf("column bound: %v", err)
	}
	copyX := *x
	if _, err := copyX.Slice(0, 0, 32, 1); !errors.Is(err, cudatmem.ErrClosed) {
		t.Fatalf("copied parent admitted: %v", err)
	}
	finishChecked(t, a, p, ids, x)
}

func TestCheckedSliceRowChunks(t *testing.T) {
	a, ids, p := checkedAllocator(t, 256, 7)
	x := checkedAllocation(t, a, p, ids, 32, 128)
	for warp := uint32(0); warp < 8; warp++ {
		for _, cols := range []uint32{1, 2} {
			s := view(t, x, cudatmem.Row(32*(warp%4)), cudatmem.Column(30), 32, cols)
			layout, issuers, bases := accessInputs(t, s, warp, int(cols))
			use, err := s.BeginAccess(warp, layout, issuers, bases)
			if err != nil {
				t.Fatalf("warp %d columns %d: %v", warp, cols, err)
			}
			if err := a.ReleaseCollective(p, x, ids, columnArgs(32)); !errors.Is(err, cudatmem.ErrBusy) {
				t.Fatalf("active use released: %v", err)
			}
			if err := use.Complete(); err != nil {
				t.Fatal(err)
			}
		}
	}
	finishChecked(t, a, p, ids, x)
}

func TestCheckedSliceAccessRefusals(t *testing.T) {
	a, ids, p := checkedAllocator(t, 160, 1)
	x := checkedAllocation(t, a, p, ids, 32, 128)
	s := view(t, x, 32, 0, 32, 2)
	good, issuers, bases := accessInputs(t, s, 1, 2)
	for _, kind := range []string{"warp", "outside block", "overflow warp", "cta", "duplicate", "missing", "base count", "nonuniform base", "wrong uniform base", "row chunk"} {
		t.Run(kind, func(t *testing.T) {
			badIDs := append([]cudagroup.Issuer(nil), issuers...)
			badBases := append([]cudatmem.Address(nil), bases...)
			warp := uint32(1)
			target := s
			switch kind {
			case "warp":
				warp = 0
			case "outside block":
				warp = 5
				badIDs = issuingWarp(warp)
			case "overflow warp":
				warp = ^uint32(0)
				badIDs = issuingWarp(warp)
			case "cta":
				badIDs[0].CTA = 1
			case "duplicate":
				badIDs[31] = badIDs[0]
			case "missing":
				badIDs = badIDs[:31]
			case "base count":
				badBases = badBases[:31]
			case "nonuniform base":
				badBases[31].Column++
			case "wrong uniform base":
				for i := range badBases {
					badBases[i].Column++
				}
			case "row chunk":
				target = view(t, x, 64, 0, 32, 2)
				_, badIDs, badBases = accessInputs(t, target, 1, 2)
			}
			if use, err := target.BeginAccess(warp, good, badIDs, badBases); !errors.Is(err, cudatmem.ErrParticipation) || use != nil {
				t.Fatalf("invalid access admitted: %v %v", use, err)
			}
		})
	}
	for _, tc := range []struct {
		name              string
		rows, cols, bytes int
		opts              []cudalayout.Option
	}{
		{"rows", 16, 2, 4, nil}, {"width", 32, 1, 4, nil}, {"wide", 32, 4, 4, nil}, {"packed", 32, 2, 2, nil}, {"stride", 32, 2, 4, []cudalayout.Option{cudalayout.WithStride(8)}}, {"software swizzle", 32, 2, 4, []cudalayout.Option{cudalayout.WithStride(8), cudalayout.WithSwizzle(cudalayout.Swizzle32B)}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			layout, err := cudalayout.NewTile(tc.rows, tc.cols, tc.bytes, tc.opts...)
			if err != nil {
				t.Fatal(err)
			}
			if use, err := s.BeginAccess(1, layout, issuers, bases); !errors.Is(err, cudatmem.ErrInvalidUnit) || use != nil {
				t.Fatalf("unsupported layout admitted: %v %v", use, err)
			}
		})
	}
	if use, err := s.BeginAccess(1, cudalayout.Tile{}, issuers, bases); !errors.Is(err, cudatmem.ErrInvalidUnit) || use != nil {
		t.Fatalf("zero layout: %v %v", use, err)
	}
	partial := view(t, x, 32, 0, 16, 2)
	if use, err := partial.BeginAccess(1, good, issuers, bases); !errors.Is(err, cudatmem.ErrInvalidUnit) || use != nil {
		t.Fatalf("partial view: %v %v", use, err)
	}
	// Every refusal is side-effect free: a legal use and deallocation still work.
	use, err := s.BeginAccess(1, good, issuers, bases)
	if err != nil {
		t.Fatal(err)
	}
	if err := use.Complete(); err != nil {
		t.Fatal(err)
	}
	finishChecked(t, a, p, ids, x)
}

func TestCheckedSliceCompletionOwnership(t *testing.T) {
	a, ids, p := checkedAllocator(t, 32, 0)
	x := checkedAllocation(t, a, p, ids, 32, 32)
	s := view(t, x, 0, 0, 32, 2)
	layout, issuers, bases := accessInputs(t, s, 0, 2)
	u, err := s.BeginAccess(0, layout, issuers, bases)
	if err != nil {
		t.Fatal(err)
	}
	v, err := s.BeginAccess(0, layout, issuers, bases)
	if err != nil {
		t.Fatal(err)
	}
	alias := *u
	if err := u.Complete(); err != nil {
		t.Fatal(err)
	}
	if err := alias.Complete(); err != nil {
		t.Fatal(err)
	}
	if err := a.ReleaseCollective(p, x, ids, columnArgs(32)); !errors.Is(err, cudatmem.ErrBusy) {
		t.Fatalf("copy consumed another access: %v", err)
	}
	if err := a.RelinquishCollective(p, ids); err != nil {
		t.Fatal(err)
	}
	w, err := s.BeginAccess(0, layout, issuers, bases)
	if err != nil {
		t.Fatalf("live access after relinquish: %v", err)
	}
	if err := v.Complete(); err != nil {
		t.Fatal(err)
	}
	if err := a.Close(); !errors.Is(err, cudatmem.ErrBusy) {
		t.Fatalf("outstanding use closed: %v", err)
	}
	if err := w.Complete(); err != nil {
		t.Fatal(err)
	}
	if err := a.ReleaseCollective(p, x, ids, columnArgs(32)); err != nil {
		t.Fatal(err)
	}
	if err := a.Close(); err != nil {
		t.Fatal(err)
	}
	if err := alias.Complete(); err != nil {
		t.Fatalf("completed alias after close: %v", err)
	}
}

func TestCheckedSliceParentLifetimeAndReuse(t *testing.T) {
	a, ids, p := checkedAllocator(t, 32, 0)
	x := checkedAllocation(t, a, p, ids, 32, 32)
	s := view(t, x, 0, 0, 32, 2)
	alias := s
	layout, issuers, bases := accessInputs(t, s, 0, 2)
	if err := a.ReleaseCollective(p, x, ids, columnArgs(32)); err != nil {
		t.Fatal(err)
	}
	y := checkedAllocation(t, a, p, ids, 32, 32)
	if y.Offset() != x.Offset() {
		t.Fatalf("expected same offset: %d %d", y.Offset(), x.Offset())
	}
	for _, old := range []cudatmem.Slice{s, alias} {
		if _, err := old.Address(0, 0); !errors.Is(err, cudatmem.ErrClosed) {
			t.Fatalf("released parent address: %v", err)
		}
		if _, err := old.Sub(0, 0, 32, 1); !errors.Is(err, cudatmem.ErrClosed) {
			t.Fatalf("released parent subview: %v", err)
		}
		if use, err := old.BeginAccess(0, layout, issuers, bases); !errors.Is(err, cudatmem.ErrClosed) || use != nil {
			t.Fatalf("reused offset revived old view: %v %v", use, err)
		}
	}
	if _, err := x.Slice(0, 0, 32, 1); !errors.Is(err, cudatmem.ErrClosed) {
		t.Fatalf("released parent sliced: %v", err)
	}
	newView := view(t, y, 0, 0, 32, 2)
	layout, issuers, bases = accessInputs(t, newView, 0, 2)
	use, err := newView.BeginAccess(0, layout, issuers, bases)
	if err != nil {
		t.Fatal(err)
	}
	if err := use.Complete(); err != nil {
		t.Fatal(err)
	}
	finishChecked(t, a, p, ids, y)
}

func TestCheckedSliceConcurrentCompletion(t *testing.T) {
	a, ids, p := checkedAllocator(t, 32, 0)
	x := checkedAllocation(t, a, p, ids, 32, 32)
	s := view(t, x, 0, 0, 32, 1)
	layout, issuers, bases := accessInputs(t, s, 0, 1)
	use, err := s.BeginAccess(0, layout, issuers, bases)
	if err != nil {
		t.Fatal(err)
	}
	var wg sync.WaitGroup
	for i := 0; i < 16; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			alias := *use
			if err := alias.Complete(); err != nil {
				t.Error(err)
			}
			other, err := s.BeginAccess(0, layout, issuers, bases)
			if err != nil {
				t.Error(err)
				return
			}
			if err := other.Complete(); err != nil {
				t.Error(err)
			}
		}()
	}
	wg.Wait()
	finishChecked(t, a, p, ids, x)
}

func TestCheckedSliceZeroValues(t *testing.T) {
	var s cudatmem.Slice
	var x *cudatmem.Allocation
	var use cudatmem.Access
	var nilUse *cudatmem.Access
	if _, err := x.Slice(0, 0, 1, 1); !errors.Is(err, cudatmem.ErrClosed) {
		t.Fatal(err)
	}
	if _, err := s.Sub(0, 0, 1, 1); !errors.Is(err, cudatmem.ErrClosed) {
		t.Fatal(err)
	}
	if _, err := s.Address(0, 0); !errors.Is(err, cudatmem.ErrClosed) {
		t.Fatal(err)
	}
	if _, err := s.BeginAccess(0, cudalayout.Tile{}, nil, nil); !errors.Is(err, cudatmem.ErrClosed) {
		t.Fatal(err)
	}
	if err := use.Complete(); !errors.Is(err, cudatmem.ErrClosed) {
		t.Fatal(err)
	}
	if err := nilUse.Complete(); !errors.Is(err, cudatmem.ErrClosed) {
		t.Fatal(err)
	}
}
