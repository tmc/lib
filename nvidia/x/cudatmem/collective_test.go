package cudatmem_test

import (
	"errors"
	"testing"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cudagroup"
	"github.com/tmc/lib/nvidia/x/cudatmem"
)

func issuingWarp(warp uint32) []cudagroup.Issuer {
	ids := make([]cudagroup.Issuer, 32)
	for i := range ids {
		ids[i] = cudagroup.Issuer{CTA: 0, Thread: uint64(warp)*32 + uint64(i)}
	}
	return ids
}

func columnArgs(n uint32) []uint32 {
	args := make([]uint32, 32)
	for i := range args {
		args[i] = n
	}
	return args
}

func checkedAllocator(t *testing.T, threads, warp uint32) (*cudatmem.Allocator, []cudagroup.Issuer, *cudatmem.Permit) {
	t.Helper()
	one := driver.Dim3{X: 1, Y: 1, Z: 1}
	cluster, err := cudagroup.New(cudagroup.Cluster, one)
	if err != nil {
		t.Fatal(err)
	}
	block := driver.Dim3{X: threads, Y: 1, Z: 1}
	c, err := cudagroup.NewTMEMCollective("sm_100a", cudagroup.TMEMGroup1, cluster, block, 0, warp)
	if err != nil {
		t.Fatal(err)
	}
	a, err := cudatmem.NewCollective(cudatmem.SM100A, c, cudagroup.LaunchRequirements{Grid: one, Block: block})
	if err != nil {
		t.Fatal(err)
	}
	ids := issuingWarp(warp)
	p, err := a.AcquireCollectivePermit(ids)
	if err != nil {
		t.Fatal(err)
	}
	return a, ids, p
}

func checkedAllocation(t *testing.T, a *cudatmem.Allocator, p *cudatmem.Permit, ids []cudagroup.Issuer, columns, rows uint32) *cudatmem.Allocation {
	t.Helper()
	x, err := a.AllocateCollective(p, ids, columnArgs(columns), rows)
	if err != nil {
		t.Fatal(err)
	}
	return x
}

func finishChecked(t *testing.T, a *cudatmem.Allocator, p *cudatmem.Permit, ids []cudagroup.Issuer, x *cudatmem.Allocation) {
	t.Helper()
	if err := a.RelinquishCollective(p, ids); err != nil {
		t.Fatal(err)
	}
	if x != nil {
		if err := a.ReleaseCollective(p, x, ids, columnArgs(x.Columns())); err != nil {
			t.Fatal(err)
		}
	}
	if err := a.Close(); err != nil {
		t.Fatal(err)
	}
}

func TestCheckedCollectiveLaunch(t *testing.T) {
	one := driver.Dim3{X: 1, Y: 1, Z: 1}
	block := driver.Dim3{X: 64, Y: 1, Z: 1}
	cluster, _ := cudagroup.New(cudagroup.Cluster, one)
	c, _ := cudagroup.NewTMEMCollective("sm_100a", cudagroup.TMEMGroup1, cluster, block, 0, 1)
	for _, dims := range []driver.Dim3{{}, one} {
		req := cudagroup.LaunchRequirements{Grid: one, Block: block, Cluster: dims}
		original := req
		a, err := cudatmem.NewCollective(cudatmem.SM100A, c, req)
		if err != nil || req != original {
			t.Fatalf("launch %v: %v; changed=%v", dims, err, req != original)
		}
		p, err := a.AcquireCollectivePermit(issuingWarp(1))
		if err != nil {
			t.Fatal(err)
		}
		finishChecked(t, a, p, issuingWarp(1), nil)
	}
	two, _ := cudagroup.New(cudagroup.Cluster, driver.Dim3{X: 2, Y: 1, Z: 1})
	c2, _ := cudagroup.NewTMEMCollective("sm_100a", cudagroup.TMEMGroup2, two, block, 0, 0)
	cWide, _ := cudagroup.NewTMEMCollective("sm_100a", cudagroup.TMEMGroup1, two, block, 0, 0)
	good := cudagroup.LaunchRequirements{Grid: one, Block: block}
	for _, tc := range []struct {
		name   string
		target cudatmem.Target
		c      cudagroup.TMEMCollective
		req    cudagroup.LaunchRequirements
	}{
		{"target", cudatmem.SM121A, c, good},
		{"zero collective", cudatmem.SM100A, cudagroup.TMEMCollective{}, good},
		{"group2", cudatmem.SM100A, c2, good},
		{"wide collective", cudatmem.SM100A, cWide, good},
		{"zero grid", cudatmem.SM100A, c, cudagroup.LaunchRequirements{Block: block}},
		{"wide grid", cudatmem.SM100A, c, cudagroup.LaunchRequirements{Grid: driver.Dim3{X: 2, Y: 1, Z: 1}, Block: block}},
		{"block mismatch", cudatmem.SM100A, c, cudagroup.LaunchRequirements{Grid: one, Block: driver.Dim3{X: 32, Y: 1, Z: 1}}},
		{"wide cluster", cudatmem.SM100A, c, cudagroup.LaunchRequirements{Grid: one, Block: block, Cluster: driver.Dim3{X: 2, Y: 1, Z: 1}}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if a, err := cudatmem.NewCollective(tc.target, tc.c, tc.req); err == nil || a != nil {
				t.Fatalf("unsupported launch admitted: %v %v", a, err)
			}
		})
	}
}

func TestCheckedCollectiveExactIssuers(t *testing.T) {
	a, ids, p := checkedAllocator(t, 128, 2)
	for _, kind := range []string{"cta", "warp", "duplicate", "missing", "arguments", "nonuniform"} {
		t.Run(kind, func(t *testing.T) {
			bad := append([]cudagroup.Issuer(nil), ids...)
			args := columnArgs(32)
			switch kind {
			case "cta":
				bad[0].CTA = 1
			case "warp":
				bad[0].Thread = 0
			case "duplicate":
				bad[31] = bad[0]
			case "missing":
				bad = bad[:31]
			case "arguments":
				args = args[:31]
			case "nonuniform":
				args[31] = 64
			}
			if x, err := a.AllocateCollective(p, bad, args, 128); !errors.Is(err, cudatmem.ErrParticipation) || x != nil {
				t.Fatalf("invalid issuing set admitted: %v %v", x, err)
			}
			if err := a.ReleaseCollective(p, nil, bad, args); !errors.Is(err, cudatmem.ErrParticipation) {
				t.Fatalf("release issuing set: %v", err)
			}
			if kind != "arguments" && kind != "nonuniform" {
				if err := a.RelinquishCollective(p, bad); !errors.Is(err, cudatmem.ErrParticipation) {
					t.Fatalf("relinquish issuing set: %v", err)
				}
			}
		})
	}
	// Input mutation after admission cannot change the stored collective.
	x := checkedAllocation(t, a, p, ids, 64, 128)
	ids[0].Thread = 0
	if err := a.ReleaseCollective(p, x, ids, columnArgs(64)); !errors.Is(err, cudatmem.ErrParticipation) {
		t.Fatalf("retained input alias: %v", err)
	}
	ids = issuingWarp(2)
	if err := a.ReleaseCollective(p, x, ids, columnArgs(32)); !errors.Is(err, cudatmem.ErrInvalidUnit) {
		t.Fatalf("wrong deallocation columns: %v", err)
	}
	finishChecked(t, a, p, ids, x)
}

func TestCheckedCollectiveNoCountBypass(t *testing.T) {
	a, ids, p := checkedAllocator(t, 32, 0)
	x := checkedAllocation(t, a, p, ids, 32, 32)
	if q, err := a.AcquirePermit(32); !errors.Is(err, cudatmem.ErrPermission) || q != nil {
		t.Fatalf("count-only permit: %v %v", q, err)
	}
	if y, err := a.Allocate(p, 32, 32, 32); !errors.Is(err, cudatmem.ErrPermission) || y != nil {
		t.Fatalf("count-only allocation: %v %v", y, err)
	}
	for name, err := range map[string]error{"release": a.Release(p, x, 32), "relinquish": a.Relinquish(p), "begin": x.BeginUser(), "complete": x.CompleteUser()} {
		if !errors.Is(err, cudatmem.ErrPermission) {
			t.Errorf("count-only %s: %v", name, err)
		}
	}
	legacy, _ := cudatmem.New(cudatmem.SM100A)
	q, _ := legacy.AcquirePermit(32)
	y, _ := legacy.Allocate(q, 32, 32, 32)
	if _, err := legacy.AcquireCollectivePermit(ids); !errors.Is(err, cudatmem.ErrPermission) {
		t.Fatalf("legacy checked permit: %v", err)
	}
	if _, err := legacy.AllocateCollective(q, ids, columnArgs(32), 32); !errors.Is(err, cudatmem.ErrPermission) {
		t.Fatalf("legacy checked allocation: %v", err)
	}
	if err := legacy.ReleaseCollective(q, y, ids, columnArgs(32)); !errors.Is(err, cudatmem.ErrPermission) {
		t.Fatalf("legacy checked release: %v", err)
	}
	if err := legacy.RelinquishCollective(q, ids); !errors.Is(err, cudatmem.ErrPermission) {
		t.Fatalf("legacy checked relinquish: %v", err)
	}
	if _, err := y.Slice(0, 0, 32, 1); !errors.Is(err, cudatmem.ErrPermission) {
		t.Fatalf("legacy slice: %v", err)
	}
	if err := legacy.Release(q, y, 32); err != nil {
		t.Fatal(err)
	}
	if err := legacy.Relinquish(q); err != nil {
		t.Fatal(err)
	}
	if err := legacy.Close(); err != nil {
		t.Fatal(err)
	}
	finishChecked(t, a, p, ids, x)
}

func TestCheckedCollectiveExecutionHistory(t *testing.T) {
	a, ids, p := checkedAllocator(t, 64, 1)
	if _, err := a.AcquireCollectivePermit(ids); !errors.Is(err, cudatmem.ErrPermission) {
		t.Fatalf("second execution permit: %v", err)
	}
	for _, dims := range [][2]uint32{{0, 32}, {31, 32}, {96, 32}, {1024, 32}, {32, 0}, {32, 129}, {^uint32(0), ^uint32(0)}} {
		if _, err := a.AllocateCollective(p, ids, columnArgs(dims[0]), dims[1]); !errors.Is(err, cudatmem.ErrInvalidUnit) {
			t.Fatalf("invalid allocation %v: %v", dims, err)
		}
	}
	x := checkedAllocation(t, a, p, ids, 64, 128)
	if err := a.ReleaseCollective(p, x, ids, columnArgs(64)); err != nil {
		t.Fatal(err)
	}
	if _, err := a.AllocateCollective(p, ids, columnArgs(128), 128); !errors.Is(err, cudatmem.ErrInvalidUnit) {
		t.Fatalf("history reset after release: %v", err)
	}
	y := checkedAllocation(t, a, p, ids, 32, 128)
	if y.Offset() != x.Offset() {
		t.Fatalf("pool not reused: %d %d", x.Offset(), y.Offset())
	}
	pcopy := *p
	if _, err := a.AllocateCollective(&pcopy, ids, columnArgs(32), 128); !errors.Is(err, cudatmem.ErrPermission) {
		t.Fatalf("copied permit allocation: %v", err)
	}
	if err := a.ReleaseCollective(&pcopy, y, ids, columnArgs(32)); !errors.Is(err, cudatmem.ErrPermission) {
		t.Fatalf("copied permit release: %v", err)
	}
	b, bids, q := checkedAllocator(t, 64, 1)
	if _, err := b.AllocateCollective(p, bids, columnArgs(32), 32); !errors.Is(err, cudatmem.ErrPermission) {
		t.Fatalf("foreign permit: %v", err)
	}
	finishChecked(t, b, q, bids, nil)
	if err := a.RelinquishCollective(p, ids); err != nil {
		t.Fatal(err)
	}
	if _, err := a.AcquireCollectivePermit(ids); !errors.Is(err, cudatmem.ErrPermission) {
		t.Fatalf("reacquired after relinquish: %v", err)
	}
	if _, err := a.AllocateCollective(p, ids, columnArgs(32), 128); !errors.Is(err, cudatmem.ErrPermission) {
		t.Fatalf("allocation after relinquish: %v", err)
	}
	if err := a.Close(); !errors.Is(err, cudatmem.ErrBusy) {
		t.Fatalf("live allocation closed: %v", err)
	}
	if err := a.ReleaseCollective(p, y, ids, columnArgs(32)); err != nil {
		t.Fatal(err)
	}
	if err := a.Close(); err != nil {
		t.Fatal(err)
	}
}
