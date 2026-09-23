//go:build !windows

package moonep

import (
	"errors"
	"fmt"
	"sync"

	"github.com/tmc/lib/nvidia/driver"
)

// A CommPlan holds the device-side outputs of MoonEP planning, mirroring
// MoonEPCommPlan (moonep/planning.py). All tensors are int32; sizes are in
// elements. Every allocation is over-allocated to a multiple of 4 elements
// because the kernels' grid-stride writes are vectorized up to 16 bytes and
// may write one full vector past the logical tail.
//
// Dst uses MoonEP's wire encoding: a non-negative value v routes flat top-k
// entry i to rank v/NvS, slot v%NvS; a negative value marks a duplicate
// whose primary destination is -v-1.
type CommPlan struct {
	Dst            driver.DevicePtr // [N]
	CuSeqlens      driver.DevicePtr // [E+B] padded end offsets per VM group
	ExpertsToCopy  driver.DevicePtr // [R, B]
	ZeroFillRanges driver.DevicePtr // [E+B, 2]
	RemoteStats    driver.DevicePtr // [2]
	DupGroups      driver.DevicePtr // [NvS, 3]
	DupLoffs       driver.DevicePtr // [NvS]
	DupCounts      driver.DevicePtr // [2] = {n_groups, n_dup_loffs}

	N, R, E, B, NvS, K int

	mu     sync.Mutex
	closed bool
}

// AllocatePlan allocates a plan's device tensors for l. Contents are
// uninitialized: fresh planning fills them on device.
func AllocatePlan(l *Layout) (*CommPlan, error) {
	p := &CommPlan{N: l.N, R: l.R, E: l.E, B: l.B, NvS: l.NvS, K: l.K}
	allocs := []struct {
		dst   *driver.DevicePtr
		elems int
	}{
		{&p.Dst, l.N},
		{&p.CuSeqlens, l.E + l.B},
		{&p.ExpertsToCopy, l.R * l.B},
		{&p.ZeroFillRanges, (l.E + l.B) * 2},
		{&p.RemoteStats, 2},
		{&p.DupGroups, l.NvS * 3},
		{&p.DupLoffs, l.NvS},
		{&p.DupCounts, 2},
	}
	for _, a := range allocs {
		ptr, err := driver.MemAlloc(uint(alignUp(a.elems, 4) * int32Bytes))
		if err != nil {
			p.Close()
			return nil, fmt.Errorf("moonep: allocate plan (%d elems): %w", a.elems, err)
		}
		*a.dst = ptr
	}
	return p, nil
}

// Close frees the plan's device memory. The plan must not be referenced by
// any in-flight kernel. Close is idempotent.
func (p *CommPlan) Close() error {
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.closed {
		return nil
	}
	p.closed = true
	var errs []error
	for _, d := range []*driver.DevicePtr{
		&p.Dst, &p.CuSeqlens, &p.ExpertsToCopy, &p.ZeroFillRanges,
		&p.RemoteStats, &p.DupGroups, &p.DupLoffs, &p.DupCounts,
	} {
		if *d != 0 {
			errs = append(errs, (*d).MemFree())
			*d = 0
		}
	}
	return errors.Join(errs...)
}
