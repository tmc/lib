package moonep

import (
	"errors"
	"fmt"
	"sync"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/nvlshm"
)

// A Buffer owns MoonEP's communication state on one rank: the NVLink shared
// hidden and meta buffers plus the local scratch allocations the kernels
// expect. Create it collectively with [NewBuffer]; tear it down collectively
// with [Buffer.Close].
type Buffer struct {
	Layout *Layout

	// Hidden is the token payload buffer: R chunks of [NvSPadded, H] bf16;
	// dispatch writes rows directly into remote ranks' chunks.
	Hidden *nvlshm.DistBuffer
	// Meta is the metadata buffer: R chunks of [MetaChunkPadded] int32
	// with the region layout described in the package documentation, plus
	// a multicast mapping used by planning's one-shot plan broadcast.
	Meta *nvlshm.DistBuffer

	// Local scratch (device allocations, sizes in int32 elements):
	Alloc         driver.DevicePtr // [E*R]
	GroupTokens   driver.DevicePtr // [R]
	Z             driver.DevicePtr // [R*R]
	LocalHist     driver.DevicePtr // [E*NumVBlocks]
	GridSyncBar   driver.DevicePtr // [1], zeroed once, self-resetting
	PrimaryPacked driver.DevicePtr // [R*S]
	Kmask         driver.DevicePtr // [R*S]
	KidxToLoff    driver.DevicePtr // [R*S*K]
	BuilderBar    driver.DevicePtr // [1], zeroed once, self-resetting

	boot nvlshm.Bootstrap
	dev  driver.Device

	mu     sync.Mutex
	closed bool
}

// NewBuffer collectively allocates all MoonEP buffers for cfg on dev.
// Every rank must call NewBuffer together with an identical cfg. The device
// must support NVSwitch multicast (the plan broadcast requires it); check
// nvlshm.MulticastSupported first if a fallback is needed.
func NewBuffer(b nvlshm.Bootstrap, dev driver.Device, cfg Config) (*Buffer, error) {
	if cfg.R != b.WorldSize() {
		return nil, fmt.Errorf("moonep: cfg.R (%d) must equal world size (%d)", cfg.R, b.WorldSize())
	}
	vmmGran, err := nvlshm.Granularity(dev)
	if err != nil {
		return nil, err
	}
	mcGran, err := nvlshm.MulticastGranularity(cfg.R)
	if err != nil {
		return nil, err
	}
	l, err := NewLayout(cfg, vmmGran, mcGran)
	if err != nil {
		return nil, err
	}

	buf := &Buffer{Layout: l, boot: b, dev: dev}
	fail := func(err error) (*Buffer, error) {
		buf.freeLocal()
		if buf.Meta != nil {
			buf.Meta.Close()
		}
		if buf.Hidden != nil {
			buf.Hidden.Close()
		}
		return nil, err
	}

	buf.Hidden, err = nvlshm.DistAlloc(b, dev, l.HiddenChunkBytes())
	if err != nil {
		return fail(fmt.Errorf("moonep: hidden_buf: %w", err))
	}
	buf.Meta, err = nvlshm.DistAllocMulticast(b, dev, l.MetaChunkBytes())
	if err != nil {
		return fail(fmt.Errorf("moonep: meta_buf: %w", err))
	}

	// The cross-rank barrier protocol needs its slots all-zero on every
	// rank before the first kernel. Every rank zeroes the barrier region
	// in every rank's chunk (redundant but harmless, as in MoonEP), then
	// all ranks synchronize.
	for r := 0; r < l.R; r++ {
		p := buf.Meta.ChunkPtr(r) + driver.DevicePtr(l.BarrierOff*int32Bytes)
		if err := p.MemsetD32(0, barrierSlots); err != nil {
			return fail(fmt.Errorf("moonep: zero barrier region (rank %d): %w", r, err))
		}
	}
	if err := driver.CtxSynchronize(); err != nil {
		return fail(fmt.Errorf("moonep: sync after barrier init: %w", err))
	}
	if err := b.Barrier(); err != nil {
		return fail(fmt.Errorf("moonep: barrier after init: %w", err))
	}

	// Local scratch. The two barrier counters are zeroed once and never
	// again: both are self-resetting (sentinel-bit / TAG protocols).
	locals := []struct {
		dst   *driver.DevicePtr
		elems int
		zero  bool
	}{
		{&buf.Alloc, l.E * l.R, false},
		{&buf.GroupTokens, l.R, false},
		{&buf.Z, l.R * l.R, false},
		{&buf.LocalHist, l.E * l.NumVBlocks, false},
		{&buf.GridSyncBar, 1, true},
		{&buf.PrimaryPacked, l.R * l.S, false},
		{&buf.Kmask, l.R * l.S, false},
		{&buf.KidxToLoff, l.R * l.S * l.K, false},
		{&buf.BuilderBar, 1, true},
	}
	for _, a := range locals {
		p, err := driver.MemAlloc(uint(a.elems * int32Bytes))
		if err != nil {
			return fail(fmt.Errorf("moonep: local scratch (%d elems): %w", a.elems, err))
		}
		*a.dst = p
		if a.zero {
			if err := p.MemsetD32(0, uint(a.elems)); err != nil {
				return fail(fmt.Errorf("moonep: zero local scratch: %w", err))
			}
		}
	}
	if err := driver.CtxSynchronize(); err != nil {
		return fail(fmt.Errorf("moonep: sync after local init: %w", err))
	}
	return buf, nil
}

// HiddenLocal returns this rank's [NvS, H] shard within Hidden.
func (b *Buffer) HiddenLocal() driver.DevicePtr {
	return b.Hidden.LocalPtr()
}

// WeightsLocal returns this rank's [NvS] fp32 route-weight region within Meta.
func (b *Buffer) WeightsLocal() driver.DevicePtr {
	return b.Meta.LocalPtr() + driver.DevicePtr(b.Layout.WeightsOff*int32Bytes)
}

// MetaMc returns the multicast address of the meta buffer (one chunk wide):
// a store at offset off lands at every rank's chunk offset off.
func (b *Buffer) MetaMc() driver.DevicePtr {
	mc, _ := b.Meta.McPtr()
	return mc
}

func (b *Buffer) freeLocal() {
	for _, p := range []*driver.DevicePtr{
		&b.Alloc, &b.GroupTokens, &b.Z, &b.LocalHist, &b.GridSyncBar,
		&b.PrimaryPacked, &b.Kmask, &b.KidxToLoff, &b.BuilderBar,
	} {
		if *p != 0 {
			(*p).MemFree()
			*p = 0
		}
	}
}

// Close collectively releases the buffer: it synchronizes the device,
// barriers with all ranks, then closes the shared views (meta with its
// multicast mapping first, then hidden) and frees the local scratch. Every
// rank must call Close together, before its Bootstrap is torn down. Close is
// idempotent.
func (b *Buffer) Close() error {
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.closed {
		return nil
	}
	b.closed = true
	errs := []error{driver.CtxSynchronize(), b.boot.Barrier()}
	if b.Meta != nil {
		errs = append(errs, b.Meta.Close())
	}
	if b.Hidden != nil {
		errs = append(errs, b.Hidden.Close())
	}
	b.freeLocal()
	return errors.Join(errs...)
}
