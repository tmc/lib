//go:build darwin && arm64

package dsv4expertcache

import (
	"context"
	"errors"
	"fmt"
	"os"

	"github.com/tmc/apple/metal"
	libmetal "github.com/tmc/lib/apple/x/metal"
	"github.com/tmc/lib/apple/x/metal/metalmem"
	"golang.org/x/sys/unix"
)

type metalExpertSlab struct {
	start, count int
	data         metalmem.Slice[byte]
}
type metalUploadDevice struct {
	device            *libmetal.Device
	slots             int
	strides           [3]int
	slotBytes, target int
	staging           []byte
	stageLocked       bool
	slabs             []metalExpertSlab
	locked            map[int]bool
}

// NewMetalUpload allocates pinned staging and lazily grows shared Metal slabs.
// Each used page-aligned expert slot is pinned before its first write. The
// dominant triplet size class is immutable; mixed-size layers need mapped views.
// Budget bounds all page-padded expert slots, not just logical tensor bytes.
// Staging reserves an additional page-rounded maximum component stride.
// A non-nil owner returned with an error retains unfinished cleanup.
func NewMetalUpload(device *libmetal.Device, slots int, strides [3]int, budget uint64) (*MetalUpload, error) {
	if device == nil {
		return nil, errors.New("missing Metal device")
	}
	return newMetalUpload(&metalUploadDevice{device: device}, slots, strides, budget)
}

func (b *metalUploadDevice) pageSize() uint64 { return uint64(os.Getpagesize()) }

func (b *metalUploadDevice) allocate(slots int, strides [3]int) error {
	b.slots, b.strides = slots, strides
	b.locked = make(map[int]bool)
	var total uint64
	for _, n := range strides {
		total += uint64(n)
	}
	page := uint64(os.Getpagesize())
	if total > uint64(^uint(0)>>1)-page+1 {
		return errors.New("Metal slot extent overflow")
	}
	b.slotBytes = int((total + page - 1) / page * page)
	maxBuffer := uint64(b.device.Raw().MaxBufferLength())
	if uint64(b.slotBytes) > maxBuffer {
		return errors.New("Metal expert exceeds maximum buffer")
	}
	b.target = int(min(metalSlabTarget(), maxBuffer, uint64(^uint(0)>>1)))
	n := max(strides[0], strides[1], strides[2])
	var err error
	b.staging, err = unix.Mmap(-1, 0, n, unix.PROT_READ|unix.PROT_WRITE, unix.MAP_ANON|unix.MAP_PRIVATE)
	if err != nil {
		return fmt.Errorf("map Metal staging: %w", err)
	}
	if err = unix.Mlock(b.staging); err != nil {
		return fmt.Errorf("pin Metal staging: %w", err)
	}
	b.stageLocked = true
	return nil
}
func (b *metalUploadDevice) stage() []byte { return b.staging }

func (b *metalUploadDevice) slot(slot int) (metalmem.Slice[byte], error) {
	if slot < 0 || slot >= b.slots {
		return metalmem.Slice[byte]{}, errors.New("invalid Metal slot")
	}
	for {
		for _, s := range b.slabs {
			if slot >= s.start && slot < s.start+s.count {
				off := (slot - s.start) * b.slotBytes
				v, err := s.data.Slice(off, off+b.slotBytes)
				if err != nil {
					return metalmem.Slice[byte]{}, err
				}
				if !b.locked[slot] {
					if err := unix.Mlock(v.HostSlice()); err != nil {
						return metalmem.Slice[byte]{}, &MetalPinFailure{LockedSlots: uint32(len(b.locked)), LockedBytes: uint64(len(b.locked)) * uint64(b.slotBytes), Err: err}
					}
					b.locked[slot] = true
				}
				return v, nil
			}
		}
		start := 0
		if len(b.slabs) > 0 {
			s := b.slabs[len(b.slabs)-1]
			start = s.start + s.count
		}
		count := min(max(1, b.target/b.slotBytes), b.slots-start)
		if len(b.slabs) >= 256 {
			return metalmem.Slice[byte]{}, errors.New("Metal expert slab limit")
		}
		var data metalmem.Slice[byte]
		var err error
		for count > 0 {
			data, err = metalmem.Alloc[byte](b.device, count*b.slotBytes)
			if err == nil {
				break
			}
			count /= 2
		}
		if count == 0 {
			return metalmem.Slice[byte]{}, fmt.Errorf("grow Metal expert slab: %w", err)
		}
		b.slabs = append(b.slabs, metalExpertSlab{start, count, data})
	}
}

func (b *metalUploadDevice) write(slot, component int, src []byte) error {
	v, err := b.slot(slot)
	if err != nil {
		return err
	}
	off := 0
	for i := 0; i < component; i++ {
		off += b.strides[i]
	}
	copy(v.HostSlice()[off:off+len(src)], src)
	return nil
}

func (b *metalUploadDevice) close() error {
	for slot := range b.locked {
		for _, s := range b.slabs {
			if slot >= s.start && slot < s.start+s.count {
				off := (slot - s.start) * b.slotBytes
				v, err := s.data.Slice(off, off+b.slotBytes)
				if err != nil {
					return err
				}
				if err := unix.Munlock(v.HostSlice()); err != nil {
					return fmt.Errorf("unpin Metal slot: %w", err)
				}
				delete(b.locked, slot)
				break
			}
		}
	}
	for _, s := range b.slabs {
		s.data.Free()
	}
	b.slabs = nil
	if b.stageLocked {
		if err := unix.Munlock(b.staging); err != nil {
			return fmt.Errorf("unpin Metal stage: %w", err)
		}
		b.stageLocked = false
	}
	if b.staging != nil {
		if err := unix.Munmap(b.staging); err != nil {
			return fmt.Errorf("unmap Metal stage: %w", err)
		}
		b.staging = nil
	}
	return nil
}

// Views returns bounded borrowed gate/up/down slices for one route position.
// Their backing owner belongs to the uploader: callers must not Free them.
// They are valid only until this selection's Release completes.
func (s *MetalSelection) Views(route int) ([3]metalmem.Slice[byte], error) {
	var out [3]metalmem.Slice[byte]
	if err := s.check(); err != nil {
		return out, err
	}
	ids := s.lease.Slots()
	if route < 0 || route >= len(ids) {
		return out, errors.New("invalid Metal route")
	}
	b, ok := s.owner.backend.(*metalUploadDevice)
	if !ok {
		return out, errors.New("Metal backend lacks shared views")
	}
	v, err := b.slot(ids[route])
	if err != nil {
		return out, err
	}
	off := 0
	for i, n := range b.strides {
		out[i], err = v.Slice(off, off+n)
		if err != nil {
			return [3]metalmem.Slice[byte]{}, err
		}
		off += n
	}
	return out, nil
}

// MetalCommandCompletion waits for a committed command buffer's terminal
// status, including errors. It never releases the command buffer owner.
type MetalCommandCompletion struct {
	Context context.Context
	Buffer  metal.MTLCommandBuffer
}

func (c MetalCommandCompletion) Wait() error { return libmetal.WaitCommandBuffer(c.Context, c.Buffer) }
