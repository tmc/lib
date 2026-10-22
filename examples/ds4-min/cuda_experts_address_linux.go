//go:build linux

package main

import (
	"errors"
	"fmt"

	"github.com/tmc/lib/examples/ds4-min/dsv4expertcache"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudalaunch"
	"github.com/tmc/lib/nvidia/x/cudamem"
)

// cudaExpertAddressLaunch owns the route tables and the borrowed selection
// until every reader on stream completes. It is request-local, not concurrent
// and not graph-capturable. The caller keeps kernels and activations live too.
type cudaExpertAddressLaunch struct {
	selection            *dsv4expertcache.CUDASelection
	stream               *cuda.Stream
	tables               [6]cudamem.Slice[uint64]
	width, hidden, slots int
	closing, finished    bool
}

// newCUDAExpertAddressLaunch takes exclusive selection ownership. On failure
// the returned owner must be retained and closed; no table is discarded while
// cleanup is uncertain. No consumer has been launched by this constructor.
func newCUDAExpertAddressLaunch(selection *dsv4expertcache.CUDASelection, stream *cuda.Stream, slots, width, hidden int) (*cudaExpertAddressLaunch, error) {
	if selection == nil || stream == nil {
		return nil, errors.New("missing selected expert owner or stream")
	}
	if err := selection.CheckStream(stream); err != nil {
		return nil, err
	}
	a, err := cudaSelectedExpertAddresses(selection, slots, width, hidden)
	if err != nil {
		return nil, err
	}
	o := &cudaExpertAddressLaunch{selection: selection, stream: stream, width: width, hidden: hidden, slots: slots}
	for i, host := range [][]uint64{a.gate, a.up, a.down, a.gateBytes, a.upBytes, a.downBytes} {
		o.tables[i], err = cudamem.Alloc[uint64](stream.Device(), len(host))
		if err == nil {
			err = o.tables[i].CopyFromHost(host)
		}
		if err != nil {
			o.closing = true
			return o, fmt.Errorf("upload selected expert table: %w", err)
		}
	}
	return o, nil
}

func (o *cudaExpertAddressLaunch) gateUp(kernel *cuda.Kernel, input, gate, up cudamem.Slice[float32]) error {
	if o == nil || o.closing || o.finished || kernel == nil {
		return errors.New("selected expert launch closed")
	}
	if input.Len() != o.width || gate.Len() != o.hidden*o.slots || up.Len() != gate.Len() {
		return errors.New("selected gate/up activation shape differs")
	}
	// Revalidate the actual committed views before each submission. A snapshot
	// of an address never permits launching after selection completion.
	if _, err := cudaSelectedExpertAddresses(o.selection, o.slots, o.width, o.hidden); err != nil {
		return err
	}
	if err := o.selection.CheckStream(o.stream); err != nil {
		return err
	}
	return cudalaunch.OwnedKernel(kernel).Grid(uint32(gate.Len()), 1, 1).Block(256, 1, 1).On(o.stream).Launch(o.tables[0], o.tables[1], o.tables[3], o.tables[4], input, gate, up, uint32(o.width), uint32(o.hidden), uint32(o.slots), uint64(input.Len()), uint64(gate.Len()))
}

func (o *cudaExpertAddressLaunch) down(kernel *cuda.Kernel, input, out cudamem.Slice[float32]) error {
	if o == nil || o.closing || o.finished || kernel == nil {
		return errors.New("selected expert launch closed")
	}
	if input.Len() != o.hidden*o.slots || out.Len() != o.width*o.slots {
		return errors.New("selected down activation shape differs")
	}
	if _, err := cudaSelectedExpertAddresses(o.selection, o.slots, o.width, o.hidden); err != nil {
		return err
	}
	if err := o.selection.CheckStream(o.stream); err != nil {
		return err
	}
	return cudalaunch.OwnedKernel(kernel).Grid(uint32(out.Len()), 1, 1).Block(256, 1, 1).On(o.stream).Launch(o.tables[2], o.tables[5], input, out, uint32(o.hidden), uint32(o.width), uint32(o.slots), uint64(input.Len()), uint64(out.Len()))
}

// Close records final-reader completion before releasing either the expert
// slots or argument tables. Every failure retains the remaining owners.
func (o *cudaExpertAddressLaunch) Close() error {
	if o == nil {
		return nil
	}
	o.closing = true
	var free [6]func() error
	for i := range o.tables {
		free[i] = func() error {
			if err := o.tables[i].Free(); err != nil {
				return err
			}
			o.tables[i] = cudamem.Slice[uint64]{}
			return nil
		}
	}
	return cudaCloseExpertAddresses(&o.finished, func() error { return o.selection.Finish(o.stream) }, free)
}
