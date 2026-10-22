package main

import (
	_ "embed"
	"errors"
	"math"

	"github.com/tmc/lib/examples/ds4-min/dsv4expertcache"
)

//go:embed cuda_experts_address.cu
var cudaExpertAddressSource string

// cudaExpertAddresses contains only exact expert-component extents. The
// selection must remain live until its final consumer finishes; these values
// do not own storage or extend the borrowed views' lifetime.
type cudaExpertAddresses struct {
	gate, up, down                []uint64
	gateBytes, upBytes, downBytes []uint64
}

func cudaCloseExpertAddresses(finished *bool, finish func() error, free [6]func() error) error {
	if !*finished {
		if err := finish(); err != nil {
			return err
		}
		*finished = true
	}
	for _, f := range free {
		if err := f(); err != nil {
			return err
		}
	}
	return nil
}

func cudaExpertAddressExtent(cols, rows, slots int, blockBytes uint64) (uint64, error) {
	if cols <= 0 || cols%256 != 0 || uint64(cols) > math.MaxUint32 || rows <= 0 || uint64(rows) > math.MaxUint32 || slots <= 0 || uint64(slots) > math.MaxInt32/uint64(rows) {
		return 0, errors.New("invalid selected expert dimensions")
	}
	if blockBytes == 0 || uint64(cols/256) > math.MaxUint64/blockBytes {
		return 0, errors.New("selected expert extent overflow")
	}
	n := uint64(cols/256) * blockBytes
	if n == 0 || uint64(rows) > math.MaxUint64/n {
		return 0, errors.New("selected expert extent overflow")
	}
	return n * uint64(rows), nil
}

func cudaSelectedExpertAddresses(s *dsv4expertcache.CUDASelection, slots, width, hidden int) (cudaExpertAddresses, error) {
	return cudaCollectExpertAddresses(slots, width, hidden, func(slot int) ([3]uint64, [3]uint64, error) {
		var ptr, size [3]uint64
		views, err := s.Views(slot)
		if err != nil {
			return ptr, size, err
		}
		for i, v := range views {
			ptr[i], size[i], err = v.Address()
			if err != nil {
				return ptr, size, err
			}
		}
		return ptr, size, nil
	})
}

func cudaCollectExpertAddresses(slots, width, hidden int, resolve func(int) ([3]uint64, [3]uint64, error)) (cudaExpertAddresses, error) {
	var a cudaExpertAddresses
	gateBytes, err := cudaExpertAddressExtent(width, hidden, slots, 66)
	if err != nil {
		return a, err
	}
	downBytes, err := cudaExpertAddressExtent(hidden, width, slots, 84)
	if err != nil {
		return a, err
	}
	for slot := 0; slot < slots; slot++ {
		ptr, size, err := resolve(slot)
		if err != nil {
			return cudaExpertAddresses{}, err
		}
		for i := range ptr {
			if ptr[i] == 0 || ptr[i] > math.MaxUint64-size[i] {
				return cudaExpertAddresses{}, errors.New("invalid selected expert address")
			}
		}
		if size != [3]uint64{gateBytes, gateBytes, downBytes} {
			return cudaExpertAddresses{}, errors.New("selected expert component extent differs")
		}
		a.gate = append(a.gate, ptr[0])
		a.up = append(a.up, ptr[1])
		a.down = append(a.down, ptr[2])
		a.gateBytes = append(a.gateBytes, size[0])
		a.upBytes = append(a.upBytes, size[1])
		a.downBytes = append(a.downBytes, size[2])
	}
	return a, nil
}
