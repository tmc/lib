package main

import "errors"

// Successful unload consumes the handle before restoring the previous CUDA
// context. A later context-restore error must not resurrect that handle.
func mxfp4UnloadInContext(context func(func() error) error, unload func() error, clear func()) error {
	return context(func() error {
		if err := unload(); err != nil {
			return err
		}
		clear()
		return nil
	})
}

type mxfp4DeviceRange struct{ address, bytes uint64 }

func checkMXFP4DeviceRanges(ranges ...mxfp4DeviceRange) error {
	for i, a := range ranges {
		if a.address == 0 || a.bytes == 0 || a.bytes > ^uint64(0)-a.address {
			return errors.New("cuda MXFP4: invalid device range")
		}
		for _, b := range ranges[:i] {
			if a.address < b.address+b.bytes && b.address < a.address+a.bytes {
				return errors.New("cuda MXFP4: overlapping operands")
			}
		}
	}
	return nil
}

// mxfp4ModuleLifetime keeps a failed CUDA release reachable for retry. Both
// callbacks are bound to the owning context; no borrowed operand is freed here.
type mxfp4ModuleLifetime struct {
	closed      bool
	synchronize func() error
	unload      func() error
}

func (s *mxfp4ModuleLifetime) close() error {
	s.closed = true
	if s.unload == nil {
		return nil
	}
	if err := s.synchronize(); err != nil {
		return err
	}
	if err := s.unload(); err != nil {
		return err
	}
	s.unload = nil
	return nil
}
