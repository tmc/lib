package nccltest

import (
	"errors"
	"fmt"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/nccl"
)

func drainRanks(states []rankState, defaultStream bool, selectContext func(driver.Context) error, synchronize func(driver.Stream) error) error {
	var result error
	for i, s := range states {
		if s.retainResources {
			result = errors.Join(result, fmt.Errorf("rank %d resources retained after incomplete cleanup", i))
			continue
		}
		if s.context == 0 {
			continue
		}
		if err := selectContext(s.context); err != nil {
			result = errors.Join(result, fmt.Errorf("select rank %d context: %w", i, err))
			continue
		}
		if s.stream != 0 || defaultStream {
			if err := synchronize(s.stream); err != nil {
				result = errors.Join(result, fmt.Errorf("synchronize rank %d: %w", i, err))
			}
		}
	}
	return result
}

// A failed deregistration leaves the registration and its backing memory live.
// Do not destroy the communicator or release rank resources until every
// registration has been removed. A failed destroy likewise retains resources.
func releaseCommunicator(s *rankState, deregister func(nccl.Comm, nccl.Registration) error, destroy func(nccl.Comm) error) error {
	var result error
	for _, registration := range []*nccl.Registration{&s.sendRegistration, &s.recvRegistration} {
		if *registration == 0 {
			continue
		}
		if err := deregister(s.comm, *registration); err != nil {
			result = errors.Join(result, fmt.Errorf("deregister buffer: %w", err))
		} else {
			*registration = 0
		}
	}
	if result != nil {
		return result
	}
	if s.comm != 0 {
		if err := destroy(s.comm); err != nil {
			return fmt.Errorf("destroy communicator: %w", err)
		}
		s.comm = 0
	}
	return nil
}
