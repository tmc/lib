// Package cuptiowner coordinates cooperating metric collectors by CUDA device.
// Exclusion is a library measurement policy, not a claim that every native API
// pair is incompatible. It does not reserve external processes or activity tracing.
package cuptiowner

import (
	"fmt"
	"sync"
)

var owners = struct {
	sync.Mutex
	devices map[uint32]*Lease
}{devices: make(map[uint32]*Lease)}

// Lease reserves a device until native cleanup succeeds. Do not copy it.
type Lease struct{ device uint32 }

// AcquireDevice rejects another cooperating metric collector on this ordinal.
// Device zero is valid. All participants must use CUDA device ordinals.
func AcquireDevice(device uint32) (*Lease, error) {
	owners.Lock()
	defer owners.Unlock()
	if owners.devices[device] != nil {
		return nil, fmt.Errorf("cuptiowner: device %d already collecting", device)
	}
	lease := &Lease{device: device}
	owners.devices[device] = lease
	return lease, nil
}

// Release relinquishes ownership after native cleanup succeeds. It is nil-safe
// and cannot release a newer owner's lease.
func (l *Lease) Release() {
	if l == nil {
		return
	}
	owners.Lock()
	defer owners.Unlock()
	if owners.devices[l.device] == l {
		delete(owners.devices, l.device)
	}
}
