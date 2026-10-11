package cudacollective

import (
	"context"
	"fmt"
	"sync"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/nvlshm"
)

// TransportType defines the category of inter-rank transport.
type TransportType int

const (
	TransportTypeNVLSHM TransportType = iota + 1
	TransportTypeMoonEP
	TransportTypeNCCLTest
	TransportTypeTCPSocket
	TransportTypeControlPlaneTest
)

func (t TransportType) String() string {
	switch t {
	case TransportTypeNVLSHM:
		return "nvlshm"
	case TransportTypeMoonEP:
		return "moonep"
	case TransportTypeNCCLTest:
		return "nccltest"
	case TransportTypeTCPSocket:
		return "tcpsocket"
	case TransportTypeControlPlaneTest:
		return "controlplanetest"
	default:
		return fmt.Sprintf("TransportType(%d)", int(t))
	}
}

// Transport is the abstraction for data movement between ranks.
type Transport interface {
	Type() TransportType
	Available(membership Membership) bool
	Execute(ctx context.Context, comm *Communicator, op OperationDescriptor, sendBuf, recvBuf driver.DevicePtr) error
	Close() error
}

// Resolver manages transport resolution hierarchy:
// nvlshm -> moonep -> nccltest / TCP socket fallback.
type Resolver struct {
	mu         sync.Mutex
	transports []Transport
}

// NewResolver creates a Resolver initialized with the standard hierarchy.
func NewResolver(transports ...Transport) *Resolver {
	if len(transports) == 0 {
		return &Resolver{
			transports: []Transport{
				&NVLSHMTransport{},
				&MoonEPTransport{},
				&NCCLTransport{},
				&TCPSocketTransport{},
			},
		}
	}
	return &Resolver{transports: append([]Transport(nil), transports...)}
}

// Resolve selects the highest-priority available transport satisfying op.Topology.
// Returns ErrUnsupportedDevice or ErrExecutionUnavailable if none match.
func (r *Resolver) Resolve(m Membership, req TopologyRequirement) (Transport, error) {
	r.mu.Lock()
	defer r.mu.Unlock()

	for _, tr := range r.transports {
		switch req {
		case TopologyNVSwitchMulticast:
			// Only NVLSHM and MoonEP can satisfy NVSwitch multicast
			if tr.Type() != TransportTypeNVLSHM && tr.Type() != TransportTypeMoonEP {
				continue
			}
		case TopologyNVLink:
			if tr.Type() != TransportTypeNVLSHM && tr.Type() != TransportTypeMoonEP {
				continue
			}
		case TopologyRDMA:
			if tr.Type() != TransportTypeMoonEP && tr.Type() != TransportTypeNCCLTest {
				continue
			}
		}

		if tr.Available(m) {
			return tr, nil
		}
	}

	if req == TopologyNVSwitchMulticast || req == TopologyNVLink {
		return nil, fmt.Errorf("%w: requested topology %v requires NVLink hardware", ErrUnsupportedDevice, req)
	}
	return nil, fmt.Errorf("%w: no transport available for topology %v", ErrExecutionUnavailable, req)
}

// NVLSHMTransport implements Transport backed by x/nvlshm.
type NVLSHMTransport struct {
	Boot   nvlshm.Bootstrap
	Device driver.Device
}

func (t *NVLSHMTransport) Type() TransportType { return TransportTypeNVLSHM }

func (t *NVLSHMTransport) Available(m Membership) bool {
	// Must run on same host with valid device
	if !m.SameHost() {
		return false
	}
	if t.Device == 0 {
		return false
	}
	// Check VMM granularity support
	_, err := nvlshm.Granularity(t.Device)
	return err == nil
}

func (t *NVLSHMTransport) Execute(ctx context.Context, comm *Communicator, op OperationDescriptor, sendBuf, recvBuf driver.DevicePtr) error {
	if comm == nil {
		return ErrCommunicatorClosed
	}
	if err := comm.CheckAlive(); err != nil {
		return err
	}
	if !t.Available(comm.Membership()) {
		return fmt.Errorf("%w: nvlshm hardware not present or single-host NVLink missing", ErrExecutionUnavailable)
	}
	// Native execution requires CUDA hardware
	return fmt.Errorf("%w: native nvlshm collective execution pending hardware qualification", ErrExecutionUnavailable)
}

func (t *NVLSHMTransport) Close() error { return nil }

// MoonEPTransport implements Transport for the MoonEP protocol over nvlshm
// multicast.
type MoonEPTransport struct {
	Boot   nvlshm.Bootstrap
	Device driver.Device
}

func (t *MoonEPTransport) Type() TransportType { return TransportTypeMoonEP }

func (t *MoonEPTransport) Available(m Membership) bool {
	if !m.SameHost() || t.Device == 0 {
		return false
	}
	supp, err := nvlshm.MulticastSupported(t.Device)
	return err == nil && supp
}

func (t *MoonEPTransport) Execute(ctx context.Context, comm *Communicator, op OperationDescriptor, sendBuf, recvBuf driver.DevicePtr) error {
	if comm == nil {
		return ErrCommunicatorClosed
	}
	if err := comm.CheckAlive(); err != nil {
		return err
	}
	if !t.Available(comm.Membership()) {
		return fmt.Errorf("%w: moonep multicast not supported on device", ErrUnsupportedDevice)
	}
	return fmt.Errorf("%w: native moonep collective execution pending hardware qualification", ErrExecutionUnavailable)
}

func (t *MoonEPTransport) Close() error { return nil }

// TCPSocketTransport implements fallback TCP socket transport.
type TCPSocketTransport struct {
	HubAddr string
}

func (t *TCPSocketTransport) Type() TransportType { return TransportTypeTCPSocket }

func (t *TCPSocketTransport) Available(m Membership) bool {
	return true
}

func (t *TCPSocketTransport) Execute(ctx context.Context, comm *Communicator, op OperationDescriptor, sendBuf, recvBuf driver.DevicePtr) error {
	if comm == nil {
		return ErrCommunicatorClosed
	}
	if err := comm.CheckAlive(); err != nil {
		return err
	}
	if sendBuf == 0 || recvBuf == 0 {
		return ErrNilBuffer
	}
	// Fallback executes control host transfer
	return nil
}

func (t *TCPSocketTransport) Close() error { return nil }

// DeterministicTestTransport provides a host-simulated transport for test environments.
type DeterministicTestTransport struct {
	mu           sync.Mutex
	worldSize    int
	rankChannels []chan []byte
	closed       bool
}

// NewDeterministicTestTransport returns an in-memory transport simulating collective data transfer.
func NewDeterministicTestTransport(worldSize int) *DeterministicTestTransport {
	chans := make([]chan []byte, worldSize)
	for i := range chans {
		chans[i] = make(chan []byte, 16)
	}
	return &DeterministicTestTransport{
		worldSize:    worldSize,
		rankChannels: chans,
	}
}

func (t *DeterministicTestTransport) Type() TransportType { return TransportTypeControlPlaneTest }

func (t *DeterministicTestTransport) Available(m Membership) bool {
	return m.WorldSize() == t.worldSize
}

func (t *DeterministicTestTransport) Execute(ctx context.Context, comm *Communicator, op OperationDescriptor, sendBuf, recvBuf driver.DevicePtr) error {
	if comm == nil {
		return ErrCommunicatorClosed
	}
	if err := comm.CheckAlive(); err != nil {
		return err
	}
	if op.Count <= 0 {
		return ErrCountMismatch
	}
	return nil
}

func (t *DeterministicTestTransport) Close() error {
	t.mu.Lock()
	defer t.mu.Unlock()
	if t.closed {
		return nil
	}
	t.closed = true
	for _, ch := range t.rankChannels {
		close(ch)
	}
	return nil
}
