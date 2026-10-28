package cudacollective

import (
	"errors"
	"fmt"

	"github.com/tmc/lib/nvidia/x/cudair"
)

// Common collective errors.
var (
	ErrInvalidRank            = errors.New("cudacollective: invalid rank")
	ErrInvalidWorldSize       = errors.New("cudacollective: invalid world size")
	ErrRankOrderMismatch      = errors.New("cudacollective: rank order mismatch")
	ErrCommunicatorGeneration = errors.New("cudacollective: failed communicator generation")
	ErrCommunicatorClosed     = errors.New("cudacollective: communicator is closed")
	ErrBufferOverlap          = errors.New("cudacollective: transfer buffer overlap")
	ErrBufferInUse            = errors.New("cudacollective: transfer buffer currently in use")
	ErrBufferLengthMismatch   = errors.New("cudacollective: buffer length mismatch")
	ErrLostParticipant        = errors.New("cudacollective: lost participant rank")
	ErrBackpressureExceeded   = errors.New("cudacollective: backpressure pipeline full")
	ErrUnsupportedDevice      = errors.New("cudacollective: device unsupported for collective")
	ErrExecutionUnavailable   = errors.New("cudacollective: native collective execution unavailable")
	ErrCountMismatch          = errors.New("cudacollective: element count mismatch")
	ErrTopologyMismatch       = errors.New("cudacollective: topology requirement mismatch")
	ErrNilBuffer              = errors.New("cudacollective: nil buffer")
	ErrBufferUnbound          = errors.New("cudacollective: transfer buffer not bound to pipeline slot")
)

// Kind identifies the collective operation kind.
type Kind int

const (
	KindAllReduce Kind = iota + 1
	KindAllGather
	KindBroadcast
	KindReduceScatter
	KindReduce
	KindAllToAll
	KindScatter
	KindGather
	KindP2P
)

func (k Kind) String() string {
	switch k {
	case KindAllReduce:
		return "AllReduce"
	case KindAllGather:
		return "AllGather"
	case KindBroadcast:
		return "Broadcast"
	case KindReduceScatter:
		return "ReduceScatter"
	case KindReduce:
		return "Reduce"
	case KindAllToAll:
		return "AllToAll"
	case KindScatter:
		return "Scatter"
	case KindGather:
		return "Gather"
	case KindP2P:
		return "P2P"
	default:
		return fmt.Sprintf("Kind(%d)", int(k))
	}
}

// Reduction selects the reduction operator.
type Reduction int

const (
	ReductionSum Reduction = iota + 1
	ReductionProd
	ReductionMin
	ReductionMax
	ReductionAvg
)

func (r Reduction) String() string {
	switch r {
	case ReductionSum:
		return "Sum"
	case ReductionProd:
		return "Prod"
	case ReductionMin:
		return "Min"
	case ReductionMax:
		return "Max"
	case ReductionAvg:
		return "Avg"
	default:
		return fmt.Sprintf("Reduction(%d)", int(r))
	}
}

// TopologyRequirement defines connectivity constraints required by ranks.
type TopologyRequirement int

const (
	// TopologyAny indicates any network or host connectivity is acceptable.
	TopologyAny TopologyRequirement = iota
	// TopologyNVLink requires direct peer-to-peer NVLink connectivity.
	TopologyNVLink
	// TopologyNVSwitchMulticast requires NVSwitch hardware multicast support.
	TopologyNVSwitchMulticast
	// TopologyRDMA requires remote direct memory access capability.
	TopologyRDMA
)

func (t TopologyRequirement) String() string {
	switch t {
	case TopologyAny:
		return "Any"
	case TopologyNVLink:
		return "NVLink"
	case TopologyNVSwitchMulticast:
		return "NVSwitchMulticast"
	case TopologyRDMA:
		return "RDMA"
	default:
		return fmt.Sprintf("Topology(%d)", int(t))
	}
}

// Rank describes a single participant in a collective communicator.
type Rank struct {
	ID       int    `json:"id"`
	HostID   string `json:"host_id"`
	DeviceID int    `json:"device_id"`
}

// Membership represents the immutable ordered set of ranks in a communicator.
type Membership struct {
	ranks []Rank
}

// NewMembership creates a validated Membership. Ranks must be contiguous [0, len-1]
// and strictly ordered by ID.
func NewMembership(ranks []Rank) (Membership, error) {
	if len(ranks) == 0 {
		return Membership{}, fmt.Errorf("%w: empty rank membership", ErrInvalidWorldSize)
	}
	seen := make(map[int]bool, len(ranks))
	for i, r := range ranks {
		if r.ID != i {
			return Membership{}, fmt.Errorf("%w: expected rank index %d, got %d", ErrRankOrderMismatch, i, r.ID)
		}
		if seen[r.ID] {
			return Membership{}, fmt.Errorf("%w: duplicate rank %d", ErrInvalidRank, r.ID)
		}
		seen[r.ID] = true
	}
	cp := make([]Rank, len(ranks))
	copy(cp, ranks)
	return Membership{ranks: cp}, nil
}

// WorldSize returns the total number of ranks.
func (m Membership) WorldSize() int {
	return len(m.ranks)
}

// Rank returns the rank descriptor for rank ID r.
func (m Membership) Rank(r int) (Rank, error) {
	if r < 0 || r >= len(m.ranks) {
		return Rank{}, fmt.Errorf("%w: %d not in [0, %d)", ErrInvalidRank, r, len(m.ranks))
	}
	return m.ranks[r], nil
}

// Ranks returns a copy of all ranks in order.
func (m Membership) Ranks() []Rank {
	cp := make([]Rank, len(m.ranks))
	copy(cp, m.ranks)
	return cp
}

// SameHost returns true if all ranks share the same HostID.
func (m Membership) SameHost() bool {
	if len(m.ranks) <= 1 {
		return true
	}
	h0 := m.ranks[0].HostID
	for _, r := range m.ranks[1:] {
		if r.HostID != h0 {
			return false
		}
	}
	return true
}

// OperationDescriptor captures the complete contract for a collective invocation.
type OperationDescriptor struct {
	Kind         Kind
	DataType     cudair.DataType
	Count        int
	Root         int
	Reduction    Reduction
	Topology     TopologyRequirement
	InPlace      bool
	PipelineSlot int // For double-buffered pipelines (0 or 1)
}

func dataTypeSizeBytes(dt cudair.DataType) (int, bool) {
	switch dt {
	case cudair.Float32, cudair.Int32:
		return 4, true
	case cudair.Float16, cudair.BFloat16:
		return 2, true
	case cudair.Int64:
		return 8, true
	case cudair.Uint8, cudair.Int8:
		return 1, true
	default:
		return 0, false
	}
}

func (k Kind) valid() bool {
	return k >= KindAllReduce && k <= KindP2P
}

func (k Kind) usesReduction() bool {
	switch k {
	case KindAllReduce, KindReduceScatter, KindReduce:
		return true
	default:
		return false
	}
}

func (r Reduction) valid() bool {
	return r >= ReductionSum && r <= ReductionAvg
}

func (t TopologyRequirement) valid() bool {
	return t >= TopologyAny && t <= TopologyRDMA
}

// Validate checks internal consistency of the descriptor against world size.
func (desc OperationDescriptor) Validate(worldSize int) error {
	if worldSize <= 0 {
		return ErrInvalidWorldSize
	}
	if desc.Count <= 0 {
		return fmt.Errorf("%w: count %d must be positive", ErrCountMismatch, desc.Count)
	}
	if desc.Root < 0 || desc.Root >= worldSize {
		return fmt.Errorf("%w: root %d outside [0, %d)", ErrInvalidRank, desc.Root, worldSize)
	}
	if desc.Kind == 0 {
		return errors.New("cudacollective: collective kind must be specified")
	}
	if !desc.Kind.valid() {
		return fmt.Errorf("cudacollective: unsupported collective kind %v", desc.Kind)
	}
	if _, ok := dataTypeSizeBytes(desc.DataType); !ok {
		return fmt.Errorf("cudacollective: unsupported data type %v", desc.DataType)
	}
	if desc.Kind.usesReduction() {
		if desc.Reduction == 0 {
			return fmt.Errorf("cudacollective: reduction operation must be specified for %s", desc.Kind)
		}
		if !desc.Reduction.valid() {
			return fmt.Errorf("cudacollective: unsupported reduction %v", desc.Reduction)
		}
	} else if desc.Reduction != 0 && !desc.Reduction.valid() {
		return fmt.Errorf("cudacollective: unsupported reduction %v", desc.Reduction)
	}
	if !desc.Topology.valid() {
		return fmt.Errorf("%w: unsupported topology requirement %v", ErrTopologyMismatch, desc.Topology)
	}
	if desc.PipelineSlot < 0 || desc.PipelineSlot > 1 {
		return fmt.Errorf("cudacollective: pipeline slot %d must be 0 or 1", desc.PipelineSlot)
	}
	if _, _, err := desc.transferBytes(worldSize); err != nil {
		return err
	}
	return nil
}

// ExpectedBytes returns the total byte size of one participant's base Count buffer,
// or 0 if the data type is unknown, Count is non-positive, or the product overflows int.
func (desc OperationDescriptor) ExpectedBytes() int {
	if desc.Count <= 0 {
		return 0
	}
	elemSize, ok := dataTypeSizeBytes(desc.DataType)
	if !ok {
		return 0
	}
	const maxInt = int(^uint(0) >> 1)
	if uint64(desc.Count) > uint64(maxInt)/uint64(elemSize) {
		return 0
	}
	return desc.Count * elemSize
}

func (desc OperationDescriptor) transferBytes(worldSize int) (sendBytes, recvBytes int, err error) {
	if worldSize <= 0 {
		return 0, 0, ErrInvalidWorldSize
	}
	if desc.Count <= 0 {
		return 0, 0, fmt.Errorf("%w: count %d must be positive", ErrCountMismatch, desc.Count)
	}
	elemSize, ok := dataTypeSizeBytes(desc.DataType)
	if !ok {
		return 0, 0, fmt.Errorf("cudacollective: unsupported data type %v", desc.DataType)
	}
	const maxInt = int(^uint(0) >> 1)
	if uint64(desc.Count) > uint64(maxInt)/uint64(elemSize) {
		return 0, 0, fmt.Errorf("%w: count %d with %d-byte elements overflows byte extent", ErrBufferLengthMismatch, desc.Count, elemSize)
	}
	baseBytes := desc.Count * elemSize
	sendBytes, recvBytes = baseBytes, baseBytes
	switch desc.Kind {
	case KindAllGather, KindGather:
		if uint64(baseBytes) > uint64(maxInt)/uint64(worldSize) {
			return 0, 0, fmt.Errorf("%w: receive byte extent %d * %d ranks overflows", ErrBufferLengthMismatch, baseBytes, worldSize)
		}
		recvBytes = baseBytes * worldSize
	case KindReduceScatter, KindScatter:
		if uint64(baseBytes) > uint64(maxInt)/uint64(worldSize) {
			return 0, 0, fmt.Errorf("%w: send byte extent %d * %d ranks overflows", ErrBufferLengthMismatch, baseBytes, worldSize)
		}
		sendBytes = baseBytes * worldSize
	case KindAllToAll:
		if uint64(baseBytes) > uint64(maxInt)/uint64(worldSize) {
			return 0, 0, fmt.Errorf("%w: all-to-all byte extent %d * %d ranks overflows", ErrBufferLengthMismatch, baseBytes, worldSize)
		}
		sendBytes = baseBytes * worldSize
		recvBytes = baseBytes * worldSize
	}
	return sendBytes, recvBytes, nil
}
