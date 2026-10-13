package cudatma

import (
	"errors"
	"fmt"
	"math"
	"sort"
	"sync"

	"github.com/tmc/lib/nvidia/driver"
)

const (
	TargetSM100a = "sm_100a"
	TargetSM121a = "sm_121a"
)

type Direction uint8

const (
	GlobalToShared Direction = iota + 1
	SharedToGlobal
	GlobalToTMEM
)

// MaxRank is the greatest tensor rank a tiled tensor map admits.
const MaxRank = 5

// ElementType names the element format a descriptor encodes. ElementBytes
// alone cannot distinguish the two-byte float formats, which share a width but
// not an exponent range, so a bfloat16 tensor has to say so.
type ElementType uint8

// The element formats a tiled tensor map admits here. ElementTypeDefault is the
// zero value and keeps the format ElementBytes implies on its own: float16 at
// two bytes and float32 at four.
const (
	ElementTypeDefault ElementType = iota
	ElementTypeFloat16
	ElementTypeBfloat16
	ElementTypeFloat32
)

// String implements [fmt.Stringer].
func (e ElementType) String() string {
	switch e {
	case ElementTypeDefault:
		return "default"
	case ElementTypeFloat16:
		return "float16"
	case ElementTypeBfloat16:
		return "bfloat16"
	case ElementTypeFloat32:
		return "float32"
	}
	return fmt.Sprintf("ElementType(%d)", uint8(e))
}

// bytes reports the width in bytes an element format occupies, and whether the
// format is one this package admits.
func (e ElementType) bytes() (uint32, bool) {
	switch e {
	case ElementTypeFloat16, ElementTypeBfloat16:
		return 2, true
	case ElementTypeFloat32:
		return 4, true
	}
	return 0, false
}

// driverType maps an element format onto the driver enum, resolving
// ElementTypeDefault against the descriptor's element width.
func (e ElementType) driverType(elementBytes uint32) (driver.TensorMapDataType, error) {
	switch e {
	case ElementTypeDefault:
		switch elementBytes {
		case 2:
			return driver.TensorMapDataTypeFloat16, nil
		case 4:
			return driver.TensorMapDataTypeFloat32, nil
		}
		return 0, fmt.Errorf("element size %d has no admitted driver format", elementBytes)
	case ElementTypeFloat16:
		return driver.TensorMapDataTypeFloat16, nil
	case ElementTypeBfloat16:
		return driver.TensorMapDataTypeBfloat16, nil
	case ElementTypeFloat32:
		return driver.TensorMapDataTypeFloat32, nil
	}
	return 0, fmt.Errorf("unsupported element type %d", uint8(e))
}

// Descriptor is validated tensor metadata. The CUDA Driver API owns creation
// of the eventual opaque CUtensorMap.
//
// Dims, Strides and Box carry MaxRank entries; only the first Rank of each
// are read. Strides holds the Rank-1 global strides the driver expects, so
// Strides[Rank-1] is unused and [Descriptor.Validate] requires it to be zero.
//
// The zero values of Interleave, L2Promotion and OOBFill select the driver's
// None for each, which is what the descriptor used before they were
// selectable.
type Descriptor struct {
	Target       string
	Rank         int
	Base         uint64
	Dims         [MaxRank]uint64
	Strides      [MaxRank]uint64
	Box          [MaxRank]uint32
	ElementBytes uint32
	ElementType  ElementType
	Swizzle      uint32
	Interleave   driver.TensorMapInterleave
	L2Promotion  driver.TensorMapL2promotion
	OOBFill      driver.TensorMapFloatOOBfill
}

// EncodeTiled asks the CUDA Driver API to construct the opaque descriptor.
// No bytes are packed or interpreted by this package. The driver library must
// be loaded and expose cuTensorMapEncodeTiled at call time.
func (d Descriptor) EncodeTiled() (*driver.TensorMap, error) {
	if err := d.Validate(); err != nil {
		return nil, err
	}
	if d.Base > uint64(^uintptr(0)) {
		return nil, fmt.Errorf("base address exceeds host pointer width")
	}
	dt, err := d.ElementType.driverType(d.ElementBytes)
	if err != nil {
		return nil, err
	}
	m := driver.NewTensorMap()
	dims := d.Dims
	strides := d.Strides
	box := d.Box
	elems := [MaxRank]uint32{1, 1, 1, 1, 1}
	sw := driver.TensorMapSwizzleNone
	switch d.Swizzle {
	case 32:
		sw = driver.TensorMapSwizzle32B
	case 64:
		sw = driver.TensorMapSwizzle64B
	case 128:
		sw = driver.TensorMapSwizzle128B
	}
	if err := driver.TensorMapEncodeTiled(m, dt, uint32(d.Rank), driver.DevicePtr(d.Base), &dims[0], &strides[0], &box[0], &elems[0], d.Interleave, sw, d.L2Promotion, d.OOBFill); err != nil {
		return nil, fmt.Errorf("encode tensor map: %w", err)
	}
	return m, nil
}

func (d Descriptor) Validate() error {
	if d.Target != TargetSM100a && d.Target != TargetSM121a {
		return fmt.Errorf("unsupported target %q", d.Target)
	}
	if d.Rank < 1 || d.Rank > MaxRank {
		return fmt.Errorf("rank must be between 1 and %d", MaxRank)
	}
	if d.Base%16 != 0 {
		return fmt.Errorf("base 0x%x is not 16-byte aligned", d.Base)
	}
	if d.ElementBytes == 0 || d.ElementBytes&(d.ElementBytes-1) != 0 {
		return fmt.Errorf("element size must be a positive power of two")
	}
	for i := 0; i < d.Rank; i++ {
		if d.Dims[i] == 0 || d.Box[i] == 0 || uint64(d.Box[i]) > d.Dims[i] {
			return fmt.Errorf("dimension %d is zero", i)
		}
		if d.Dims[i] > 1<<32 || d.Box[i] > 256 {
			return fmt.Errorf("dimension %d exceeds tensor-map limits", i)
		}
	}
	// The driver reads Rank-1 global strides; the innermost dimension is
	// implicitly contiguous and carries no stride.
	for i := 0; i < d.Rank-1; i++ {
		if d.Strides[i] == 0 || d.Strides[i]%16 != 0 || d.Strides[i] >= 1<<40 {
			return fmt.Errorf("stride %d must be a nonzero 16-byte multiple", i)
		}
	}
	// Reject a stride the driver will never read rather than ignore it: a
	// caller who filled every entry believes it is describing the innermost
	// dimension, and nothing downstream would ever say otherwise.
	if d.Strides[d.Rank-1] != 0 {
		return fmt.Errorf("stride %d is unused at rank %d and must be zero", d.Rank-1, d.Rank)
	}
	width := uint64(d.Box[0]) * uint64(d.ElementBytes)
	if width > 128 || width == 0 || width%16 != 0 {
		return fmt.Errorf("box width %d is outside the 16..128 byte range", width)
	}
	if d.Base%128+width > 128 {
		return fmt.Errorf("box crosses a 128-byte boundary")
	}
	if d.Swizzle != 0 && d.Swizzle != 32 && d.Swizzle != 64 && d.Swizzle != 128 {
		return fmt.Errorf("unsupported swizzle %d", d.Swizzle)
	}
	if d.ElementBytes != 2 && d.ElementBytes != 4 {
		return fmt.Errorf("unsupported element size %d", d.ElementBytes)
	}
	if d.ElementType != ElementTypeDefault {
		typeWidth, ok := d.ElementType.bytes()
		if !ok {
			return fmt.Errorf("unsupported element type %d", uint8(d.ElementType))
		}
		if typeWidth != d.ElementBytes {
			return fmt.Errorf("element type %v is %d bytes, not the %d declared", d.ElementType, typeWidth, d.ElementBytes)
		}
	}
	switch d.Interleave {
	case driver.TensorMapInterleaveNone:
	case driver.TensorMapInterleave16B, driver.TensorMapInterleave32B:
		// An interleaved layout addresses a trailing element axis, which a
		// rank-2 tensor does not have.
		if d.Rank < 3 {
			return fmt.Errorf("interleaved layout requires rank 3 or more")
		}
	default:
		return fmt.Errorf("unsupported interleave %d", d.Interleave)
	}
	if d.Interleave == driver.TensorMapInterleave32B {
		if d.Base%32 != 0 || d.Swizzle != 32 {
			return fmt.Errorf("32-byte interleave requires 32-byte base alignment and swizzle")
		}
		for i := 0; i < d.Rank-1; i++ {
			if d.Strides[i]%32 != 0 {
				return fmt.Errorf("stride %d is not 32-byte aligned", i)
			}
		}
	}
	if d.Interleave == driver.TensorMapInterleaveNone && d.Swizzle != 0 && width > uint64(d.Swizzle) {
		return fmt.Errorf("box width %d exceeds swizzle %d", width, d.Swizzle)
	}
	switch d.L2Promotion {
	case driver.TensorMapL2PromotionNone, driver.TensorMapL2PromotionL264B,
		driver.TensorMapL2PromotionL2128B, driver.TensorMapL2PromotionL2256B:
	default:
		return fmt.Errorf("unsupported L2 promotion %d", d.L2Promotion)
	}
	switch d.OOBFill {
	case driver.TensorMapFloatOobFillNone, driver.TensorMapFloatOobFillNanRequestZeroFma:
	default:
		return fmt.Errorf("unsupported out-of-bounds fill %d", d.OOBFill)
	}
	return nil
}

// Transfer is a checked operation description. Staged must be true for
// sm_121a global-to-TMEM because that target has no direct path.
type Transfer struct {
	Descriptor          Descriptor
	Direction           Direction
	Staged              bool
	SharedOffset, Bytes uint64
}

func (t Transfer) Validate() error {
	if err := t.Descriptor.Validate(); err != nil {
		return err
	}
	if t.Direction < GlobalToShared || t.Direction > GlobalToTMEM {
		return fmt.Errorf("unsupported transfer direction")
	}
	if t.Bytes == 0 || t.Bytes > math.MaxUint64-t.SharedOffset {
		return fmt.Errorf("invalid transfer extent")
	}
	if t.Direction == GlobalToTMEM && t.Descriptor.Target != TargetSM100a {
		return fmt.Errorf("global-to-tmem is unsupported on %s", t.Descriptor.Target)
	}
	return nil
}

// ErrDuplicateCompletion reports a second completion from a multicast
// destination rank that has already completed.
var ErrDuplicateCompletion = errors.New("duplicate completion")

// Multicast is a handle for one host-side multicast transaction. Copying a
// Multicast copies the handle, not the transaction: both values share one
// completion state. Call NewMulticast to create a separate transaction.
//
// The exported fields are descriptive snapshots from construction. Changing
// them does not reconfigure the transaction. The zero value is not usable:
// Complete reports an error, and CompleteAll and Participating report false.
// Methods synchronize shared completion state across copied handles. Direct
// concurrent access to the exported fields requires caller synchronization.
type Multicast struct {
	// SourceRank is the source CTA rank recorded at construction. It is a
	// descriptive snapshot; methods use the private transaction state.
	SourceRank uint32
	// Destinations is a sorted copy of the ranks passed to NewMulticast;
	// later changes to the caller's slice do not reach it. Value copies of
	// a Multicast share this slice's backing array, so an element written
	// through one copy is visible through the others. The transaction's
	// destinations, byte count and completion state are private,
	// synchronized state that this slice never reconfigures.
	Destinations []uint32
	// Mask is the destination bit mask recorded at construction. It is a
	// descriptive snapshot; methods use the private transaction state.
	Mask uint64
	// Bytes is the byte count recorded at construction. It is a descriptive
	// snapshot; methods use the private transaction state.
	Bytes uint64
	state *multicastState
}

// multicastState is shared by copied handles. Its constructor fields never
// change; completed is protected by mu.
type multicastState struct {
	mu              sync.Mutex
	sourceRank      uint32
	destinations    []uint32
	destinationMask uint64
	bytes           uint64
	completedMask   uint64
}

// NewMulticast tracks transactions for CTA ranks 0 through 15.
// The source and every destination must be in that range.
func NewMulticast(source uint32, destinations []uint32, bytes uint64) (Multicast, error) {
	if source >= 16 {
		return Multicast{}, fmt.Errorf("source rank %d must be between 0 and 15", source)
	}
	if len(destinations) == 0 || bytes == 0 {
		return Multicast{}, fmt.Errorf("multicast requires destinations and bytes")
	}
	canonical := append([]uint32(nil), destinations...)
	var mask uint64
	for _, r := range canonical {
		if r >= 16 {
			return Multicast{}, fmt.Errorf("destination rank %d must be between 0 and 15", r)
		}
		if mask&(uint64(1)<<r) != 0 {
			return Multicast{}, fmt.Errorf("duplicate destination rank %d", r)
		}
		mask |= uint64(1) << r
	}
	sort.Slice(canonical, func(i, j int) bool { return canonical[i] < canonical[j] })
	state := &multicastState{
		sourceRank:      source,
		destinations:    append([]uint32(nil), canonical...),
		destinationMask: mask,
		bytes:           bytes,
	}
	return Multicast{
		SourceRank:   source,
		Destinations: append([]uint32(nil), canonical...),
		Mask:         mask,
		Bytes:        bytes,
		state:        state,
	}, nil
}

// Complete records the transaction for one destination rank. Each rank
// completes exactly once across all copied handles; concurrent attempts are
// serialized, and a second completion returns an error wrapping
// [ErrDuplicateCompletion] without changing the recorded state.
func (m *Multicast) Complete(rank uint32, bytes uint64) error {
	if m == nil {
		return fmt.Errorf("nil multicast")
	}
	if m.state == nil {
		return fmt.Errorf("uninitialized multicast")
	}
	s := m.state
	s.mu.Lock()
	defer s.mu.Unlock()
	if rank >= 16 || s.destinationMask&(uint64(1)<<rank) == 0 {
		return fmt.Errorf("rank %d is not a destination", rank)
	}
	bit := uint64(1) << rank
	if s.completedMask&bit != 0 {
		return fmt.Errorf("rank %d: %w", rank, ErrDuplicateCompletion)
	}
	if bytes != s.bytes {
		return fmt.Errorf("rank %d completed %d bytes, want %d", rank, bytes, s.bytes)
	}
	s.completedMask |= bit
	return nil
}

// CompleteAll reports whether every constructor destination completed.
// Completion reads are synchronized across copied handles.
func (m Multicast) CompleteAll() bool {
	if m.state == nil {
		return false
	}
	s := m.state
	s.mu.Lock()
	done := s.destinationMask != 0 && s.completedMask == s.destinationMask
	s.mu.Unlock()
	return done
}

// Participating reports whether rank was a destination at construction.
func (m Multicast) Participating(rank uint32) bool {
	return m.state != nil && rank < 16 && m.state.destinationMask&(uint64(1)<<rank) != 0
}

// FixtureSource emits an ordinary synchronous global-to-shared fixture.
func FixtureSource(target string, elements int) (string, error) {
	if target != TargetSM100a && target != TargetSM121a {
		return "", fmt.Errorf("unsupported target %q", target)
	}
	if elements <= 0 {
		return "", fmt.Errorf("elements must be positive")
	}
	return fmt.Sprintf("// target=%s elements=%d\nextern \"C\" __global__ void tma_fixture(const float* in,float* out){ extern __shared__ float s[]; unsigned long long i=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x; if(i<%dULL)s[threadIdx.x]=in[i]; __syncthreads(); if(i<%dULL)out[i]=s[threadIdx.x];}\n", target, elements, elements, elements), nil
}

// TMAFixtureSource emits the documented tensor-map/mbarrier protocol. It is
// compile-gated because instruction availability and syntax are target and
// toolkit specific; this source is never treated as native evidence by
// itself.
//
// Two details of the emitted source are load-bearing and are easy to get
// wrong in ways that do not announce themselves.
//
// The descriptor parameter is __grid_constant__. Without it the compiler
// copies all 128 bytes out of parameter space onto the stack and hands
// cp.async.bulk.tensor a generic address of that copy, which is not where
// the instruction reads a tensor map from. That version still compiles, so
// the compile gate cannot see it; [TestTMAFixturePassesTheMapInParamSpace]
// reads the PTX instead and requires cvta.param.
//
// mbarrier.try_wait.parity writes a predicate, not a 32-bit register. The
// inline asm therefore declares a .pred and moves it with selp. Naming an
// "=r" operand as the destination is an arguments mismatch that ptxas
// rejects outright.
func TMAFixtureSource(target string) (string, error) {
	if target != TargetSM100a && target != TargetSM121a {
		return "", fmt.Errorf("unsupported target %q", target)
	}
	return fmt.Sprintf(`#include <cuda.h>
// target=%s; requires PTX cp.async.bulk.tensor and mbarrier support
extern "C" __global__ void tma_fixture(__grid_constant__ const CUtensorMap map) {
  __shared__ __align__(128) unsigned char tile[128];
  __shared__ __align__(8) unsigned long long bar;
  unsigned smem = __cvta_generic_to_shared(tile);
  unsigned barr = __cvta_generic_to_shared(&bar);
  if (threadIdx.x == 0) {
    asm volatile("mbarrier.init.shared::cta.b64 [%%0], 1;" :: "r"(barr));
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%%0], 128;" :: "r"(barr));
    asm volatile("cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes"
                 " [%%0], [%%1, {0, 0}], [%%2];" :: "r"(smem), "l"(&map), "r"(barr) : "memory");
  }
  int ready = 0;
  while (!ready) {
    asm volatile("{\n\t.reg .pred p;\n\t"
                 "mbarrier.try_wait.parity.shared::cta.b64 p, [%%1], %%2;\n\t"
                 "selp.b32 %%0, 1, 0, p;\n\t}"
                 : "=r"(ready) : "r"(barr), "r"(0));
  }
}
`, target), nil
}

// MulticastSource emits a real two-CTA, two-dimensional FP32 fixture. Both
// CTAs receive a 128-byte tile. Launch one 2x1x1 cluster with 136 shared bytes,
// encode the returned fixture's equivalent tensor map, and pass coordinates
// (0, 0). Compiler and native qualification are separate from emission.
func MulticastSource(target string) (string, error) {
	d := Descriptor{Target: target, Rank: 2, Base: 0x1000,
		Dims: [MaxRank]uint64{32, 1}, Strides: [MaxRank]uint64{128},
		Box: [MaxRank]uint32{32, 1}, ElementBytes: 4, ElementType: ElementTypeFloat32}
	c := MulticastConfig{Descriptor: d, Cluster: [3]uint32{2, 1, 1},
		Destinations: []uint32{0, 1}, AvailableSharedBytes: 136,
		SharedLimitProvenance: "fixture requested allocation, not queried device capability"}
	for rank := uint32(0); rank < 2; rank++ {
		c.Windows = append(c.Windows, SharedWindow{Rank: rank, Descriptor: d,
			BarrierOffset: 128, ReservedBytes: 136})
	}
	p, err := NewMulticastPlan(c)
	if err != nil {
		return "", err
	}
	return p.Source()
}
