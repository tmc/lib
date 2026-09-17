package cudatma

import (
	"fmt"
	"github.com/tmc/lib/nvidia/driver"
	"math"
	"sort"
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

// Descriptor is validated tensor metadata. The CUDA Driver API owns creation
// of the eventual opaque CUtensorMap.
type Descriptor struct {
	Target       string
	Rank         int
	Base         uint64
	Dims         [3]uint64
	Strides      [3]uint64
	Box          [3]uint32
	ElementBytes uint32
	Swizzle      uint32
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
	var dt driver.TensorMapDataType
	switch d.ElementBytes {
	case 2:
		dt = driver.TensorMapDataTypeFloat16
	case 4:
		dt = driver.TensorMapDataTypeFloat32
	default:
		return nil, fmt.Errorf("element size %d has no admitted driver format", d.ElementBytes)
	}
	m := driver.NewTensorMap()
	dims := d.Dims
	strides := d.Strides
	box := d.Box
	elems := [3]uint32{1, 1, 1}
	sw := driver.TensorMapSwizzleNone
	switch d.Swizzle {
	case 32:
		sw = driver.TensorMapSwizzle32B
	case 64:
		sw = driver.TensorMapSwizzle64B
	case 128:
		sw = driver.TensorMapSwizzle128B
	}
	if err := driver.TensorMapEncodeTiled(m, dt, uint32(d.Rank), uintptr(d.Base), &dims[0], &strides[0], &box[0], &elems[0], driver.TensorMapInterleaveNone, sw, driver.TensorMapL2PromotionNone, driver.TensorMapFloatOobFillNone); err != nil {
		return nil, fmt.Errorf("encode tensor map: %w", err)
	}
	return m, nil
}

func (d Descriptor) Validate() error {
	if d.Target != TargetSM100a && d.Target != TargetSM121a {
		return fmt.Errorf("unsupported target %q", d.Target)
	}
	if d.Rank != 2 && d.Rank != 3 {
		return fmt.Errorf("rank must be 2 or 3")
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
	}
	if d.Strides[0] == 0 || d.Strides[0]%16 != 0 || (d.Rank == 3 && (d.Strides[1] == 0 || d.Strides[1]%16 != 0)) {
		return fmt.Errorf("strides must be nonzero 16-byte multiples")
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

// Multicast tracks one transaction per destination CTA rank.
type Multicast struct {
	SourceRank   uint32
	Destinations []uint32
	Mask         uint64
	Bytes        uint64
	completed    map[uint32]uint64
}

func NewMulticast(source uint32, destinations []uint32, bytes uint64) (Multicast, error) {
	if len(destinations) == 0 || bytes == 0 {
		return Multicast{}, fmt.Errorf("multicast requires destinations and bytes")
	}
	m := Multicast{SourceRank: source, Destinations: append([]uint32(nil), destinations...), Bytes: bytes, completed: make(map[uint32]uint64)}
	for _, r := range destinations {
		if r >= 64 {
			return Multicast{}, fmt.Errorf("destination rank %d exceeds mask", r)
		}
		if m.Mask&(uint64(1)<<r) != 0 {
			return Multicast{}, fmt.Errorf("duplicate destination rank %d", r)
		}
		m.Mask |= uint64(1) << r
	}
	sort.Slice(m.Destinations, func(i, j int) bool { return m.Destinations[i] < m.Destinations[j] })
	return m, nil
}
func (m *Multicast) Complete(rank uint32, bytes uint64) error {
	if m == nil {
		return fmt.Errorf("nil multicast")
	}
	if m.completed == nil {
		return fmt.Errorf("uninitialized multicast")
	}
	if rank >= 16 || m.Mask&(uint64(1)<<rank) == 0 {
		return fmt.Errorf("rank %d is not a destination", rank)
	}
	if bytes != m.Bytes {
		return fmt.Errorf("rank %d completed %d bytes, want %d", rank, bytes, m.Bytes)
	}
	m.completed[rank] = bytes
	return nil
}
func (m Multicast) CompleteAll() bool {
	return len(m.Destinations) > 0 && len(m.completed) == len(m.Destinations)
}
func (m Multicast) Participating(rank uint32) bool { return rank < 16 && m.Mask&(uint64(1)<<rank) != 0 }

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
func TMAFixtureSource(target string) (string, error) {
	if target != TargetSM100a && target != TargetSM121a {
		return "", fmt.Errorf("unsupported target %q", target)
	}
	return fmt.Sprintf(`#include <cuda.h>
// target=%s; requires PTX cp.async.bulk.tensor and mbarrier support
extern "C" __global__ void tma_fixture(const CUtensorMap map) {
  __shared__ __align__(16) unsigned char tile[128];
  __shared__ unsigned long long bar;
  unsigned smem = __cvta_generic_to_shared(tile);
  unsigned barr = __cvta_generic_to_shared(&bar);
  if (threadIdx.x == 0) {
    asm volatile("mbarrier.init.shared.b64 [%%0], 1;" :: "r"(barr));
    asm volatile("mbarrier.arrive.expect_tx.shared.b64 _, [%%0], 128;" :: "r"(barr));
    asm volatile("cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes [%%0], [%%1, {0, 0}], [%%2];" :: "r"(smem), "l"(&map), "r"(barr));
  }
  int ready = 0;
  while (!ready) asm volatile("mbarrier.try_wait.parity.shared.b64 %%0, [%%1], 0;" : "=r"(ready) : "r"(barr));
}
`, target), nil
}

// MulticastSource is intentionally unsupported until the exact PTX multicast
// qualifier, cluster mask, and completion protocol are qualified together.
// shared::cluster addressing alone is not multicast.
func MulticastSource(target string) (string, error) {
	return "", fmt.Errorf("multicast source unsupported for %s: target-specific PTX protocol is unqualified", target)
}
