package cudatma

import (
	"fmt"
	"math"
	"strings"

	"github.com/tmc/lib/nvidia/driver"
)

// PTX 9.4 Table 43 bounds the magnitude of an mbarrier's tx-count to
// 2^20-1. complete_tx::bytes counts bytes, so the entire one-shot tile must
// fit this bound independently of the 32-bit instruction operand and any
// selected shared-memory resource limit.
const maxMulticastTransactionBytes = 1<<20 - 1

// SharedWindow describes a CTA's reservation. Every CTA in Cluster must name
// the same tensor map, tile offset and barrier offset. ReservedBytes is the
// size of that CTA's shared allocation, not a device capability default.
type SharedWindow struct {
	Rank          uint32
	Descriptor    Descriptor
	TileOffset    uint64
	BarrierOffset uint64
	ReservedBytes uint64
}

// MulticastConfig describes a one-shot tiled load. Negative Coordinates and
// tails use the descriptor's zero-fill mode. No driver or device is consulted
// by NewMulticastPlan.
type MulticastConfig struct {
	Descriptor            Descriptor
	Coordinates           [MaxRank]int32
	Cluster               [3]uint32
	SourceRank            uint32
	Destinations          []uint32
	Windows               []SharedWindow
	AvailableSharedBytes  uint64
	SharedLimitProvenance string
}

// MulticastRequirements is the launch contract of a source fixture. Cluster
// describes exactly one cluster, which must also be the launch grid.
// SharedBytes must be admitted against the actual target's queried resources.
// Input, tensor map and output ownership lasts through stream completion and
// the final cluster rendezvous. Source generation alone proves none of these.
type MulticastRequirements struct {
	Coordinates   [MaxRank]int32
	Cluster       [3]uint32
	CTAs          uint32
	Mask          uint16
	TileBytes     uint32
	SharedBytes   uint32
	TileOffset    uint32
	BarrierOffset uint32
	OutputBytes   uint64
}

// MulticastPlan owns immutable validated metadata. Its zero value is unusable.
// It presently establishes physical storage for noninterleaved full-width
// swizzles and unswizzled boxes. Interleaved and narrow swizzled storage need
// an independently sourced physical byte-span contract before admission.
type MulticastPlan struct {
	config       MulticastConfig
	requirements MulticastRequirements
	valid        bool
}

// NewMulticastPlan validates a multicast load without encoding its tensor map.
// A 16-bit mask represents at most sixteen CTAs; it does not imply that the
// chosen device admits a cluster of that size.
func NewMulticastPlan(c MulticastConfig) (*MulticastPlan, error) {
	d := c.Descriptor
	if err := d.Validate(); err != nil {
		return nil, fmt.Errorf("descriptor: %w", err)
	}
	if d.Target != TargetSM100a {
		return nil, fmt.Errorf("multicast target %q is unsupported", d.Target)
	}
	if d.Base == 0 {
		return nil, fmt.Errorf("tensor base must be nonzero")
	}
	if d.Interleave != driver.TensorMapInterleaveNone {
		return nil, fmt.Errorf("interleaved physical storage contract is unresolved")
	}
	width := uint64(d.Box[0]) * uint64(d.ElementBytes)
	if d.Swizzle != 0 && width != uint64(d.Swizzle) {
		return nil, fmt.Errorf("narrow swizzled physical storage contract is unresolved")
	}
	if d.OOBFill != driver.TensorMapFloatOobFillNone {
		return nil, fmt.Errorf("multicast source requires zero-fill mode")
	}
	for i := d.Rank; i < MaxRank; i++ {
		if d.Dims[i] != 0 || d.Box[i] != 0 || d.Strides[i] != 0 || c.Coordinates[i] != 0 {
			return nil, fmt.Errorf("inactive axis %d must be zero", i)
		}
	}
	// Noninterleaved global strides include all storage of the preceding axis.
	footprint := d.Dims[0] * uint64(d.ElementBytes)
	for i := 0; i < d.Rank-1; i++ {
		if d.Strides[i] < footprint {
			return nil, fmt.Errorf("stride %d overlaps preceding axis", i)
		}
		if d.Strides[i] > math.MaxUint64/d.Dims[i+1] {
			return nil, fmt.Errorf("global footprint overflows at axis %d", i+1)
		}
		footprint = d.Strides[i] * d.Dims[i+1]
	}
	if d.Base > math.MaxUint64-footprint {
		return nil, fmt.Errorf("global address extent overflows")
	}
	n := uint64(1)
	for _, dim := range c.Cluster {
		if dim == 0 || dim > 16 || n > 16/uint64(dim) {
			return nil, fmt.Errorf("cluster must contain between 1 and 16 CTAs")
		}
		n *= uint64(dim)
	}
	if uint64(c.SourceRank) >= n {
		return nil, fmt.Errorf("source rank is outside cluster")
	}
	if len(c.Destinations) == 0 {
		return nil, fmt.Errorf("multicast requires a destination")
	}
	var mask uint16
	for _, rank := range c.Destinations {
		if uint64(rank) >= n {
			return nil, fmt.Errorf("destination rank %d is outside cluster", rank)
		}
		bit := uint16(1) << rank
		if mask&bit != 0 {
			return nil, fmt.Errorf("duplicate destination rank %d", rank)
		}
		mask |= bit
	}
	bytes := uint64(d.ElementBytes)
	for i := 0; i < d.Rank; i++ {
		if bytes > maxMulticastTransactionBytes/uint64(d.Box[i]) {
			return nil, fmt.Errorf("transaction bytes exceed mbarrier tx-count limit %d", maxMulticastTransactionBytes)
		}
		bytes *= uint64(d.Box[i])
	}
	if c.AvailableSharedBytes == 0 || c.SharedLimitProvenance == "" {
		return nil, fmt.Errorf("shared limit and provenance are required")
	}
	if len(c.Windows) != int(n) {
		return nil, fmt.Errorf("every cluster CTA requires a shared window")
	}
	seen := make([]bool, n)
	var first SharedWindow
	var shared uint64
	for i, w := range c.Windows {
		if uint64(w.Rank) >= n || seen[w.Rank] {
			return nil, fmt.Errorf("invalid or duplicate window rank %d", w.Rank)
		}
		seen[w.Rank] = true
		if w.Descriptor != d {
			return nil, fmt.Errorf("window rank %d has a different descriptor", w.Rank)
		}
		alignment := uint64(128)
		if d.Swizzle != 0 {
			alignment = uint64(d.Swizzle) * 8
		}
		if w.TileOffset%alignment != 0 || w.BarrierOffset%8 != 0 {
			return nil, fmt.Errorf("window rank %d is misaligned", w.Rank)
		}
		if w.TileOffset > math.MaxUint32-bytes || w.BarrierOffset > math.MaxUint32-8 {
			return nil, fmt.Errorf("window rank %d exceeds shared address width", w.Rank)
		}
		tileEnd, barEnd := w.TileOffset+bytes, w.BarrierOffset+8
		if w.TileOffset < barEnd && w.BarrierOffset < tileEnd {
			return nil, fmt.Errorf("window rank %d overlaps tile and barrier", w.Rank)
		}
		end := tileEnd
		if barEnd > end {
			end = barEnd
		}
		if w.ReservedBytes < end || w.ReservedBytes > c.AvailableSharedBytes || w.ReservedBytes > math.MaxUint32 {
			return nil, fmt.Errorf("window rank %d exceeds reservation or selected shared limit", w.Rank)
		}
		if i == 0 {
			first = w
		} else if w.TileOffset != first.TileOffset || w.BarrierOffset != first.BarrierOffset {
			return nil, fmt.Errorf("window rank %d uses different multicast offsets", w.Rank)
		}
		if w.ReservedBytes > shared {
			shared = w.ReservedBytes
		}
	}
	c.Destinations = append([]uint32(nil), c.Destinations...)
	c.Windows = append([]SharedWindow(nil), c.Windows...)
	return &MulticastPlan{config: c, valid: true, requirements: MulticastRequirements{Coordinates: c.Coordinates, Cluster: c.Cluster, CTAs: uint32(n), Mask: mask, TileBytes: uint32(bytes), SharedBytes: uint32(shared), TileOffset: uint32(first.TileOffset), BarrierOffset: uint32(first.BarrierOffset), OutputBytes: n * bytes}}, nil
}

// Requirements returns a copy of the checked launch contract.
func (p *MulticastPlan) Requirements() (MulticastRequirements, error) {
	if p == nil || !p.valid {
		return MulticastRequirements{}, fmt.Errorf("uninitialized multicast plan")
	}
	return p.requirements, nil
}

// Descriptor returns the tensor-map metadata to pass to EncodeTiled. The
// caller retains that opaque map and its allocation through stream completion.
func (p *MulticastPlan) Descriptor() (Descriptor, error) {
	if p == nil || !p.valid {
		return Descriptor{}, fmt.Errorf("uninitialized multicast plan")
	}
	return p.config.Descriptor, nil
}

// Source emits a real one-shot multicast kernel for the checked plan. Launch
// one cluster, a one-dimensional block of positive size, and SharedBytes of
// dynamic shared memory. Pass Coordinates in order and an opaque EncodeTiled
// map made from Descriptor. Output holds CTAs*TileBytes logical-order bytes;
// unselected output slots retain their caller-provided canaries. Status is
// published after the final retirement rendezvous. No host completion receipt
// can substitute for successful stream completion.
//
// This is source emission, not compiler or native qualification. Its opcode
// and operand widths follow CCCL c339a526 and CUTLASS f614dc40. Swizzle reads
// use PTX's 128-byte pattern rows with a repeat-boundary-aligned tile base.
func (p *MulticastPlan) Source() (string, error) {
	if p == nil || !p.valid {
		return "", fmt.Errorf("uninitialized multicast plan")
	}
	d, r := p.config.Descriptor, p.requirements
	if r.TileBytes > maxMulticastTransactionBytes {
		return "", fmt.Errorf("transaction bytes exceed mbarrier tx-count limit %d", maxMulticastTransactionBytes)
	}
	var b strings.Builder
	fmt.Fprintf(&b, `#include <cuda.h>
#include <stdint.h>
#include <cooperative_groups.h>
// target=sm_100a; one-shot phase=0; grid equals cluster; block is one-dimensional
struct multicast_status {
  uint64_t generation;
  uint32_t rank, selected, phase, bytes, retired;
};
extern "C" __global__ void __cluster_dims__(%d, %d, %d)
tma_multicast(__grid_constant__ const CUtensorMap map,
             unsigned char* output, multicast_status* status, uint64_t generation`, r.Cluster[0], r.Cluster[1], r.Cluster[2])
	for i := 0; i < d.Rank; i++ {
		fmt.Fprintf(&b, ", int32_t c%d", i)
	}
	fmt.Fprintf(&b, `) {
  namespace cg = cooperative_groups;
  cg::cluster_group cluster = cg::this_cluster();
  const unsigned rank = cluster.block_rank();
  const uint16_t mask = %d;
  const bool selected = (mask & (uint16_t(1) << rank)) != 0;
  extern __shared__ __align__(1024) unsigned char storage[];
  const unsigned tile = static_cast<unsigned>(__cvta_generic_to_shared(storage + %d));
  const unsigned bar = static_cast<unsigned>(__cvta_generic_to_shared(storage + %d));
  if (threadIdx.x == 0) {
    asm volatile("mbarrier.init.shared::cta.b64 [%%0], 1;" :: "r"(bar) : "memory");
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();
`, r.Mask, r.TileOffset, r.BarrierOffset)
	// Emit each selected expectation explicitly so a dropped destination has
	// an identifiable source control, not one shared text-presence check.
	for rank := uint32(0); rank < r.CTAs; rank++ {
		if r.Mask&(uint16(1)<<rank) != 0 {
			fmt.Fprintf(&b, "  if (rank == %d && threadIdx.x == 0) {\n    asm volatile(\"mbarrier.arrive.expect_tx.shared::cta.b64 _, [%%0], %%1;\" :: \"r\"(bar), \"r\"(%du) : \"memory\");\n  }\n", rank, r.TileBytes)
		}
	}
	fmt.Fprintf(&b, "  __syncthreads();\n  cluster.sync(); // initialized expectations before issue\n  if (rank == %d && threadIdx.x == 0) {\n", p.config.SourceRank)
	fmt.Fprintf(&b, "    asm volatile(\"cp.async.bulk.tensor.%dd.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster [%%0], [%%1, {", d.Rank)
	for i := 0; i < d.Rank; i++ {
		if i != 0 {
			b.WriteString(", ")
		}
		fmt.Fprintf(&b, "%%%d", i+2)
	}
	fmt.Fprintf(&b, "}], [%%%d], %%%d;\"\n      :: \"r\"(tile), \"l\"(&map)", d.Rank+2, d.Rank+3)
	for i := 0; i < d.Rank; i++ {
		fmt.Fprintf(&b, ", \"r\"(c%d)", i)
	}
	b.WriteString(", \"r\"(bar), \"h\"(mask) : \"memory\");\n  }\n")
	b.WriteString(`  if (selected) {
    unsigned ready = 0;
    while (!ready) {
      asm volatile("{ .reg .pred p;\n\t"
                   "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 p, [%1], %2;\n\t"
                   "selp.b32 %0, 1, 0, p; }"
                   : "=r"(ready) : "r"(bar), "r"(0) : "memory");
    }
`)
	xorMask := d.Swizzle / 16
	if xorMask != 0 {
		xorMask--
	}
	fmt.Fprintf(&b, `    for (uint64_t i = threadIdx.x; i < %du; i += blockDim.x) {
      // PTX 16-byte cells: pattern row is byte offset / 128, not tensor row.
      const uint64_t physical = i ^ (((i >> 7) & %du) << 4);
      output[uint64_t(rank) * %du + i] = storage[%du + physical];
    }
  }
  __syncthreads();
  cluster.sync(); // retirement after all selected waits and reads
  if (threadIdx.x == 0) {
    status[rank] = {generation, rank, uint32_t(selected), 0,
                    selected ? %du : 0u, 1};
  }
}
`, r.TileBytes, xorMask, r.TileBytes, r.TileOffset, r.TileBytes)
	return b.String(), nil
}
