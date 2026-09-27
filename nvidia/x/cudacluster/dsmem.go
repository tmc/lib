package cudacluster

import (
	"fmt"

	"github.com/tmc/lib/nvidia/x/cudashared"
)

// ClusterPtr names a PTX register holding a pointer into another CTA's shared
// memory, together with the provenance that makes it checkable.
//
// It is deliberately not a string and deliberately not interchangeable with a
// CTA-local shared address. A local shared pointer and a mapped cluster pointer
// live in different address spaces -- 3 and 7 in the LLVM lowering -- and a
// dereference of one as the other is not a type error in PTX, so it must be a
// type error here. The zero value is unusable.
type ClusterPtr struct {
	reg    string
	rank   string
	region cudashared.Region
	mut    bool
}

// Reg returns the PTX register holding the mapped address.
//
// On the zero value it returns ""; [ClusterPtr.Valid] reports whether p came
// from a mapping call.
func (p ClusterPtr) Reg() string {
	if p.reg == "" {
		return "" // unreachable for values from MapSharedRank; see Valid.
	}
	return p.reg
}

// Valid reports whether p came from a mapping call rather than a zero value.
func (p ClusterPtr) Valid() bool { return p.reg != "" }

// Region returns the CTA-local shared region the mapping was derived from.
// The region describes the peer CTA's layout, which is the same layout: every
// CTA in a cluster runs the same kernel with the same shared plan.
func (p ClusterPtr) Region() cudashared.Region { return p.region }

// Mutable reports whether the mapping was created for writing.
func (p ClusterPtr) Mutable() bool { return p.mut }

// MapSharedRank returns PTX mapping a region of the shared plan into the
// window of the cluster CTA named by the rank register, and a [ClusterPtr]
// describing the result.
//
// dst and rank are .u32 registers; dst receives the mapped address. base is
// the operand naming the CTA-local shared address to be mapped: a register
// holding one, as produced by [EmitSharedAddr]. It is not an offset. mapa takes
// an address operand, so passing a bare integer emits an instruction that does
// not assemble -- which is why the region offset is folded into base by
// [EmitSharedAddr] rather than emitted here.
//
// The returned pointer is read-only: see [MapSharedRankMut] for the writable
// form.
//
// The mapped address is only meaningful while the target CTA is live. A
// cluster has no mechanism to keep a peer alive, so the caller must place
// cluster synchronization appropriate to the data dependency around every use
// -- see [EmitClusterSync]. This package does not insert it, because the
// required placement depends on the dependency and a library that guesses
// produces either a silent race or a barrier the caller cannot remove.
func MapSharedRank(dst, base string, plan *cudashared.Plan, region cudashared.Region, rank string) (string, ClusterPtr, error) {
	return mapShared(dst, base, plan, region, rank, false)
}

// MapSharedRankMut is [MapSharedRank] for a mapping that will be written
// through. The emitted instruction is identical; the distinction is carried in
// the returned pointer so that a store through a read-only mapping is caught
// before it is emitted.
func MapSharedRankMut(dst, base string, plan *cudashared.Plan, region cudashared.Region, rank string) (string, ClusterPtr, error) {
	return mapShared(dst, base, plan, region, rank, true)
}

func mapShared(dst, base string, plan *cudashared.Plan, region cudashared.Region, rank string, mut bool) (string, ClusterPtr, error) {
	if plan == nil {
		return "", ClusterPtr{}, fmt.Errorf("cudacluster: nil shared plan")
	}
	if dst == "" || base == "" || rank == "" {
		return "", ClusterPtr{}, fmt.Errorf("cudacluster: dst, base and rank registers are required")
	}
	if !planHas(plan, region) {
		return "", ClusterPtr{}, fmt.Errorf("cudacluster: region %q is not in the shared plan", region.Name)
	}
	ptx := fmt.Sprintf("mapa.shared::cluster.u32 %s, %s, %s;", dst, base, rank)
	return ptx, ClusterPtr{reg: dst, rank: rank, region: region, mut: mut}, nil
}

func planHas(plan *cudashared.Plan, region cudashared.Region) bool {
	for _, r := range plan.Regions() {
		if r.Name == region.Name && r.Offset == region.Offset {
			return true
		}
	}
	return false
}

// EmitLoadU32 returns PTX loading a 32-bit word from a mapped cluster pointer
// into dst, a .u32 register.
func EmitLoadU32(dst string, p ClusterPtr) (string, error) {
	if !p.Valid() {
		return "", fmt.Errorf("cudacluster: load through an unmapped cluster pointer")
	}
	return fmt.Sprintf("ld.shared::cluster.u32 %s, [%s];", dst, p.Reg()), nil
}

// EmitStoreU32 returns PTX storing a 32-bit word from src, a .u32 register,
// through a mapped cluster pointer.
//
// It fails on a pointer from [MapSharedRank]. The read-only form exists so that
// this check is possible.
func EmitStoreU32(p ClusterPtr, src string) (string, error) {
	if !p.Valid() {
		return "", fmt.Errorf("cudacluster: store through an unmapped cluster pointer")
	}
	if !p.Mutable() {
		return "", fmt.Errorf("cudacluster: store through a read-only mapping of region %q", p.region.Name)
	}
	return fmt.Sprintf("st.shared::cluster.u32 [%s], %s;", p.Reg(), src), nil
}

// EmitReadU32 returns the combined mapping-and-load sequence for one 32-bit
// word from the CTA named by the rank register.
//
// It is a convenience over [MapSharedRank] plus [EmitLoadU32] and not a
// separate mechanism: it emits exactly those two instructions, using addr as a
// scratch .u32 register for the mapped address. Reach for it when a single word
// is wanted and the mapping is not reused; reach for the pair when it is.
func EmitReadU32(dst, addr, base string, plan *cudashared.Plan, region cudashared.Region, rank string) ([]string, error) {
	mapPTX, p, err := MapSharedRank(addr, base, plan, region, rank)
	if err != nil {
		return nil, err
	}
	ldPTX, err := EmitLoadU32(dst, p)
	if err != nil {
		return nil, err
	}
	return []string{mapPTX, ldPTX}, nil
}

// EmitClusterSync returns the cluster-wide barrier, as the two instructions
// that this tree has actually assembled.
//
// cudadevicetable/ptxprobe/cluster.ptx asserts barrier.cluster.arrive followed
// by barrier.cluster.wait, and ptxas accepted that pair for every target at or
// above sm_90. NVlabs/cuda-oxide's intrinsic table instead names a single
// barrier.cluster.sync.aligned for cluster_sync(). That form is not assembled
// anywhere in this tree and is not emitted here. The discrepancy is recorded,
// not resolved: emitting the unverified spelling to match a secondhand table
// would trade a measurement for a citation.
//
// Both instructions are cluster-wide and must be executed by every thread of
// every CTA in the cluster. Placing one inside divergent control flow hangs the
// cluster.
func EmitClusterSync() []string {
	return []string{"barrier.cluster.arrive;", "barrier.cluster.wait;"}
}

// EmitSharedAddr returns PTX materializing the CTA-local address of a region
// into dst, a .u32 register, suitable as the base operand of [MapSharedRank].
//
// symbol names the .shared array the plan was laid out over. The region's
// offset within it is folded in here, so that the mapping call takes an address
// rather than a number.
func EmitSharedAddr(dst, symbol string, plan *cudashared.Plan, region cudashared.Region) (string, error) {
	if plan == nil {
		return "", fmt.Errorf("cudacluster: nil shared plan")
	}
	if dst == "" || symbol == "" {
		return "", fmt.Errorf("cudacluster: dst register and shared symbol are required")
	}
	if !planHas(plan, region) {
		return "", fmt.Errorf("cudacluster: region %q is not in the shared plan", region.Name)
	}
	if region.Offset == 0 {
		return fmt.Sprintf("mov.u32 %s, %s;", dst, symbol), nil
	}
	return fmt.Sprintf("mov.u32 %s, %s+%d;", dst, symbol, region.Offset), nil
}
