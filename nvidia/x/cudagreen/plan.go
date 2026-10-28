package cudagreen

import (
	"errors"
	"fmt"
	"math"
	"math/bits"
	"strconv"
	"strings"
)

// MaxSMs is the largest SM count a [Mask] can describe.
const MaxSMs = 256

// A Role names the workload a partition isolates.
type Role uint8

const (
	RoleCompute     Role = iota // prefill or main persistent kernel
	RoleDecode                  // latency-sensitive decode or speculative verification
	RoleFabric                  // NVLink transfer and replication kernels
	RoleMaintenance             // background KV eviction and compression
)

func (r Role) String() string {
	switch r {
	case RoleCompute:
		return "compute"
	case RoleDecode:
		return "decode"
	case RoleFabric:
		return "fabric"
	case RoleMaintenance:
		return "maintenance"
	}
	return "Role(" + strconv.Itoa(int(r)) + ")"
}

// A Spec requests SMs for one role.
type Spec struct {
	Role     Role
	MinSMs   int     // hard minimum
	Fraction float64 // target share of the device's SMs; fractions sum to at most 1
	// ClusterSize is the thread-block cluster size the role launches,
	// at one CTA per SM. The partition's SM count is a multiple of it.
	// Zero means 1.
	ClusterSize int
}

// A Granularity is how a device's SMs can be split. The driver reports Min
// and Alignment in the minSmPartitionSize and smCoscheduledAlignment fields
// of an SM resource. A partition is a whole number of groups of
// [Granularity.Granule] SMs.
//
// The driver splits along the GPU's hardware hierarchy and makes only
// groups that are alike, so it may make fewer groups than the SM count
// divided by the granule: on GB10 (48 SMs, granule 8) it makes 4. Groups,
// when nonzero, is that limit, as [DeviceSplit] discovers it; the SMs left
// over are not assigned.
type Granularity struct {
	Min       int
	Alignment int
	Groups    int
}

// Granule returns the group size: Min rounded up to a multiple of
// Alignment.
func (g Granularity) Granule() int {
	return roundUp(g.Min, g.Alignment)
}

// DefaultGranularity returns the split quantum the CUDA documentation gives
// for arch's compute capability (sm_70 through sm_8x: 2 SMs, sm_90 and
// later: 8 SMs) without SM coscheduling disabled. A live device's own
// report, used by [Open], takes precedence.
func DefaultGranularity(arch string) (Granularity, error) {
	cc, err := computeCapability(arch)
	if err != nil {
		return Granularity{}, err
	}
	switch {
	case cc >= 90:
		return Granularity{Min: 8, Alignment: 8}, nil
	case cc >= 70:
		return Granularity{Min: 2, Alignment: 2}, nil
	}
	return Granularity{}, fmt.Errorf("cudagreen: %s predates green contexts", arch)
}

// computeCapability returns 10*major+minor for an sm_ target such as sm_90a
// (90) or sm_121a (121).
func computeCapability(arch string) (int, error) {
	s, ok := strings.CutPrefix(arch, "sm_")
	if !ok {
		return 0, fmt.Errorf("cudagreen: arch %q: missing sm_ prefix", arch)
	}
	s = strings.TrimRight(s, "af")
	n, err := strconv.Atoi(s)
	if err != nil || n < 10 {
		return 0, fmt.Errorf("cudagreen: arch %q: bad compute capability", arch)
	}
	return n, nil
}

// A Mask is a set of logical SM indices in [0, MaxSMs). Bit i of word i/64
// is SM i.
type Mask [MaxSMs / 64]uint64

// Range returns the mask of SMs [first, first+n).
func Range(first, n int) Mask {
	var m Mask
	for i := first; i < first+n; i++ {
		m[i/64] |= 1 << (i % 64)
	}
	return m
}

// Count returns the number of SMs in m.
func (m Mask) Count() int {
	n := 0
	for _, w := range m {
		n += bits.OnesCount64(w)
	}
	return n
}

// Overlaps reports whether m and o share an SM.
func (m Mask) Overlaps(o Mask) bool {
	for i := range m {
		if m[i]&o[i] != 0 {
			return true
		}
	}
	return false
}

// A Partition is one role's share of a [Plan].
type Partition struct {
	Spec     Spec
	SMs      int  // assigned SM count
	First    int  // first logical SM
	Mask     Mask // logical SMs [First, First+SMs)
	Clusters int  // SMs / cluster size: the persistent grid, in clusters
}

// Grid returns the number of CTAs a persistent kernel launches to fill the
// partition at ctasPerSM CTAs per SM.
func (p Partition) Grid(ctasPerSM int) int {
	return p.SMs * ctasPerSM
}

// A Plan is a disjoint carving of one device's SMs.
type Plan struct {
	Arch        string
	TotalSMs    int
	Granularity Granularity
	Partitions  []Partition
	Unassigned  int // SMs no partition holds
}

// Partition returns the first partition for role.
func (p Plan) Partition(role Role) (Partition, bool) {
	for _, q := range p.Partitions {
		if q.Spec.Role == role {
			return q, true
		}
	}
	return Partition{}, false
}

// ErrInfeasible reports that a plan's minimums do not fit on the device.
var ErrInfeasible = errors.New("cudagreen: minimum partitions exceed device")

// ComputePlan carves totalSMs SMs among specs, in order. Each partition
// starts at its target fraction rounded to the nearest legal size, and is
// raised to its minimum. If the total exceeds the SMs available in whole
// granules, or in the groups the device makes, the partition furthest above
// its target shrinks by one unit at a time until the plan fits; ComputePlan
// returns an error wrapping [ErrInfeasible] if the minimums alone do not
// fit.
func ComputePlan(arch string, totalSMs int, g Granularity, specs []Spec) (Plan, error) {
	if totalSMs <= 0 || totalSMs > MaxSMs {
		return Plan{}, fmt.Errorf("cudagreen: %d SMs out of range [1, %d]", totalSMs, MaxSMs)
	}
	if g.Min <= 0 || g.Alignment <= 0 {
		return Plan{}, fmt.Errorf("cudagreen: bad granularity %+v", g)
	}
	sum := 0.0
	for i, s := range specs {
		if s.Fraction < 0 || s.Fraction > 1 || math.IsNaN(s.Fraction) {
			return Plan{}, fmt.Errorf("cudagreen: spec %d (%v): fraction %v out of range", i, s.Role, s.Fraction)
		}
		if s.MinSMs < 0 || s.ClusterSize < 0 {
			return Plan{}, fmt.Errorf("cudagreen: spec %d (%v): negative size", i, s.Role)
		}
		sum += s.Fraction
	}
	if sum > 1+1e-9 {
		return Plan{}, fmt.Errorf("cudagreen: fractions sum to %v > 1", sum)
	}
	if g.Groups < 0 {
		return Plan{}, fmt.Errorf("cudagreen: bad granularity %+v", g)
	}
	usable := totalSMs / g.Granule() * g.Granule()
	if g.Groups > 0 {
		usable = min(usable, g.Groups*g.Granule())
	}

	units := make([]int, len(specs))
	mins := make([]int, len(specs))
	want := make([]int, len(specs))
	total := 0
	for i, s := range specs {
		u := unit(g, s)
		units[i] = u
		mins[i] = roundUp(max(s.MinSMs, g.Min, u), u)
		target := s.Fraction * float64(totalSMs)
		want[i] = max(int(math.Round(target/float64(u)))*u, mins[i])
		total += want[i]
	}
	for total > usable {
		// Shrink the partition furthest above its target.
		best, over := -1, math.Inf(-1)
		for i, s := range specs {
			if want[i]-units[i] < mins[i] {
				continue
			}
			o := float64(want[i]) - s.Fraction*float64(totalSMs)
			if o > over {
				best, over = i, o
			}
		}
		if best < 0 {
			return Plan{}, fmt.Errorf("%w: need %d SMs, have %d usable", ErrInfeasible, total, usable)
		}
		want[best] -= units[best]
		total -= units[best]
	}

	plan := Plan{Arch: arch, TotalSMs: totalSMs, Granularity: g}
	first := 0
	for i, s := range specs {
		plan.Partitions = append(plan.Partitions, Partition{
			Spec:     s,
			SMs:      want[i],
			First:    first,
			Mask:     Range(first, want[i]),
			Clusters: want[i] / max(s.ClusterSize, 1),
		})
		first += want[i]
	}
	plan.Unassigned = totalSMs - first
	return plan, nil
}

// unit returns the size step for s: a multiple of both the granule and the
// cluster size.
func unit(g Granularity, s Spec) int {
	c, u := max(s.ClusterSize, 1), g.Granule()
	return u / gcd(u, c) * c
}

func gcd(a, b int) int {
	for b != 0 {
		a, b = b, a%b
	}
	return a
}

func roundUp(n, u int) int {
	return (n + u - 1) / u * u
}

// VerifyPlan checks that no two partitions share an SM, that each mask
// holds exactly the partition's SM count and lies on the device, that each
// count meets its minimum, the granule and its cluster size, that the
// partitions fit in the groups the device makes, and that partitions and
// unassigned SMs account for the whole device.
func VerifyPlan(p Plan) error {
	if p.TotalSMs <= 0 || p.TotalSMs > MaxSMs {
		return fmt.Errorf("cudagreen: %d SMs out of range", p.TotalSMs)
	}
	if g := p.Granularity; g.Min <= 0 || g.Alignment <= 0 || g.Groups < 0 {
		return fmt.Errorf("cudagreen: bad granularity %+v", g)
	}
	device := Range(0, p.TotalSMs)
	assigned := 0
	for i, q := range p.Partitions {
		if n := q.Mask.Count(); n != q.SMs {
			return fmt.Errorf("cudagreen: partition %d (%v): mask holds %d SMs, want %d", i, q.Spec.Role, n, q.SMs)
		}
		for w := range q.Mask {
			if q.Mask[w]&^device[w] != 0 {
				return fmt.Errorf("cudagreen: partition %d (%v): mask beyond SM %d", i, q.Spec.Role, p.TotalSMs-1)
			}
		}
		if q.SMs < q.Spec.MinSMs || q.SMs < p.Granularity.Min {
			return fmt.Errorf("cudagreen: partition %d (%v): %d SMs below minimum", i, q.Spec.Role, q.SMs)
		}
		if u := unit(p.Granularity, q.Spec); q.SMs%u != 0 {
			return fmt.Errorf("cudagreen: partition %d (%v): %d SMs not a multiple of %d", i, q.Spec.Role, q.SMs, u)
		}
		if q.Clusters != q.SMs/max(q.Spec.ClusterSize, 1) {
			return fmt.Errorf("cudagreen: partition %d (%v): %d clusters, want %d", i, q.Spec.Role, q.Clusters, q.SMs/max(q.Spec.ClusterSize, 1))
		}
		for j := range i {
			if q.Mask.Overlaps(p.Partitions[j].Mask) {
				return fmt.Errorf("cudagreen: partitions %d (%v) and %d (%v) share SMs", j, p.Partitions[j].Spec.Role, i, q.Spec.Role)
			}
		}
		assigned += q.SMs
	}
	if g := p.Granularity; g.Groups > 0 && assigned > g.Groups*g.Granule() {
		return fmt.Errorf("cudagreen: partitions need %d groups of %d SMs, device makes %d", assigned/g.Granule(), g.Granule(), g.Groups)
	}
	if assigned+p.Unassigned != p.TotalSMs {
		return fmt.Errorf("cudagreen: %d assigned + %d unassigned != %d SMs", assigned, p.Unassigned, p.TotalSMs)
	}
	return nil
}

// PrefillHeavy returns specs that give most SMs to the compute role, in the
// proportions 80/10/5/5 percent.
func PrefillHeavy() []Spec {
	return []Spec{
		{Role: RoleCompute, Fraction: 0.80},
		{Role: RoleDecode, Fraction: 0.10},
		{Role: RoleFabric, Fraction: 0.05},
		{Role: RoleMaintenance, Fraction: 0.05},
	}
}

// DecodeHeavy returns specs that give most SMs to the decode role, in the
// proportions 20/70/5/5 percent.
func DecodeHeavy() []Spec {
	return []Spec{
		{Role: RoleCompute, Fraction: 0.20},
		{Role: RoleDecode, Fraction: 0.70},
		{Role: RoleFabric, Fraction: 0.05},
		{Role: RoleMaintenance, Fraction: 0.05},
	}
}
