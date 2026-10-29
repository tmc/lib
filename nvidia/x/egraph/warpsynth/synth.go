package warpsynth

import (
	"errors"
	"fmt"
	"slices"
	"sort"

	"github.com/tmc/lib/nvidia/x/cudatmem"
)

// A WarpRoleKind identifies a specialized warp role inside a persistent CTA.
// The roles form a chain in the order of the constants below: each role
// consumes what the previous one produced through a ring of stages.
type WarpRoleKind string

const (
	RoleTMAProducer  WarpRoleKind = "TMAProducer"  // issues cp.async.bulk.tensor
	RoleMMAIssuer    WarpRoleKind = "MMAIssuer"    // issues tcgen05.cp and tcgen05.mma
	RoleEpilogueMath WarpRoleKind = "EpilogueMath" // tcgen05.ld, softmax or SwiGLU, quantization
	RoleFabricStorer WarpRoleKind = "FabricStorer" // TMA store or fabric put
)

var roleOrder = []WarpRoleKind{RoleTMAProducer, RoleMMAIssuer, RoleEpilogueMath, RoleFabricStorer}

func roleRank(k WarpRoleKind) int { return slices.Index(roleOrder, k) }

// A RoleDemand is what one role needs: its minimum warps and registers, and
// its work per tile at the minimum warp count.
type RoleDemand struct {
	Kind            WarpRoleKind
	MinWarps        int    // rounded up to a multiple of 4
	MinRegs         int    // registers per thread for zero spills, rounded up to a multiple of 8
	WorkCyclesTile  uint64 // cycles per tile at MinWarps
	ScalesWithWarps bool   // work divides across extra warps
}

// A TMEMBufferReq asks for a TMEM buffer.
type TMEMBufferReq struct {
	Name      string
	WidthCols uint16 // 32, 64, 128, 256 or 512
	DoubleBuf bool   // two disjoint buffers, for overlap
}

// A Problem is the input to [Synthesize].
type Problem struct {
	Arch              string // exact target, such as "sm_100a"
	SMEMCapBytes      int    // shared memory per CTA
	TileSMEMBytes     int    // bytes one ring stage holds
	EpilogueSMEMBytes int    // shared memory outside the ring
	TMALatency        uint64 // cycles from TMA issue to data in shared memory
	Roles             []RoleDemand
	TMEM              []TMEMBufferReq
	Totals            []int // allowed total warp counts; nil means 8, 12, 16
}

// A RoleAssignment is one role's place in a [Schedule].
type RoleAssignment struct {
	Kind      WarpRoleKind
	FirstWarp int // index of the role's first warp in the CTA
	Warps     int
	Regs      int    // setmaxnreg target
	Cycles    uint64 // modelled cycles per tile
}

// A TMEMPlacement is one TMEM buffer's columns.
type TMEMPlacement struct {
	Name  string
	Width uint16
	Cols  [2]uint16 // first column of each buffer; equal when single-buffered
}

// A Schedule is a warp-specialization plan that satisfies the invariants
// listed in the package documentation.
type Schedule struct {
	Arch                string
	Roles               []RoleAssignment // in chain order
	TotalWarps          int
	LaunchRegs          int // .maxnreg at launch; the pool is 32·TotalWarps·LaunchRegs
	TotalRegsUsed       int // 32·Σ Warps·Regs
	PipelineStagesD     int // shared-memory ring depth
	RingDepths          []int
	TMEM                []TMEMPlacement
	TotalTMEMCols       uint16 // columns to allocate; 0 if no TMEM
	SMEMBytes           int
	SMEMCapBytes        int
	UseEarlySMEMRelease bool   // the MMA issuer releases a stage on its commit
	BottleneckCycTile   uint64 // slowest role
	SteadyStateCycTile  uint64 // modelled period per tile
	TMASlack            int64  // D·period minus stage occupancy; negative means exposed latency
	Candidates          int    // configurations enumerated
}

// Role returns the assignment for kind, and false if the schedule has none.
func (s Schedule) Role(kind WarpRoleKind) (RoleAssignment, bool) {
	for _, r := range s.Roles {
		if r.Kind == kind {
			return r, true
		}
	}
	return RoleAssignment{}, false
}

const (
	minRegs     = 24
	maxRegs     = 256
	regStep     = 8
	regFile     = 65536
	mbarBytes   = 16 // per mbarrier, with alignment
	accStages   = 2  // MMA→epilogue ring depth when an accumulator is double-buffered
	storeStages = 2  // epilogue→storer staging depth
	minDepth    = 2
	maxDepth    = 8
)

// ErrInfeasible is returned when no configuration satisfies the constraints.
var ErrInfeasible = errors.New("no feasible warp-specialization schedule")

// TMEMColumns returns the TMEM column limit the package assumes for arch:
// 512 on sm_100 and sm_103, 576 on sm_107 (exclusive allocation, PTX ISA 9.4
// Table 58, as [cudatmem.MaxColumnsExclusive] records), and 0 on targets
// without tcgen05 TMEM, including sm_90a and sm_121a.
func TMEMColumns(arch string) uint16 {
	switch arch {
	case "sm_100a", "sm_100f", "sm_103a", "sm_103f":
		return 512
	case "sm_107a", "sm_107f":
		return cudatmem.MaxColumnsExclusive
	}
	return 0
}

func knownArch(arch string) bool {
	switch arch {
	case "sm_90a", "sm_100a", "sm_100f", "sm_103a", "sm_103f", "sm_107a", "sm_107f", "sm_120a", "sm_121a":
		return true
	}
	return false
}

func roundUp(x, m int) int { return (x + m - 1) / m * m }

// Synthesize searches the configurations described in the package
// documentation and returns the one with the smallest modelled period, then
// the largest TMA slack, then the fewest warps, the shallowest ring, early
// release before late, and the lexicographically smallest warp split.
func Synthesize(p Problem) (Schedule, error) {
	if !knownArch(p.Arch) {
		return Schedule{}, fmt.Errorf("synthesize: unknown target %q", p.Arch)
	}
	roles, err := normalize(p.Roles)
	if err != nil {
		return Schedule{}, fmt.Errorf("synthesize: %w", err)
	}
	if p.TileSMEMBytes <= 0 || p.SMEMCapBytes <= 0 {
		return Schedule{}, fmt.Errorf("synthesize: shared-memory cap and tile bytes must be positive")
	}
	tmem, cols, err := packTMEM(p.TMEM, TMEMColumns(p.Arch))
	if err != nil {
		return Schedule{}, fmt.Errorf("synthesize: %w", err)
	}
	totals := p.Totals
	if totals == nil {
		totals = []int{8, 12, 16}
	}
	acc := 1
	for _, b := range p.TMEM {
		if b.DoubleBuf {
			acc = accStages
		}
	}
	hasMMA := slices.ContainsFunc(roles, func(r RoleDemand) bool { return r.Kind == RoleMMAIssuer })

	var best *Schedule
	n := 0
	warps := make([]int, len(roles))
	var walk func(i, used int)
	walk = func(i, used int) {
		if i == len(roles) {
			if !slices.Contains(totals, used) {
				return
			}
			for d := minDepth; d <= maxDepth; d++ {
				for _, early := range []bool{true, false} {
					if early && !hasMMA {
						continue
					}
					n++
					s, ok := build(p, roles, warps, used, d, acc, early)
					if !ok {
						continue
					}
					s.TMEM, s.TotalTMEMCols = tmem, cols
					if best == nil || better(s, *best) {
						best = &s
					}
				}
			}
			return
		}
		for w := roundUp(max(roles[i].MinWarps, 1), 4); used+w <= slices.Max(totals); w += 4 {
			warps[i] = w
			walk(i+1, used+w)
		}
	}
	walk(0, 0)
	if best == nil {
		return Schedule{}, fmt.Errorf("synthesize %s: %w", p.Arch, ErrInfeasible)
	}
	best.Candidates = n
	if err := best.Check(); err != nil {
		return Schedule{}, fmt.Errorf("synthesize: internal: %w", err)
	}
	return *best, nil
}

func normalize(in []RoleDemand) ([]RoleDemand, error) {
	if len(in) < 2 {
		return nil, fmt.Errorf("need at least two roles, have %d", len(in))
	}
	out := slices.Clone(in)
	seen := map[WarpRoleKind]bool{}
	for _, r := range out {
		if roleRank(r.Kind) < 0 {
			return nil, fmt.Errorf("unknown role %q", r.Kind)
		}
		if seen[r.Kind] {
			return nil, fmt.Errorf("role %s listed twice", r.Kind)
		}
		seen[r.Kind] = true
		if r.MinRegs > maxRegs {
			return nil, fmt.Errorf("role %s needs %d registers, more than %d", r.Kind, r.MinRegs, maxRegs)
		}
	}
	sort.Slice(out, func(i, j int) bool { return roleRank(out[i].Kind) < roleRank(out[j].Kind) })
	if out[0].Kind != RoleTMAProducer {
		return nil, fmt.Errorf("the chain must start with %s", RoleTMAProducer)
	}
	return out, nil
}

func roleCycles(r RoleDemand, w int) uint64 {
	if !r.ScalesWithWarps {
		return r.WorkCyclesTile
	}
	base := uint64(roundUp(max(r.MinWarps, 1), 4))
	return (r.WorkCyclesTile*base + uint64(w) - 1) / uint64(w)
}

// build evaluates one configuration, reporting false if it is infeasible.
func build(p Problem, roles []RoleDemand, warps []int, total, d, acc int, early bool) (Schedule, bool) {
	s := Schedule{Arch: p.Arch, TotalWarps: total, PipelineStagesD: d, UseEarlySMEMRelease: early, SMEMCapBytes: p.SMEMCapBytes}
	s.LaunchRegs = min(regFile/(32*total)/regStep*regStep, maxRegs)
	pool := 32 * total * s.LaunchRegs
	first := 0
	for i, r := range roles {
		c := roleCycles(r, warps[i])
		s.Roles = append(s.Roles, RoleAssignment{
			Kind: r.Kind, FirstWarp: first, Warps: warps[i],
			Regs: max(roundUp(r.MinRegs, regStep), minRegs), Cycles: c,
		})
		first += warps[i]
		s.BottleneckCycTile = max(s.BottleneckCycTile, c)
	}
	used := 0
	for _, r := range s.Roles {
		used += 32 * r.Warps * r.Regs
	}
	if used > pool {
		return Schedule{}, false
	}
	// Grant leftover registers, hungriest role first.
	order := make([]int, len(s.Roles))
	for i := range order {
		order[i] = i
	}
	sort.SliceStable(order, func(a, b int) bool { return roles[order[a]].MinRegs > roles[order[b]].MinRegs })
	for _, i := range order {
		r := &s.Roles[i]
		for r.Regs+regStep <= maxRegs && used+32*r.Warps*regStep <= pool {
			r.Regs += regStep
			used += 32 * r.Warps * regStep
		}
	}
	s.TotalRegsUsed = used

	s.RingDepths = []int{d}
	if len(roles) > 2 {
		s.RingDepths = append(s.RingDepths, acc)
	}
	if len(roles) > 3 {
		s.RingDepths = append(s.RingDepths, storeStages)
	}
	mbars := 0
	for _, x := range s.RingDepths {
		mbars += 2 * x
	}
	s.SMEMBytes = d*p.TileSMEMBytes + p.EpilogueSMEMBytes + mbarBytes*mbars
	if s.SMEMBytes > p.SMEMCapBytes {
		return Schedule{}, false
	}

	// A stage is occupied from TMA issue until its release.
	occ := p.TMALatency + s.Roles[1].Cycles
	if !early && len(s.Roles) > 2 {
		occ += s.Roles[2].Cycles
	}
	s.SteadyStateCycTile = max(s.BottleneckCycTile, (occ+uint64(d)-1)/uint64(d))
	s.TMASlack = int64(d)*int64(s.SteadyStateCycTile) - int64(occ)
	return s, true
}

func better(a, b Schedule) bool {
	if a.SteadyStateCycTile != b.SteadyStateCycTile {
		return a.SteadyStateCycTile < b.SteadyStateCycTile
	}
	if a.TMASlack != b.TMASlack {
		return a.TMASlack > b.TMASlack
	}
	if a.TotalWarps != b.TotalWarps {
		return a.TotalWarps < b.TotalWarps
	}
	if a.PipelineStagesD != b.PipelineStagesD {
		return a.PipelineStagesD < b.PipelineStagesD
	}
	if a.UseEarlySMEMRelease != b.UseEarlySMEMRelease {
		return a.UseEarlySMEMRelease
	}
	for i := range a.Roles {
		if a.Roles[i].Warps != b.Roles[i].Warps {
			return a.Roles[i].Warps < b.Roles[i].Warps
		}
	}
	return false
}

// packTMEM places the buffers largest first, each at an offset aligned to its
// width, and returns the placements and the column count to allocate: the
// smallest power of two of at least 32 that covers them, or on a 576-column
// target the used span rounded to 32 when that exceeds 512.
func packTMEM(reqs []TMEMBufferReq, limit uint16) ([]TMEMPlacement, uint16, error) {
	if len(reqs) == 0 {
		return nil, 0, nil
	}
	if limit == 0 {
		return nil, 0, fmt.Errorf("target has no tcgen05 TMEM but %d buffers were requested", len(reqs))
	}
	type piece struct{ req, half int }
	var pieces []piece
	names := map[string]bool{}
	for i, r := range reqs {
		switch r.WidthCols {
		case 32, 64, 128, 256, 512:
		default:
			return nil, 0, fmt.Errorf("TMEM buffer %q: width %d is not a power of two in 32..512", r.Name, r.WidthCols)
		}
		if names[r.Name] {
			return nil, 0, fmt.Errorf("TMEM buffer %q requested twice", r.Name)
		}
		names[r.Name] = true
		pieces = append(pieces, piece{i, 0})
		if r.DoubleBuf {
			pieces = append(pieces, piece{i, 1})
		}
	}
	sort.SliceStable(pieces, func(a, b int) bool { return reqs[pieces[a].req].WidthCols > reqs[pieces[b].req].WidthCols })
	out := make([]TMEMPlacement, len(reqs))
	var end int
	for _, p := range pieces {
		r := reqs[p.req]
		w := int(r.WidthCols)
		off := roundUp(end, w)
		if off+w > int(limit) {
			return nil, 0, fmt.Errorf("TMEM buffers need more than %d columns", limit)
		}
		out[p.req].Name, out[p.req].Width = r.Name, r.WidthCols
		out[p.req].Cols[p.half] = uint16(off)
		if !r.DoubleBuf {
			out[p.req].Cols[1] = uint16(off)
		}
		end = off + w
	}
	alloc := 32
	for alloc < end {
		alloc *= 2
	}
	if alloc > int(limit) {
		alloc = roundUp(end, 32)
	}
	return out, uint16(alloc), nil
}

// Check verifies the hardware invariants listed in the package
// documentation. [Synthesize] calls it on every schedule it returns.
func (s Schedule) Check() error {
	if s.TotalWarps <= 0 || s.LaunchRegs <= 0 {
		return fmt.Errorf("schedule has no warps")
	}
	if len(s.Roles) < 2 {
		return fmt.Errorf("schedule has %d roles, want at least 2", len(s.Roles))
	}
	if s.LaunchRegs%regStep != 0 || s.LaunchRegs > maxRegs || 32*s.TotalWarps*s.LaunchRegs > regFile {
		return fmt.Errorf("launch register count %d for %d warps exceeds the register file", s.LaunchRegs, s.TotalWarps)
	}
	used, warps := 0, 0
	for i, r := range s.Roles {
		if r.Warps <= 0 || r.Warps%4 != 0 {
			return fmt.Errorf("role %s has %d warps, not a positive multiple of 4", r.Kind, r.Warps)
		}
		if r.Regs%regStep != 0 || r.Regs < minRegs || r.Regs > maxRegs {
			return fmt.Errorf("role %s setmaxnreg %d is not a multiple of %d in %d..%d", r.Kind, r.Regs, regStep, minRegs, maxRegs)
		}
		if r.FirstWarp != warps {
			return fmt.Errorf("role %s starts at warp %d, want %d", r.Kind, r.FirstWarp, warps)
		}
		if i > 0 && roleRank(r.Kind) <= roleRank(s.Roles[i-1].Kind) {
			return fmt.Errorf("roles out of chain order at %s", r.Kind)
		}
		warps += r.Warps
		used += 32 * r.Warps * r.Regs
	}
	if warps != s.TotalWarps {
		return fmt.Errorf("roles hold %d warps, schedule says %d", warps, s.TotalWarps)
	}
	if used != s.TotalRegsUsed {
		return fmt.Errorf("roles use %d registers, schedule says %d", used, s.TotalRegsUsed)
	}
	if used > 32*s.TotalWarps*s.LaunchRegs {
		return fmt.Errorf("roles use %d registers, more than the %d-register pool", used, 32*s.TotalWarps*s.LaunchRegs)
	}
	if len(s.RingDepths) != len(s.Roles)-1 || s.RingDepths[0] != s.PipelineStagesD {
		return fmt.Errorf("ring depths %v do not match %d roles and D=%d", s.RingDepths, len(s.Roles), s.PipelineStagesD)
	}
	if s.PipelineStagesD < minDepth || s.PipelineStagesD > maxDepth {
		return fmt.Errorf("ring depth %d outside %d..%d", s.PipelineStagesD, minDepth, maxDepth)
	}
	if s.SMEMBytes > s.SMEMCapBytes {
		return fmt.Errorf("shared memory %d bytes exceeds the %d-byte cap", s.SMEMBytes, s.SMEMCapBytes)
	}
	if s.UseEarlySMEMRelease && s.Roles[1].Kind != RoleMMAIssuer {
		return fmt.Errorf("early shared-memory release needs %s to consume the ring", RoleMMAIssuer)
	}
	limit := TMEMColumns(s.Arch)
	if s.TotalTMEMCols > limit {
		return fmt.Errorf("TMEM allocation of %d columns exceeds %d", s.TotalTMEMCols, limit)
	}
	type iv struct{ lo, hi int }
	var ivs []iv
	for _, t := range s.TMEM {
		for h, c := range t.Cols {
			if h == 1 && t.Cols[1] == t.Cols[0] {
				continue
			}
			if int(c)%int(t.Width) != 0 {
				return fmt.Errorf("TMEM buffer %q at column %d is not aligned to %d", t.Name, c, t.Width)
			}
			ivs = append(ivs, iv{int(c), int(c) + int(t.Width)})
		}
	}
	sort.Slice(ivs, func(i, j int) bool { return ivs[i].lo < ivs[j].lo })
	for i, v := range ivs {
		if v.hi > int(s.TotalTMEMCols) {
			return fmt.Errorf("TMEM interval [%d,%d) ends past the %d allocated columns", v.lo, v.hi, s.TotalTMEMCols)
		}
		if i > 0 && v.lo < ivs[i-1].hi {
			return fmt.Errorf("TMEM intervals [%d,%d) and [%d,%d) overlap", ivs[i-1].lo, ivs[i-1].hi, v.lo, v.hi)
		}
	}
	return nil
}
