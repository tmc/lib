package cudashared

import (
	"fmt"
	"math"
	"sort"

	"github.com/tmc/lib/nvidia/x/cudadevicetable"
	"github.com/tmc/lib/nvidia/x/cudalayout"
)

// Limits bounds one block's shared allocation. All values are bytes.
// A zero Limits is intentionally unusable: resource figures must come from
// the actual target record or an explicitly recorded test target.
type Limits struct {
	StaticMax  int64
	DynamicMax int64
	TotalMax   int64
	Source     string
}

// LimitsFromDevice returns limits only when the embedded record has probed
// values for the relevant attributes. It never substitutes a neighboring
// architecture or a documented-but-unmeasured default.
func LimitsFromDevice(arch string) (Limits, error) {
	a, ok := cudadevicetable.Lookup(arch)
	if !ok {
		return Limits{}, fmt.Errorf("no device record for %q", arch)
	}
	total, err := a.ProbedAttr("maxSharedMemoryPerBlockOptin")
	if err != nil {
		return Limits{}, fmt.Errorf("shared limit: %w", err)
	}
	static, err := a.ProbedAttr("maxSharedMemoryPerBlock")
	if err != nil {
		return Limits{}, fmt.Errorf("static shared limit: %w", err)
	}
	return Limits{StaticMax: static, DynamicMax: total, TotalMax: total, Source: a.Architecture}, nil
}

func (l Limits) validate() error {
	if l.StaticMax <= 0 || l.DynamicMax <= 0 || l.TotalMax <= 0 || l.Source == "" {
		return fmt.Errorf("shared limits require positive values and a source")
	}
	if l.StaticMax > l.TotalMax || l.DynamicMax > l.TotalMax {
		return fmt.Errorf("shared limits exceed total limit")
	}
	return nil
}

// Request describes one region. Stages are inclusive; a region ending at
// stage 2 may be reused by one beginning at stage 3.
type Request struct {
	Name       string
	Size       int64
	Alignment  int64
	FirstStage int
	LastStage  int
	Dynamic    bool
}

// RequestForTile adapts the physical footprint and alignment of cudalayout's
// checked tile into an allocation request. Storage remains owned by this
// planner; cudalayout still owns indexing and software swizzle semantics.
func RequestForTile(tile cudalayout.Tile, name string, firstStage, lastStage int, dynamic bool) (Request, error) {
	size, err := tile.FootprintBytes()
	if err != nil {
		return Request{}, fmt.Errorf("tile footprint: %w", err)
	}
	return Request{Name: name, Size: int64(size), Alignment: int64(tile.AlignmentBytes()), FirstStage: firstStage, LastStage: lastStage, Dynamic: dynamic}, nil
}

// Region is a range relative to its static or dynamic allocation base.
type Region struct {
	Name                    string
	Offset, Size, Alignment int64
	FirstStage, LastStage   int
	Dynamic                 bool
}

// Plan assigns regions in separate static and dynamic allocation pools.
type Plan struct {
	limits                               Limits
	regions                              []Region
	staticBytes, dynamicBytes, footprint int64
}

// NewPlan validates target resource limits.
func NewPlan(limits Limits) (*Plan, error) {
	if err := limits.validate(); err != nil {
		return nil, err
	}
	return &Plan{limits: limits}, nil
}

// Add assigns a checked aligned offset using first-fit stage reuse.
func (p *Plan) Add(r Request) (Region, error) {
	if p == nil {
		return Region{}, fmt.Errorf("nil shared plan")
	}
	if r.Name == "" || r.Size <= 0 || r.Alignment <= 0 || r.Alignment&(r.Alignment-1) != 0 {
		return Region{}, fmt.Errorf("invalid shared request %q", r.Name)
	}
	if r.FirstStage < 0 || r.LastStage < r.FirstStage {
		return Region{}, fmt.Errorf("invalid lifetime for %q", r.Name)
	}
	for _, old := range p.regions {
		if old.Name == r.Name {
			return Region{}, fmt.Errorf("duplicate shared region %q", r.Name)
		}
	}
	if r.Size > math.MaxInt64-r.Alignment {
		return Region{}, fmt.Errorf("shared size overflow for %q", r.Name)
	}
	for off := int64(0); off <= p.limits.TotalMax; {
		var aligned bool
		off, aligned = alignUp(off, r.Alignment)
		if !aligned {
			break
		}
		if off > p.limits.TotalMax || r.Size > p.limits.TotalMax-off {
			break
		}
		end := off + r.Size
		conflict := false
		next := off
		for _, old := range p.regions {
			live := old.Dynamic == r.Dynamic && old.LastStage >= r.FirstStage && r.LastStage >= old.FirstStage
			if live && off < old.Offset+old.Size && old.Offset < end {
				conflict = true
				if old.Offset+old.Size > next {
					next = old.Offset + old.Size
				}
			}
		}
		if !conflict {
			if r.Dynamic {
				candidate := append(append([]Region(nil), p.regions...), Region{Offset: off, Size: r.Size, FirstStage: r.FirstStage, LastStage: r.LastStage, Dynamic: true})
				if peak := dynamicPeak(candidate); peak > p.limits.DynamicMax || categoryFootprint(candidate, true) > p.limits.DynamicMax {
					return Region{}, fmt.Errorf("dynamic shared limit exceeded by %q", r.Name)
				}
			} else if end > p.limits.StaticMax {
				return Region{}, fmt.Errorf("static shared limit exceeded by %q", r.Name)
			}
			staticBytes, dynamicBytes := p.staticBytes, p.dynamicBytes
			if r.Dynamic && end > dynamicBytes {
				dynamicBytes = end
			}
			if !r.Dynamic && end > staticBytes {
				staticBytes = end
			}
			if staticBytes > p.limits.TotalMax-dynamicBytes {
				return Region{}, fmt.Errorf("total shared limit exceeded by %q", r.Name)
			}
			v := Region{Name: r.Name, Offset: off, Size: r.Size, Alignment: r.Alignment, FirstStage: r.FirstStage, LastStage: r.LastStage, Dynamic: r.Dynamic}
			p.regions = append(p.regions, v)
			p.staticBytes, p.dynamicBytes = staticBytes, dynamicBytes
			p.footprint = staticBytes + dynamicBytes
			return v, nil
		}
		if next <= off || next > p.limits.TotalMax {
			break
		}
		off = next
	}
	return Region{}, fmt.Errorf("unable to place shared region %q", r.Name)
}

func dynamicPeak(regions []Region) int64 {
	type event struct {
		stage int
		delta int64
	}
	events := make([]event, 0, len(regions)*2)
	for _, r := range regions {
		if !r.Dynamic {
			continue
		}
		events = append(events, event{r.FirstStage, r.Size})
		if r.LastStage < math.MaxInt {
			events = append(events, event{r.LastStage + 1, -r.Size})
		}
	}
	sort.Slice(events, func(i, j int) bool { return events[i].stage < events[j].stage })
	var cur, peak int64
	for i := 0; i < len(events); {
		stage := events[i].stage
		var delta int64
		for i < len(events) && events[i].stage == stage {
			if events[i].delta > 0 && delta > math.MaxInt64-events[i].delta {
				return math.MaxInt64
			}
			delta += events[i].delta
			i++
		}
		if delta > 0 && cur > math.MaxInt64-delta {
			return math.MaxInt64
		}
		cur += delta
		if cur > peak {
			peak = cur
		}
	}
	return peak
}

func categoryFootprint(regions []Region, dynamic bool) int64 {
	var max int64
	for _, r := range regions {
		if r.Dynamic != dynamic {
			continue
		}
		if r.Offset > math.MaxInt64-r.Size {
			return math.MaxInt64
		}
		if end := r.Offset + r.Size; end > max {
			max = end
		}
	}
	return max
}

func alignUp(v, a int64) (int64, bool) {
	rem := v % a
	if rem == 0 {
		return v, true
	}
	inc := a - rem
	if v > math.MaxInt64-inc {
		return 0, false
	}
	return v + inc, true
}

// Regions returns a copy of the assigned regions.
func (p *Plan) Regions() []Region {
	if p == nil {
		return nil
	}
	return append([]Region(nil), p.regions...)
}
func (p *Plan) StaticBytes() int64 {
	if p == nil {
		return 0
	}
	return p.staticBytes
}
func (p *Plan) DynamicBytes() int64 {
	if p == nil {
		return 0
	}
	return p.dynamicBytes
}
func (p *Plan) FootprintBytes() int64 {
	if p == nil {
		return 0
	}
	return p.footprint
}
func (p *Plan) Limits() Limits {
	if p == nil {
		return Limits{}
	}
	return p.limits
}

// ValidateBase checks the actual shared allocation base and every region's
// resulting address. Base alignment is a caller contract; it is never fixed
// by silently adjusting the address.
func (p *Plan) ValidateBase(base uint64) error {
	if p == nil {
		return fmt.Errorf("nil shared plan")
	}
	for _, r := range p.regions {
		if uint64(r.Offset) > math.MaxUint64-base || uint64(r.Size) > math.MaxUint64-base-uint64(r.Offset) || (base+uint64(r.Offset))%uint64(r.Alignment) != 0 {
			return fmt.Errorf("base 0x%x is unaligned for region %q", base, r.Name)
		}
	}
	return nil
}
