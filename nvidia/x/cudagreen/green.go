package cudagreen

import (
	"errors"
	"fmt"

	"github.com/tmc/lib/nvidia/driver"
)

// MinDriverVersion is the first CUDA driver version, 12.4, with green
// contexts, in the encoding of cuDriverGetVersion.
const MinDriverVersion = 12040

// ErrUnsupported reports a driver without green contexts.
var ErrUnsupported = errors.New("cudagreen: driver has no green contexts")

// Partitions holds the green contexts and streams [Open] created for a plan.
type Partitions struct {
	Plan    Plan
	ctxs    []driver.GreenCtx
	streams []driver.Stream
}

// Supported reports whether the loaded driver has green contexts. The
// caller must have called [driver.Init].
func Supported() error {
	v, err := driver.DriverGetVersion()
	if err != nil {
		return fmt.Errorf("cudagreen: driver version: %w", err)
	}
	if v < MinDriverVersion {
		return fmt.Errorf("%w: version %d < %d", ErrUnsupported, v, MinDriverVersion)
	}
	for _, sym := range []string{"cuDeviceGetDevResource", "cuDevSmResourceSplitByCount", "cuDevResourceGenerateDesc", "cuGreenCtxCreate", "cuGreenCtxStreamCreate"} {
		if !driver.Has(sym) {
			return fmt.Errorf("%w: missing %s", ErrUnsupported, sym)
		}
	}
	return nil
}

// DeviceGranularity returns dev's SM count and split granularity as the
// driver reports them. A driver that reports no minimum partition size
// (before CUDA 13) yields the zero Granularity; use [DefaultGranularity].
func DeviceGranularity(dev driver.Device) (int, Granularity, error) {
	root, err := dev.DevResource(driver.DevResourceTypeSm)
	if err != nil {
		return 0, Granularity{}, fmt.Errorf("cudagreen: device sm resource: %w", err)
	}
	sm := root.Sm()
	return int(sm.SmCount), Granularity{Min: int(sm.MinSmPartitionSize), Alignment: int(sm.SmCoscheduledAlignment)}, nil
}

// DeviceSplit returns g with Groups set to the number of groups of
// g.Granule() SMs the driver makes from dev's SMs, as a simulated
// cuDevSmResourceSplitByCount reports it.
func DeviceSplit(dev driver.Device, g Granularity) (Granularity, error) {
	if g.Min <= 0 || g.Alignment <= 0 {
		return Granularity{}, fmt.Errorf("cudagreen: bad granularity %+v", g)
	}
	root, err := dev.DevResource(driver.DevResourceTypeSm)
	if err != nil {
		return Granularity{}, fmt.Errorf("cudagreen: device sm resource: %w", err)
	}
	var n uint32
	if err := driver.DevSmResourceSplitByCount(nil, &n, &root, nil, 0, uint32(g.Granule())); err != nil {
		return Granularity{}, fmt.Errorf("cudagreen: simulate split by %d: %w", g.Granule(), err)
	}
	g.Groups = int(n)
	return g, nil
}

// assignGroups returns, for each partition of plan, the index range
// [lo, hi) of the split groups it combines, given the SM count of each
// group the driver made. It fails if the groups are not all one granule or
// are too few for the plan.
func assignGroups(plan Plan, groups []int) ([][2]int, error) {
	granule := plan.Granularity.Granule()
	need := 0
	for i, q := range plan.Partitions {
		if q.SMs%granule != 0 {
			return nil, fmt.Errorf("cudagreen: partition %d (%v): %d SMs not a multiple of granule %d", i, q.Spec.Role, q.SMs, granule)
		}
		need += q.SMs / granule
	}
	if len(groups) < need {
		return nil, fmt.Errorf("cudagreen: driver made %d groups of %d SMs, plan needs %d", len(groups), granule, need)
	}
	for i, c := range groups {
		if c != granule {
			return nil, fmt.Errorf("cudagreen: group %d holds %d SMs, want %d", i, c, granule)
		}
	}
	ranges := make([][2]int, len(plan.Partitions))
	next := 0
	for i, q := range plan.Partitions {
		k := q.SMs / granule
		ranges[i] = [2]int{next, next + k}
		next += k
	}
	return ranges, nil
}

// Open creates one green context, with a non-blocking stream, for each of
// plan's partitions on dev. It asks the driver to split the device's SMs
// into as many groups of the plan's granule as the partitions need, and
// combines each partition's groups into one resource descriptor. A plan
// computed without the device's group limit (see [DeviceSplit]) may need
// more groups than the driver makes, and fails here. Open checks that each
// green context reports the planned SM count. The caller must have called
// [driver.Init].
func Open(dev driver.Device, plan Plan) (_ *Partitions, err error) {
	if err := VerifyPlan(plan); err != nil {
		return nil, err
	}
	if err := Supported(); err != nil {
		return nil, err
	}
	root, err := dev.DevResource(driver.DevResourceTypeSm)
	if err != nil {
		return nil, fmt.Errorf("cudagreen: device sm resource: %w", err)
	}
	sm := root.Sm()
	if int(sm.SmCount) != plan.TotalSMs {
		return nil, fmt.Errorf("cudagreen: device has %d SMs, plan %d", sm.SmCount, plan.TotalSMs)
	}
	g := plan.Granularity
	if m := int(sm.MinSmPartitionSize); m > g.Min {
		return nil, fmt.Errorf("cudagreen: device minimum partition %d > plan %d", m, g.Min)
	}
	if a := int(sm.SmCoscheduledAlignment); a > 0 && g.Alignment%a != 0 {
		return nil, fmt.Errorf("cudagreen: plan alignment %d not a multiple of device %d", g.Alignment, a)
	}
	need := 0
	for _, q := range plan.Partitions {
		need += q.SMs / g.Granule()
	}
	if need == 0 {
		return &Partitions{Plan: plan}, nil
	}

	groups := make([]driver.DevResource, need)
	n := uint32(need)
	if err := driver.DevSmResourceSplitByCount(&groups[0], &n, &root, nil, 0, uint32(g.Granule())); err != nil {
		return nil, fmt.Errorf("cudagreen: split %d SMs into %d groups of %d: %w", plan.TotalSMs, need, g.Granule(), err)
	}
	sizes := make([]int, n)
	for i := range sizes {
		sizes[i] = int(groups[i].Sm().SmCount)
	}
	ranges, err := assignGroups(plan, sizes)
	if err != nil {
		return nil, err
	}

	p := &Partitions{Plan: plan}
	defer func() {
		if err != nil {
			p.Close()
		}
	}()
	for i, q := range plan.Partitions {
		lo, hi := ranges[i][0], ranges[i][1]
		desc, err := driver.DevResourceGenerateDesc(&groups[lo], uint32(hi-lo))
		if err != nil {
			return nil, fmt.Errorf("cudagreen: partition %d (%v): descriptor: %w", i, q.Spec.Role, err)
		}
		ctx, err := driver.NewGreenCtx(desc, dev, driver.GreenCtxDefaultStream)
		if err != nil {
			return nil, fmt.Errorf("cudagreen: partition %d (%v): create: %w", i, q.Spec.Role, err)
		}
		p.ctxs = append(p.ctxs, ctx)
		res, err := ctx.DevResource(driver.DevResourceTypeSm)
		if err != nil {
			return nil, fmt.Errorf("cudagreen: partition %d (%v): resource: %w", i, q.Spec.Role, err)
		}
		if c := int(res.Sm().SmCount); c != q.SMs {
			return nil, fmt.Errorf("cudagreen: partition %d (%v): green context holds %d SMs, want %d", i, q.Spec.Role, c, q.SMs)
		}
		s, err := ctx.StreamCreate(driver.StreamNonBlocking, 0)
		if err != nil {
			return nil, fmt.Errorf("cudagreen: partition %d (%v): stream: %w", i, q.Spec.Role, err)
		}
		p.streams = append(p.streams, s)
	}
	return p, nil
}

// GreenCtx returns the green context of the i'th partition.
func (p *Partitions) GreenCtx(i int) driver.GreenCtx { return p.ctxs[i] }

// Stream returns the stream of the i'th partition.
func (p *Partitions) Stream(i int) driver.Stream { return p.streams[i] }

// Close destroys the streams and green contexts and returns the first error.
func (p *Partitions) Close() error {
	var first error
	for i := range p.streams {
		if err := p.streams[i].Close(); err != nil && first == nil {
			first = err
		}
	}
	for i := range p.ctxs {
		if err := p.ctxs[i].Close(); err != nil && first == nil {
			first = err
		}
	}
	p.streams, p.ctxs = nil, nil
	return first
}
