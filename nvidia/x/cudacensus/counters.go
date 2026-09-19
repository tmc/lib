package cudacensus

import (
	"fmt"
	"sort"
	"time"

	"github.com/tmc/lib/nvidia/nvml"
)

// Granularity says what population a counter's value describes. Counters are
// never combined across granularities: the fraction of SMs that were busy and
// the fraction of warp slots that were occupied are not two measurements of one
// thing, and a sum or an average over both is a number about nothing.
type Granularity string

const (
	DeviceWide Granularity = "device" // one value for the whole GPU
	PerSM      Granularity = "sm"     // a population of streaming multiprocessors
	PerWarp    Granularity = "warp"   // a population of warp slots
)

// Counter names a quantity the device can report. The names describe activity,
// not throughput: an active pipe is a pipe that issued something, and how much
// it issued is not in the counter. Turning activity into FLOPs requires a peak
// and a precision, and those come from cudaroofline and from Attribution, not
// from here.
type Counter string

const (
	// GraphicsActive is the fraction of time any context was resident.
	GraphicsActive Counter = "graphics_active"

	// SMActive is the fraction of SMs that were busy. It is not occupancy and
	// it is not efficiency: an SM issuing one instruction per kilocycle is
	// busy.
	SMActive Counter = "sm_active"

	// SMOccupancy is the fraction of warp slots that held a resident warp.
	// Occupancy is a residency, not work.
	SMOccupancy Counter = "sm_occupancy"

	// DRAMBandwidthUtil is the fraction of theoretical DRAM bandwidth used.
	// The theoretical figure is the driver's, not this repository's measured
	// ceiling, so it does not agree with cudaroofline's bandwidth and must not
	// be substituted for it.
	DRAMBandwidthUtil Counter = "dram_bandwidth_util"

	// The non-tensor math pipes.
	FP64Active Counter = "fp64_active"
	FP32Active Counter = "fp32_active"
	FP16Active Counter = "fp16_active"
	IntActive  Counter = "int_active"

	// AnyTensorActive is the fraction of time the tensor pipes did anything.
	// It does not say which format, which is the whole difficulty: see
	// Attribution.
	AnyTensorActive Counter = "any_tensor_active"

	// The per-format tensor pipes. A device may support AnyTensorActive and
	// none of these.
	HMMATensorActive Counter = "hmma_tensor_active" // half
	IMMATensorActive Counter = "imma_tensor_active" // integer
	DMMATensorActive Counter = "dmma_tensor_active" // double
	DFMATensorActive Counter = "dfma_tensor_active" // double FMA
)

// counterMeta maps a counter to the GPM metric that supplies it and to the
// population its value describes.
type counterMeta struct {
	metric      nvml.GpmMetricId
	granularity Granularity
}

// counters is the one table binding this package's counter names to GPM metric
// identifiers. A second mapping elsewhere would be a second opinion about what
// a counter means.
var counters = map[Counter]counterMeta{
	GraphicsActive:    {nvml.GpmMetricGraphicsUtil, DeviceWide},
	SMActive:          {nvml.GpmMetricSmUtil, PerSM},
	SMOccupancy:       {nvml.GpmMetricSmOccupancy, PerWarp},
	DRAMBandwidthUtil: {nvml.GpmMetricDramBwUtil, DeviceWide},
	FP64Active:        {nvml.GpmMetricFP64Util, PerSM},
	FP32Active:        {nvml.GpmMetricFP32Util, PerSM},
	FP16Active:        {nvml.GpmMetricFP16Util, PerSM},
	IntActive:         {nvml.GpmMetricIntegerUtil, PerSM},
	AnyTensorActive:   {nvml.GpmMetricAnyTensorUtil, PerSM},
	HMMATensorActive:  {nvml.GpmMetricHmmaTensorUtil, PerSM},
	IMMATensorActive:  {nvml.GpmMetricImmaTensorUtil, PerSM},
	DMMATensorActive:  {nvml.GpmMetricDmmaTensorUtil, PerSM},
	DFMATensorActive:  {nvml.GpmMetricDfmaTensorUtil, PerSM},
}

// AllCounters returns every counter this package knows, sorted, for capability
// discovery.
func AllCounters() []Counter {
	out := make([]Counter, 0, len(counters))
	for c := range counters {
		out = append(out, c)
	}
	sort.Slice(out, func(i, j int) bool { return out[i] < out[j] })
	return out
}

// GranularityOf returns the population a counter describes.
func GranularityOf(c Counter) (Granularity, error) {
	m, ok := counters[c]
	if !ok {
		return "", fmt.Errorf("cudacensus: unknown counter %q", c)
	}
	return m.granularity, nil
}

// Reading is one counter's value, or the reason there is no value.
//
// Value is meaningful only when Available is true. When it is false Value is
// zero and must not be read: an unsupported counter, a counter the driver
// refused, and a counter that genuinely read zero are three different facts,
// and a pipeline that spells the first two as zero reports an idle device where
// it has no information at all.
type Reading struct {
	Counter   Counter
	Value     float64
	Available bool

	// Reason names what the driver said when Available is false: the NVML
	// result for this specific metric, since GPM reports availability per
	// metric rather than per query.
	Reason string
}

// Get returns the value and whether there is one. Prefer it to reading Value.
func (r Reading) Get() (float64, bool) {
	if !r.Available {
		return 0, false
	}
	return r.Value, true
}

// Sample is one set of readings taken over one interval. The interval is the
// span between the two GPM samples the values were differenced from, not a host
// wall-clock duration around the call: a sampler that reports the latter is
// reporting its own scheduling delay as device time.
type Sample struct {
	Readings   []Reading
	IntervalNs int64
	TakenAt    time.Time
}

// Reading returns the reading for c, and whether c was requested at all.
func (s Sample) Reading(c Counter) (Reading, bool) {
	for _, r := range s.Readings {
		if r.Counter == c {
			return r, true
		}
	}
	return Reading{}, false
}

// Support is what capability discovery found: whether the device implements GPM
// at all, and which of this package's counters it answers for.
//
// A device with GPM absent supports nothing, and that is reported rather than
// worked around. Supported is keyed by counter and holds the NVML result for
// the ones that failed, so an unsupported metric and a metric refused for
// permissions are distinguishable.
type Support struct {
	GPM        bool
	Supported  map[Counter]bool
	Unverified map[Counter]string // counter -> reason it is not supported
}

// Has reports whether the device answers for c.
func (s Support) Has(c Counter) bool { return s.Supported[c] }

// Identity is the P0 record: what was measured, on what, with which software,
// and which counters that device actually answers for.
//
// Every FLOPs or overhead figure derived from counters must be published with
// one of these. A throughput number without the identity of the part that
// produced it, the driver that reported it, and the list of counters that were
// absent is not reproducible and not checkable.
type Identity struct {
	DeviceName   string // as the driver reports it
	Architecture string // sm_XY, derived from the CUDA compute capability
	ComputeMajor int32
	ComputeMinor int32

	DriverVersion string
	NVMLVersion   string

	MaxSMClockMHz     uint32
	MaxMemoryClockMHz uint32
	PowerLimitMW      uint32

	Support Support
}

// StructVersions carries the ABI version values the driver validates on the GPM
// parameter blocks. The caller supplies them because this repository cannot
// state them: the nvml binding exposes GpmSupport and GpmMetricsGetParams with
// their Version fields, and GpmQueryDeviceSupport, GpmMetricsGet, NewGpmSample
// and GpmSample.Get to use them, but binds no constant for either version --
// nvml.h's NVML_GPM_SUPPORT_VERSION and NVML_GPM_METRICS_GET_VERSION are not
// among the generated identifiers, and no nvml.h is vendored here to read them
// from.
//
// Hardcoding a number that happens to work on one driver is how an ABI check
// turns into an ABI guess, and nvidia/nvml is a binding package this package
// does not own and may not extend. So the values cross the boundary explicitly,
// from a caller that read them out of the header it built against. Zero is
// refused rather than treated as "unset means 1".
type StructVersions struct {
	GpmSupport    uint32 // NVML_GPM_SUPPORT_VERSION
	GpmMetricsGet uint32 // NVML_GPM_METRICS_GET_VERSION
}

func (v StructVersions) validate() error {
	if v.GpmSupport == 0 || v.GpmMetricsGet == 0 {
		return fmt.Errorf("cudacensus: GPM struct versions must be supplied from nvml.h (got support=%d metrics=%d)",
			v.GpmSupport, v.GpmMetricsGet)
	}
	return nil
}

// gpmSampler reads GPM counters from one device. The sample buffers and the
// parameter block are allocated once; Read reuses them, so a steady-state
// sampling loop allocates nothing.
type gpmSampler struct {
	dev    nvml.Device
	want   []Counter
	vers   StructVersions
	prev   nvml.GpmSample
	next   nvml.GpmSample
	warm   bool
	params nvml.GpmMetricsGetParams
	prevAt time.Time
}

// OpenSampler prepares to read want from dev, discovering which of them the
// device answers for. It does not read: the first Read is a warm-up that
// establishes the base sample and returns no values, because a GPM value is a
// difference between two samples and one sample is not a difference.
func OpenSampler(dev nvml.Device, want []Counter, vers StructVersions) (*gpmSampler, Support, error) {
	if err := vers.validate(); err != nil {
		return nil, Support{}, err
	}
	for _, c := range want {
		if _, ok := counters[c]; !ok {
			return nil, Support{}, fmt.Errorf("cudacensus: unknown counter %q", c)
		}
	}
	sup, err := DiscoverSupport(dev, want, vers)
	if err != nil {
		return nil, Support{}, err
	}
	if !sup.GPM {
		return nil, sup, fmt.Errorf("cudacensus: device does not support GPM")
	}
	s := &gpmSampler{dev: dev, want: append([]Counter(nil), want...), vers: vers}
	if s.prev, err = nvml.NewGpmSample(); err != nil {
		return nil, sup, fmt.Errorf("cudacensus: allocating GPM sample: %w", err)
	}
	if s.next, err = nvml.NewGpmSample(); err != nil {
		s.prev.Free()
		return nil, sup, fmt.Errorf("cudacensus: allocating GPM sample: %w", err)
	}
	return s, sup, nil
}

// Close releases the sample buffers.
func (s *gpmSampler) Close() error {
	var first error
	for _, h := range []nvml.GpmSample{s.prev, s.next} {
		if h == 0 {
			continue
		}
		if err := h.Free(); err != nil && first == nil {
			first = err
		}
	}
	s.prev, s.next = 0, 0
	return first
}

// Read fills dst with one reading per requested counter and returns the sample.
// dst must have room for len(want) readings; pass the same slice every call and
// the loop allocates nothing after warm-up.
//
// The first call establishes the base sample and reports every counter absent
// with the reason "warm-up"; that is not an error.
func (s *gpmSampler) Read(dst []Reading) (Sample, error) {
	if len(dst) < len(s.want) {
		return Sample{}, fmt.Errorf("cudacensus: destination holds %d readings, need %d", len(dst), len(s.want))
	}
	dst = dst[:len(s.want)]
	now := time.Now()

	if err := s.next.Get(s.dev); err != nil {
		return Sample{}, fmt.Errorf("cudacensus: GPM sample: %w", err)
	}
	if !s.warm {
		s.prev, s.next = s.next, s.prev
		s.prevAt, s.warm = now, true
		for i, c := range s.want {
			dst[i] = Reading{Counter: c, Reason: "warm-up: no previous sample"}
		}
		return Sample{Readings: dst, TakenAt: now}, nil
	}

	s.params.Version = s.vers.GpmMetricsGet
	s.params.NumMetrics = uint32(len(s.want))
	s.params.Sample1 = s.prev
	s.params.Sample2 = s.next
	for i, c := range s.want {
		s.params.Metrics[i].MetricId = uint32(counters[c].metric)
	}
	if err := nvml.GpmMetricsGet(&s.params); err != nil {
		return Sample{}, fmt.Errorf("cudacensus: GPM metrics: %w", err)
	}
	fillReadings(dst, s.want, &s.params)

	sample := Sample{Readings: dst, IntervalNs: now.Sub(s.prevAt).Nanoseconds(), TakenAt: now}
	s.prev, s.next = s.next, s.prev
	s.prevAt = now
	return sample, nil
}

// fillReadings decodes one GPM result block into readings. It is separate from
// Read because this is where absence is decided, and absence has to be testable
// without a device: a recorded parameter block is enough to show that an
// unsupported metric produces no value and that a stale value from the previous
// tick cannot survive into this one.
//
// Every element of dst is assigned, never updated in place. A decoder that
// writes only the successful metrics leaves the last tick's value sitting in the
// slice, and a carried-over counter reads exactly like a steady one.
func fillReadings(dst []Reading, want []Counter, params *nvml.GpmMetricsGetParams) {
	for i, c := range want {
		m := params.Metrics[i]
		if m.NvmlReturn == nvml.Success {
			dst[i] = Reading{Counter: c, Value: m.Value, Available: true}
			continue
		}
		// Absent. No value is written: see Reading. m.Value may hold
		// whatever the driver left in the block and is deliberately dropped.
		dst[i] = Reading{Counter: c, Reason: m.NvmlReturn.String()}
	}
}

// SumByGranularity adds readings that describe the same population and refuses a
// mixed set. Utilization figures at different granularities are not addends:
// the fraction of SMs that were busy and the fraction of warp slots that were
// occupied answer different questions, and a total over both answers none.
//
// An unavailable reading makes the sum unavailable rather than contributing
// zero.
func SumByGranularity(readings []Reading) (float64, Granularity, error) {
	var sum float64
	var g Granularity
	for _, r := range readings {
		rg, err := GranularityOf(r.Counter)
		if err != nil {
			return 0, "", err
		}
		if g == "" {
			g = rg
		} else if rg != g {
			return 0, "", fmt.Errorf("cudacensus: cannot sum %s (%s) with %s readings", r.Counter, rg, g)
		}
		v, ok := r.Get()
		if !ok {
			return 0, g, fmt.Errorf("cudacensus: %s is absent (%s); a sum over it would report zero for no information", r.Counter, r.Reason)
		}
		sum += v
	}
	if g == "" {
		return 0, "", fmt.Errorf("cudacensus: no readings to sum")
	}
	return sum, g, nil
}

// DiscoverSupport asks the device which counters it answers for, one metric at a
// time, because GPM reports support per metric and a device that implements GPM
// need not implement every metric in it.
func DiscoverSupport(dev nvml.Device, want []Counter, vers StructVersions) (Support, error) {
	sup := Support{
		Supported:  make(map[Counter]bool, len(want)),
		Unverified: make(map[Counter]string, len(want)),
	}
	if err := vers.validate(); err != nil {
		return sup, err
	}
	var q nvml.GpmSupport
	q.Version = vers.GpmSupport
	if err := dev.GpmQueryDeviceSupport(&q); err != nil {
		return sup, fmt.Errorf("cudacensus: querying GPM support: %w", err)
	}
	sup.GPM = q.IsSupportedDevice != 0
	if !sup.GPM {
		for _, c := range want {
			sup.Unverified[c] = "device does not support GPM"
		}
		return sup, nil
	}
	// Per-metric support is only observable by asking for the metric: GPM
	// returns a per-metric result inside the query. Discovery therefore
	// happens on the first real Read, and until then every requested counter
	// is unverified rather than assumed present.
	for _, c := range want {
		sup.Supported[c] = true
	}
	return sup, nil
}

// ReadIdentity collects the P0 record for dev. Fields the driver refuses are
// left zero and named in the error, rather than silently defaulted: an identity
// record that does not say what it could not read is not an identity record.
func ReadIdentity(dev nvml.Device, want []Counter, vers StructVersions) (Identity, error) {
	var id Identity
	var errs []string
	record := func(what string, err error) {
		if err != nil {
			errs = append(errs, what+": "+err.Error())
		}
	}

	name, err := dev.GetName()
	record("name", err)
	id.DeviceName = name

	var minor int32
	major, err := dev.GetCudaComputeCapability(&minor)
	record("compute capability", err)
	if err == nil {
		id.ComputeMajor, id.ComputeMinor = major, minor
		id.Architecture = fmt.Sprintf("sm_%d%d", major, minor)
	}

	id.DriverVersion, err = nvml.SystemGetDriverVersion()
	record("driver version", err)
	id.NVMLVersion, err = nvml.SystemGetNVMLVersion()
	record("NVML version", err)

	id.MaxSMClockMHz, err = dev.GetMaxClockInfo(nvml.ClockSm)
	record("max SM clock", err)
	id.MaxMemoryClockMHz, err = dev.GetMaxClockInfo(nvml.ClockMem)
	record("max memory clock", err)
	id.PowerLimitMW, err = dev.GetPowerManagementLimit()
	record("power limit", err)

	sup, err := DiscoverSupport(dev, want, vers)
	record("GPM support", err)
	id.Support = sup

	if len(errs) > 0 {
		return id, fmt.Errorf("cudacensus: incomplete identity: %v", errs)
	}
	return id, nil
}
