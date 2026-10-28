package advise

import (
	"errors"
	"fmt"
	"strings"

	"github.com/tmc/lib/nvidia/x/cudacapture"
	"github.com/tmc/lib/nvidia/x/cudadevicetable"
	"github.com/tmc/lib/nvidia/x/cudaprof/timeline"
)

// EvidenceModeled marks a finding computed from a cost model rather than
// read from the capture: a projection onto hardware the capture did not run
// on. It is weaker than suspected, because no instruction or timestamp of the
// target stands behind it.
const EvidenceModeled EvidenceLevel = "modeled"

// A Projection models a capture's step on another device.
//
// The model is the bandwidth floor plus the measured idle: compute kernels
// scale by the ratio of the two devices' memory bandwidths, as a
// bandwidth-bound decode step does, and everything else -- the idle between
// launches and the communication -- is carried over unchanged. It is the
// same floor + idle model as nsightsystems.LaunchGaps.Project. Launch and
// drain latency differ between devices and a compute-bound kernel does not
// scale with bandwidth, so every figure here is a model, not a measurement.
type Projection struct {
	Target        string `json:"target"`  // PTX target, such as "sm_103a"
	Product       string `json:"product"` // the target product, such as "GB300"
	SourceProduct string `json:"source_product"`

	SourceGBps float64 `json:"source_gbps"`
	TargetGBps float64 `json:"target_gbps"`

	// BandwidthSource says where the two bandwidths came from; datasheet
	// figures are cudadevicetable Unverified values.
	BandwidthSource string `json:"bandwidth_source"`

	Launches     int   `json:"launches"`
	SpanNanos    int64 `json:"span_nanos"`
	ComputeNanos int64 `json:"compute_nanos"`
	CommNanos    int64 `json:"comm_nanos"`
	IdleNanos    int64 `json:"idle_nanos"`

	// FloorNanos is the compute time scaled to the target, and
	// ProjectedNanos the step the model projects: floor + communication +
	// idle, where communication is the exposed part only.
	FloorNanos     int64 `json:"floor_nanos"`
	ProjectedNanos int64 `json:"projected_nanos"`

	// MeanGapNanos is the idle per launch, and MeanKernelNanos the mean
	// compute kernel on the target.
	MeanGapNanos    int64 `json:"mean_gap_nanos"`
	MeanKernelNanos int64 `json:"mean_kernel_nanos"`

	// Remedy is "none", "pdl" or "persistent"; see [Report.ProjectTo].
	Remedy string `json:"remedy"`
}

// ProjectTo models cap's step on target, a PTX target or product name that
// cudadevicetable.LookupProfile accepts, records the [Projection] in
// r.Target and adds a finding stating it.
//
// The finding says which remedy the launch gaps need on the target. When the
// mean projected kernel outlasts the mean gap, programmatic dependent launch
// can issue each kernel while its predecessor runs, and the gap hides behind
// work. When kernels finish faster than the next can launch there is nothing
// to hide the gap behind, and only removing launches -- a persistent kernel
// or a megakernel -- recovers it.
func (r *Report) ProjectTo(cap *cudacapture.Capture, target string) error {
	tp, err := cudadevicetable.LookupProfile(target)
	if err != nil {
		return fmt.Errorf("project: %w", err)
	}
	sp, ok := sourceProfile(cap)
	if !ok {
		return errors.New("project: the capture names no device with a known memory bandwidth; record the device in capture-provenance.json")
	}
	if sp.MemoryGBps.Value <= 0 || tp.MemoryGBps.Value <= 0 {
		return fmt.Errorf("project: no memory bandwidth for %s or %s", sp.Product, tp.Product)
	}
	streams, _ := timeline.Events(cap)
	_, span, busy, ok := profileDutyCycle(streams)
	if !ok {
		return errors.New("project: the capture holds no timed device activity")
	}
	comm, _ := summarizeCommunication(streams)
	launches := 0
	for _, evs := range streams {
		for _, ev := range evs {
			if strings.EqualFold(ev.Subtype, "kernel") && ev.End > ev.Start && commKind(ev) == "" {
				launches++
			}
		}
	}
	compute := busy - comm.comm
	if comm.comm == 0 {
		compute = busy
	}
	p := &Projection{
		Target:        tp.Target,
		Product:       tp.Product,
		SourceProduct: sp.Product,
		SourceGBps:    float64(sp.MemoryGBps.Value),
		TargetGBps:    float64(tp.MemoryGBps.Value),
		BandwidthSource: fmt.Sprintf("%s %s (%s); %s %s (%s)",
			sp.Product, sp.MemoryGBps.Source, sp.MemoryGBps.Note, tp.Product, tp.MemoryGBps.Source, tp.MemoryGBps.Note),
		Launches:     launches,
		SpanNanos:    int64(span),
		ComputeNanos: int64(compute),
		CommNanos:    int64(comm.comm),
		IdleNanos:    int64(span - busy),
	}
	p.FloorNanos = int64(float64(compute) * p.SourceGBps / p.TargetGBps)
	p.ProjectedNanos = p.FloorNanos + int64(comm.exposed) + p.IdleNanos
	if launches > 0 {
		p.MeanGapNanos = p.IdleNanos / int64(launches)
		p.MeanKernelNanos = p.FloorNanos / int64(launches)
	}
	switch {
	case p.IdleNanos == 0 || p.ProjectedNanos == 0 || float64(p.IdleNanos)/float64(p.ProjectedNanos) < 0.05:
		p.Remedy = "none"
	case p.MeanKernelNanos >= p.MeanGapNanos:
		p.Remedy = "pdl"
	default:
		p.Remedy = "persistent"
	}
	r.Target = p

	idleShare := 0.0
	if p.ProjectedNanos > 0 {
		idleShare = float64(p.IdleNanos) / float64(p.ProjectedNanos) * 100
	}
	finding := fmt.Sprintf("Projected onto %s (%s): compute %.3f ms on %s scales to a %.3f ms bandwidth floor at %.0f/%.0f GB/s; with this capture's %.3f ms of launch gaps (%d launches, %.2f us each)",
		p.Product, p.Target, float64(p.ComputeNanos)/1e6, p.SourceProduct, float64(p.FloorNanos)/1e6, p.TargetGBps, p.SourceGBps,
		float64(p.IdleNanos)/1e6, p.Launches, float64(p.MeanGapNanos)/1e3)
	if comm.exposed > 0 {
		finding += fmt.Sprintf(" and %.3f ms of exposed communication carried over unscaled", float64(comm.exposed)/1e6)
	}
	finding += fmt.Sprintf(", the step models at %.3f ms, of which launch gaps are %.1f%%.", float64(p.ProjectedNanos)/1e6, idleShare)
	var rec string
	severity := SeverityCost
	switch p.Remedy {
	case "none":
		severity = SeverityContext
		finding += " The gaps stay a small share of the step on the target."
		rec = "The launch gaps do not dominate the projected step; the kernels' own time on the target is the quantity to work on."
	case "pdl":
		finding += fmt.Sprintf(" The mean kernel on the target, %.2f us, outlasts the mean gap, so the gap can hide behind the preceding kernel.", float64(p.MeanKernelNanos)/1e3)
		rec = "Launch with programmatic dependent launch (cudaLaunchAttributeProgrammaticStreamSerialization, griddepcontrol.wait in the kernel) so each kernel is issued while its predecessor runs; a CUDA graph alone removes host submission but not the device-side gap."
	case "persistent":
		finding += fmt.Sprintf(" The mean kernel on the target, %.2f us, is shorter than the mean gap, so there is no running kernel to hide the next launch behind.", float64(p.MeanKernelNanos)/1e3)
		rec = "Remove launches rather than overlapping them: fuse the step into a persistent kernel or megakernel that loops over the work on device. PDL and CUDA graphs both still pay a gap per kernel, and on the target the gaps outlast the kernels."
	}
	r.Items = append(r.Items, Item{
		rule:                     "target-projection",
		Severity:                 severity,
		Category:                 CategoryArchitectureDispatch,
		EvidenceLevel:            EvidenceModeled,
		magnitude:                fmt.Sprintf("modeled on %s: %.3f ms step, %.1f%% launch gaps", p.Product, float64(p.ProjectedNanos)/1e6, idleShare),
		Finding:                  finding,
		ActionableRecommendation: rec + " This is a model: it assumes bandwidth-bound kernels and this device's launch gaps, and needs a capture on the target to confirm.",
		NextMeasurement:          "a capture of the same step on " + p.Product,
	})
	return nil
}

// sourceProfile returns the profile of the device cap ran on.
func sourceProfile(cap *cudacapture.Capture) (cudadevicetable.Profile, bool) {
	if p, ok := productProfile(captureDeviceName(cap)); ok {
		return p, true
	}
	arch := parseArchFromCapture(cap)
	if p, ok := productProfile(arch); ok {
		return p, true
	}
	return soleProbedProfile(arch)
}
