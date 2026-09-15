package cudaroofline_test

import (
	"bytes"
	"regexp"
	"strconv"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudaroofline"
)

func TestLookupCeiling(t *testing.T) {
	tests := []struct {
		name       string
		arch       string
		residency  cudaroofline.Residency
		wantPeak   float64
		wantMem    float64
		wantMemSrc cudaroofline.Provenance
		shouldFail bool
	}{
		{name: "sm_90/device", arch: "sm_90", residency: cudaroofline.DeviceResident,
			wantPeak: 67.0, wantMem: 3350.0, wantMemSrc: cudaroofline.Unverified},
		{name: "sm_121/device", arch: "sm_121", residency: cudaroofline.DeviceResident,
			wantPeak: 40.0, wantMem: 239.6, wantMemSrc: cudaroofline.Measured},
		{name: "sm_121a/device", arch: "sm_121a", residency: cudaroofline.DeviceResident,
			wantPeak: 40.0, wantMem: 239.6, wantMemSrc: cudaroofline.Measured},
		{name: "sm_121a/host_page_table", arch: "sm_121a", residency: cudaroofline.HostPageTableResident,
			wantPeak: 40.0, wantMem: 162.4, wantMemSrc: cudaroofline.Measured},
		{name: "sm_80/device", arch: "sm_80", residency: cudaroofline.DeviceResident,
			wantPeak: 19.5, wantMem: 1935.0, wantMemSrc: cudaroofline.Unverified},

		// A residency nobody characterized on this part must fail rather than
		// hand back the device-resident number.
		{name: "sm_90/host_page_table", arch: "sm_90", residency: cudaroofline.HostPageTableResident, shouldFail: true},
		// An unset Residency is not a default.
		{name: "sm_121/unset", arch: "sm_121", shouldFail: true},
		{name: "unknown arch", arch: "sm_999", residency: cudaroofline.DeviceResident, shouldFail: true},
		{name: "empty arch", arch: "", residency: cudaroofline.DeviceResident, shouldFail: true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			c, err := cudaroofline.LookupCeiling(tt.arch, tt.residency)
			if tt.shouldFail {
				if err == nil {
					t.Fatalf("LookupCeiling(%q, %q) = %+v, want error", tt.arch, tt.residency, c)
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if c.PeakTFLOPS != tt.wantPeak {
				t.Errorf("PeakTFLOPS = %v, want %v", c.PeakTFLOPS, tt.wantPeak)
			}
			if c.PeakMemoryGBps != tt.wantMem {
				t.Errorf("PeakMemoryGBps = %v, want %v", c.PeakMemoryGBps, tt.wantMem)
			}
			if c.MemoryProvenance != tt.wantMemSrc {
				t.Errorf("MemoryProvenance = %q, want %q", c.MemoryProvenance, tt.wantMemSrc)
			}
			if c.Residency != tt.residency {
				t.Errorf("Residency = %q, want %q", c.Residency, tt.residency)
			}
		})
	}
}

func TestModelKneeIntensity(t *testing.T) {
	model, err := cudaroofline.NewModel("sm_121", cudaroofline.DeviceResident)
	if err != nil {
		t.Fatal(err)
	}
	// PeakTFLOPS: 40, PeakMemoryGBps: 239.6
	knee := model.KneeIntensity()
	want := (40.0 * 1000.0) / 239.6
	if diff := knee - want; diff < -0.01 || diff > 0.01 {
		t.Errorf("knee intensity = %f, want %f", knee, want)
	}
}

func TestEvaluateClassifications(t *testing.T) {
	model, err := cudaroofline.NewModel("sm_121", cudaroofline.DeviceResident)
	if err != nil {
		t.Fatal(err)
	}

	tests := []struct {
		name       string
		kernel     string
		flops      uint64
		bytes      uint64
		durationNs uint64
		wantClass  string
	}{
		{
			name:       "memory_bound_q8_matvec",
			kernel:     "ds4_q8_0_matvec",
			flops:      1000000,
			bytes:      1000000, // Intensity = 1.0 FLOP/B (well below knee ~166.9)
			durationNs: 4500,    // 1 MB / 4.5us = 222.2 GB/s (93% of the measured 239.6 GB/s)
			wantClass:  cudaroofline.ClassMemoryBound,
		},
		{
			name:       "compute_bound_dense_gemm",
			kernel:     "dense_gemm",
			flops:      1000000000, // 1 GFLOP
			bytes:      1000000,    // 1 MB -> Intensity = 1000 FLOP/B (well above knee)
			durationNs: 28000,      // 1 GFLOP / 28us = 35.7 TFLOPS (89% of 40 TFLOPS)
			wantClass:  cudaroofline.ClassComputeBound,
		},
		{
			name:       "unconstrained_slow_kernel",
			kernel:     "ragged_slow_kernel",
			flops:      1000000,
			bytes:      1000000, // Intensity = 1.0 FLOP/B
			durationNs: 100000,  // 1 MB / 100us = 10 GB/s -> 4% of ceiling
			wantClass:  cudaroofline.ClassUnconstrained,
		},
		{
			name:       "above_ceiling_beats_device_bandwidth",
			kernel:     "impossible_matvec",
			flops:      2000000,
			bytes:      1000000, // Intensity = 2.0 FLOP/B
			durationNs: 4000,    // 1 MB / 4us = 250 GB/s > the 239.6 GB/s measured ceiling
			wantClass:  cudaroofline.ClassAboveCeiling,
		},
		{
			name:       "zero_duration_edge_case",
			kernel:     "noop",
			flops:      0,
			bytes:      0,
			durationNs: 0,
			wantClass:  cudaroofline.ClassUnconstrained,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			pt := model.Evaluate(tt.kernel, tt.flops, tt.bytes, tt.durationNs)
			if pt.Classification != tt.wantClass {
				t.Errorf("Classification = %q, want %q (attained=%.3f TFLOPS, intensity=%.2f, eff=%.4f)",
					pt.Classification, tt.wantClass, pt.AttainedTFLOPS, pt.Intensity, pt.Efficiency)
			}
		})
	}
}

// TestEfficiencyAboveCeilingIsReported pins the reading that matters most: a
// launch that beats its ceiling refutes the ceiling, so the ratio must survive
// intact and the point must not be filed as a healthy memory-bound kernel.
//
// The numbers are real. ds4's Q8_0 matvecs attained 164.8 GB/s against a
// 162.0 GB/s host-page-table ceiling; a clamp to 1.0 is what hid the wrong
// denominator.
func TestEfficiencyAboveCeilingIsReported(t *testing.T) {
	ceiling := cudaroofline.Ceiling{
		Architecture:      "sm_121a",
		Residency:         cudaroofline.HostPageTableResident,
		PeakTFLOPS:        40.0,
		PeakMemoryGBps:    162.0,
		ComputeProvenance: cudaroofline.Unverified,
		MemoryProvenance:  cudaroofline.Measured,
	}
	if err := cudaroofline.ValidateCeiling(ceiling); err != nil {
		t.Fatal(err)
	}
	model := cudaroofline.NewModelWithCeiling(ceiling)

	// 1,648,000 bytes in 10,000 ns = 164.8 GB/s, intensity 2.0 FLOP/B.
	pt := model.Evaluate("ds4_q8_0_matvec", 3296000, 1648000, 10000)

	if got, want := pt.AttainedGBps, 164.8; got != want {
		t.Fatalf("AttainedGBps = %v, want %v", got, want)
	}
	if pt.Efficiency <= 1.0 {
		t.Errorf("Efficiency = %v, want > 1.0 (attained %v GB/s against a %v GB/s ceiling)",
			pt.Efficiency, pt.AttainedGBps, ceiling.PeakMemoryGBps)
	}
	if diff := pt.Efficiency - 164.8/162.0; diff < -1e-9 || diff > 1e-9 {
		t.Errorf("Efficiency = %v, want %v", pt.Efficiency, 164.8/162.0)
	}
	if pt.Classification != cudaroofline.ClassAboveCeiling {
		t.Errorf("Classification = %q, want %q", pt.Classification, cudaroofline.ClassAboveCeiling)
	}
	for _, healthy := range []string{cudaroofline.ClassMemoryBound, cudaroofline.ClassComputeBound} {
		if pt.Classification == healthy {
			t.Errorf("a launch above the ceiling was classified %q", healthy)
		}
	}
}

// TestResidencyChangesTheVerdict shows why Residency is part of the lookup key:
// the identical launch is a healthy memory-bound kernel against device-resident
// bandwidth and a refutation against host-page-table bandwidth, and a kernel
// that looks latency-bound against the device ceiling is in fact saturating the
// path it actually uses.
func TestResidencyChangesTheVerdict(t *testing.T) {
	device, err := cudaroofline.NewModel("sm_121a", cudaroofline.DeviceResident)
	if err != nil {
		t.Fatal(err)
	}
	host, err := cudaroofline.NewModel("sm_121a", cudaroofline.HostPageTableResident)
	if err != nil {
		t.Fatal(err)
	}
	if device.Ceiling.PeakMemoryGBps == host.Ceiling.PeakMemoryGBps {
		t.Fatalf("residency ceilings are identical (%v GB/s); the distinction is not expressible",
			device.Ceiling.PeakMemoryGBps)
	}

	tests := []struct {
		name       string
		flops      uint64
		bytes      uint64
		durationNs uint64
		wantDevice string
		wantHost   string
	}{
		{
			// 164.8 GB/s: 69% of 239.6, but 101% of 162.4.
			name: "ds4_q8_0_matvec", flops: 3296000, bytes: 1648000, durationNs: 10000,
			wantDevice: cudaroofline.ClassMemoryBound,
			wantHost:   cudaroofline.ClassAboveCeiling,
		},
		{
			// 70 GB/s: 29% of 239.6 (reads as latency-bound), but 43% of 162.4.
			name: "narrow_reader", flops: 140000, bytes: 70000, durationNs: 1000,
			wantDevice: cudaroofline.ClassUnconstrained,
			wantHost:   cudaroofline.ClassMemoryBound,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			d := device.Evaluate(tt.name, tt.flops, tt.bytes, tt.durationNs)
			h := host.Evaluate(tt.name, tt.flops, tt.bytes, tt.durationNs)
			if d.Classification != tt.wantDevice {
				t.Errorf("device-resident: Classification = %q, want %q (eff=%.4f)",
					d.Classification, tt.wantDevice, d.Efficiency)
			}
			if h.Classification != tt.wantHost {
				t.Errorf("host-page-table: Classification = %q, want %q (eff=%.4f)",
					h.Classification, tt.wantHost, h.Efficiency)
			}
			if d.AttainedGBps != h.AttainedGBps {
				t.Errorf("attained bandwidth differs across models: %v vs %v", d.AttainedGBps, h.AttainedGBps)
			}
		})
	}
}

func TestValidateCeiling(t *testing.T) {
	ok := cudaroofline.Ceiling{
		Architecture: "sm_121", Residency: cudaroofline.DeviceResident,
		PeakTFLOPS: 40, PeakMemoryGBps: 239.6,
		ComputeProvenance: cudaroofline.Unverified, MemoryProvenance: cudaroofline.Measured,
	}

	tests := []struct {
		name    string
		mutate  func(*cudaroofline.Ceiling)
		wantErr bool
	}{
		{name: "valid", mutate: func(*cudaroofline.Ceiling) {}},
		{name: "no compute", mutate: func(c *cudaroofline.Ceiling) { c.PeakTFLOPS = 0 }, wantErr: true},
		{name: "no bandwidth", mutate: func(c *cudaroofline.Ceiling) { c.PeakMemoryGBps = 0 }, wantErr: true},
		{name: "no residency", mutate: func(c *cudaroofline.Ceiling) { c.Residency = "" }, wantErr: true},
		{name: "bogus residency", mutate: func(c *cudaroofline.Ceiling) { c.Residency = "somewhere" }, wantErr: true},
		{name: "unmarked compute", mutate: func(c *cudaroofline.Ceiling) { c.ComputeProvenance = "" }, wantErr: true},
		{name: "unmarked memory", mutate: func(c *cudaroofline.Ceiling) { c.MemoryProvenance = "" }, wantErr: true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			c := ok
			tt.mutate(&c)
			err := cudaroofline.ValidateCeiling(c)
			if (err != nil) != tt.wantErr {
				t.Errorf("ValidateCeiling(%+v) error = %v, wantErr %v", c, err, tt.wantErr)
			}
		})
	}
}

func TestRenderSVG(t *testing.T) {
	model, err := cudaroofline.NewModel("sm_90", cudaroofline.DeviceResident)
	if err != nil {
		t.Fatal(err)
	}

	pt1 := model.Evaluate("matvec", 2000000, 1000000, 500)
	pt2 := model.Evaluate("gemm", 10000000000, 5000000, 160000)
	// 4 TB/s against a 3.35 TB/s ceiling: above the roof, and it must still be
	// drawn on the canvas.
	pt3 := model.Evaluate("impossible", 8000000, 4000000, 1000)
	if pt3.Classification != cudaroofline.ClassAboveCeiling {
		t.Fatalf("setup: Classification = %q, want %q", pt3.Classification, cudaroofline.ClassAboveCeiling)
	}

	var buf bytes.Buffer
	if err := model.RenderSVG(&buf, []cudaroofline.KernelPoint{pt1, pt2, pt3}); err != nil {
		t.Fatalf("RenderSVG failed: %v", err)
	}

	svg := buf.String()
	for _, expected := range []string{
		"<?xml version=",
		"<svg",
		"Roofline Model (sm_90, device memory)",
		"matvec",
		"gemm",
		"impossible",
		"pt-above-ceiling",
		"Memory Bound",
		"Compute Bound",
		"Above Ceiling",
		"(unverified)", // the ceiling labels must say what kind of number they are
		"</svg>",
	} {
		if !strings.Contains(svg, expected) {
			t.Errorf("SVG missing expected substring: %q", expected)
		}
	}

	// Every plotted mark stays inside the 800x520 viewBox, the above-ceiling
	// point included.
	coord := regexp.MustCompile(`c([xy])="([-0-9.]+)"`)
	for _, m := range coord.FindAllStringSubmatch(svg, -1) {
		v, err := strconv.ParseFloat(m[2], 64)
		if err != nil {
			t.Fatalf("unparsable c%s=%q: %v", m[1], m[2], err)
		}
		limit := 800.0
		if m[1] == "y" {
			limit = 520.0
		}
		if v < 0 || v > limit {
			t.Errorf("c%s = %v outside viewBox [0, %v]", m[1], v, limit)
		}
	}
}

// TestCeilingProvenanceTravelsWithTheVerdict pins the marking to the verdict it
// supports. On GB10 one ceiling carries two different kinds of number: the
// memory figure was measured, the 40 TFLOPS compute figure was not. A
// memory-bound reading of that ceiling is therefore a reading of the hardware
// and a compute-bound reading of the same ceiling is a hypothesis, and a
// consumer holding only the KernelPoint has to be able to tell them apart.
func TestCeilingProvenanceTravelsWithTheVerdict(t *testing.T) {
	tests := []struct {
		name       string
		arch       string
		residency  cudaroofline.Residency
		flops      uint64
		bytes      uint64
		durationNs uint64
		wantClass  string
		wantAxis   cudaroofline.LimitingAxis
		wantSrc    cudaroofline.Provenance
	}{
		{
			// 222.2 GB/s of a measured 239.6 GB/s, intensity 1 FLOP/B.
			name: "memory_bound_rests_on_the_measured_ceiling",
			arch: "sm_121", residency: cudaroofline.DeviceResident,
			flops: 1000000, bytes: 1000000, durationNs: 4500,
			wantClass: cudaroofline.ClassMemoryBound,
			wantAxis:  cudaroofline.MemoryLimited,
			wantSrc:   cudaroofline.Measured,
		},
		{
			// 35.7 TFLOPS of an unverified 40 TFLOPS, intensity 1000 FLOP/B.
			name: "compute_bound_rests_on_the_unverified_ceiling",
			arch: "sm_121", residency: cudaroofline.DeviceResident,
			flops: 1000000000, bytes: 1000000, durationNs: 28000,
			wantClass: cudaroofline.ClassComputeBound,
			wantAxis:  cudaroofline.ComputeLimited,
			wantSrc:   cudaroofline.Unverified,
		},
		{
			// 164.8 GB/s against the measured 162.4 GB/s host-page-table
			// ceiling: the exceeded limit is the memory one.
			name: "above_ceiling_names_the_ceiling_it_refutes",
			arch: "sm_121a", residency: cudaroofline.HostPageTableResident,
			flops: 3296000, bytes: 1648000, durationNs: 10000,
			wantClass: cudaroofline.ClassAboveCeiling,
			wantAxis:  cudaroofline.MemoryLimited,
			wantSrc:   cudaroofline.Measured,
		},
		{
			// Same launch, judged against a part whose every number is a
			// datasheet figure.
			name: "sm_90_memory_bound_is_still_unverified",
			arch: "sm_90", residency: cudaroofline.DeviceResident,
			flops: 2000000, bytes: 1000000, durationNs: 400,
			wantClass: cudaroofline.ClassMemoryBound,
			wantAxis:  cudaroofline.MemoryLimited,
			wantSrc:   cudaroofline.Unverified,
		},
		{
			// Nothing was divided by anything: no ceiling backs this verdict,
			// and the point must not claim one does.
			name: "no_duration_consults_no_ceiling",
			arch: "sm_121", residency: cudaroofline.DeviceResident,
			flops: 0, bytes: 0, durationNs: 0,
			wantClass: cudaroofline.ClassUnconstrained,
			wantAxis:  cudaroofline.NoLimit,
			wantSrc:   "",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			model, err := cudaroofline.NewModel(tt.arch, tt.residency)
			if err != nil {
				t.Fatal(err)
			}
			pt := model.Evaluate(tt.name, tt.flops, tt.bytes, tt.durationNs)
			if pt.Classification != tt.wantClass {
				t.Fatalf("Classification = %q, want %q (eff=%.4f)", pt.Classification, tt.wantClass, pt.Efficiency)
			}
			if pt.LimitingAxis != tt.wantAxis {
				t.Errorf("LimitingAxis = %q, want %q", pt.LimitingAxis, tt.wantAxis)
			}
			if pt.CeilingProvenance != tt.wantSrc {
				t.Errorf("CeilingProvenance = %q, want %q", pt.CeilingProvenance, tt.wantSrc)
			}
		})
	}
}

// TestProvenanceMatchesTheAxisItNames checks the dependency rather than the
// table: whatever axis a point names, its recorded provenance must be the
// provenance of that axis on the model's own ceiling. Stapling both markings on,
// or recording the wrong one, fails here for every entry in knownCeilings.
func TestProvenanceMatchesTheAxisItNames(t *testing.T) {
	// One launch per axis, reused across every part: intensity 1 FLOP/B is
	// below every knee in the table, intensity 1e6 is above every one.
	launches := []struct {
		name  string
		flops uint64
		bytes uint64
		axis  cudaroofline.LimitingAxis
	}{
		{"low_intensity", 1000000, 1000000, cudaroofline.MemoryLimited},
		{"high_intensity", 1000000000000, 1000000, cudaroofline.ComputeLimited},
	}

	for _, arch := range []string{"sm_70", "sm_75", "sm_80", "sm_86", "sm_89", "sm_90", "sm_121", "sm_121a"} {
		for _, res := range []cudaroofline.Residency{cudaroofline.DeviceResident, cudaroofline.HostPageTableResident} {
			model, err := cudaroofline.NewModel(arch, res)
			if err != nil {
				continue // this part has no ceiling for this memory path
			}
			for _, l := range launches {
				t.Run(arch+"/"+string(res)+"/"+l.name, func(t *testing.T) {
					// One nanosecond, so the launch lands above the
					// unconstrained cutoff and below no ceiling in particular.
					pt := model.Evaluate(l.name, l.flops, l.bytes, 1)
					if pt.LimitingAxis != l.axis {
						t.Fatalf("LimitingAxis = %q, want %q (intensity %.3g, knee %.3g)",
							pt.LimitingAxis, l.axis, pt.Intensity, model.KneeIntensity())
					}
					want := model.Ceiling.MemoryProvenance
					if pt.LimitingAxis == cudaroofline.ComputeLimited {
						want = model.Ceiling.ComputeProvenance
					}
					if pt.CeilingProvenance != want {
						t.Errorf("CeilingProvenance = %q, want %q for the %q axis",
							pt.CeilingProvenance, want, pt.LimitingAxis)
					}
				})
			}
		}
	}
}
