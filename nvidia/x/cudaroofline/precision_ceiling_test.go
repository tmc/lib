package cudaroofline

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"strings"
	"testing"
)

func constructedPrecisionCeiling() Ceiling {
	return Ceiling{
		Architecture:      "sm_121a",
		Residency:         DeviceResident,
		DeviceName:        "NVIDIA GB10 (constructed test profile)",
		MaxSMClockMHz:     2400,
		PeakTFLOPS:        40.0,
		PeakMemoryGBps:    239.6,
		ComputeProvenance: Unverified,
		MemoryProvenance:  Measured,
		PrecisionPeaks: map[Precision]PrecisionPeak{
			FP64: {
				PeakTFLOPS: 1.25,
				Provenance: Unverified,
				Citation:   "constructed test profile: dense FP64 pipe assertion",
			},
			FP16: {
				PeakTFLOPS: 80.0,
				Provenance: Unverified,
				Citation:   "constructed test profile: dense FP16 non-tensor pipe assertion",
			},
			TF32Tensor: {
				PeakTFLOPS: 80.0,
				Provenance: Unverified,
				Citation:   "constructed test profile: dense TF32 tensor assertion",
			},
			FP16Tensor: {
				PeakTFLOPS: 160.0,
				Provenance: Unverified,
				Citation:   "constructed test profile: dense FP16 tensor assertion",
			},
			BF16Tensor: {
				PeakTFLOPS: 160.0,
				Provenance: Unverified,
				Citation:   "constructed test profile: dense BF16 tensor assertion",
			},
			FP8Tensor: {
				PeakTFLOPS: 320.0,
				Provenance: Unverified,
				Citation:   "constructed test profile: dense FP8 tensor assertion",
			},
		},
	}
}

func TestPrecisionClassification(t *testing.T) {
	for _, tc := range []struct {
		p         Precision
		wantValid bool
		wantFloat bool
	}{
		{FP64, true, true},
		{FP32, true, true},
		{FP16, true, true},
		{TF32Tensor, true, true},
		{FP16Tensor, true, true},
		{BF16Tensor, true, true},
		{FP8Tensor, true, true},
		{INT8Tensor, true, false},
		{Precision("fp4_tensor"), false, false},
		{Precision(""), false, false},
	} {
		if got := tc.p.IsValid(); got != tc.wantValid {
			t.Errorf("%q.IsValid() = %v, want %v", tc.p, got, tc.wantValid)
		}
		if got := tc.p.IsFloatingPoint(); got != tc.wantFloat {
			t.Errorf("%q.IsFloatingPoint() = %v, want %v", tc.p, got, tc.wantFloat)
		}
	}
}

func TestPackagePeakTFLOPSForBuiltInRemainsFP32Only(t *testing.T) {
	for k, c := range knownCeilings {
		if len(c.PrecisionPeaks) != 0 {
			t.Errorf("knownCeilings[%+v] has %d PrecisionPeaks; built-in table must not carry unanchored non-FP32 peaks", k, len(c.PrecisionPeaks))
		}
	}

	peak, prov, err := PeakTFLOPSFor("sm_121a", FP32)
	if err != nil || peak != 40.0 || prov != Unverified {
		t.Fatalf("PeakTFLOPSFor(sm_121a, FP32) = (%v, %q, %v), want (40.0, %q, nil)", peak, prov, err, Unverified)
	}

	for _, p := range []Precision{FP64, FP16, TF32Tensor, FP16Tensor, BF16Tensor, FP8Tensor} {
		if got, _, err := PeakTFLOPSFor("sm_121a", p); err == nil {
			t.Errorf("PeakTFLOPSFor(sm_121a, %s) = %v, want error", p, got)
		} else if !strings.Contains(err.Error(), "FP32 non-tensor peaks only") {
			t.Errorf("PeakTFLOPSFor(sm_121a, %s) error = %q, want FP32-only message", p, err)
		}
	}

	if got, _, err := PeakTFLOPSFor("sm_121a", INT8Tensor); err == nil {
		t.Errorf("PeakTFLOPSFor(sm_121a, INT8Tensor) = %v, want error", got)
	} else if !strings.Contains(err.Error(), "integer TOPS") || !strings.Contains(err.Error(), "FP32 non-tensor peaks only") {
		t.Errorf("PeakTFLOPSFor(sm_121a, INT8Tensor) error = %q, want integer TOPS and FP32-only message", err)
	}
}

func TestModelPeakTFLOPSForAndCeilingAtFor(t *testing.T) {
	c := constructedPrecisionCeiling()
	if err := ValidateCeiling(c); err != nil {
		t.Fatalf("ValidateCeiling: %v", err)
	}
	m := NewModelWithCeiling(c)

	for _, tc := range []struct {
		p        Precision
		wantPeak float64
	}{
		{FP64, 1.25},
		{FP32, 40.0},
		{FP16, 80.0},
		{TF32Tensor, 80.0},
		{FP16Tensor, 160.0},
		{BF16Tensor, 160.0},
		{FP8Tensor, 320.0},
	} {
		peak, prov, err := m.PeakTFLOPSFor(tc.p)
		if err != nil {
			t.Fatalf("PeakTFLOPSFor(%s): %v", tc.p, err)
		}
		if peak != tc.wantPeak || prov != Unverified {
			t.Errorf("PeakTFLOPSFor(%s) = (%v, %q), want (%v, %q)", tc.p, peak, prov, tc.wantPeak, Unverified)
		}

		// Low arithmetic intensity (1 FLOP/Byte -> 0.2396 TFLOPS) is memory-limited for all precisions.
		memBound, err := m.CeilingAtFor(tc.p, 1.0)
		if err != nil {
			t.Fatalf("CeilingAtFor(%s, 1.0): %v", tc.p, err)
		}
		if want := 0.2396; math.Abs(memBound-want) > 1e-9 {
			t.Errorf("CeilingAtFor(%s, 1.0) = %v, want %v", tc.p, memBound, want)
		}

		// High arithmetic intensity (10000 FLOP/Byte -> 2396 TFLOPS memory slope) is bounded by tc.wantPeak, never FP32.
		compBound, err := m.CeilingAtFor(tc.p, 10000.0)
		if err != nil {
			t.Fatalf("CeilingAtFor(%s, 10000.0): %v", tc.p, err)
		}
		if compBound != tc.wantPeak {
			t.Errorf("CeilingAtFor(%s, 10000.0) = %v, want %v", tc.p, compBound, tc.wantPeak)
		}
	}

	// Removing one entry makes only that precision unavailable rather than falling back to FP32 (40.0).
	delete(m.Ceiling.PrecisionPeaks, BF16Tensor)
	if got, _, err := m.PeakTFLOPSFor(BF16Tensor); err == nil {
		t.Fatalf("PeakTFLOPSFor(BF16Tensor) = %v after deletion, want error", got)
	} else if !strings.Contains(err.Error(), "no bf16_tensor peak in model") {
		t.Errorf("PeakTFLOPSFor(BF16Tensor) error = %q, want missing bf16_tensor peak in model", err)
	}

	// INT8Tensor is always refused as an integer TOPS format.
	if got, _, err := m.PeakTFLOPSFor(INT8Tensor); err == nil {
		t.Fatalf("PeakTFLOPSFor(INT8Tensor) = %v, want error", got)
	} else if !strings.Contains(err.Error(), "integer TOPS") {
		t.Errorf("PeakTFLOPSFor(INT8Tensor) error = %q, want integer TOPS refusal", err)
	}
}

func TestValidateCeilingPrecisionPeaksAndIdentity(t *testing.T) {
	// Legacy FP32-only ceiling remains valid without DeviceName or MaxSMClockMHz,
	// and omitempty keeps JSON output free of empty precision-ceiling fields.
	legacy, err := LookupCeiling("sm_121a", DeviceResident)
	if err != nil {
		t.Fatalf("LookupCeiling: %v", err)
	}
	if err := ValidateCeiling(legacy); err != nil {
		t.Fatalf("ValidateCeiling(legacy): %v", err)
	}
	raw, err := json.Marshal(legacy)
	if err != nil {
		t.Fatalf("json.Marshal(legacy): %v", err)
	}
	for _, unexpected := range []string{"device_name", "max_sm_clock_mhz", "precision_peaks"} {
		if strings.Contains(string(raw), unexpected) {
			t.Errorf("legacy Ceiling JSON %s unexpectedly contains %q", raw, unexpected)
		}
	}

	// Consistent duplicate FP32 entry in PrecisionPeaks is allowed.
	withFP32 := constructedPrecisionCeiling()
	withFP32.PrecisionPeaks[FP32] = PrecisionPeak{
		PeakTFLOPS: withFP32.PeakTFLOPS,
		Provenance: withFP32.ComputeProvenance,
		Citation:   "constructed test profile: matching FP32 entry",
	}
	if err := ValidateCeiling(withFP32); err != nil {
		t.Fatalf("ValidateCeiling(with matching FP32 entry): %v", err)
	}

	for _, tc := range []struct {
		name    string
		mutate  func(*Ceiling)
		wantSub string
	}{
		{
			name: "missing_device_name",
			mutate: func(c *Ceiling) {
				c.DeviceName = "   "
			},
			wantSub: "DeviceName is required",
		},
		{
			name: "zero_max_sm_clock",
			mutate: func(c *Ceiling) {
				c.MaxSMClockMHz = 0
			},
			wantSub: "MaxSMClockMHz must be positive",
		},
		{
			name: "non_canonical_architecture",
			mutate: func(c *Ceiling) {
				c.Architecture = "gb10"
			},
			wantSub: "canonical sm_XY architecture",
		},
		{
			name: "sparse_peak_rejected",
			mutate: func(c *Ceiling) {
				pk := c.PrecisionPeaks[FP16Tensor]
				pk.Sparse = true
				c.PrecisionPeaks[FP16Tensor] = pk
			},
			wantSub: "sparse (2:4) peaks are unsupported",
		},
		{
			name: "nonpositive_precision_peak",
			mutate: func(c *Ceiling) {
				pk := c.PrecisionPeaks[FP16Tensor]
				pk.PeakTFLOPS = 0
				c.PrecisionPeaks[FP16Tensor] = pk
			},
			wantSub: "PeakTFLOPS must be finite and positive",
		},
		{
			name: "unrecognized_precision_provenance",
			mutate: func(c *Ceiling) {
				pk := c.PrecisionPeaks[FP16Tensor]
				pk.Provenance = Provenance("guessed")
				c.PrecisionPeaks[FP16Tensor] = pk
			},
			wantSub: "unknown provenance",
		},
		{
			name: "missing_citation",
			mutate: func(c *Ceiling) {
				pk := c.PrecisionPeaks[FP16Tensor]
				pk.Citation = ""
				c.PrecisionPeaks[FP16Tensor] = pk
			},
			wantSub: "Citation is required",
		},
		{
			name: "int8_tensor_in_precision_peaks",
			mutate: func(c *Ceiling) {
				c.PrecisionPeaks[INT8Tensor] = PrecisionPeak{
					PeakTFLOPS: 640.0,
					Provenance: Unverified,
					Citation:   "invalid integer TOPS entry in TFLOPS map",
				}
			},
			wantSub: "measures integer TOPS, not floating-point TFLOPS",
		},
		{
			name: "unknown_precision_key",
			mutate: func(c *Ceiling) {
				c.PrecisionPeaks[Precision("fp4_tensor")] = PrecisionPeak{
					PeakTFLOPS: 640.0,
					Provenance: Unverified,
					Citation:   "unknown precision",
				}
			},
			wantSub: "unknown precision",
		},
		{
			name: "contradictory_fp32_entry",
			mutate: func(c *Ceiling) {
				c.PrecisionPeaks[FP32] = PrecisionPeak{
					PeakTFLOPS: 99.0,
					Provenance: Unverified,
					Citation:   "contradicts Ceiling.PeakTFLOPS",
				}
			},
			wantSub: "disagrees with Ceiling.PeakTFLOPS",
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			bad := constructedPrecisionCeiling()
			tc.mutate(&bad)
			if err := ValidateCeiling(bad); err == nil || !strings.Contains(err.Error(), tc.wantSub) {
				t.Fatalf("ValidateCeiling = %v, want error containing %q", err, tc.wantSub)
			}
			m := NewModelWithCeiling(bad)
			if _, _, err := m.PeakTFLOPSFor(FP16Tensor); err == nil || !strings.Contains(err.Error(), tc.wantSub) {
				t.Fatalf("PeakTFLOPSFor(FP16Tensor) = %v, want error containing %q", err, tc.wantSub)
			}
		})
	}
}

func TestModelCeilingSingleAuthorityAndDefensiveClone(t *testing.T) {
	inputPeaks := map[Precision]PrecisionPeak{
		FP16Tensor: {
			PeakTFLOPS: 160.0,
			Provenance: Unverified,
			Citation:   "constructed test profile: initial FP16 tensor peak",
		},
	}
	c := Ceiling{
		Architecture:      "sm_121a",
		Residency:         DeviceResident,
		DeviceName:        "NVIDIA GB10 (constructed test profile)",
		MaxSMClockMHz:     2400,
		PeakTFLOPS:        40.0,
		PeakMemoryGBps:    239.6,
		ComputeProvenance: Unverified,
		MemoryProvenance:  Measured,
		PrecisionPeaks:    inputPeaks,
	}
	m := NewModelWithCeiling(c)

	// Mutating the caller's input map after construction must not affect m.
	inputPeaks[FP16Tensor] = PrecisionPeak{
		PeakTFLOPS: 999.0,
		Provenance: Unverified,
		Citation:   "mutated caller input map",
	}
	inputPeaks[BF16Tensor] = PrecisionPeak{
		PeakTFLOPS: 888.0,
		Provenance: Unverified,
		Citation:   "injected into caller input map",
	}

	got, _, err := m.PeakTFLOPSFor(FP16Tensor)
	if err != nil || got != 160.0 {
		t.Fatalf("PeakTFLOPSFor(FP16Tensor) after caller map mutation = (%v, %v), want (160.0, nil)", got, err)
	}
	if _, _, err := m.PeakTFLOPSFor(BF16Tensor); err == nil {
		t.Fatal("PeakTFLOPSFor(BF16Tensor) succeeded from mutated caller map alias")
	}

	// Model.Ceiling is the single authoritative state on Model: explicit mutations
	// to m.Ceiling and m.Ceiling.PrecisionPeaks affect future reads consistently.
	m.Ceiling.PeakTFLOPS = 44.0
	fp32Peak, _, err := m.PeakTFLOPSFor(FP32)
	if err != nil || fp32Peak != 44.0 {
		t.Fatalf("PeakTFLOPSFor(FP32) after m.Ceiling.PeakTFLOPS update = (%v, %v), want (44.0, nil)", fp32Peak, err)
	}
	if got := m.CeilingAt(10000.0); got != 44.0 {
		t.Fatalf("CeilingAt(10000.0) after m.Ceiling.PeakTFLOPS update = %v, want 44.0", got)
	}

	m.Ceiling.PrecisionPeaks[BF16Tensor] = PrecisionPeak{
		PeakTFLOPS: 175.0,
		Provenance: Measured,
		Citation:   "constructed test profile: explicitly updated BF16 tensor peak",
	}
	bf16Peak, bf16Prov, err := m.PeakTFLOPSFor(BF16Tensor)
	if err != nil || bf16Peak != 175.0 || bf16Prov != Measured {
		t.Fatalf("PeakTFLOPSFor(BF16Tensor) after m.Ceiling.PrecisionPeaks update = (%v, %q, %v), want (175.0, %q, nil)",
			bf16Peak, bf16Prov, err, Measured)
	}
}

func TestPrecisionCeilingExportedSurface(t *testing.T) {
	raw, err := os.ReadFile("testdata/exported-surface.txt")
	if err != nil {
		t.Fatalf("reading testdata/exported-surface.txt: %v", err)
	}
	gotLines := strings.Split(strings.TrimSuffix(string(raw), "\n"), "\n")
	gotSet := make(map[string]bool, len(gotLines))
	for _, ln := range gotLines {
		gotSet[ln] = true
	}
	for _, want := range []string{
		"  field Ceiling.DeviceName string",
		"  field Ceiling.MaxSMClockMHz uint32",
		"  field Ceiling.PrecisionPeaks map[Precision]PrecisionPeak",
		"  field PrecisionPeak.Citation string",
		"  field PrecisionPeak.PeakTFLOPS float64",
		"  field PrecisionPeak.Provenance Provenance",
		"  field PrecisionPeak.Sparse bool",
		"func (*Model) CeilingAtFor(p Precision, intensity float64) (float64, error)",
		"func (*Model) PeakTFLOPSFor(p Precision) (float64, Provenance, error)",
		"func (Precision) IsFloatingPoint() bool",
		"func (Precision) IsValid() bool",
		"type PrecisionPeak struct",
	} {
		if !gotSet[want] {
			t.Errorf("testdata/exported-surface.txt missing %q", want)
		}
	}
	for _, astLine := range exportedSurface(t) {
		if strings.Contains(astLine, "GEMMSpatial") || strings.Contains(astLine, "GEMMSplitK") ||
			strings.Contains(astLine, "GEMMReduction") || strings.Contains(astLine, "SpatialHierarchy") ||
			strings.Contains(astLine, "SpatialTensor") {
			continue
		}
		if !gotSet[astLine] {
			t.Errorf("exported declaration %q missing from testdata/exported-surface.txt", astLine)
		}
	}
}

func ExampleModel_PeakTFLOPSFor() {
	m := NewModelWithCeiling(Ceiling{
		Architecture:      "sm_121a",
		Residency:         DeviceResident,
		DeviceName:        "NVIDIA GB10 (constructed example profile)",
		MaxSMClockMHz:     2400,
		PeakTFLOPS:        40.0,
		PeakMemoryGBps:    239.6,
		ComputeProvenance: Unverified,
		MemoryProvenance:  Measured,
		PrecisionPeaks: map[Precision]PrecisionPeak{
			FP16Tensor: {
				PeakTFLOPS: 160.0,
				Provenance: Unverified,
				Citation:   "constructed example profile: dense FP16 tensor assertion",
			},
		},
	})

	peak, prov, err := m.PeakTFLOPSFor(FP16Tensor)
	if err != nil {
		panic(err)
	}
	ceilingAtHighIntensity, err := m.CeilingAtFor(FP16Tensor, 1000.0)
	if err != nil {
		panic(err)
	}
	_, _, int8Err := m.PeakTFLOPSFor(INT8Tensor)
	fmt.Printf("fp16_tensor peak=%.1f (%s) ceiling=%.1f int8_refused=%v\n",
		peak, prov, ceilingAtHighIntensity, int8Err != nil)
	// Output:
	// fp16_tensor peak=160.0 (unverified) ceiling=160.0 int8_refused=true
}

func ExamplePrecision_IsValid() {
	fmt.Println(FP16Tensor.IsValid())
	fmt.Println(Precision("fp128").IsValid())
	// Output:
	// true
	// false
}

func ExamplePrecision_IsFloatingPoint() {
	fmt.Println(FP16Tensor.IsFloatingPoint())
	fmt.Println(INT8Tensor.IsFloatingPoint())
	// Output:
	// true
	// false
}

func ExampleModel_CeilingAtFor() {
	// Every profile number below is an illustrative caller assertion, not
	// a hardware measurement or a built-in peak for this architecture.
	m := NewModelWithCeiling(Ceiling{
		Architecture:      "sm_90",
		Residency:         DeviceResident,
		DeviceName:        "illustrative device",
		MaxSMClockMHz:     1000,
		PeakTFLOPS:        2,
		PeakMemoryGBps:    100,
		ComputeProvenance: Unverified,
		MemoryProvenance:  Unverified,
		PrecisionPeaks: map[Precision]PrecisionPeak{
			FP16Tensor: {
				PeakTFLOPS: 4,
				Provenance: Unverified,
				Citation:   "illustrative dense-peak assertion, not measured",
			},
		},
	})
	for _, intensity := range []float64{10, 100} {
		peak, err := m.CeilingAtFor(FP16Tensor, intensity)
		if err != nil {
			fmt.Println(err)
			return
		}
		fmt.Printf("%.1f TFLOPS\n", peak)
	}
	// Output:
	// 1.0 TFLOPS
	// 4.0 TFLOPS
}

func ExamplePrecisionPeak() {
	// This made-up dense peak illustrates metadata, not device performance.
	peak := PrecisionPeak{
		PeakTFLOPS: 4,
		Provenance: Unverified,
		Citation:   "illustrative caller assertion, not measured",
	}
	fmt.Printf("%.1f TFLOPS (%s), sparse=%t\n", peak.PeakTFLOPS, peak.Provenance, peak.Sparse)
	// Output:
	// 4.0 TFLOPS (unverified), sparse=false
}
