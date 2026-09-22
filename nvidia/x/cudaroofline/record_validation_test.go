package cudaroofline

import (
	"bytes"
	"encoding/json"
	"math"
	"strings"
	"testing"
)

func TestRecordRejectsContradictoryFloors(t *testing.T) {
	for _, field := range []string{"negative", "nan", "infinity", "basis", "axis", "max", "verdict", "zero_duration", "provenance", "peak"} {
		t.Run(field, func(t *testing.T) {
			m, judgments := goldenJudgments(t)
			r := NewRecord(m, judgments)
			x := &r.Triples[0]
			switch field {
			case "negative":
				x.FloorNs = -2
			case "nan":
				x.MemoryFloorNs = math.NaN()
			case "infinity":
				x.ComputeFloorNs = math.Inf(1)
			case "basis":
				x.Basis = string(FloorFromComputeOnly)
			case "axis":
				x.Axis = string(ComputeLimited)
			case "max":
				x.FloorNs++
			case "verdict":
				x.Verdict = string(VerdictRejected)
			case "zero_duration":
				x.DurationNs = 0
			case "provenance":
				x.CeilingProvenance = "invented"
			case "peak":
				r.Device.PeakTFLOPS = -1
			}
			if err := r.Validate(); err == nil {
				t.Fatal("accepted contradiction")
			}
			var out bytes.Buffer
			if err := r.Encode(&out); err == nil || out.Len() != 0 {
				t.Fatal("wrote invalid record")
			}
			if data, err := json.Marshal(r); err == nil {
				if _, err := DecodeRecord(bytes.NewReader(data)); err == nil {
					t.Fatal("decoded contradiction")
				}
			}
		})
	}
}

func TestRecordRequiresSingleJSONValue(t *testing.T) {
	m, js := goldenJudgments(t)
	var out bytes.Buffer
	if err := NewRecord(m, js).Encode(&out); err != nil {
		t.Fatal(err)
	}
	for _, suffix := range []string{"{}", "null", "garbage"} {
		if _, err := DecodeRecord(strings.NewReader(out.String() + suffix)); err == nil {
			t.Fatalf("accepted suffix %q", suffix)
		}
	}
	if _, err := DecodeRecord(strings.NewReader(out.String() + " \n\t")); err != nil {
		t.Fatal(err)
	}
}

func TestRecordAbsentFloorIsNotAcceptance(t *testing.T) {
	r := Record{Version: RecordVersion, Device: DeviceRecord{Architecture: "unknown", Residency: "device"}, Triples: []Triple{{FloorNs: -1, MemoryFloorNs: -1, ComputeFloorNs: -1, Basis: string(FloorAbsent), Verdict: string(VerdictUnjudged)}}}
	if err := r.Validate(); err != nil {
		t.Fatal(err)
	}
	r.Triples[0].Verdict = string(VerdictAccepted)
	if err := r.Validate(); err == nil {
		t.Fatal("accepted absent floor")
	}
}

func TestRecordFloorBoundaries(t *testing.T) {
	for _, tt := range []struct {
		name            string
		memory, compute float64
		floor           float64
		basis           FloorBasis
		axis            LimitingAxis
	}{
		{"memory", 2, -1, 2, FloorFromMemoryOnly, MemoryLimited},
		{"compute", -1, 2, 2, FloorFromComputeOnly, ComputeLimited},
		{"tie", 2, 2, 2, FloorFromBothAxes, MemoryLimited},
		{"zero", 0, 0, 0, FloorFromBothAxes, MemoryLimited},
	} {
		t.Run(tt.name, func(t *testing.T) {
			for _, duration := range []uint64{0, 1, 2, 3} {
				verdict := VerdictAccepted
				if duration == 0 || float64(duration) < tt.floor {
					verdict = VerdictRejected
				}
				r := Record{
					Version: RecordVersion,
					Device:  DeviceRecord{Architecture: "opaque", Residency: "opaque", ComputeProvenance: "compute", MemoryProvenance: "memory"},
					Triples: []Triple{{Name: tt.name, DurationNs: duration, FloorNs: tt.floor, MemoryFloorNs: tt.memory, ComputeFloorNs: tt.compute, Basis: string(tt.basis), Axis: string(tt.axis), Verdict: string(verdict), CeilingProvenance: string(tt.axis)}},
				}
				var out bytes.Buffer
				if err := r.Encode(&out); err != nil {
					t.Fatalf("duration %d: %v", duration, err)
				}
				if _, err := DecodeRecord(&out); err != nil {
					t.Fatalf("duration %d: %v", duration, err)
				}
			}
		})
	}
}
