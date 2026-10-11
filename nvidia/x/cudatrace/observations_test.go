package cudatrace_test

import (
	"bytes"
	"encoding/binary"
	"errors"
	"fmt"
	"github.com/tmc/lib/nvidia/x/cudaprobe"
	"github.com/tmc/lib/nvidia/x/cudatrace"
	"github.com/tmc/lib/nvidia/x/nvtelemetry"
	"github.com/tmc/lib/nvidia/x/perfetto"
	"io"
	"math"
	"testing"
)

type observationField struct {
	number, integer uint64
	data            []byte
}

func observationFields(t *testing.T, b []byte) []observationField {
	t.Helper()
	var fs []observationField
	for len(b) > 0 {
		tag, n := binary.Uvarint(b)
		if n <= 0 {
			t.Fatal("bad tag")
		}
		b = b[n:]
		f := observationField{number: tag >> 3}
		switch tag & 7 {
		case 0:
			v, n := binary.Uvarint(b)
			if n <= 0 {
				t.Fatal("bad value")
			}
			f.integer = v
			b = b[n:]
		case 1:
			if len(b) < 8 {
				t.Fatal("truncated fixed64")
			}
			f.data = b[:8]
			b = b[8:]
		case 2:
			sz, n := binary.Uvarint(b)
			if n <= 0 || sz > uint64(len(b)-n) {
				t.Fatal("bad length")
			}
			b = b[n:]
			f.data = b[:sz]
			b = b[sz:]
		default:
			t.Fatal("bad wire type")
		}
		fs = append(fs, f)
	}
	return fs
}
func TestHostObservationClock(t *testing.T) {
	var out bytes.Buffer
	e := cudaprobe.Event{Identity: cudaprobe.Identity{Process: cudaprobe.Process{PID: 1, Generation: 2}, Symbol: "cuLaunchKernel"}, Sequence: 1, Timestamp: 5, ReturnKnown: true}
	r, err := cudatrace.WriteHostObservations(&out, cudatrace.ObservationClock{Domain: "bpf_monotonic", Epoch: "boot-A"}, []cudaprobe.Event{e}, math.MaxUint64, 1)
	if err != nil || r.Records != 1 || r.LostRecords != math.MaxUint64 || r.JoinStatus != "unmapped_separate_artifact" {
		t.Fatalf("report %+v err %v", r, err)
	}
	instants := 0
	for _, packet := range observationFields(t, out.Bytes()) {
		fs := observationFields(t, packet.data)
		clock := false
		for _, f := range fs {
			if f.number == perfetto.FieldPacketTimestampClockId {
				clock = f.integer == 128
			}
			if f.number == perfetto.FieldPacketClockSnapshot {
				t.Fatal("invented mapping")
			}
		}
		for _, f := range fs {
			if f.number == perfetto.FieldPacketTrackEvent {
				if !clock {
					t.Fatal("missing clock")
				}
				for _, e := range observationFields(t, f.data) {
					if e.number == perfetto.FieldTrackEventType {
						if e.integer != perfetto.TrackEventTypeInstant {
							t.Fatal("invented duration")
						}
						instants++
					}
				}
			}
		}
	}
	if instants != 1 {
		t.Fatalf("instants=%d", instants)
	}
}
func observationTelemetry() (nvtelemetry.Manifest, []nvtelemetry.Sample) {
	m := nvtelemetry.Manifest{ClockMonotonicDomain: "process_local", ClockMonotonicEpochID: "proc-A", Devices: []nvtelemetry.DeviceMetadata{{UUID: "GPU"}}}
	rs := []nvtelemetry.Reading{{Name: nvtelemetry.MetricMemoryUsedBytes, Unit: nvtelemetry.UnitBytes, DeviceUUID: "GPU", Evidence: nvtelemetry.EvidenceMeasured, IsInt: true, IntValue: 1<<53 + 1}, {Name: nvtelemetry.MetricTotalEnergyMJ, Unit: nvtelemetry.UnitMillijoules, DeviceUUID: "GPU", Evidence: nvtelemetry.EvidenceMeasured, IsInt: true, IntValue: math.MaxUint64}, {Name: nvtelemetry.MetricPowerUsageMW, Unit: nvtelemetry.UnitMilliwatts, DeviceUUID: "GPU", Evidence: nvtelemetry.EvidenceUnavailable, UnavailableReason: "not_supported"}}
	return m, []nvtelemetry.Sample{{QueryStartNanos: 0, QueryEndNanos: 10, Readings: rs}}
}
func TestTelemetryExactIntegers(t *testing.T) {
	m, s := observationTelemetry()
	var out bytes.Buffer
	r, err := cudatrace.WriteTelemetryCounters(&out, m, s, 3)
	if err != nil || r.Records != 3 {
		t.Fatalf("%+v %v", r, err)
	}
	counters := 0
	exact, unavailable := false, false
	for _, packet := range observationFields(t, out.Bytes()) {
		for _, f := range observationFields(t, packet.data) {
			if f.number != perfetto.FieldPacketTrackEvent {
				continue
			}
			for _, e := range observationFields(t, f.data) {
				if e.number == perfetto.FieldTrackEventCounterValue {
					counters++
					if e.integer != 1<<53+1 {
						t.Fatalf("rounded counter %d", e.integer)
					}
				}
				if e.number == perfetto.FieldTrackEventDoubleCounterVal {
					t.Fatal("float integer")
				}
				if e.number == perfetto.FieldTrackEventName && string(e.data) == "unavailable" {
					unavailable = true
				}
				if e.number == perfetto.FieldTrackEventDebugAnnotations {
					for _, a := range observationFields(t, e.data) {
						if a.number == perfetto.FieldDebugAnnotationUintValue && a.integer == math.MaxUint64 {
							exact = true
						}
					}
				}
			}
		}
	}
	if counters != 1 || !exact || !unavailable {
		t.Fatalf("counters %d exact %v unavailable %v", counters, exact, unavailable)
	}
}

type observationShortWriter struct{}

func (observationShortWriter) Write(p []byte) (int, error) { return len(p) - 1, nil }
func TestObservationRefusal(t *testing.T) {
	for _, name := range []string{"budget", "clock", "bounds", "identity", "nan", "kind", "unit", "status", "evidence", "sequence"} {
		t.Run(name, func(t *testing.T) {
			m, s := observationTelemetry()
			limit := 3
			switch name {
			case "budget":
				limit = 2
			case "clock":
				m.ClockMonotonicEpochID = ""
			case "bounds":
				s[0].Readings[0].HasQueryBounds = true
				s[0].Readings[0].QueryEndNanos = 11
			case "identity":
				s[0].Readings[0].DeviceUUID = "other"
			case "nan":
				s[0].Readings[0].IsInt = false
				s[0].Readings[0].Value = math.NaN()
			case "kind":
				s[0].Readings[0].Name = "unknown"
			case "unit":
				s[0].Readings[0].Unit = nvtelemetry.UnitMilliwatts
			case "status":
				s[0].Readings[0].Evidence = nvtelemetry.EvidenceMeasured
				s[0].Readings[0].UnavailableReason = "unsupported"
			case "evidence":
				s[0].Readings[0].Evidence = "invented"
			case "sequence":
				s[0].Seq = 2
				s = append(s, nvtelemetry.Sample{Seq: 1, QueryStartNanos: 10, QueryEndNanos: 11})
			}
			var out bytes.Buffer
			if _, err := cudatrace.WriteTelemetryCounters(&out, m, s, limit); err == nil {
				t.Fatal("admitted invalid input")
			}
			if out.Len() != 0 {
				t.Fatal("wrote invalid input")
			}
		})
	}
	m, s := observationTelemetry()
	if _, err := cudatrace.WriteTelemetryCounters(observationShortWriter{}, m, s, 3); !errors.Is(err, io.ErrShortWrite) {
		t.Fatalf("short write %v", err)
	}
}
func ExampleWriteTelemetryCounters() {
	m := nvtelemetry.Manifest{ClockMonotonicDomain: "process_local", ClockMonotonicEpochID: "example", Devices: []nvtelemetry.DeviceMetadata{{UUID: "GPU"}}}
	s := []nvtelemetry.Sample{{QueryStartNanos: 1, QueryEndNanos: 2, Readings: []nvtelemetry.Reading{{Name: nvtelemetry.MetricPowerUsageMW, Unit: nvtelemetry.UnitMilliwatts, DeviceUUID: "GPU", Evidence: nvtelemetry.EvidenceMeasured, Value: 1000}}}}
	report, err := cudatrace.WriteTelemetryCounters(io.Discard, m, s, 10)
	fmt.Println(report.Records, report.JoinStatus, err)
	// Output: 1 unmapped_separate_artifact <nil>
}
func ExampleWriteHostObservations() {
	e := cudaprobe.Event{Identity: cudaprobe.Identity{Process: cudaprobe.Process{PID: 1, Generation: 1}, Symbol: "cuCtxSynchronize"}, Sequence: 1, Timestamp: 1}
	report, err := cudatrace.WriteHostObservations(io.Discard, cudatrace.ObservationClock{Domain: "host_monotonic", Epoch: "example"}, []cudaprobe.Event{e}, 0, 10)
	fmt.Println(report.Records, err)
	// Output: 1 <nil>
}

func TestTelemetryCounterSegments(t *testing.T) {
	for _, boundary := range []string{"unavailable", "missing_evidence", "omitted", "clock_discontinuity", "unsigned_overflow"} {
		t.Run(boundary, func(t *testing.T) {
			m, s := observationTelemetry()
			reading := s[0].Readings[0]
			samples := []nvtelemetry.Sample{{Seq: 1, QueryStartNanos: 0, QueryEndNanos: 1, Readings: []nvtelemetry.Reading{reading}}, {Seq: 2, QueryStartNanos: 2, QueryEndNanos: 3, Readings: []nvtelemetry.Reading{reading}}, {Seq: 3, QueryStartNanos: 4, QueryEndNanos: 5, Readings: []nvtelemetry.Reading{reading}}}
			switch boundary {
			case "unavailable":
				samples[1].Readings[0].Evidence = nvtelemetry.EvidenceUnavailable
				samples[1].Readings[0].UnavailableReason = "unsupported"
			case "missing_evidence":
				samples[1].Readings[0].Evidence = ""
			case "omitted":
				samples[1].Readings = nil
			case "clock_discontinuity":
				samples[1].ClockDiscontinuity = true
			case "unsigned_overflow":
				samples[1].Readings[0].IntValue = math.MaxUint64
			}
			var out bytes.Buffer
			if _, err := cudatrace.WriteTelemetryCounters(&out, m, samples, 3); err != nil {
				t.Fatal(err)
			}
			var counterTracks []uint64
			for _, packet := range observationFields(t, out.Bytes()) {
				for _, f := range observationFields(t, packet.data) {
					if f.number != perfetto.FieldPacketTrackEvent {
						continue
					}
					var track uint64
					counter := false
					for _, e := range observationFields(t, f.data) {
						if e.number == perfetto.FieldTrackEventCounterValue {
							counter = true
						}
						if e.number == perfetto.FieldTrackEventTrackUUID {
							track = e.integer
						}
					}
					if counter {
						counterTracks = append(counterTracks, track)
					}
				}
			}
			if len(counterTracks) < 2 || counterTracks[0] == counterTracks[len(counterTracks)-1] {
				t.Fatalf("counter crossed %s: %v", boundary, counterTracks)
			}
		})
	}
}

func TestTelemetryEvidenceStatuses(t *testing.T) {
	for _, tt := range []struct {
		name, evidence, event string
		numeric               bool
	}{
		{"measured", nvtelemetry.EvidenceMeasured, "reading_metadata", true},
		{"derived", nvtelemetry.EvidenceDerived, "reading_metadata", true},
		{"unavailable", nvtelemetry.EvidenceUnavailable, "unavailable", false},
		{"missing evidence", "", "unknown", false},
	} {
		t.Run(tt.name, func(t *testing.T) {
			m, s := observationTelemetry()
			s[0].Readings = s[0].Readings[:1]
			r := &s[0].Readings[0]
			r.Evidence = tt.evidence
			r.IntValue = 9
			if tt.evidence == nvtelemetry.EvidenceUnavailable {
				r.UnavailableReason = "not_supported"
			}
			var out bytes.Buffer
			if _, err := cudatrace.WriteTelemetryCounters(&out, m, s, 1); err != nil {
				t.Fatal(err)
			}
			numeric, status := false, false
			for _, packet := range observationFields(t, out.Bytes()) {
				for _, f := range observationFields(t, packet.data) {
					if f.number != perfetto.FieldPacketTrackEvent {
						continue
					}
					for _, event := range observationFields(t, f.data) {
						if event.number == perfetto.FieldTrackEventCounterValue || event.number == perfetto.FieldTrackEventDoubleCounterVal {
							numeric = true
						}
						if event.number == perfetto.FieldTrackEventName && string(event.data) == tt.event {
							status = true
						}
					}
				}
			}
			if numeric != tt.numeric || !status {
				t.Fatalf("numeric=%v status=%v expected=%v,%q", numeric, status, tt.numeric, tt.event)
			}
		})
	}
}
