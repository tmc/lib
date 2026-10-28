package nvtelemetry_test

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"math"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/nvml"
	"github.com/tmc/lib/nvidia/x/nvtelemetry"
)

type counterDevice struct {
	*linkDevice
	clock  *nvtelemetry.FakeClock
	values [5]uint64
	errors map[nvml.NvLinkErrorCounter]error
	calls  int
	change func()
}

func (d *counterDevice) NvLinkErrorCounter(_ uint32, c nvml.NvLinkErrorCounter) (uint64, error) {
	d.calls++
	d.clock.Advance(time.Nanosecond)
	if d.change != nil {
		d.change()
	}
	return d.values[int(c)], d.errors[c]
}

func counterFixture() (*counterDevice, *nvtelemetry.FakeClock, nvtelemetry.NVLinkCounterConfig) {
	d, clock, topology := linkFixture()
	device := &counterDevice{linkDevice: d, clock: clock, values: [5]uint64{1<<53 + 1, 0, 2, 3, 4}, errors: map[nvml.NvLinkErrorCounter]error{}}
	config := nvtelemetry.NVLinkCounterConfig{Topology: topology, Counters: []nvml.NvLinkErrorCounter{nvml.NvlinkErrorDlReplay, nvml.NvlinkErrorDlRecovery, nvml.NvlinkErrorDlCrcFlit, nvml.NvlinkErrorDlCrcData, nvml.NvlinkErrorDlEccData}}
	return device, clock, config
}

func TestNVLinkCounterExplicitIdentity(t *testing.T) {
	d, clock, config := counterFixture()
	config.Counters = config.Counters[:1]
	reader, err := nvtelemetry.NewNVLinkCounterReader(d, clock, config)
	if err != nil {
		t.Fatal(err)
	}
	snapshot, err := reader.Read(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	raw, err := json.Marshal(snapshot)
	if err != nil {
		t.Fatal(err)
	}
	for _, field := range []string{"link", "counter"} {
		for _, state := range []string{"zero", "missing", "null"} {
			t.Run(field+"/"+state, func(t *testing.T) {
				var body map[string]json.RawMessage
				if err := json.Unmarshal(raw, &body); err != nil {
					t.Fatal(err)
				}
				var observations []map[string]json.RawMessage
				if err := json.Unmarshal(body["observations"], &observations); err != nil {
					t.Fatal(err)
				}
				switch state {
				case "zero":
					observations[0][field] = json.RawMessage("0")
				case "missing":
					delete(observations[0], field)
				case "null":
					observations[0][field] = json.RawMessage("null")
				}
				body["observations"], err = json.Marshal(observations)
				if err != nil {
					t.Fatal(err)
				}
				changed, err := json.Marshal(body)
				if err != nil {
					t.Fatal(err)
				}
				_, readErr := nvtelemetry.ReadNVLinkCounterSnapshot(bytes.NewReader(changed), 256<<10)
				_, convertErr := nvtelemetry.NVLinkCounterReadings(changed, "synthetic")
				if state == "zero" {
					if readErr != nil || convertErr != nil {
						t.Fatalf("explicit zero identity refused: read=%v convert=%v", readErr, convertErr)
					}
				} else if readErr == nil || convertErr == nil {
					t.Fatalf("%s identity accepted: read=%v convert=%v", state, readErr, convertErr)
				}
			})
		}
	}
}

type counterSampler struct {
	reader *nvtelemetry.NVLinkCounterReader
	raw    []byte
}

func (s *counterSampler) Devices() []nvtelemetry.DeviceMetadata {
	return []nvtelemetry.DeviceMetadata{{UUID: "GPU-0", MIGScope: nvtelemetry.MIGScopeDisabled}}
}
func (s *counterSampler) Sample(ctx context.Context) ([]nvtelemetry.Reading, error) {
	snapshot, err := s.reader.Read(ctx)
	if err != nil {
		return nil, err
	}
	s.raw, err = json.Marshal(snapshot)
	if err != nil {
		return nil, err
	}
	return nvtelemetry.NVLinkCounterReadings(s.raw, "synthetic")
}

func TestNVLinkCounterCanonicalReplay(t *testing.T) {
	for _, value := range []uint64{1<<53 + 1, 0, math.MaxUint64} {
		t.Run(stringValue(value), func(t *testing.T) {
			d, clock, config := counterFixture()
			d.values[0] = value
			reader, err := nvtelemetry.NewNVLinkCounterReader(d, clock, config)
			if err != nil {
				t.Fatal(err)
			}
			sampler := &counterSampler{reader: reader}
			var out bytes.Buffer
			cfg := nvtelemetry.CollectorConfig{SchemaVersion: nvtelemetry.SchemaVersionV3}
			collector := nvtelemetry.NewCollector("nvlink-host-control", sampler, clock, cfg, nvtelemetry.NewWriter(&out, cfg))
			if err := collector.InitStream(); err != nil {
				t.Fatal(err)
			}
			if err := collector.Step(context.Background()); err != nil {
				t.Fatal(err)
			}
			if _, err := collector.Stop(); err != nil {
				t.Fatal(err)
			}
			m, _, samples, err := nvtelemetry.ReadStreamStrict(&out, 1<<20)
			if err != nil {
				t.Fatal(err)
			}
			if m.MetricRegistryVersion != "2026.10-v3" || m.SchemaVersion != 3 {
				t.Fatalf("dictionary/stream versions conflated: %+v", m)
			}
			readings := samples[0].Readings
			if len(readings) != 5 || readings[0].Number == nil || readings[0].Number.Uint == nil || *readings[0].Number.Uint != value || readings[0].Number.Float != nil {
				t.Fatalf("exact raw count lost: %+v", readings)
			}
			if !readings[1].IsAvailable() || readings[1].Number.Uint == nil || *readings[1].Number.Uint != 0 {
				t.Fatal("observed zero missing")
			}
			names := []string{"nvlink_dl_replay_errors", "nvlink_dl_recovery_errors", "nvlink_dl_crc_flit_errors", "nvlink_dl_crc_data_errors", "nvlink_dl_ecc_data_errors"}
			for i, r := range readings {
				if r.Name != names[i] || r.Number.Uint == nil || *r.Number.Uint != d.values[i] {
					t.Fatalf("raw enum/value mapping changed at %d: %+v", i, r)
				}
				kind, found := nvtelemetry.LookupMetricKind(r.Name)
				if !found || kind != nvtelemetry.KindCounter || r.Unit != "count" {
					t.Fatalf("noncanonical count %+v", r)
				}
				if r.Observation.CounterEpoch != "" || r.HasSourceTimestamp || r.SourceTimestamp != 0 || !r.HasQueryBounds || r.QueryEndNanos < r.QueryStartNanos || !strings.Contains(r.Observation.Scope, "GPU-1") {
					t.Fatalf("invented reset/sensor/endpoint metadata %+v", r)
				}
			}
			if d.calls != 5 {
				t.Fatalf("unselected calls: %d", d.calls)
			}
		})
	}
}

func stringValue(v uint64) string {
	switch v {
	case 0:
		return "zero"
	case math.MaxUint64:
		return "max-uint64"
	default:
		return "above-float-exactness"
	}
}

func TestNVLinkCounterUnavailable(t *testing.T) {
	for _, tc := range []struct {
		name          string
		err           error
		state, reason string
	}{
		{"unsupported", nvml.ErrNotSupported, "unsupported", nvtelemetry.ReasonUnsupported},
		{"missing-api", nvml.ErrFunctionNotFound, "unsupported", nvtelemetry.ReasonUnsupported},
		{"missing-symbol", &nvml.SymbolError{Symbol: nvtelemetry.NVLinkErrorCounterAPI}, "unsupported", nvtelemetry.ReasonUnsupported},
		{"binding-failure", &nvml.SymbolError{Symbol: nvtelemetry.NVLinkErrorCounterAPI, Err: errors.New("host binding failure")}, "failed", nvtelemetry.ReasonQueryFailure},
		{"binding-native-like-error", &nvml.SymbolError{Symbol: nvtelemetry.NVLinkErrorCounterAPI, Err: nvml.ErrNotSupported}, "failed", nvtelemetry.ReasonQueryFailure},
		{"different-symbol", &nvml.SymbolError{Symbol: "other"}, "failed", nvtelemetry.ReasonQueryFailure},
		{"denied", nvml.ErrNoPermission, "denied", nvtelemetry.ReasonPermissionDenied},
		{"future-error", nvml.Result(12345), "failed", nvtelemetry.ReasonQueryFailure},
		{"non-native-error", errors.New("host fixture failure"), "failed", nvtelemetry.ReasonQueryFailure},
	} {
		t.Run(tc.name, func(t *testing.T) {
			d, c, config := counterFixture()
			d.errors[nvml.NvlinkErrorDlReplay] = tc.err
			r, err := nvtelemetry.NewNVLinkCounterReader(d, c, config)
			if err != nil {
				t.Fatal(err)
			}
			s, err := r.Read(context.Background())
			if err != nil {
				t.Fatal(err)
			}
			if s.Observations[0].Query.State != tc.state || s.Observations[0].Value != nil {
				t.Fatalf("failure became zero %+v", s.Observations[0])
			}
			raw, _ := json.Marshal(s)
			readings, err := nvtelemetry.NVLinkCounterReadings(raw, "synthetic")
			if err != nil {
				t.Fatal(err)
			}
			if readings[0].Number != nil || readings[0].UnavailableReason != tc.reason || readings[0].IsAvailable() {
				t.Fatalf("unavailable numeric arm fabricated %+v", readings[0])
			}
			if tc.name == "future-error" && s.Observations[0].Query.NativeCode != "12345" {
				t.Fatal("native error lost")
			}
			if (strings.Contains(tc.name, "symbol") || strings.HasPrefix(tc.name, "binding-")) && s.Observations[0].Query.NativeCode != "" {
				t.Fatal("invented native result code")
			}
		})
	}
}

func TestNVLinkCounterIdentityAndCancellation(t *testing.T) {
	for _, change := range []string{"uuid", "peer", "clock-epoch", "cancel", "expired", "lost"} {
		t.Run(change, func(t *testing.T) {
			d, c, config := counterFixture()
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			d.change = func() {
				switch change {
				case "uuid":
					d.uuid = "GPU-other"
				case "peer":
					copy(d.pci.BusId[:], "00000000:04:00.0")
					d.pci.Bus = 4
				case "clock-epoch":
					c.SetEpochID("other")
				case "cancel":
					cancel()
				case "expired":
					c.Advance(time.Second)
				}
			}
			if change == "lost" {
				d.errors[nvml.NvlinkErrorDlReplay] = nvml.ErrGpuIsLost
			}
			r, err := nvtelemetry.NewNVLinkCounterReader(d, c, config)
			if err != nil {
				t.Fatal(err)
			}
			s, err := r.Read(ctx)
			if err == nil || s != nil {
				t.Fatalf("untrusted batch returned %+v %v", s, err)
			}
			if change == "cancel" && !errors.Is(err, context.Canceled) {
				t.Fatal(err)
			}
			if change == "lost" && !errors.Is(err, nvml.ErrGpuIsLost) {
				t.Fatal(err)
			}
		})
	}
}

func TestNVLinkCounterSelectionAndCopy(t *testing.T) {
	for _, counters := range [][]nvml.NvLinkErrorCounter{nil, {nvml.NvlinkErrorDlReplay, nvml.NvlinkErrorDlReplay}, {nvml.NvLinkErrorCounter(-1)}, {nvml.NvlinkErrorCount}, {0, 1, 2, 3, 4, 5}} {
		d, c, config := counterFixture()
		config.Counters = counters
		if _, err := nvtelemetry.NewNVLinkCounterReader(d, c, config); err == nil || d.reads != 0 || d.calls != 0 {
			t.Fatal("invalid selection touched device")
		}
	}
	d, c, config := counterFixture()
	r, err := nvtelemetry.NewNVLinkCounterReader(d, c, config)
	if err != nil {
		t.Fatal(err)
	}
	config.Counters[0] = 99
	config.Topology.Links[0] = 31
	config.Topology.Peers[0].UUID = "GPU-evil"
	s, err := r.Read(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if s.Observations[0].Counter != 0 || s.Observations[0].Link != 0 || s.Before.Links[0].PeerUUID != "GPU-1" {
		t.Fatal("borrowed mutable config")
	}
	old, clock, cfg := linkFixture()
	if _, err := nvtelemetry.NewNVLinkCounterReader(old, clock, nvtelemetry.NVLinkCounterConfig{Topology: cfg, Counters: []nvml.NvLinkErrorCounter{0}}); err == nil {
		t.Fatal("optional interface claimed available")
	}
	if _, err := nvtelemetry.NewNVLinkReader(old, clock, cfg); err != nil {
		t.Fatal("existing interface compatibility lost", err)
	}
}

func TestNVLinkCounterStrictRawSnapshot(t *testing.T) {
	d, c, config := counterFixture()
	r, _ := nvtelemetry.NewNVLinkCounterReader(d, c, config)
	s, err := r.Read(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	for _, kind := range []string{"version", "api", "reset", "uuid", "peer", "topology", "missing", "counter", "bounds", "available-missing", "failed-value", "native-code"} {
		t.Run(kind, func(t *testing.T) {
			raw, _ := json.Marshal(s)
			var x nvtelemetry.NVLinkCounterSnapshot
			if err := json.Unmarshal(raw, &x); err != nil {
				t.Fatal(err)
			}
			switch kind {
			case "version":
				x.Version++
			case "api":
				x.API = "reset-counter"
			case "reset":
				x.ResetCoverage = "known-zero-reset"
			case "uuid":
				x.After.DeviceUUID = "GPU-other"
			case "peer":
				x.After.PeerAssignments[0].UUID = "GPU-other"
				x.After.Links[0].PeerUUID = "GPU-other"
			case "topology":
				x.After.TopologyEpoch = "next"
			case "missing":
				x.Observations = x.Observations[:4]
			case "counter":
				x.Observations[0].Counter = 1
			case "bounds":
				x.Observations[0].Query.StartNanos = x.Before.AtNanos - 1
			case "available-missing":
				x.Observations[0].Value = nil
			case "failed-value":
				x.Observations[0].Query.State = "failed"
			case "native-code":
				x.Observations[0].Query.NativeCode = "0"
			}
			raw, _ = json.Marshal(x)
			if _, err := nvtelemetry.ReadNVLinkCounterSnapshot(bytes.NewReader(raw), 256<<10); err == nil {
				t.Fatal("invalid snapshot admitted")
			}
		})
	}
	raw, _ := json.Marshal(s)
	if _, err := nvtelemetry.ReadNVLinkCounterSnapshot(bytes.NewReader(raw), 2); err == nil {
		t.Fatal("byte cap ignored")
	}
	duplicate := append([]byte(`{"version":2,`), raw[1:]...)
	if _, err := nvtelemetry.ReadNVLinkCounterSnapshot(bytes.NewReader(duplicate), 256<<10); err == nil {
		t.Fatal("duplicate version admitted")
	}
}

func TestNVLinkCounterCanonicalUnitsRefuse(t *testing.T) {
	d, c, config := counterFixture()
	r, _ := nvtelemetry.NewNVLinkCounterReader(d, c, config)
	s, _ := r.Read(context.Background())
	raw, _ := json.Marshal(s)
	readings, _ := nvtelemetry.NVLinkCounterReadings(raw, "synthetic")
	for _, change := range []string{"unchanged", "unit", "registered-name"} {
		t.Run(change, func(t *testing.T) {
			reading := readings[0]
			if change == "unit" {
				reading.Unit = nvtelemetry.UnitBytesPerSec
			} else if change == "registered-name" {
				reading.Name = nvtelemetry.MetricNVLinkRxBytesPerSec
			}
			var out bytes.Buffer
			w := nvtelemetry.NewWriter(&out, nvtelemetry.CollectorConfig{})
			m := nvtelemetry.Manifest{SchemaVersion: 3, RunID: "bad-unit-control", Devices: []nvtelemetry.DeviceMetadata{{UUID: "GPU-0"}}, ClockMonotonicDomain: s.Before.ClockDomain, ClockMonotonicEpochID: s.Before.ClockEpoch, MetricRegistryVersion: nvtelemetry.DefaultMetricRegistryVersion, StartTimeMonotonic: s.Before.AtNanos}
			if err := w.WriteManifest(&m); err != nil {
				t.Fatal(err)
			}
			if err := w.WriteSample(&nvtelemetry.Sample{Seq: 1, QueryStartNanos: s.Before.AtNanos, QueryEndNanos: s.After.AtNanos, Readings: []nvtelemetry.Reading{reading}}); err != nil {
				t.Fatal(err)
			}
			m.EndTimeMonotonic = s.After.AtNanos
			m.TotalSamples = 1
			m.IsComplete = true
			m.LostRecordsByFamily = w.FamilyLoss()
			if err := w.WriteManifest(&m); err != nil {
				t.Fatal(err)
			}
			_, _, _, err := nvtelemetry.ReadStreamStrict(&out, 1<<20)
			if change == "unchanged" && err != nil {
				t.Fatalf("valid positive refused: %v", err)
			}
			if change != "unchanged" && (err == nil || !errors.Is(err, nvtelemetry.ErrMalformedStream) || !strings.Contains(err.Error(), "unexpected unit")) {
				t.Fatalf("unit contradiction not causally refused: %v", err)
			}
		})
	}
}

func TestNVLinkCounterUnknownPeerStaysUnknown(t *testing.T) {
	d, c, config := counterFixture()
	config.Topology.Peers = nil
	r, err := nvtelemetry.NewNVLinkCounterReader(d, c, config)
	if err != nil {
		t.Fatal(err)
	}
	s, err := r.Read(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if s.Before.Links[0].PeerState != "unknown" || s.Before.Links[0].PeerUUID != "" || s.Observations[0].Value == nil {
		t.Fatal("counter fabricated peer identity")
	}
	health := nvtelemetry.HealthSnapshot{Version: 1, NodeID: s.Before.NodeID, TopologyEpoch: s.Before.TopologyEpoch, ClockDomain: s.Before.ClockDomain, ClockEpoch: s.Before.ClockEpoch, AtNanos: s.After.AtNanos, DeviceStates: map[string]string{s.Before.DeviceUUID: "present"}}
	limits := nvtelemetry.HealthLimits{MaxBytes: 65536, MaxDevices: 2, MaxSources: 2, MaxObservations: 4, MaxEdges: 4}
	joined, err := nvtelemetry.JoinNVLinkHealth(health, s.Before, limits)
	if err != nil {
		t.Fatal(err)
	}
	if len(joined.Edges) != 0 {
		t.Fatal("raw errors fabricated transfer route")
	}
}

func TestNVLinkCounterLegacyDictionaryReplay(t *testing.T) {
	raw, err := os.ReadFile("testdata/canonical_v2.jsonl")
	if err != nil {
		t.Fatal(err)
	}
	m, phases, samples, err := nvtelemetry.ReadStreamStrict(bytes.NewReader(raw), 1<<20)
	if err != nil {
		t.Fatal(err)
	}
	if m.MetricRegistryVersion != "2026.10-v2" || m.SchemaVersion != 2 {
		t.Fatal("historical fixture reinterpreted")
	}
	var out bytes.Buffer
	w := nvtelemetry.NewWriter(&out, nvtelemetry.CollectorConfig{})
	if err := w.WriteManifest(m); err != nil {
		t.Fatal(err)
	}
	for i := range samples {
		if err := w.WriteSample(&samples[i]); err != nil {
			t.Fatal(err)
		}
	}
	for i := range phases {
		if err := w.WritePhase(&phases[i]); err != nil {
			t.Fatal(err)
		}
	}
	if err := w.WriteManifest(m); err != nil {
		t.Fatal(err)
	}
	got, _, decoded, err := nvtelemetry.ReadStreamStrict(&out, 1<<20)
	if err != nil {
		t.Fatal(err)
	}
	if got.MetricRegistryVersion != "2026.10-v2" || len(decoded) != len(samples) {
		t.Fatal("legacy dictionary changed")
	}
}
