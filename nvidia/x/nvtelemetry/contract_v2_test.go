package nvtelemetry_test

import (
	"bytes"
	"errors"
	"os"
	"path/filepath"
	"testing"

	"github.com/tmc/lib/nvidia/x/nvtelemetry"
)

func TestStreamV1_BackwardCompatibility(t *testing.T) {
	fixturePath := filepath.Join("testdata", "canonical_v1.jsonl")
	data, err := os.ReadFile(fixturePath)
	if err != nil {
		t.Fatalf("read fixture canonical_v1.jsonl: %v", err)
	}

	manifest, phases, samples, err := nvtelemetry.ReadStream(bytes.NewReader(data), 0)
	if err != nil {
		t.Fatalf("ReadStream failed on canonical_v1: %v", err)
	}

	if manifest.SchemaVersion != 1 {
		t.Errorf("manifest.SchemaVersion = %d, want 1", manifest.SchemaVersion)
	}
	if !manifest.IsComplete {
		t.Errorf("manifest.IsComplete = false, want true")
	}
	if len(samples) != 2 {
		t.Fatalf("len(samples) = %d, want 2", len(samples))
	}
	if len(manifest.Devices) != 1 {
		t.Fatalf("len(manifest.Devices) = %d, want 1", len(manifest.Devices))
	}

	dev := manifest.Devices[0]
	if dev.UUID != "GPU-7b81f3d2-0001-0000-0000-000000000001" {
		t.Errorf("dev.UUID = %q, want GPU-7b81f3d2-0001-0000-0000-000000000001", dev.UUID)
	}
	if dev.MIGScope != "" {
		t.Errorf("dev.MIGScope = %q, want empty (unassessed in v1)", dev.MIGScope)
	}

	// Verify all v2 additive flags on legacy readings default to unassessed (false/zero)
	for i, s := range samples {
		for j, r := range s.Readings {
			if r.HasSourceTimestamp {
				t.Errorf("sample[%d].reading[%d].HasSourceTimestamp = true, want false for legacy", i, j)
			}
			if r.SourceTimestampDomain != "" {
				t.Errorf("sample[%d].reading[%d].SourceTimestampDomain = %q, want empty", i, j, r.SourceTimestampDomain)
			}
			if r.HasWallClockUncertainty {
				t.Errorf("sample[%d].reading[%d].HasWallClockUncertainty = true, want false for legacy", i, j)
			}
			if r.HasQueryBounds {
				t.Errorf("sample[%d].reading[%d].HasQueryBounds = true, want false for legacy", i, j)
			}
		}
	}

	// Legacy stream aggregates cleanly
	summaries, _ := nvtelemetry.AggregateWindow("test", 1000000000, 1105000000, samples, nvtelemetry.WindowOptions{})
	if len(summaries) != 2 {
		t.Errorf("len(summaries) = %d, want 2", len(summaries))
	}
	_ = phases
}

func TestStreamV2_PresenceAndAdditiveMetadata(t *testing.T) {
	fixturePath := filepath.Join("testdata", "canonical_v2.jsonl")
	data, err := os.ReadFile(fixturePath)
	if err != nil {
		t.Fatalf("read fixture canonical_v2.jsonl: %v", err)
	}

	manifest, _, samples, err := nvtelemetry.ReadStream(bytes.NewReader(data), 0)
	if err != nil {
		t.Fatalf("ReadStream failed on canonical_v2: %v", err)
	}

	if manifest.SchemaVersion != 2 {
		t.Errorf("manifest.SchemaVersion = %d, want 2", manifest.SchemaVersion)
	}
	if manifest.CollectorVersion != "nvtelemetry-2.0.0" {
		t.Errorf("manifest.CollectorVersion = %q, want nvtelemetry-2.0.0", manifest.CollectorVersion)
	}
	if manifest.MetricRegistryVersion != "2026.10-v2" {
		t.Errorf("manifest.MetricRegistryVersion = %q, want 2026.10-v2", manifest.MetricRegistryVersion)
	}
	if manifest.ClockMonotonicDomain != "process_local_go_monotonic" {
		t.Errorf("manifest.ClockMonotonicDomain = %q, want process_local_go_monotonic", manifest.ClockMonotonicDomain)
	}
	if manifest.ClockMonotonicEpochID != "epoch-20261010-v2-001" {
		t.Errorf("manifest.ClockMonotonicEpochID = %q, want epoch-20261010-v2-001", manifest.ClockMonotonicEpochID)
	}
	if manifest.ClockWallDomain != "unix_utc" {
		t.Errorf("manifest.ClockWallDomain = %q, want unix_utc", manifest.ClockWallDomain)
	}
	if manifest.ClockProvenance != "system_kernel_ntp" {
		t.Errorf("manifest.ClockProvenance = %q, want system_kernel_ntp", manifest.ClockProvenance)
	}
	if !manifest.HasClockCalibration {
		t.Errorf("manifest.HasClockCalibration = false, want true")
	}
	if manifest.ClockMappingValidUntil != 1800000000000000000 {
		t.Errorf("manifest.ClockMappingValidUntil = %d, want 1800000000000000000", manifest.ClockMappingValidUntil)
	}
	if manifest.ClockCalibrationBoundsNanos != 50000 {
		t.Errorf("manifest.ClockCalibrationBoundsNanos = %d, want 50000", manifest.ClockCalibrationBoundsNanos)
	}
	if !manifest.IsComplete {
		t.Errorf("manifest.IsComplete = false, want true")
	}

	if len(manifest.Devices) != 1 {
		t.Fatalf("len(manifest.Devices) = %d, want 1", len(manifest.Devices))
	}
	dev := manifest.Devices[0]
	if dev.Status != nvtelemetry.DeviceStatusOk {
		t.Errorf("dev.Status = %q, want ok", dev.Status)
	}
	if dev.MIGScope != nvtelemetry.MIGScopeDisabled {
		t.Errorf("dev.MIGScope = %q, want disabled", dev.MIGScope)
	}

	if len(samples) != 2 {
		t.Fatalf("len(samples) = %d, want 2", len(samples))
	}

	// Verify Sample 1
	s1 := samples[0]
	if s1.ClockDiscontinuity {
		t.Errorf("s1.ClockDiscontinuity = true, want false")
	}
	if len(s1.Readings) != 2 {
		t.Fatalf("len(s1.Readings) = %d, want 2", len(s1.Readings))
	}

	r1 := s1.Readings[0] // gpu_utilization_pct with hardware source timestamp
	if !r1.HasSourceTimestamp {
		t.Errorf("r1.HasSourceTimestamp = false, want true")
	}
	if r1.SourceTimestamp != 1000500000 {
		t.Errorf("r1.SourceTimestamp = %d, want 1000500000", r1.SourceTimestamp)
	}
	if r1.SourceTimestampDomain != "nvml_device_monotonic_ns" {
		t.Errorf("r1.SourceTimestampDomain = %q, want nvml_device_monotonic_ns", r1.SourceTimestampDomain)
	}
	if !r1.HasWallClockUncertainty {
		t.Errorf("r1.HasWallClockUncertainty = false, want true")
	}
	if r1.WallClockUncertaintyNanos != 25000 {
		t.Errorf("r1.WallClockUncertaintyNanos = %d, want 25000", r1.WallClockUncertaintyNanos)
	}
	if !r1.HasQueryBounds {
		t.Errorf("r1.HasQueryBounds = false, want true")
	}
	if r1.QueryStartNanos != 1000000000 || r1.QueryEndNanos != 1002000000 {
		t.Errorf("r1 query bounds [%d, %d], want [1000000000, 1002000000]", r1.QueryStartNanos, r1.QueryEndNanos)
	}

	r2 := s1.Readings[1] // power_milliwatts with unknown sensor observation time
	if r2.HasSourceTimestamp {
		t.Errorf("r2.HasSourceTimestamp = true, want false (omitted sensor timestamp)")
	}
	if r2.HasWallClockUncertainty {
		t.Errorf("r2.HasWallClockUncertainty = true, want false (unknown uncertainty)")
	}
	if !r2.HasQueryBounds {
		t.Errorf("r2.HasQueryBounds = false, want true (query bounds known)")
	}
}

func TestStreamV2_NegativeFixtures(t *testing.T) {
	tests := []struct {
		name        string
		fixtureFile string
		checkErr    func(error) bool
		desc        string
	}{
		{
			name:        "truncated",
			fixtureFile: "negative_truncated.jsonl",
			checkErr: func(err error) bool {
				return errors.Is(err, nvtelemetry.ErrTruncatedRecord)
			},
			desc: "expected ErrTruncatedRecord on truncated stream",
		},
		{
			name:        "contradictory_manifest",
			fixtureFile: "negative_contradictory_manifest.jsonl",
			checkErr: func(err error) bool {
				return errors.Is(err, nvtelemetry.ErrMalformedStream) || errors.Is(err, nvtelemetry.ErrIdentityMismatch)
			},
			desc: "expected ErrMalformedStream or ErrIdentityMismatch on contradictory manifest",
		},
		{
			name:        "missing_uuid",
			fixtureFile: "negative_missing_uuid.jsonl",
			checkErr: func(err error) bool {
				return errors.Is(err, nvtelemetry.ErrIdentityMismatch) || errors.Is(err, nvtelemetry.ErrUnidentifiableDevice)
			},
			desc: "expected ErrUnidentifiableDevice or ErrIdentityMismatch on missing device UUID",
		},
		{
			name:        "invalid_bounds",
			fixtureFile: "negative_invalid_bounds.jsonl",
			checkErr: func(err error) bool {
				return errors.Is(err, nvtelemetry.ErrMalformedStream)
			},
			desc: "expected ErrMalformedStream on inverted query bounds",
		},
		{
			name:        "epoch_mismatch",
			fixtureFile: "negative_epoch_mismatch.jsonl",
			checkErr: func(err error) bool {
				return errors.Is(err, nvtelemetry.ErrIdentityMismatch)
			},
			desc: "expected ErrIdentityMismatch on trailing manifest epoch mismatch",
		},
		{
			name:        "schema_downgrade",
			fixtureFile: "negative_schema_downgrade.jsonl",
			checkErr: func(err error) bool {
				return errors.Is(err, nvtelemetry.ErrIdentityMismatch)
			},
			desc: "expected ErrIdentityMismatch on trailing manifest schema downgrade",
		},
		{
			name:        "out_of_pass_bounds",
			fixtureFile: "negative_out_of_pass_bounds.jsonl",
			checkErr: func(err error) bool {
				return errors.Is(err, nvtelemetry.ErrMalformedStream)
			},
			desc: "expected ErrMalformedStream on reading query bounds exceeding sample pass interval",
		},
		{
			name:        "provenance_mismatch",
			fixtureFile: "negative_provenance_mismatch.jsonl",
			checkErr: func(err error) bool {
				return errors.Is(err, nvtelemetry.ErrIdentityMismatch)
			},
			desc: "expected ErrIdentityMismatch on trailing manifest provenance contradiction",
		},
		{
			name:        "binary_digest_mismatch",
			fixtureFile: "negative_binary_digest_mismatch.jsonl",
			checkErr: func(err error) bool {
				return errors.Is(err, nvtelemetry.ErrIdentityMismatch)
			},
			desc: "expected ErrIdentityMismatch on trailing manifest binary digest contradiction",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			path := filepath.Join("testdata", tt.fixtureFile)
			data, err := os.ReadFile(path)
			if err != nil {
				t.Fatalf("failed to read fixture %s: %v", tt.fixtureFile, err)
			}

			_, _, _, err = nvtelemetry.ReadStream(bytes.NewReader(data), 0)
			if err == nil {
				t.Fatalf("ReadStream unexpectedly succeeded on %s, %s", tt.fixtureFile, tt.desc)
			}
			if !tt.checkErr(err) {
				t.Fatalf("ReadStream error %v did not match expectation for %s (%s)", err, tt.fixtureFile, tt.desc)
			}
		})
	}

	// negative_bitmask_averaging.jsonl: parses as valid stream, but refuses numeric averaging
	t.Run("bitmask_averaging_refusal", func(t *testing.T) {
		path := filepath.Join("testdata", "negative_bitmask_averaging.jsonl")
		data, err := os.ReadFile(path)
		if err != nil {
			t.Fatalf("failed to read negative_bitmask_averaging.jsonl: %v", err)
		}

		manifest, _, samples, err := nvtelemetry.ReadStream(bytes.NewReader(data), 0)
		if err != nil {
			t.Fatalf("ReadStream failed on bitmask fixture: %v", err)
		}
		if len(samples) != 2 {
			t.Fatalf("got %d samples, want 2", len(samples))
		}

		// AggregateWindow must refuse to produce summaries for bitmasks
		summaries, _ := nvtelemetry.AggregateWindow("test", 1000000000, 1105000000, samples, nvtelemetry.WindowOptions{})
		if len(summaries) != 0 {
			t.Fatalf("AggregateWindow produced %d summaries, want 0 (bitmasks must not be averaged)", len(summaries))
		}

		// AggregateWindowStrict returns explicit refusal error
		_, _, err = nvtelemetry.AggregateWindowStrict("test", 1000000000, 1105000000, samples, nvtelemetry.WindowOptions{})
		if !errors.Is(err, nvtelemetry.ErrBitmaskAveragingRefused) {
			t.Fatalf("expected ErrBitmaskAveragingRefused, got: %v", err)
		}
		_ = manifest
	})
}

func TestBitmaskSummary_RefusesAveraging(t *testing.T) {
	// 1. Registry validation
	kind, found := nvtelemetry.LookupMetricKind(nvtelemetry.MetricThrottleReasons)
	if !found || kind != nvtelemetry.KindBitmask {
		t.Fatalf("LookupMetricKind(%s) = (%q, %v), want (bitmask, true)", nvtelemetry.MetricThrottleReasons, kind, found)
	}

	kind, found = nvtelemetry.LookupMetricKind(nvtelemetry.MetricClocksEventReasons)
	if !found || kind != nvtelemetry.KindBitmask {
		t.Fatalf("LookupMetricKind(%s) = (%q, %v), want (bitmask, true)", nvtelemetry.MetricClocksEventReasons, kind, found)
	}

	kind, found = nvtelemetry.LookupMetricKind(nvtelemetry.MetricXIDEvent)
	if !found || kind != nvtelemetry.KindEvent {
		t.Fatalf("LookupMetricKind(%s) = (%q, %v), want (event, true)", nvtelemetry.MetricXIDEvent, kind, found)
	}

	if err := nvtelemetry.ValidateMetricAveraging(nvtelemetry.MetricThrottleReasons); !errors.Is(err, nvtelemetry.ErrBitmaskAveragingRefused) {
		t.Errorf("ValidateMetricAveraging(throttle_reasons) = %v, want ErrBitmaskAveragingRefused", err)
	}

	if err := nvtelemetry.ValidateMetricAveraging(nvtelemetry.MetricClocksEventReasons); !errors.Is(err, nvtelemetry.ErrBitmaskAveragingRefused) {
		t.Errorf("ValidateMetricAveraging(clocks_event_reasons) = %v, want ErrBitmaskAveragingRefused", err)
	}

	if err := nvtelemetry.ValidateMetricAveraging(nvtelemetry.MetricGPUUtilization); err != nil {
		t.Errorf("ValidateMetricAveraging(gpu_utilization_pct) = %v, want nil", err)
	}

	// 2. Window aggregation skips bitmask metrics
	samples := []nvtelemetry.Sample{
		{
			Seq:             1,
			QueryStartNanos: 1000,
			QueryEndNanos:   1010,
			Readings: []nvtelemetry.Reading{
				{
					Name:       nvtelemetry.MetricGPUUtilization,
					Unit:       nvtelemetry.UnitPercent,
					DeviceUUID: "GPU-TEST",
					Value:      50.0,
					Evidence:   nvtelemetry.EvidenceMeasured,
				},
				{
					Name:       nvtelemetry.MetricThrottleReasons,
					Unit:       nvtelemetry.UnitBitmask,
					DeviceUUID: "GPU-TEST",
					IntValue:   4,
					IsInt:      true,
					Evidence:   nvtelemetry.EvidenceMeasured,
				},
			},
		},
		{
			Seq:             2,
			QueryStartNanos: 2000,
			QueryEndNanos:   2010,
			Readings: []nvtelemetry.Reading{
				{
					Name:       nvtelemetry.MetricGPUUtilization,
					Unit:       nvtelemetry.UnitPercent,
					DeviceUUID: "GPU-TEST",
					Value:      60.0,
					Evidence:   nvtelemetry.EvidenceMeasured,
				},
				{
					Name:       nvtelemetry.MetricThrottleReasons,
					Unit:       nvtelemetry.UnitBitmask,
					DeviceUUID: "GPU-TEST",
					IntValue:   0,
					IsInt:      true,
					Evidence:   nvtelemetry.EvidenceMeasured,
				},
			},
		},
	}

	summaries, _ := nvtelemetry.AggregateWindow("test", 1000, 2010, samples, nvtelemetry.WindowOptions{})
	if len(summaries) != 1 {
		t.Fatalf("len(summaries) = %d, want 1 (only gpu_utilization_pct)", len(summaries))
	}
	if summaries[0].MetricName != nvtelemetry.MetricGPUUtilization {
		t.Errorf("summary metric = %q, want %s", summaries[0].MetricName, nvtelemetry.MetricGPUUtilization)
	}

	// 3. Strict aggregation returns ErrBitmaskAveragingRefused
	_, _, err := nvtelemetry.AggregateWindowStrict("test", 1000, 2010, samples, nvtelemetry.WindowOptions{})
	if !errors.Is(err, nvtelemetry.ErrBitmaskAveragingRefused) {
		t.Fatalf("AggregateWindowStrict error = %v, want ErrBitmaskAveragingRefused", err)
	}
}

func TestDeviceIdentity_MissingUUIDRefusal(t *testing.T) {
	// Manifest with empty UUID
	streamEmptyUUID := `{"manifest":{"run_id":"test-missing-uuid","schema_version":2,"devices":[{"uuid":"","index":0,"model":"NVIDIA L4"}]}}` + "\n"
	_, _, _, err := nvtelemetry.ReadStream(bytes.NewBufferString(streamEmptyUUID), 0)
	if err == nil {
		t.Fatalf("ReadStream unexpectedly succeeded for empty UUID")
	}
	if !errors.Is(err, nvtelemetry.ErrIdentityMismatch) || !errors.Is(err, nvtelemetry.ErrUnidentifiableDevice) {
		t.Fatalf("expected ErrIdentityMismatch and ErrUnidentifiableDevice, got: %v", err)
	}

	// Manifest with unidentifiable status
	streamUnidentifiable := `{"manifest":{"run_id":"test-unidentifiable","schema_version":2,"devices":[{"uuid":"GPU-TEMP","index":0,"status":"unidentifiable"}]}}` + "\n"
	_, _, _, err = nvtelemetry.ReadStream(bytes.NewBufferString(streamUnidentifiable), 0)
	if err == nil {
		t.Fatalf("ReadStream unexpectedly succeeded for unidentifiable status")
	}
	if !errors.Is(err, nvtelemetry.ErrIdentityMismatch) || !errors.Is(err, nvtelemetry.ErrUnidentifiableDevice) {
		t.Fatalf("expected ErrIdentityMismatch and ErrUnidentifiableDevice, got: %v", err)
	}

	// Writer.WriteManifest refuses unidentifiable device
	var buf bytes.Buffer
	w := nvtelemetry.NewWriter(&buf, nvtelemetry.CollectorConfig{})
	badManifest := &nvtelemetry.Manifest{
		RunID:         "test-write-missing-uuid",
		SchemaVersion: 2,
		Devices: []nvtelemetry.DeviceMetadata{
			{UUID: "", Index: 0, Status: nvtelemetry.DeviceStatusUnidentifiable},
		},
	}
	if err := w.WriteManifest(badManifest); !errors.Is(err, nvtelemetry.ErrUnidentifiableDevice) {
		t.Fatalf("WriteManifest error = %v, want ErrUnidentifiableDevice", err)
	}

	// Writer.WriteManifest refuses unsupported schema version
	badSchemaManifest := &nvtelemetry.Manifest{
		RunID:         "test-bad-schema",
		SchemaVersion: 3,
		Devices: []nvtelemetry.DeviceMetadata{
			{UUID: "GPU-001", Index: 0, Status: nvtelemetry.DeviceStatusOk},
		},
	}
	if err := w.WriteManifest(badSchemaManifest); !errors.Is(err, nvtelemetry.ErrUnsupportedSchema) {
		t.Fatalf("WriteManifest error = %v, want ErrUnsupportedSchema", err)
	}
}
