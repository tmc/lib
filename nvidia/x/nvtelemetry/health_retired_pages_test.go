package nvtelemetry

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"math"
	"reflect"
	"strconv"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/nvml"
)

var retiredPageNames = []string{"retired_pages_multiple_single_bit_ecc", "retired_pages_double_bit_ecc"}

type retiredPageReply struct {
	count uint32
	err   error
}

type retiredPageDevice struct {
	operationNVMLDevice
	t      *testing.T
	reply  []retiredPageReply
	causes []nvml.PageRetirementCause
	after  func()
}

func (d *retiredPageDevice) RetiredPagesV1(cause nvml.PageRetirementCause, count *uint32, addresses *uint64) error {
	d.t.Helper()
	if count == nil || addresses == nil {
		d.t.Fatal("retired-page count query passed a nil pointer")
	}
	if *count != 0 {
		d.t.Fatalf("retired-page query capacity = %d, want 0", *count)
	}
	i := len(d.causes)
	if i >= len(d.reply) || i >= 2 {
		d.t.Fatal("retired-page query retried or enumerated")
	}
	want := []nvml.PageRetirementCause{nvml.PageRetirementCauseMultipleSingleBitEccErrors, nvml.PageRetirementCauseDoubleBitEccError}
	if cause != want[i] {
		d.t.Fatalf("retired-page cause = %v, want %v", cause, want[i])
	}
	d.causes = append(d.causes, cause)
	*count = d.reply[i].count
	*addresses = 0x123456789abcdef0 // Must never become exported address evidence.
	if d.after != nil {
		d.after()
	}
	return d.reply[i].err
}

func retiredPageReader(t *testing.T, reply []retiredPageReply) (*NVMLHealthReader, *retiredPageDevice) {
	t.Helper()
	d := &retiredPageDevice{operationNVMLDevice: operationNVMLDevice{uuid: "GPU-retired", ecc: 91}, t: t, reply: reply}
	r, err := newNVMLHealthReader(d, NewFakeClock(1, 100), "node", "topology", MIGScopeUnknown, nil, "fixture-v1")
	if err != nil {
		t.Fatal(err)
	}
	return r, d
}

func retiredPageBundle(t *testing.T, r *NVMLHealthReader, d *retiredPageDevice) (*HealthSnapshot, []Reading) {
	t.Helper()
	s, err := r.Read(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if len(d.causes) != 2 {
		t.Fatalf("retired-page producer made %d queries, want both causes once", len(d.causes))
	}
	limits := HealthLimits{MaxBytes: 64 << 10, MaxDevices: 1, MaxSources: 1, MaxObservations: 32, MaxEdges: 1}
	var wire bytes.Buffer
	if err := WriteHealthSnapshot(&wire, *s, limits); err != nil {
		t.Fatal(err)
	}
	got, err := ReadHealthSnapshot(&wire, limits)
	if err != nil {
		t.Fatal(err)
	}
	var readings []Reading
	for _, o := range got.Observations {
		for _, name := range retiredPageNames {
			if o.Reading.Name == name {
				if o.Scope != MIGScopeUnknown || o.SourceID != "nvml" || o.Class != "raw" {
					t.Fatal("retired-page scope changed", o)
				}
				readings = append(readings, o.Reading)
			}
		}
	}
	if len(readings) != 2 || readings[0].Name != retiredPageNames[0] || readings[1].Name != retiredPageNames[1] {
		t.Fatal("missing or merged retirement causes", readings)
	}
	return got, readings
}

func TestRetiredPagesProducerExport(t *testing.T) {
	for _, tc := range []struct {
		name  string
		reply []retiredPageReply
	}{
		{"zero", []retiredPageReply{{}, {}}},
		{"success", []retiredPageReply{{count: 17}, {count: 19}}},
		{"needed", []retiredPageReply{{count: 23, err: nvml.ErrInsufficientSize}, {count: math.MaxUint32, err: fmt.Errorf("count-only: %w", nvml.ErrInsufficientSize)}}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			r, d := retiredPageReader(t, tc.reply)
			s, readings := retiredPageBundle(t, r, d)
			for i, reading := range readings {
				if reading.Evidence != EvidenceMeasured || !reading.IsInt || reading.IntValue != uint64(tc.reply[i].count) || reading.Value != float64(tc.reply[i].count) || reading.Unit != "count" || reading.DeviceUUID != "GPU-retired" {
					t.Fatal("retired-page value lost or replaced with remapped rows", reading)
				}
				if !reading.HasQueryBounds || reading.QueryEndNanos < reading.QueryStartNanos || reading.TimestampMonotonic != reading.QueryEndNanos || reading.HasSourceTimestamp || reading.IntegrationIntervalNanos != 0 {
					t.Fatal("invented retired-page sensor time", reading)
				}
				if unit, ok := LookupMetricUnit(reading.Name); !ok || unit != "count" {
					t.Fatal("retired-page unit missing")
				}
				if kind, ok := LookupMetricKind(reading.Name); !ok || kind != KindGauge {
					t.Fatal("inventory became a monotonic counter")
				}
			}
			remapped := map[string]uint64{}
			for _, o := range s.Observations {
				if o.Reading.Name == "corrected_remapped_rows" || o.Reading.Name == "uncorrected_remapped_rows" {
					remapped[o.Reading.Name] = o.Reading.IntValue
				}
			}
			if remapped["corrected_remapped_rows"] != 2 || remapped["uncorrected_remapped_rows"] != 3 || readings[1].QueryStartNanos < readings[0].QueryEndNanos {
				t.Fatal("remapped rows or separate query intervals changed")
			}
			var metrics, logs bytes.Buffer
			// Inventory gauges require no cumulative counter epoch.
			if err := WriteOTLPMetrics(&metrics, readings, ExportOptions{}); err != nil {
				t.Fatal(err)
			}
			mg := decodeEntityOTLP(t, metrics.Bytes(), "resourceMetrics")
			items := mg[0].(map[string]any)["scopeMetrics"].([]any)[0].(map[string]any)["metrics"].([]any)
			if len(items) != 2 {
				t.Fatal("lost inventory series")
			}
			for i, item := range items {
				m := item.(map[string]any)
				gauge, ok := m["gauge"].(map[string]any)
				if !ok || m["sum"] != nil {
					t.Fatal("inventory did not export as a gauge", m)
				}
				point := gauge["dataPoints"].([]any)[0].(map[string]any)
				if m["name"] != "gpu."+readings[i].Name || m["unit"] != "1" || point["asInt"] != strconv.FormatUint(uint64(tc.reply[i].count), 10) || point["startTimeUnixNano"] != nil {
					t.Fatal("changed inventory namespace, unit, value or epoch", m)
				}
			}
			if err := WriteOTLPLogs(&logs, readings, ExportOptions{}); err != nil {
				t.Fatal(err)
			}
			lg := decodeEntityOTLP(t, logs.Bytes(), "resourceLogs")
			for i, reading := range readings {
				fields := entityLogFields(t, lg[0], i)
				if fields["name"] != reading.Name || fields["kind"] != KindGauge || fields["unit"] != "count" || fields["status"] != "available" || fields["value_uint64"] != strconv.FormatUint(reading.IntValue, 10) {
					t.Fatal("lost exact inventory diagnostic", fields)
				}
			}
			if !reflect.DeepEqual(entityResourceAttrs(t, mg[0]), entityResourceAttrs(t, lg[0])) || entityAttrText(entityResourceAttrs(t, mg[0]), "device.uuid") != "GPU-retired" || entityAttrText(entityResourceAttrs(t, mg[0]), "assignment.status") != "absent" {
				t.Fatal("retired-page signals disagree on device resource")
			}
			if strings.Contains(metrics.String()+logs.String(), "1311768467463790320") || strings.Contains(metrics.String()+logs.String(), "123456789abcdef0") {
				t.Fatal("address placeholder leaked")
			}
		})
	}
}

func TestRetiredPagesUnavailable(t *testing.T) {
	for _, tc := range []struct {
		name, reason string
		err          error
		count        uint32
	}{
		{"unsupported", ReasonUnsupported, nvml.ErrNotSupported, 999},
		{"missing", ReasonUnsupported, nvml.ErrFunctionNotFound, 999},
		{"denied", ReasonPermissionDenied, nvml.ErrNoPermission, 999},
		{"lost", ReasonGPULost, nvml.ErrGpuIsLost, 999},
		{"invalid", ReasonQueryFailure, nvml.ErrInvalidArgument, 999},
		{"uninitialized", ReasonQueryFailure, nvml.ErrUninitialized, 999},
		{"unknown", ReasonQueryFailure, nvml.ErrUnknown, 999},
		{"zero_needed", ReasonQueryFailure, nvml.ErrInsufficientSize, 0},
	} {
		t.Run(tc.name, func(t *testing.T) {
			err := fmt.Errorf("retirement-native-canary: %w", tc.err)
			r, d := retiredPageReader(t, []retiredPageReply{{count: tc.count, err: err}, {count: tc.count, err: err}})
			s, readings := retiredPageBundle(t, r, d)
			for _, o := range s.Observations {
				if o.Reading.Name == retiredPageNames[0] || o.Reading.Name == retiredPageNames[1] {
					if !strings.Contains(o.NativeError, "retirement-native-canary") {
						t.Fatal("native error identity text lost")
					}
				}
				if tc.reason == ReasonGPULost && (o.Reading.Evidence != EvidenceUnavailable || o.Reading.UnavailableReason != ReasonGPULost) {
					t.Fatal("lost device retained earlier measured observations")
				}
			}
			for _, reading := range readings {
				if reading.Evidence != EvidenceUnavailable || reading.UnavailableReason != tc.reason || reading.IntValue != 0 || reading.Value != 0 {
					t.Fatal("retired-page failure became a measured zero or poisoned value", reading)
				}
			}
			var metrics, logs bytes.Buffer
			if err := WriteOTLPMetrics(&metrics, readings, ExportOptions{}); err == nil || metrics.Len() != 0 {
				t.Fatal("unavailable inventory became metrics")
			}
			if err := WriteOTLPLogs(&logs, readings, ExportOptions{}); err != nil {
				t.Fatal(err)
			}
			groups := decodeEntityOTLP(t, logs.Bytes(), "resourceLogs")
			for i := range readings {
				fields := entityLogFields(t, groups[0], i)
				if fields["status"] != "unavailable" || fields["unavailable_reason"] != tc.reason || fields["value_uint64"] != "" {
					t.Fatal("inventory failure lost typed absence", fields)
				}
			}
			if strings.Contains(logs.String(), "retirement-native-canary") || s.DeviceStates["GPU-retired"] != map[bool]string{true: "lost", false: "present"}[tc.reason == ReasonGPULost] {
				t.Fatal("native error leaked or unsupported became device loss")
			}
		})
	}
}

func TestRetiredPagesInventoryCanDecrease(t *testing.T) {
	r, d := retiredPageReader(t, []retiredPageReply{{count: 37}, {count: 41}})
	_, before := retiredPageBundle(t, r, d)
	d.causes = nil
	d.reply = []retiredPageReply{{}, {count: 1}}
	_, after := retiredPageBundle(t, r, d)
	if before[0].IntValue != 37 || before[1].IntValue != 41 || after[0].IntValue != 0 || after[1].IntValue != 1 {
		t.Fatal("inventory was clamped or converted to event deltas")
	}
}

func TestRetiredPagesCancellationAndIdentity(t *testing.T) {
	for _, mode := range []string{"before", "between", "identity"} {
		t.Run(mode, func(t *testing.T) {
			r, d := retiredPageReader(t, []retiredPageReply{{count: 1}, {count: 2}})
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			switch mode {
			case "before":
				cancel()
			case "between":
				d.after = cancel
			case "identity":
				d.after = func() { d.uuid = "GPU-replaced" }
			}
			s, err := r.Read(ctx)
			wantCalls := map[string]int{"before": 0, "between": 1, "identity": 2}[mode]
			if s != nil || err == nil || len(d.causes) != wantCalls {
				t.Fatal("canceled or rebound inventory admitted", s, err, d.causes)
			}
			if mode != "identity" && !errors.Is(err, context.Canceled) {
				t.Fatal("cancellation identity lost", err)
			}
		})
	}
}
