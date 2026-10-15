package cudadiagdemo

import (
	"bytes"
	"context"
	"encoding/json"
	"math"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/nvrtc"
	"github.com/tmc/lib/nvidia/x/cudadiag"
	"github.com/tmc/lib/nvidia/x/cudaflightrecorder"
)

func timingHostOnly(t *testing.T) {
	t.Helper()
	if driver.Loaded() || nvrtc.Loaded() {
		t.Fatal("driver/compiler loaded")
	}
	t.Cleanup(func() {
		if driver.Loaded() || nvrtc.Loaded() {
			t.Error("driver/compiler loaded")
		}
	})
}

func capturedService(t *testing.T) (*ServiceFixture, Response, []byte, timingExpectation) {
	t.Helper()
	sf := newTestFixture(t)
	response, err := sf.Process(context.Background(), Request{ID: "observed", Mode: ModeHostSubmissionDelay, Delay: 2 * time.Millisecond, Data: []float64{-3, 0, 2.5}})
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(response.Output, []float64{-5, 1, 6}) {
		t.Fatalf("independent output oracle: %v", response.Output)
	}
	data, expect, err := sf.captureTiming(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	expect.oraclePassed, expect.baselinePassed = true, true
	return sf, response, data, expect
}

func timingThresholds() cudadiag.Thresholds {
	t := cudadiag.DefaultThresholds()
	t.BaselinePreEntryDuration = time.Nanosecond
	return t
}

func TestTimingCapturedPublicProducerToDiagnose(t *testing.T) {
	timingHostOnly(t)
	_, response, data, expect := capturedService(t)
	in, got := diagnoseTiming(data, expect, response, timingThresholds())
	if !got.Admitted || got.Mechanism != cudadiag.MechanismHostSubmissionDelay || got.PrimaryFinding.Rule != "host_pre_entry_delay" || got.PrimaryFinding.Class != "associated" {
		t.Fatalf("captured finding %+v", got)
	}
	observed := response.Observation
	if !reflect.DeepEqual(in.HostCalls, []cudadiag.HostCallEvent{observed.API}) || in.PreEntries[0].Span != observed.PreEntry || in.WallInterval != observed.Service {
		t.Fatal("not the captured source phases")
	}
	if got.TimeAccounting.HostAPIDuration != time.Duration(observed.API.Span.Duration()) || *got.TimeAccounting.PreEntryDuration != time.Duration(observed.PreEntry.Duration()) || in.Sched.ProbePresent || len(in.Sched.Intervals) != 0 {
		t.Fatal("configured API or scheduler fabrication")
	}
	// Ground truth is independently scored after diagnosis, not an input path.
	response.GroundTruth.HostPrepDelay = 0
	response.GroundTruth.HostAPIDuration = time.Hour
	response.GroundTruth.Mode = ModeDeviceBottleneck
	_, other := diagnoseTiming(data, expect, response, timingThresholds())
	if !reflect.DeepEqual(got, other) {
		t.Fatal("ground truth changed diagnostic evidence")
	}
	if in.Receipt.InputSHA256 != expect.digest || in.PreEntries[0].Submission.EvidenceRef != expect.digest || in.Receipt.Backends[0].Name != "gpudiagdemo-portable-service-v1" {
		t.Fatal("source bytes/identity not bound")
	}
}

func rewriteTiming(t *testing.T, data []byte, expect timingExpectation, change func(*timingCapture)) ([]byte, timingExpectation) {
	t.Helper()
	bundle, err := cudaflightrecorder.ReadBundle(bytes.NewReader(data))
	if err != nil {
		t.Fatal(err)
	}
	var capture timingCapture
	if err := json.Unmarshal(bundle.Files[timingFile], &capture); err != nil {
		t.Fatal(err)
	}
	change(&capture)
	raw, err := json.Marshal(capture)
	if err != nil {
		t.Fatal(err)
	}
	bundle.Files[timingFile] = raw
	expect.digest = timingDigest(raw) // Separate semantic admission from hash failure.
	var out bytes.Buffer
	if err := cudaflightrecorder.WriteBundle(&out, bundle); err != nil {
		t.Fatal(err)
	}
	return out.Bytes(), expect
}

func TestTimingCaptureAdmissionMutations(t *testing.T) {
	timingHostOnly(t)
	_, response, data, expect := capturedService(t)
	for _, tc := range []struct {
		name   string
		change func(*timingCapture)
	}{
		{"source", func(c *timingCapture) { c.Source.SourceClass = "native" }},
		{"observer", func(c *timingCapture) { c.Source.Observer.Pass = "other" }},
		{"epoch", func(c *timingCapture) { c.Source.ClockEpoch = "other" }},
		{"lost", func(c *timingCapture) { c.Dropped = 1 }},
		{"incomplete", func(c *timingCapture) { c.Complete = false }},
		{"missing", func(c *timingCapture) { c.Records = nil }},
		{"duplicate", func(c *timingCapture) { c.Records = append(c.Records, c.Records[0]) }},
		{"population-alias-conflict", func(c *timingCapture) {
			other := c.Records[0]
			other.Host.OpID++
			other.RequestID = "other"
			other.Device.ExternalCorrelationID = uint64(other.Host.OpID)
			c.Records = append(c.Records, other)
		}},
		{"op", func(c *timingCapture) { c.Records[0].Host.OpID++ }},
		{"process", func(c *timingCapture) { c.Records[0].Device.Process.Generation++ }},
		{"workload-process", func(c *timingCapture) { c.Records[0].Host.Workload.Process.PID++ }},
		{"run", func(c *timingCapture) { c.Records[0].Host.Workload.RunID = "" }},
		{"other-run", func(c *timingCapture) { c.Records[0].Host.Workload.RunID = "other" }},
		{"request", func(c *timingCapture) { c.Records[0].RequestID = "other" }},
		{"thread", func(c *timingCapture) { c.Records[0].Host.TID++ }},
		{"clock", func(c *timingCapture) { c.Records[0].Host.TimestampClock = "unix_wall" }},
		{"host-epoch", func(c *timingCapture) { c.Records[0].Host.TimestampEpoch = "other" }},
		{"API-completion-timestamp", func(c *timingCapture) { c.Records[0].Host.TimestampNs = uint64(c.Records[0].API.Span.End) }},
		{"API-includes-wait", func(c *timingCapture) { c.Records[0].API.Span.Start = c.Records[0].PreEntry.Start }},
		{"pre-entry-end", func(c *timingCapture) { c.Records[0].PreEntry.End-- }},
		{"negative", func(c *timingCapture) { c.Records[0].PreEntry.Start = -1 }},
		{"service-bounds", func(c *timingCapture) { c.Records[0].Service.End = c.Records[0].API.Span.Start }},
		{"missing-return", func(c *timingCapture) { c.Records[0].API.HasReturn = false }},
		{"cancelled", func(c *timingCapture) { c.Records[0].Complete = false }},
		{"device-op", func(c *timingCapture) { c.Records[0].Device.ExternalCorrelationID++ }},
		{"device-correlation", func(c *timingCapture) { c.Records[0].Device.NativeCorrelationID++ }},
		{"device-overflow", func(c *timingCapture) { c.Records[0].Device.EndNs = math.MaxUint64 }},
		{"device-before-return", func(c *timingCapture) { c.Records[0].Device.StartNs = uint64(c.Records[0].PreEntry.Start) }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			changed, expected := rewriteTiming(t, data, expect, tc.change)
			_, got := diagnoseTiming(changed, expected, response, timingThresholds())
			if got.Admitted || got.Mechanism != cudadiag.MechanismWithheld {
				t.Fatalf("mutation admitted %+v", got)
			}
		})
	}
	for _, name := range []string{"raw-digest", "oracle", "baseline", "truncated", "target-thread"} {
		t.Run(name, func(t *testing.T) {
			changed, expected := data, expect
			thresholds := timingThresholds()
			switch name {
			case "raw-digest":
				expected.digest = strings.Repeat("b", 64)
			case "oracle":
				expected.oraclePassed = false
			case "baseline":
				expected.baselinePassed = false
			case "truncated":
				changed = data[:len(data)/2]
			case "target-thread":
				thresholds.TargetTID = response.Observation.Host.TID + 1
			}
			_, got := diagnoseTiming(changed, expected, response, thresholds)
			if got.Admitted {
				t.Fatal("independent gate ignored")
			}
		})
	}
}

func TestTimingObservationAndGroundTruthCustody(t *testing.T) {
	timingHostOnly(t)
	sf, response, _, _ := capturedService(t)
	expect, err := sf.attachTiming()
	if err != nil {
		t.Fatal(err)
	}
	response.GroundTruth.Workload.Labels["foreign"] = "response"
	logs := sf.GroundTruthLogs()
	if logs[0].Workload.Labels["foreign"] != "" {
		t.Fatal("returned ground truth aliases retained owner")
	}
	logs[0].Workload.Labels["foreign"] = "snapshot"
	if sf.GroundTruthLogs()[0].Workload.Labels["foreign"] != "" {
		t.Fatal("ground truth snapshot aliases retained owner")
	}
	response.Observation.Host.Workload.Labels["foreign"] = "source response"
	after, err := sf.attachTiming()
	if err != nil {
		t.Fatal(err)
	}
	if after.digest != expect.digest {
		t.Fatal("returned source observation mutated retained capture bytes")
	}
}

func TestTimingDeviceNegativeSameCapturePath(t *testing.T) {
	timingHostOnly(t)
	_, response, data, expect := capturedService(t)
	// Deterministic portable replay endpoints isolate classification from host
	// load; these are explicitly synthetic records, not observed device timing.
	data, expect = rewriteTiming(t, data, expect, func(c *timingCapture) {
		r := &c.Records[0]
		r.PreEntry.Start = r.PreEntry.End
		r.Device.EndNs = r.Device.StartNs + 500000
		r.Service.End = int64(r.Device.EndNs) + 1
	})
	in, got := diagnoseTiming(data, expect, response, timingThresholds())
	if !got.Admitted || got.Mechanism != cudadiag.MechanismDeviceBottleneck || in.Sched.ProbePresent || got.TimeAccounting.PreEntryDuration == nil || *got.TimeAccounting.PreEntryDuration != 0 {
		t.Fatalf("device negative %+v", got)
	}
}

func TestTimingCancellationPressureAndBounds(t *testing.T) {
	timingHostOnly(t)
	sf := newTestFixture(t)
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	response, err := sf.Process(ctx, Request{ID: "cancelled", Data: []float64{1}, Mode: ModeHostSubmissionDelay, Delay: time.Millisecond})
	if err == nil || response.Observation.Complete {
		t.Fatal("cancellation published completed timing")
	}
	sf = newTestFixture(t)
	sf.timingLimit = 1
	for i := 0; i < 2; i++ {
		if _, err := sf.Process(context.Background(), Request{ID: "pressure", Data: []float64{1}, Mode: ModeBaseline}); err != nil {
			t.Fatal(err)
		}
	}
	data, expect, err := sf.captureTiming(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	expect.oraclePassed, expect.baselinePassed = true, true
	_, got := diagnoseTiming(data, expect, response, timingThresholds())
	if got.Admitted {
		t.Fatal("lost timing population admitted")
	}
	for _, request := range []Request{{ID: "huge-delay", Data: []float64{1}, Delay: time.Duration(math.MaxInt64)}, {ID: "wrong-tid", Data: []float64{1}, TID: math.MaxUint64}} {
		if _, err := sf.Process(context.Background(), request); err == nil {
			t.Fatal("invalid request admitted")
		}
	}
	if _, err := RunDemonstration(context.Background(), DemonstrationConfig{NumRequests: math.MaxInt}); err == nil {
		t.Fatal("unsafe extent admitted")
	}
	if addTimingLoss(math.MaxUint64, 1) != math.MaxUint64 || addTimingLoss(math.MaxUint64-1, 2) != math.MaxUint64 {
		t.Fatal("loss counter wrapped to a false known zero")
	}
}
