package cudaprof_test

import (
	"bytes"
	"encoding/json"
	"errors"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudaevidence"
	"github.com/tmc/lib/nvidia/x/cudaprof"
	"github.com/tmc/lib/nvidia/x/cuptipmsampling"
	"github.com/tmc/lib/nvidia/x/cuptirange"
	"github.com/tmc/lib/nvidia/x/cuptisass"
)

// These fixtures are synthetic owner-struct controls, not native receipts.
func ownerReceipt() cudaprof.CaptureProviderResult {
	return cudaprof.CaptureProviderResult{PassID: "pass-1", Replay: "replay-1", Process: 42, Generation: 1, Context: "target-context-1", Start: 10, Stop: 20, CleanupComplete: true,
		Identity:    cudaevidence.Identity{ArtifactDigest: strings.Repeat("1", 64), Entry: "kernel", Target: "sm_70", CompilerKey: strings.Repeat("2", 64), WorkloadKey: strings.Repeat("3", 64)},
		Environment: cudaprof.CaptureEnvironment{Device: "synthetic device", Driver: "synthetic driver", CUDA: "synthetic CUDA", CUPTI: "synthetic CUPTI", ClockMode: "unchanged", CorrelationDomain: "target-clock"}}
}
func rangeOwner() *cuptirange.Capture {
	return &cuptirange.Capture{Image: []byte{1, 2, 3}, Ranges: 1, RangeInfo: []cuptirange.RangeInfo{{Index: 0, ID: "region/1", Name: "region", Kind: "user"}}, Quality: cuptirange.CollectionQuality{PassesStarted: 2, PassesStopped: 2, AllPassesSubmitted: true, Decoded: true}}
}
func TestRangeCaptureProviderResult(t *testing.T) {
	c := rangeOwner()
	r, err := cudaprof.RangeCaptureProviderResult(ownerReceipt(), c, []byte{7}, []string{"metric"}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if r.Status != "complete" || r.Provider != "range" || r.Scope != "target-context" || !r.Decoded || *r.Counts.Retained != 1 || *r.Counts.Dropped != 0 {
		t.Fatalf("result = %+v", r)
	}
	var raw struct {
		Image     []byte
		Quality   cuptirange.CollectionQuality
		RangeInfo []cuptirange.RangeInfo `json:"range_info"`
	}
	if err := json.Unmarshal(r.Raw, &raw); err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(raw.Image, c.Image) || raw.Quality != c.Quality || raw.RangeInfo[0].ID != "region/1" {
		t.Fatal("owner evidence lost")
	}
	c.Image[0] = 99
	c.RangeInfo[0].ID = "changed"
	if bytes.Contains(r.Raw, []byte("changed")) {
		t.Fatal("result aliases capture")
	}
}
func TestRangeCaptureProviderFailures(t *testing.T) {
	cases := []struct {
		name   string
		change func(*cuptirange.Capture, *cudaprof.CaptureProviderResult)
		err    error
	}{
		{"dropped", func(c *cuptirange.Capture, _ *cudaprof.CaptureProviderResult) { c.Quality.DroppedRanges = 2 }, nil},
		{"decode", func(c *cuptirange.Capture, _ *cudaprof.CaptureProviderResult) { c.Quality.Decoded = false }, nil},
		{"pass truncated", func(c *cuptirange.Capture, _ *cudaprof.CaptureProviderResult) { c.Quality.AllPassesSubmitted = false }, nil},
		{"unbalanced", func(c *cuptirange.Capture, _ *cudaprof.CaptureProviderResult) { c.Quality.PassesStopped = 1 }, nil},
		{"zero passes", func(c *cuptirange.Capture, _ *cudaprof.CaptureProviderResult) {
			c.Quality.PassesStarted = 0
			c.Quality.PassesStopped = 0
		}, nil},
		{"empty image", func(c *cuptirange.Capture, _ *cudaprof.CaptureProviderResult) { c.Image = nil }, nil},
		{"zero ranges", func(c *cuptirange.Capture, _ *cudaprof.CaptureProviderResult) { c.Ranges = 0 }, nil},
		{"cleanup", func(_ *cuptirange.Capture, r *cudaprof.CaptureProviderResult) { r.CleanupComplete = false }, nil},
		{"native failure", func(c *cuptirange.Capture, _ *cudaprof.CaptureProviderResult) {
			c.DiagnosticImage = []byte{9}
			c.Image = nil
		}, errors.New("native failure")},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			c, r := rangeOwner(), ownerReceipt()
			tc.change(c, &r)
			got, err := cudaprof.RangeCaptureProviderResult(r, c, []byte{7}, []string{"metric"}, tc.err)
			if err != nil {
				t.Fatal(err)
			}
			if got.Status == "complete" || got.Reason == "" {
				t.Fatalf("overclaimed %+v", got)
			}
			if !c.Quality.Decoded && got.Counts.Dropped != nil {
				t.Fatal("failed decode fabricated zero loss")
			}
			if tc.err != nil && got.Counts.Retained != nil {
				t.Fatal("failed capture counted completed ranges")
			}
		})
	}
	got, err := cudaprof.RangeCaptureProviderResult(ownerReceipt(), nil, []byte{7}, []string{"metric"}, nil)
	if err != nil || got.Status != "unavailable" || got.Counts.Retained != nil {
		t.Fatalf("missing %+v, %v", got, err)
	}
}
func TestPMCaptureProviderResult(t *testing.T) {
	decoded := cuptipmsampling.DecodeResult{StopReason: cuptipmsampling.DecodeStopReasonEndOfRecords, Info: cuptipmsampling.CounterDataInfo{NumTotalSamples: 8, NumPopulatedSamples: 1, NumCompletedSamples: 1}}
	samples := []cuptipmsampling.SampleInfo{{Index: 0, StartTimestamp: 11, EndTimestamp: 19}}
	r, err := cudaprof.PMCaptureProviderResult(ownerReceipt(), decoded, samples, []byte{4, 5}, cuptipmsampling.Config{ConfigImage: []byte{7}, MetricNames: []string{"metric"}, SamplingInterval: 1000, MaxSamples: 8}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if r.Provider != "pm" || r.Scope != "device-wide" || !r.Decoded || *r.Counts.Retained != 1 || r.Status != "partial" || r.Counts.Offered != nil {
		t.Fatalf("result %+v", r)
	}
	var raw struct {
		Image   []byte
		Decode  cuptipmsampling.DecodeResult
		Samples []cuptipmsampling.SampleInfo
	}
	if err := json.Unmarshal(r.Raw, &raw); err != nil {
		t.Fatal(err)
	}
	if raw.Decode != decoded || raw.Samples[0] != samples[0] || !bytes.Equal(raw.Image, []byte{4, 5}) {
		t.Fatal("PM owner results lost")
	}
	decoded.Overflow = true
	r, err = cudaprof.PMCaptureProviderResult(ownerReceipt(), decoded, samples, []byte{4, 5}, cuptipmsampling.Config{ConfigImage: []byte{7}, MetricNames: []string{"metric"}, SamplingInterval: 1000, MaxSamples: 8}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if r.Status == "complete" || r.Counts.Dropped != nil {
		t.Fatal("overflow reported complete or exact drops")
	}
	r, err = cudaprof.PMCaptureProviderResult(ownerReceipt(), decoded, nil, []byte{4}, cuptipmsampling.Config{ConfigImage: []byte{7}, MetricNames: []string{"metric"}, SamplingInterval: 1000, MaxSamples: 8}, errors.New("decode failed"))
	if err != nil {
		t.Fatal(err)
	}
	if r.Decoded || r.Counts.Retained != nil || r.Counts.Dropped != nil {
		t.Fatal("decode failure fabricated counts")
	}
}
func TestPMCaptureProviderBoundaries(t *testing.T) {
	d := cuptipmsampling.DecodeResult{StopReason: cuptipmsampling.DecodeStopReasonEndOfRecords, Info: cuptipmsampling.CounterDataInfo{NumTotalSamples: 1, NumPopulatedSamples: 1, NumCompletedSamples: 1}}
	for _, samples := range [][]cuptipmsampling.SampleInfo{nil, {{Index: 1, StartTimestamp: 11, EndTimestamp: 12}}, {{Index: 0, StartTimestamp: 12, EndTimestamp: 11}}} {
		r, err := cudaprof.PMCaptureProviderResult(ownerReceipt(), d, samples, []byte{1}, cuptipmsampling.Config{ConfigImage: []byte{7}, MetricNames: []string{"metric"}, SamplingInterval: 1000, MaxSamples: 8}, nil)
		if err == nil && r.Status == "complete" {
			t.Fatal("accepted invalid sample boundaries")
		}
	}
	d.Info.NumCompletedSamples = 2
	if _, err := cudaprof.PMCaptureProviderResult(ownerReceipt(), d, nil, []byte{1}, cuptipmsampling.Config{ConfigImage: []byte{7}, MetricNames: []string{"metric"}, SamplingInterval: 1000, MaxSamples: 8}, nil); err == nil {
		t.Fatal("accepted invalid native counts")
	}
}
func TestCaptureOwnerReceiptRequiresProvenance(t *testing.T) {
	changes := []func(*cudaprof.CaptureProviderResult){func(r *cudaprof.CaptureProviderResult) { r.Identity.ArtifactDigest = "" }, func(r *cudaprof.CaptureProviderResult) { r.Process = 0 }, func(r *cudaprof.CaptureProviderResult) { r.Generation = 0 }, func(r *cudaprof.CaptureProviderResult) { r.Context = "" }, func(r *cudaprof.CaptureProviderResult) { r.Environment.CUPTI = "" }, func(r *cudaprof.CaptureProviderResult) { r.Start = 0 }, func(r *cudaprof.CaptureProviderResult) { r.Scope = "controller-context" }}
	for _, change := range changes {
		r := ownerReceipt()
		change(&r)
		got, err := cudaprof.RangeCaptureProviderResult(r, rangeOwner(), []byte{7}, []string{"metric"}, nil)
		if err != nil {
			t.Fatal(err)
		}
		if got.Status == "complete" || len(got.Raw) == 0 || got.Reason == "" {
			t.Fatal("unknown provenance overclaimed or raw discarded")
		}
	}
	r := cudaprof.CaptureProviderResult{PassID: "p", Replay: "r"}
	got, err := cudaprof.RangeCaptureProviderResult(r, nil, nil, nil, nil)
	if err != nil || got.Status != "unavailable" {
		t.Fatalf("unavailable = %+v, %v", got, err)
	}
	r = ownerReceipt()
	r.Identity.ArtifactDigest = "malformed"
	if _, err := cudaprof.RangeCaptureProviderResult(r, rangeOwner(), []byte{7}, []string{"metric"}, nil); err == nil {
		t.Fatal("malformed supplied digest accepted")
	}
}

func TestRangeProviderManifestConfigurationBinding(t *testing.T) {
	receipt := ownerReceipt()
	result, err := cudaprof.RangeCaptureProviderResult(receipt, rangeOwner(), []byte{7}, []string{"metric-a", "metric-b"}, nil)
	if err != nil {
		t.Fatal(err)
	}
	plan := cudaprof.CapturePlan{Version: 1, Identity: receipt.Identity, Environment: receipt.Environment, Budget: cudaprof.CaptureBudget{MaxRecords: 100, MaxBytes: 1 << 20, MaxPasses: 4, MaxWallNanos: 1000}, Passes: []cudaprof.CapturePass{{ID: receipt.PassID, Replay: receipt.Replay, Providers: []cudaprof.CaptureProvider{"range"}, Configs: map[cudaprof.CaptureProvider]cudaprof.CaptureProviderConfig{"range": result.Config}}}}
	manifest := cudaprof.CaptureManifest{Version: 1, Plan: plan, Results: []cudaprof.CaptureProviderResult{result}, ElapsedNanos: 100}
	if err := manifest.Validate(); err != nil {
		t.Fatal(err)
	}
	if result.NativePasses == nil || result.NativePasses.Started != 2 || result.NativePasses.Stopped != 2 || !result.NativePasses.AllSubmitted {
		t.Fatal("native quality not retained")
	}
	for _, tc := range []struct {
		name   string
		change func(*cudaprof.CaptureProviderResult)
	}{
		{"digest", func(r *cudaprof.CaptureProviderResult) { r.Config.ConfigDigest = strings.Repeat("9", 64) }},
		{"metric order", func(r *cudaprof.CaptureProviderResult) { r.Config.Metrics = []string{"metric-b", "metric-a"} }},
		{"metric names", func(r *cudaprof.CaptureProviderResult) { r.Config.Metrics = []string{"other"} }},
		{"native incomplete", func(r *cudaprof.CaptureProviderResult) {
			r.NativePasses = &cudaprof.CaptureNativePasses{Started: 2, Stopped: 1, AllSubmitted: false}
		}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			m := manifest
			r := result
			tc.change(&r)
			m.Results = []cudaprof.CaptureProviderResult{r}
			if m.Validate() == nil {
				t.Fatal("configuration or native progress mismatch admitted complete")
			}
		})
	}
}
func TestCaptureProviderConfigComesFromOwner(t *testing.T) {
	receipt := ownerReceipt()
	receipt.Config = cudaprof.CaptureProviderConfig{ConfigDigest: strings.Repeat("9", 64), Metrics: []string{"expected-plan-only"}}
	metrics := []string{"actual-owner"}
	r, err := cudaprof.RangeCaptureProviderResult(receipt, rangeOwner(), []byte{7}, metrics, nil)
	if err != nil {
		t.Fatal(err)
	}
	if r.Config.ConfigDigest == receipt.Config.ConfigDigest || r.Config.Metrics[0] != "actual-owner" {
		t.Fatal("copied expected configuration as observation")
	}
	metrics[0] = "changed"
	if r.Config.Metrics[0] != "actual-owner" {
		t.Fatal("configuration metric alias")
	}
	for _, tc := range []struct {
		image   []byte
		metrics []string
	}{{nil, []string{"m"}}, {[]byte{7}, nil}, {[]byte{7}, []string{" "}}, {[]byte{7}, []string{"m", "m"}}} {
		if _, err := cudaprof.RangeCaptureProviderResult(receipt, rangeOwner(), tc.image, tc.metrics, nil); err == nil {
			t.Fatal("invalid observed configuration admitted")
		}
	}
}

func TestPMProviderPreservesSamplingPolicy(t *testing.T) {
	config := cuptipmsampling.Config{ConfigImage: []byte{7}, MetricNames: []string{"metric"}, SamplingInterval: 1234, TriggerMode: cuptipmsampling.TriggerModeSysclkInterval, AppendMode: cuptipmsampling.AppendModeKeepLatest, MaxSamples: 8}
	d := cuptipmsampling.DecodeResult{StopReason: cuptipmsampling.DecodeStopReasonEndOfRecords, Info: cuptipmsampling.CounterDataInfo{NumTotalSamples: 8, NumPopulatedSamples: 1, NumCompletedSamples: 1}}
	samples := []cuptipmsampling.SampleInfo{{Index: 0, StartTimestamp: 11, EndTimestamp: 19}}
	r, err := cudaprof.PMCaptureProviderResult(ownerReceipt(), d, samples, []byte{1}, config, nil)
	if err != nil {
		t.Fatal(err)
	}
	if r.Config.SamplingInterval != 1234 || r.Config.TriggerMode != int32(config.TriggerMode) || r.Config.AppendMode != int32(config.AppendMode) {
		t.Fatal("actual PM policy lost")
	}
	if r.Counts.Offered != nil || r.Status == "complete" {
		t.Fatal("capacity became offered denominator")
	}
	config.SamplingInterval = 0
	if _, err := cudaprof.PMCaptureProviderResult(ownerReceipt(), d, samples, []byte{1}, config, nil); err == nil {
		t.Fatal("zero PM interval admitted")
	}
}

func TestSASSCaptureProviderResult(t *testing.T) {
	config := cuptisass.Config{DeviceIndex: 0, LazyPatching: true, Metrics: []cuptisass.MetricConfig{{MetricID: 7, Granularity: cuptisass.GranularityGPU}}}
	records := []cuptisass.Record{{CubinCRC: 123, FunctionName: "kernel", PCOffset: 16, Values: []cuptisass.InstanceValue{{MetricID: 7, Value: 1<<63 + 1}}}}
	zero := uint64(0)
	r, err := cudaprof.SASSCaptureProviderResult(ownerReceipt(), config, records, &zero, nil)
	if err != nil {
		t.Fatal(err)
	}
	if r.Status != "complete" || !r.Config.LazyPatching || len(r.Config.SASSMetrics) != 1 || r.Config.SASSMetrics[0].MetricID != 7 || r.Config.SASSMetrics[0].Granularity != uint8(cuptisass.GranularityGPU) {
		t.Fatalf("actual SASS policy lost: %+v", r)
	}
	var raw struct {
		Config  cuptisass.Config
		Records []cuptisass.Record
		Dropped *uint64
	}
	if err := json.Unmarshal(r.Raw, &raw); err != nil {
		t.Fatal(err)
	}
	if raw.Records[0].Values[0].Value != 1<<63+1 || len(r.Values) != 0 {
		t.Fatal("native uint64 rounded into numeric evidence")
	}
	if raw.Config.LazyPatching != config.LazyPatching || raw.Dropped == nil || *raw.Dropped != 0 {
		t.Fatal("raw SASS receipt lost")
	}
	for _, tc := range []struct {
		name    string
		dropped *uint64
		modify  func(*cudaprof.CaptureProviderResult, []cuptisass.Record)
		err     error
	}{
		{"unknown loss", nil, func(_ *cudaprof.CaptureProviderResult, _ []cuptisass.Record) {}, nil},
		{"cleanup", &zero, func(r *cudaprof.CaptureProviderResult, _ []cuptisass.Record) { r.CleanupComplete = false }, nil},
		{"unnamed", &zero, func(_ *cudaprof.CaptureProviderResult, r []cuptisass.Record) { r[0].FunctionName = "" }, nil},
		{"different function", &zero, func(_ *cudaprof.CaptureProviderResult, r []cuptisass.Record) { r[0].FunctionName = "other" }, nil},
		{"unknown metric", &zero, func(_ *cudaprof.CaptureProviderResult, r []cuptisass.Record) {
			r[0].Values = []cuptisass.InstanceValue{{MetricID: 99, Value: 4}}
		}, nil},
		{"disable failure", &zero, func(_ *cudaprof.CaptureProviderResult, _ []cuptisass.Record) {}, errors.New("disable failed")},
	} {
		t.Run(tc.name, func(t *testing.T) {
			receipt := ownerReceipt()
			copied := append([]cuptisass.Record(nil), records...)
			tc.modify(&receipt, copied)
			got, err := cudaprof.SASSCaptureProviderResult(receipt, config, copied, tc.dropped, tc.err)
			if err != nil {
				t.Fatal(err)
			}
			if got.Status == "complete" || got.Reason == "" {
				t.Fatal("partial SASS overclaimed")
			}
			if tc.dropped == nil && got.Counts.Dropped != nil {
				t.Fatal("unknown drops fabricated zero")
			}
		})
	}
	dropped := uint64(3)
	r, err = cudaprof.SASSCaptureProviderResult(ownerReceipt(), config, records, &dropped, nil)
	if err != nil {
		t.Fatal(err)
	}
	if r.Status == "complete" || *r.Counts.Offered != 4 || *r.Counts.Dropped != 3 {
		t.Fatal("dropped SASS records hidden")
	}
}

func TestRangeCaptureProviderInventoryQualification(t *testing.T) {
	for _, tc := range []struct {
		name   string
		info   []cuptirange.RangeInfo
		ranges uint
	}{
		{"legacy absent", nil, 1},
		{"duplicate index", []cuptirange.RangeInfo{{Index: 0, ID: "a"}, {Index: 0, ID: "b"}}, 2},
		{"out of range", []cuptirange.RangeInfo{{Index: 2, ID: "a"}}, 1},
		{"duplicate identity", []cuptirange.RangeInfo{{Index: 0, ID: "a"}, {Index: 1, ID: "a"}}, 2},
	} {
		t.Run(tc.name, func(t *testing.T) {
			c := rangeOwner()
			c.RangeInfo = tc.info
			c.Ranges = tc.ranges
			r, err := cudaprof.RangeCaptureProviderResult(ownerReceipt(), c, []byte{7}, []string{"m"}, nil)
			if err != nil {
				t.Fatal(err)
			}
			if r.Status == "complete" || len(r.Raw) == 0 {
				t.Fatal("ambiguous inventory qualified")
			}
		})
	}
}
func TestSASSCaptureProviderRequiresAllMetricsAndKnownInstances(t *testing.T) {
	zero := uint64(0)
	for _, tc := range []struct {
		name    string
		config  cuptisass.Config
		records []cuptisass.Record
	}{
		{"missing configured metric", cuptisass.Config{Metrics: []cuptisass.MetricConfig{{MetricID: 7, Granularity: cuptisass.GranularityGPU}, {MetricID: 8, Granularity: cuptisass.GranularityGPU}}}, []cuptisass.Record{{FunctionName: "kernel", Values: []cuptisass.InstanceValue{{MetricID: 7, Value: 1}}}}},
		{"SM instances unknown", cuptisass.Config{Metrics: []cuptisass.MetricConfig{{MetricID: 7, Granularity: cuptisass.GranularitySM}}}, []cuptisass.Record{{FunctionName: "kernel", Values: []cuptisass.InstanceValue{{MetricID: 7, Value: 1}}}}},
		{"SMSP instances unknown", cuptisass.Config{Metrics: []cuptisass.MetricConfig{{MetricID: 7, Granularity: cuptisass.GranularitySMSP}}}, []cuptisass.Record{{FunctionName: "kernel", Values: []cuptisass.InstanceValue{{MetricID: 7, Value: 1}}}}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			r, err := cudaprof.SASSCaptureProviderResult(ownerReceipt(), tc.config, tc.records, &zero, nil)
			if err != nil {
				t.Fatal(err)
			}
			if r.Status == "complete" || r.Reason == "" || len(r.Raw) == 0 {
				t.Fatal("missing metrics or instance cardinality qualified")
			}
		})
	}
}
func TestCaptureOwnerRawBudget(t *testing.T) {
	large := make([]byte, 64<<20)
	c := rangeOwner()
	c.Image = large
	if _, err := cudaprof.RangeCaptureProviderResult(ownerReceipt(), c, []byte{7}, []string{"m"}, nil); err == nil {
		t.Fatal("range raw image exceeds transport cap")
	}
	cfg := cuptipmsampling.Config{ConfigImage: []byte{7}, MetricNames: []string{"m"}, SamplingInterval: 1000, MaxSamples: 1}
	if _, err := cudaprof.PMCaptureProviderResult(ownerReceipt(), cuptipmsampling.DecodeResult{}, nil, large, cfg, nil); err == nil {
		t.Fatal("PM raw image exceeds transport cap")
	}
	records := []cuptisass.Record{{FunctionName: strings.Repeat("x", 12<<20)}}
	if _, err := cudaprof.SASSCaptureProviderResult(ownerReceipt(), cuptisass.Config{Metrics: []cuptisass.MetricConfig{{MetricID: 7, Granularity: cuptisass.GranularityGPU}}}, records, nil, nil); err == nil {
		t.Fatal("SASS unbounded raw name admitted")
	}
}
