package nvtelemetry

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"math"
	"sync"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/dcgm"
)

type fakeDCGMClient struct {
	fail             string
	closed, reads    int
	values           []dcgm.FieldValueV1
	closeErr         error
	entered, release chan struct{}
	once             sync.Once
}

func (f *fakeDCGMClient) NewGroup(name string, kind dcgm.GroupType) (dcgm.GpuGrp, error) {
	if name == "" || kind != dcgm.GroupEmpty {
		panic("invalid group")
	}
	return 1, f.err("group")
}
func (f *fakeDCGMClient) AddEntity(dcgm.GpuGrp, dcgm.FieldEntityGroup, dcgm.FieldEid) error {
	return f.err("entity")
}
func (f *fakeDCGMClient) NewFieldGroup(string, []dcgm.FieldID) (dcgm.FieldGrp, error) {
	return 2, f.err("fields")
}
func (f *fakeDCGMClient) Watch(dcgm.GpuGrp, dcgm.FieldGrp, time.Duration, time.Duration, int32) error {
	return f.err("watch")
}
func (f *fakeDCGMClient) LatestValues(uint32, []dcgm.FieldID) ([]dcgm.FieldValueV1, error) {
	f.reads++
	if f.entered != nil {
		f.once.Do(func() { close(f.entered) })
		<-f.release
	}
	return append([]dcgm.FieldValueV1(nil), f.values...), f.err("read")
}
func (f *fakeDCGMClient) Close() error { f.closed++; return f.closeErr }
func (f *fakeDCGMClient) err(stage string) error {
	if f.fail == stage {
		return errors.New(stage + " failure")
	}
	return nil
}

func dcgmTestConfig() DCGMConfig {
	return DCGMConfig{SourceID: "engine-incarnation", ProducerVersion: "mock-dcgm-v1", NativeOrigin: "engine/nvml/GPU-a", Devices: []DCGMDevice{{0, "GPU-a"}}, Fields: []DCGMField{{dcgm.FieldDevTotalEnergyConsumption, ManagedField{Metric: "DCGM_FI_DEV_TOTAL_ENERGY_CONSUMPTION", Name: "total_energy_millijoules", Type: "counter", Numerator: 1, Denominator: 1, Integer: true}}}, Interval: time.Second, MaxKeepAge: time.Minute, MaxKeepSamples: 8}
}
func dcgmTestValue() dcgm.FieldValueV1 {
	v := dcgm.FieldValueV1{Version: dcgm.FieldValueVersion1, FieldId: uint16(dcgm.FieldDevTotalEnergyConsumption), FieldType: dcgm.FieldTypeInt64, Ts: 123456}
	v.Value[0] = 9007199254740993
	return v
}

func ExampleNewDCGMSampler() {
	client := &fakeDCGMClient{values: []dcgm.FieldValueV1{dcgmTestValue()}}
	sampler, err := NewDCGMSampler(context.Background(), client, NewFakeClock(10, 100), dcgmTestConfig())
	readings, err := sampler.Sample(context.Background())
	fmt.Println(readings[0].IntValue, readings[0].Source.Native.TimestampMicroseconds, readings[0].HasSourceTimestamp, err)
	sampler.Close()
	// Output: 9007199254740993 123456 false <nil>
}

func TestDCGMPublicCollectorStream(t *testing.T) {
	client := &fakeDCGMClient{values: []dcgm.FieldValueV1{dcgmTestValue()}}
	clock := NewFakeClock(10, 100)
	sampler, e := NewDCGMSampler(t.Context(), client, clock, dcgmTestConfig())
	if e != nil {
		t.Fatal(e)
	}
	var b bytes.Buffer
	collector := NewCollector("dcgm", sampler, clock, CollectorConfig{}, NewWriter(&b, CollectorConfig{}))
	if e := collector.InitStream(); e != nil {
		t.Fatal(e)
	}
	if e := collector.Step(t.Context()); e != nil {
		t.Fatal(e)
	}
	if _, e := collector.Stop(); e != nil {
		t.Fatal(e)
	}
	_, _, samples, e := ReadStreamStrict(&b, 0)
	if e != nil {
		t.Fatal(e)
	}
	r := samples[0].Readings[0]
	if r.IntValue != 9007199254740993 || r.Source.Native.RawBits != 9007199254740993 || r.Source.Native.Status != 0 || r.HasSourceTimestamp {
		t.Fatal(r)
	}
	for _, mut := range []string{"bits", "status", "version"} {
		sample := samples[0]
		sample.Readings = cloneReadings(sample.Readings)
		switch mut {
		case "bits":
			sample.Readings[0].Source.Native.RawBits++
		case "status":
			sample.Readings[0].Source.Native.Status = -17
		case "version":
			sample.Readings[0].Source.Native.Version++
		}
		var bad bytes.Buffer
		w := NewWriter(&bad, CollectorConfig{})
		m := &Manifest{RunID: "mutation", SchemaVersion: 2, Devices: sampler.Devices()}
		w.WriteManifest(m)
		w.WriteSample(&sample)
		m.IsComplete = true
		m.TotalSamples = 1
		w.WriteManifest(m)
		if _, _, _, e := ReadStreamStrict(&bad, 0); e == nil {
			t.Fatal("accepted", mut)
		}
	}
	before := collector.Samples()[0].Readings[0].Source.Native.Status
	snapshot := collector.Samples()
	snapshot[0].Readings[0].Source.Native.Status = -17
	if collector.Samples()[0].Readings[0].Source.Native.Status != before {
		t.Fatal("native source aliases retained sample")
	}
	if e := sampler.Close(); e != nil {
		t.Fatal(e)
	}
	if e := sampler.Close(); e != nil || client.closed != 1 {
		t.Fatal(e, client.closed)
	}
	if _, e := sampler.Sample(t.Context()); !errors.Is(e, dcgm.ErrClosed) {
		t.Fatal(e)
	}
}

func TestDCGMSetupAndResults(t *testing.T) {
	for _, stage := range []string{"group", "entity", "fields", "watch"} {
		f := &fakeDCGMClient{fail: stage, closeErr: errors.New("cleanup failure")}
		if _, e := NewDCGMSampler(t.Context(), f, NewFakeClock(10, 100), dcgmTestConfig()); e == nil || f.closed != 1 || !errors.Is(e, f.closeErr) {
			t.Fatal(stage, e, f.closed)
		}
	}
	f := &fakeDCGMClient{}
	bad := dcgmTestConfig()
	bad.MaxKeepSamples = 0
	if _, e := NewDCGMSampler(t.Context(), f, NewFakeClock(10, 100), bad); e == nil || f.closed != 0 {
		t.Fatal("validation stole client")
	}
	var nilClient *fakeDCGMClient
	if _, e := NewDCGMSampler(t.Context(), nilClient, NewFakeClock(10, 100), dcgmTestConfig()); e == nil {
		t.Fatal("typed nil")
	}
	for _, tc := range []string{"permission", "blank", "nan", "layout", "duplicate", "missing", "call error", "cancel"} {
		t.Run(tc, func(t *testing.T) {
			v := dcgmTestValue()
			f := &fakeDCGMClient{values: []dcgm.FieldValueV1{v}}
			switch tc {
			case "permission":
				f.values[0].Status = int32(dcgm.ErrNoPermission)
			case "blank":
				f.values[0].Value[0] = uint64(dcgm.Int64Blank)
			case "nan":
				f.values[0].FieldType = dcgm.FieldTypeDouble
				f.values[0].Value[0] = math.Float64bits(math.NaN())
			case "layout":
				f.values[0].Version++
			case "duplicate":
				f.values = append(f.values, v)
			case "missing":
				f.values = nil
			case "call error":
				f.fail = "read"
			}
			s, e := NewDCGMSampler(t.Context(), f, NewFakeClock(10, 100), dcgmTestConfig())
			if e != nil {
				t.Fatal(e)
			}
			defer s.Close()
			ctx, cancel := context.WithCancel(t.Context())
			defer cancel()
			if tc == "cancel" {
				cancel()
			}
			r, e := s.Sample(ctx)
			switch tc {
			case "permission", "blank", "nan":
				if e != nil || r[0].IsAvailable() {
					t.Fatal(r, e)
				}
			default:
				if e == nil {
					t.Fatal("accepted")
				}
			}
		})
	}
}

func TestDCGMCloseWaitsForQuery(t *testing.T) {
	f := &fakeDCGMClient{values: []dcgm.FieldValueV1{dcgmTestValue()}, entered: make(chan struct{}), release: make(chan struct{}), closeErr: errors.New("close error")}
	s, e := NewDCGMSampler(t.Context(), f, NewFakeClock(10, 100), dcgmTestConfig())
	if e != nil {
		t.Fatal(e)
	}
	readDone := make(chan struct{})
	go func() { defer close(readDone); s.Sample(t.Context()) }()
	<-f.entered
	closeDone := make(chan error, 1)
	go func() { closeDone <- s.Close() }()
	select {
	case <-closeDone:
		t.Fatal("closed during native query")
	default:
	}
	close(f.release)
	<-readDone
	if e := <-closeDone; !errors.Is(e, f.closeErr) {
		t.Fatal(e)
	}
	if e := s.Close(); !errors.Is(e, f.closeErr) || f.closed != 1 {
		t.Fatal(e, f.closed)
	}
}

func TestDCGMFreshnessAndTerminalCollector(t *testing.T) {
	for _, tc := range []string{"unknown", "stale", "future", "overflow"} {
		t.Run(tc, func(t *testing.T) {
			v := dcgmTestValue()
			c := dcgmTestConfig()
			c.MaxSampleAge = time.Second
			c.MaxFutureSkew = time.Second
			now := int64(10 * time.Second)
			switch tc {
			case "unknown":
				v.Ts = 0
			case "stale":
				v.Ts = 1
			case "future":
				v.Ts = 20_000_000
			case "overflow":
				v.Ts = math.MaxInt64
			}
			f := &fakeDCGMClient{values: []dcgm.FieldValueV1{v}}
			s, e := NewDCGMSampler(t.Context(), f, NewFakeClock(10, now), c)
			if e != nil {
				t.Fatal(e)
			}
			defer s.Close()
			r, e := s.Sample(t.Context())
			if e != nil || r[0].IsAvailable() {
				t.Fatal(r, e)
			}
		})
	}
	f := &fakeDCGMClient{fail: "read"}
	clock := NewFakeClock(10, 100)
	s, e := NewDCGMSampler(t.Context(), f, clock, dcgmTestConfig())
	if e != nil {
		t.Fatal(e)
	}
	defer s.Close()
	var b bytes.Buffer
	c := NewCollector("failure", s, clock, CollectorConfig{}, NewWriter(&b, CollectorConfig{}))
	if e := c.InitStream(); e != nil {
		t.Fatal(e)
	}
	if e := c.Step(t.Context()); e == nil {
		t.Fatal("query error swallowed")
	}
	m, e := c.Stop()
	if e == nil || m.IsComplete {
		t.Fatal(m, e)
	}
	if _, _, _, e := ReadStreamStrict(&b, 0); !errors.Is(e, ErrIncompleteStream) {
		t.Fatal(e)
	}
}

func ExampleOpenDCGMClient() {
	_, e := OpenDCGMClient(context.Background(), "not-contacted", dcgm.ConnectOptions{Timeout: time.Second, PersistAfterDisconnect: true})
	fmt.Println(e)
	// Output: dcgm sampler: bounded nonpersistent connection required
}

func ExampleDCGMSampler_Close() {
	var s DCGMSampler
	fmt.Println(s.Close()) // Output: <nil>
}

func ExampleDCGMIMEXFields() {
	for _, field := range DCGMIMEXFields() {
		fmt.Println(field.Mapping.Name, field.Mapping.SourceUnit)
	}
	// Output:
	// imex_domain_status code
	// imex_daemon_status code
}

func TestDCGMIMEXPublicStream(t *testing.T) {
	c := dcgmTestConfig()
	c.Fields = DCGMIMEXFields()
	domain := dcgm.FieldValueV1{Version: dcgm.FieldValueVersion1, FieldId: uint16(dcgm.FieldImexDomainStatus), FieldType: dcgm.FieldTypeString, Ts: 123}
	domain.Value[0] = uint64('U') | uint64('P')<<8
	daemon := dcgm.FieldValueV1{Version: dcgm.FieldValueVersion1, FieldId: uint16(dcgm.FieldImexDaemonStatus), FieldType: dcgm.FieldTypeInt64, Ts: 124}
	daemon.Value[0] = 5
	f := &fakeDCGMClient{values: []dcgm.FieldValueV1{domain, daemon}}
	clock := NewFakeClock(10, 100)
	s, err := NewDCGMSampler(t.Context(), f, clock, c)
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	var b bytes.Buffer
	collector := NewCollector("imex", s, clock, CollectorConfig{}, NewWriter(&b, CollectorConfig{}))
	if err := collector.InitStream(); err != nil {
		t.Fatal(err)
	}
	if err := collector.Step(t.Context()); err != nil {
		t.Fatal(err)
	}
	if _, err := collector.Stop(); err != nil {
		t.Fatal(err)
	}
	_, _, samples, err := ReadStreamStrict(&b, 0)
	if err != nil {
		t.Fatal(err)
	}
	r := samples[0].Readings
	if r[0].IsAvailable() || r[0].UnavailableReason != "unsupported_field_type" || r[0].Source.Native.RawText != "UP" {
		t.Fatal(r[0])
	}
	if !r[1].IsAvailable() || r[1].IntValue != 5 || r[1].Unit != "code" || r[1].HasSourceTimestamp {
		t.Fatal(r[1])
	}
	if err := ValidateMetricAveraging(r[1].Name); err == nil {
		t.Fatal("averaged discrete status")
	}
}
