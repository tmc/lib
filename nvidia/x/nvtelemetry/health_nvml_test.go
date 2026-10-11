package nvtelemetry

import (
	"bytes"
	"context"
	"errors"
	"testing"

	"github.com/tmc/lib/nvidia/nvml"
)

type operationNVMLDevice struct {
	uuid              string
	err               error
	ecc               uint64
	linkCalls         []uint32
	scopes            []nvml.EccCounterType
	changeDuringQuery bool
}

func (d *operationNVMLDevice) UUID() (string, error) { return d.uuid, nil }
func (d *operationNVMLDevice) TotalEccErrors(_ nvml.MemoryErrorType, c nvml.EccCounterType) (uint64, error) {
	d.scopes = append(d.scopes, c)
	return d.ecc, d.err
}
func (d *operationNVMLDevice) RemappedRowsV1() (uint32, uint32, uint32, uint32, error) {
	return 2, 3, 1, 0, d.err
}
func (d *operationNVMLDevice) Temperature(nvml.TemperatureSensors) (uint32, error) { return 31, d.err }
func (d *operationNVMLDevice) PowerUsage() (uint32, error) {
	if d.changeDuringQuery {
		d.uuid = "GPU-other"
	}
	return 46000, d.err
}
func (d *operationNVMLDevice) CurrentClocksEventReasons() (uint64, error) { return 4, d.err }
func (d *operationNVMLDevice) CurrPcieLinkGeneration() (uint32, error)    { return 4, d.err }
func (d *operationNVMLDevice) CurrPcieLinkWidth() (uint32, error)         { return 16, d.err }
func (d *operationNVMLDevice) NvLinkState(link uint32) (nvml.EnableState, error) {
	d.linkCalls = append(d.linkCalls, link)
	return nvml.FeatureEnabled, d.err
}

func TestNVMLHealthPublicReadAndBundle(t *testing.T) {
	d := &operationNVMLDevice{uuid: "GPU-0", ecc: 9007199254740993}
	clock := NewFakeClock(1, 1)
	r, err := newNVMLHealthReader(d, clock, "node-0", "topology-1", MIGScopeUnknown, []uint32{0, 2}, "fixture-library-v1")
	if err != nil {
		t.Fatal(err)
	}
	s, err := r.Read(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if len(s.Observations) != 13 || len(d.linkCalls) != 2 || len(d.scopes) != 2 || d.scopes[0] != nvml.VolatileEcc {
		t.Fatal("scope or selected link contract failed")
	}
	if s.Observations[0].Reading.IntValue != d.ecc || s.Observations[0].Reading.HasSourceTimestamp || s.Observations[0].Scope != MIGScopeUnknown {
		t.Fatal("native acquisition provenance invented or lost")
	}
	l := HealthLimits{MaxBytes: 64 << 10, MaxDevices: 1, MaxSources: 1, MaxObservations: 64, MaxEdges: 1}
	var raw bytes.Buffer
	if err := WriteHealthSnapshot(&raw, *s, l); err != nil {
		t.Fatal(err)
	}
	got, err := ReadHealthSnapshot(&raw, l)
	if err != nil {
		t.Fatal(err)
	}
	if got.Sources[0].Version != "fixture-library-v1" || got.Observations[0].Reading.IntValue != d.ecc {
		t.Fatal("public round trip changed provenance")
	}
	d.uuid = "GPU-other"
	if _, err := r.Read(context.Background()); err == nil {
		t.Fatal("handle identity mutation admitted")
	}
}

func TestNVMLHealthUnavailableAndLost(t *testing.T) {
	for _, tc := range []struct {
		name          string
		err           error
		reason, state string
	}{{"unsupported", nvml.ErrNotSupported, ReasonUnsupported, "present"}, {"denied", nvml.ErrNoPermission, ReasonPermissionDenied, "present"}, {"lost", nvml.ErrGpuIsLost, ReasonGPULost, "lost"}, {"outage", errors.New("fixture outage"), ReasonQueryFailure, "present"}} {
		t.Run(tc.name, func(t *testing.T) {
			d := &operationNVMLDevice{uuid: "GPU-0", err: tc.err}
			r, err := newNVMLHealthReader(d, NewFakeClock(1, 1), "node", "epoch", MIGScopeUnknown, nil, "fixture-v1")
			if err != nil {
				t.Fatal(err)
			}
			s, err := r.Read(context.Background())
			if err != nil {
				t.Fatal(err)
			}
			if s.DeviceStates["GPU-0"] != tc.state {
				t.Fatal("source failure became device loss")
			}
			for _, o := range s.Observations {
				if o.Reading.Evidence != EvidenceUnavailable || o.Reading.UnavailableReason != tc.reason || o.Reading.IntValue != 0 || o.NativeError == "" {
					t.Fatal("unavailable evidence lost", o)
				}
			}
		})
	}
}

func TestNVMLHealthConfigurationAndCancellation(t *testing.T) {
	d := &operationNVMLDevice{uuid: "GPU-0"}
	clock := NewFakeClock(1, 1)
	for _, links := range [][]uint32{{0, 0}, {32}} {
		if _, err := newNVMLHealthReader(d, clock, "node", "epoch", MIGScopeUnknown, links, "v1"); err == nil {
			t.Fatal("bad link selection admitted")
		}
	}
	r, err := newNVMLHealthReader(d, clock, "node", "epoch", MIGScopeUnknown, nil, "v1")
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := r.Read(ctx); !errors.Is(err, context.Canceled) {
		t.Fatal(err)
	}
	if len(d.scopes) != 0 {
		t.Fatal("canceled batch queried device")
	}
}

func TestNVMLHealthIdentityChangeDuringQuery(t *testing.T) {
	d := &operationNVMLDevice{uuid: "GPU-0", changeDuringQuery: true}
	r, err := newNVMLHealthReader(d, NewFakeClock(1, 1), "node", "epoch", MIGScopeUnknown, nil, "v1")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := r.Read(context.Background()); err == nil {
		t.Fatal("identity changed during query was admitted")
	}
}

func TestNVMLXIDPublicPollAndCleanup(t *testing.T) {
	clock := NewFakeClock(1, 1)
	calls := 0
	closeCalls := 0
	r := &NVMLXIDReader{clock: clock, uuid: "GPU-0", registered: true, identity: func(nvml.Device) (string, error) { return "GPU-0", nil }, wait: func(event *nvml.EventData, timeout uint32) error {
		calls++
		if timeout != 0 {
			t.Fatal("blocking event wait")
		}
		if calls > 1 {
			return nvml.ErrTimeout
		}
		*event = nvml.EventData{Device: 1, EventType: nvml.EventTypeXidCriticalError, EventData: 31, GpuInstanceId: 2, ComputeInstanceId: 3}
		return nil
	}, closeSet: func() error {
		closeCalls++
		if closeCalls == 1 {
			return errors.New("fixture close failure")
		}
		return nil
	}}
	observations, err := r.Poll(context.Background(), 4)
	if err != nil || len(observations) != 1 {
		t.Fatal(observations, err)
	}
	if observations[0].Reading.IntValue != 31 || observations[0].Reading.HasSourceTimestamp || observations[0].Scope != "native_instance_ids/2/3" {
		t.Fatal("event provenance invented")
	}
	if err := r.Close(); err == nil || r.closed {
		t.Fatal("failed cleanup lost resource ownership")
	}
	if err := r.Close(); err != nil || !r.closed {
		t.Fatal("explicit cleanup retry failed")
	}
	if err := r.Close(); err != nil || closeCalls != 2 {
		t.Fatal("successful close repeated native cleanup")
	}
	if _, err := r.Poll(context.Background(), 4); err == nil {
		t.Fatal("closed event reader used")
	}
}

func TestNVMLXIDIdentityAndSaturation(t *testing.T) {
	for _, uuid := range []string{"GPU-0", "GPU-other"} {
		r := &NVMLXIDReader{clock: NewFakeClock(1, 1), uuid: "GPU-0", registered: true, identity: func(nvml.Device) (string, error) { return uuid, nil }, wait: func(event *nvml.EventData, _ uint32) error {
			*event = nvml.EventData{Device: 1, EventType: nvml.EventTypeXidCriticalError, EventData: 31}
			return nil
		}}
		observations, err := r.Poll(context.Background(), 2)
		if err == nil {
			t.Fatal("complete event drain falsely claimed")
		}
		if uuid == "GPU-other" && observations != nil {
			t.Fatal("cross-target event exported")
		}
		if uuid == "GPU-0" && len(observations) != 2 {
			t.Fatal("bounded partial drain lost")
		}
	}
}

func TestNVMLXIDIdentityQueryFailureRetainsPriorEvent(t *testing.T) {
	calls := 0
	r := &NVMLXIDReader{clock: NewFakeClock(1, 1), uuid: "GPU-0", registered: true, wait: func(event *nvml.EventData, _ uint32) error {
		*event = nvml.EventData{Device: 1, EventType: nvml.EventTypeXidCriticalError, EventData: 31}
		return nil
	}, identity: func(nvml.Device) (string, error) {
		calls++
		if calls == 1 {
			return "GPU-0", nil
		}
		return "", nvml.ErrGpuIsLost
	}}
	got, err := r.Poll(context.Background(), 4)
	if err == nil || len(got) != 1 || got[0].Reading.IntValue != 31 {
		t.Fatal("consumed valid event lost on subsequent identity query error")
	}
}
