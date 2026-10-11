package nvtelemetry_test

import (
	"bytes"
	"context"
	"encoding/json"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/nvml"
	"github.com/tmc/lib/nvidia/x/nvtelemetry"
)

type linkDevice struct {
	uuid    string
	changed bool
	reads   int
	state   nvml.EnableState
	typ     nvml.IntNvLinkDeviceType
	err     error
	pci     nvml.PciInfo
}

func (d *linkDevice) UUID() (string, error) {
	d.reads++
	if d.changed && d.reads >= 3 {
		return "GPU-other", nil
	}
	return d.uuid, nil
}
func (d *linkDevice) NvLinkState(uint32) (nvml.EnableState, error)     { return d.state, d.err }
func (d *linkDevice) NvLinkRemotePciInfo(uint32) (nvml.PciInfo, error) { return d.pci, d.err }
func (d *linkDevice) NvLinkRemoteDeviceType(_ uint32, p *nvml.IntNvLinkDeviceType) error {
	*p = d.typ
	return d.err
}
func (d *linkDevice) NvLinkVersion(uint32) (uint32, error) { return 4, d.err }
func linkFixture() (*linkDevice, *nvtelemetry.FakeClock, nvtelemetry.NVLinkConfig) {
	d := &linkDevice{uuid: "GPU-0", state: nvml.FeatureEnabled, typ: nvml.NvlinkDeviceTypeGpu}
	copy(d.pci.BusId[:], "00000000:03:00.0")
	d.pci.Bus = 3
	c := nvtelemetry.NewFakeClock(10, 0)
	c.SetEpochID("link-fixture")
	cfg := nvtelemetry.NVLinkConfig{NodeID: "n", SourceID: "nvml-nvlink", NVMLVersion: "fixture", TopologyEpoch: "t", Links: []uint32{0}, Peers: []nvtelemetry.NVLinkPeer{{BusID: "00000000:03:00.0", UUID: "GPU-1"}}, ValidFromNanos: 0, ValidUntilNanos: 100}
	return d, c, cfg
}
func reportFixture(t *testing.T) *nvtelemetry.NVLinkSnapshot {
	t.Helper()
	d, c, cfg := linkFixture()
	r, e := nvtelemetry.NewNVLinkReader(d, c, cfg)
	if e != nil {
		t.Fatal(e)
	}
	s, e := r.Read(context.Background())
	if e != nil {
		t.Fatal(e)
	}
	return s
}
func TestNVLinkPublicComposition(t *testing.T) {
	report := reportFixture(t)
	raw, e := json.Marshal(report)
	if e != nil {
		t.Fatal(e)
	}
	got, e := nvtelemetry.ReadNVLinkSnapshot(bytes.NewReader(raw), 8192)
	if e != nil {
		t.Fatal(e)
	}
	if got.Links[0].PeerUUID != "GPU-1" || got.Links[0].LinkVersion != 4 {
		t.Fatal(got)
	}
	s := nvtelemetry.HealthSnapshot{Version: 1, NodeID: "n", TopologyEpoch: "t", ClockDomain: report.ClockDomain, ClockEpoch: report.ClockEpoch, AtNanos: 10, DeviceStates: map[string]string{"GPU-0": "present", "GPU-1": "present"}}
	l := nvtelemetry.HealthLimits{MaxBytes: 8192, MaxDevices: 2, MaxSources: 2, MaxObservations: 4, MaxEdges: 4}
	out, e := nvtelemetry.JoinNVLinkHealth(s, *report, l)
	if e != nil {
		t.Fatal(e)
	}
	var b bytes.Buffer
	if e = nvtelemetry.WriteHealthSnapshot(&b, *out, l); e != nil {
		t.Fatal(e)
	}
	native, e := nvtelemetry.ReadHealthSnapshot(&b, l)
	if e != nil {
		t.Fatal(e)
	}
	if len(native.Edges) != 1 || native.Edges[0].Capability != "nominal_version=4" {
		t.Fatal(native)
	}
	out.DeviceStates["GPU-0"] = "lost"
	if s.DeviceStates["GPU-0"] != "present" {
		t.Fatal("aliased state")
	}
}
func TestNVLinkUnknowns(t *testing.T) {
	for _, tc := range []struct {
		name   string
		modify func(*linkDevice, *nvtelemetry.NVLinkConfig)
		state  string
	}{{"unmapped", func(d *linkDevice, c *nvtelemetry.NVLinkConfig) { c.Peers = nil }, "unknown"}, {"ambiguous", func(d *linkDevice, c *nvtelemetry.NVLinkConfig) {
		c.Peers = append(c.Peers, nvtelemetry.NVLinkPeer{BusID: c.Peers[0].BusID, UUID: "GPU-2"})
	}, "ambiguous"}, {"switch", func(d *linkDevice, c *nvtelemetry.NVLinkConfig) { d.typ = nvml.NvlinkDeviceTypeSwitch }, "non_gpu"}, {"future", func(d *linkDevice, c *nvtelemetry.NVLinkConfig) { d.typ = 12345 }, "unknown"}, {"unknown_enum", func(d *linkDevice, c *nvtelemetry.NVLinkConfig) { d.typ = nvml.NvlinkDeviceTypeUnknown }, "unknown"}, {"unsupported", func(d *linkDevice, c *nvtelemetry.NVLinkConfig) { d.err = nvml.ErrNotSupported }, "unknown"}, {"denied", func(d *linkDevice, c *nvtelemetry.NVLinkConfig) { d.err = nvml.ErrNoPermission }, "unknown"}} {
		t.Run(tc.name, func(t *testing.T) {
			d, clock, c := linkFixture()
			tc.modify(d, &c)
			r, e := nvtelemetry.NewNVLinkReader(d, clock, c)
			if e != nil {
				t.Fatal(e)
			}
			s, e := r.Read(context.Background())
			if e != nil {
				t.Fatal(e)
			}
			if s.Links[0].PeerState != tc.state || s.Links[0].PeerUUID != "" {
				t.Fatal(s)
			}
			if d.err != nil && (s.Links[0].RawState != 0 || s.Links[0].RemotePCI != "" || s.Links[0].LinkVersion != 0) {
				t.Fatal("invented unavailable value")
			}
		})
	}
}
func TestNVLinkIdentityAndBounds(t *testing.T) {
	for _, tc := range []struct {
		name   string
		modify func(*linkDevice, *nvtelemetry.FakeClock, *nvtelemetry.NVLinkConfig)
	}{{"identity", func(d *linkDevice, c *nvtelemetry.FakeClock, cfg *nvtelemetry.NVLinkConfig) { d.changed = true }}, {"pci_numeric", func(d *linkDevice, c *nvtelemetry.FakeClock, cfg *nvtelemetry.NVLinkConfig) { d.pci.Bus = 5 }}, {"expired", func(d *linkDevice, c *nvtelemetry.FakeClock, cfg *nvtelemetry.NVLinkConfig) { cfg.ValidUntilNanos = 10 }}, {"unterminated", func(d *linkDevice, c *nvtelemetry.FakeClock, cfg *nvtelemetry.NVLinkConfig) {
		for i := range d.pci.BusId {
			d.pci.BusId[i] = 'a'
		}
	}}} {
		t.Run(tc.name, func(t *testing.T) {
			d, c, cfg := linkFixture()
			tc.modify(d, c, &cfg)
			r, e := nvtelemetry.NewNVLinkReader(d, c, cfg)
			if e != nil {
				t.Fatal(e)
			}
			if s, e := r.Read(context.Background()); e == nil || s != nil {
				t.Fatal("accepted invalid source", s, e)
			}
		})
	}
	d, c, cfg := linkFixture()
	cfg.Links = []uint32{0, 0}
	if _, e := nvtelemetry.NewNVLinkReader(d, c, cfg); e == nil || d.reads != 0 {
		t.Fatal("queried invalid config")
	}
}
func TestNVLinkStrictReport(t *testing.T) {
	s := reportFixture(t)
	for _, mutate := range []func(*nvtelemetry.NVLinkSnapshot){func(s *nvtelemetry.NVLinkSnapshot) { s.Links[0].PeerUUID = "GPU-evil" }, func(s *nvtelemetry.NVLinkSnapshot) { s.Links[0].StateQuery.EndNanos = 20 }, func(s *nvtelemetry.NVLinkSnapshot) { s.Links[0].RemoteTypeKnown = false }, func(s *nvtelemetry.NVLinkSnapshot) { s.ValidUntilNanos = 10 }} {
		clone := *s
		clone.Links = append([]nvtelemetry.NVLinkObservation(nil), s.Links...)
		mutate(&clone)
		raw, _ := json.Marshal(clone)
		if _, e := nvtelemetry.ReadNVLinkSnapshot(bytes.NewReader(raw), 8192); e == nil {
			t.Fatal("accepted contradiction")
		}
	}
	raw, _ := json.Marshal(s)
	if _, e := nvtelemetry.ReadNVLinkSnapshot(bytes.NewReader(raw), 2); e == nil {
		t.Fatal("accepted excess bytes")
	}
	duplicate := strings.Replace(string(raw), `"version":1`, `"version":1,"version":1`, 1)
	if _, e := nvtelemetry.ReadNVLinkSnapshot(strings.NewReader(duplicate), 8192); e == nil {
		t.Fatal("accepted duplicate")
	}
}

func TestNVLinkLostRefusesTopology(t *testing.T) {
	d, c, cfg := linkFixture()
	d.err = nvml.ErrGpuIsLost
	r, e := nvtelemetry.NewNVLinkReader(d, c, cfg)
	if e != nil {
		t.Fatal(e)
	}
	if s, e := r.Read(context.Background()); e == nil || s != nil {
		t.Fatal("retained trusted topology after device lost", s, e)
	}
}
func TestNVLinkCanceledRead(t *testing.T) {
	d, c, cfg := linkFixture()
	r, e := nvtelemetry.NewNVLinkReader(d, c, cfg)
	if e != nil {
		t.Fatal(e)
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, e = r.Read(ctx); e == nil || d.reads != 1 {
		t.Fatal("queried canceled read")
	}
}

func TestNVLinkUnknownStateNeverActive(t *testing.T) {
	d, c, cfg := linkFixture()
	d.state = nvml.EnableState(99)
	r, e := nvtelemetry.NewNVLinkReader(d, c, cfg)
	if e != nil {
		t.Fatal(e)
	}
	report, e := r.Read(context.Background())
	if e != nil {
		t.Fatal(e)
	}
	s := nvtelemetry.HealthSnapshot{Version: 1, NodeID: "n", TopologyEpoch: "t", ClockDomain: report.ClockDomain, ClockEpoch: report.ClockEpoch, AtNanos: 10, DeviceStates: map[string]string{"GPU-0": "present", "GPU-1": "present"}}
	l := nvtelemetry.HealthLimits{MaxBytes: 8192, MaxDevices: 2, MaxSources: 2, MaxObservations: 4, MaxEdges: 4}
	out, e := nvtelemetry.JoinNVLinkHealth(s, *report, l)
	if e != nil || len(out.Edges) != 0 || report.Links[0].RawState != 99 {
		t.Fatal(out, report, e)
	}
}
func TestNVLinkUnknownClockAndSelfPeer(t *testing.T) {
	d, c, cfg := linkFixture()
	cfg.Peers[0].UUID = "GPU-0"
	r, e := nvtelemetry.NewNVLinkReader(d, c, cfg)
	if e != nil {
		t.Fatal(e)
	}
	if _, e = r.Read(context.Background()); e == nil {
		t.Fatal("accepted self peer")
	}
	report := reportFixture(t)
	s := nvtelemetry.HealthSnapshot{Version: 1, NodeID: "n", TopologyEpoch: "other", ClockDomain: report.ClockDomain, ClockEpoch: report.ClockEpoch, AtNanos: 10, DeviceStates: map[string]string{"GPU-0": "present", "GPU-1": "present"}}
	l := nvtelemetry.HealthLimits{MaxBytes: 8192, MaxDevices: 2, MaxSources: 2, MaxObservations: 4, MaxEdges: 4}
	if _, e = nvtelemetry.JoinNVLinkHealth(s, *report, l); e == nil {
		t.Fatal("accepted topology mismatch")
	}
}

func TestNVLinkMIGAssignmentRefused(t *testing.T) {
	d, c, cfg := linkFixture()
	cfg.Peers[0].UUID = "MIG-instance"
	if _, e := nvtelemetry.NewNVLinkReader(d, c, cfg); e == nil || d.reads != 0 {
		t.Fatal("accepted physical PCI mapping to MIG")
	}
	d, c, cfg = linkFixture()
	d.uuid = "MIG-instance"
	if _, e := nvtelemetry.NewNVLinkReader(d, c, cfg); e == nil {
		t.Fatal("accepted MIG source as physical link endpoint")
	}
}

func TestNVLinkNativeStatusMatchesCode(t *testing.T) {
	for _, tc := range []struct {
		name, state, code string
		wantError         bool
	}{
		{"unsupported", "unsupported", "3", false}, {"missing_function", "unsupported", "13", false}, {"denied", "denied", "4", false}, {"failed", "failed", "2", false}, {"future_failure", "failed", "12345", false}, {"opaque_failure", "failed", "", false}, {"unspecified_unsupported", "unsupported", "", false}, {"negative_future_failure", "failed", "-1", false},
		{"denied_unsupported", "denied", "3", true}, {"unsupported_denied", "unsupported", "4", true}, {"failed_unsupported", "failed", "3", true}, {"success_as_denied", "denied", "0", true}, {"success_as_failed", "failed", "0", true}, {"future_as_denied", "denied", "12345", true}, {"available_failure", "available", "2", true}, {"noncanonical_code", "unsupported", "03", true}, {"lost_device", "failed", "15", true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			report := reportFixture(t)
			o := &report.Links[0]
			o.VersionQuery.State = tc.state
			o.VersionQuery.NativeCode = tc.code
			o.LinkVersion = 0
			raw, _ := json.Marshal(report)
			parsed, e := nvtelemetry.ReadNVLinkSnapshot(bytes.NewReader(raw), 8192)
			if (e != nil) != tc.wantError {
				t.Fatalf("state=%s code=%s error=%v", tc.state, tc.code, e)
			}
			if e != nil && parsed != nil {
				t.Fatal("returned invalid trusted report")
			}
			if e == nil && parsed.Links[0].VersionQuery.NativeCode != tc.code {
				t.Fatal("lost raw code")
			}
		})
	}
}

func TestNVLinkNativeTransitions(t *testing.T) {
	d, c, cfg := linkFixture()
	r, e := nvtelemetry.NewNVLinkReader(d, c, cfg)
	if e != nil {
		t.Fatal(e)
	}
	for _, tc := range []struct {
		err               error
		state, code, peer string
	}{{nil, "available", "", "mapped"}, {nvml.ErrNoPermission, "denied", "4", "unknown"}, {nvml.ErrNotSupported, "unsupported", "3", "unknown"}, {nvml.Result(12345), "failed", "12345", "unknown"}, {nil, "available", "", "mapped"}} {
		d.err = tc.err
		report, e := r.Read(context.Background())
		if e != nil {
			t.Fatal(e)
		}
		q := report.Links[0].StateQuery
		if q.State != tc.state || q.NativeCode != tc.code || report.Links[0].PeerState != tc.peer {
			t.Fatalf("%+v", report)
		}
		if tc.err != nil && report.Links[0].LinkVersion != 0 {
			t.Fatal("retained stale version")
		}
	}
}
