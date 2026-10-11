package nvtelemetry

import (
	"bytes"
	"context"
	"testing"

	"github.com/tmc/lib/nvidia/nvml"
)

type fixtureFabric struct {
	uuid        string
	queryErr    error
	topologyErr error
	mutate      bool
}

func (d *fixtureFabric) UUID() (string, error) { return d.uuid, nil }
func (d *fixtureFabric) GpuFabricInfoV(p *nvml.GpuFabricInfoV) error {
	p.State = nvml.GpuFabricState(3)
	p.HealthMask = 0x1234
	p.HealthSummary = 2
	if d.mutate {
		d.uuid = "other"
	}
	return d.queryErr
}
func (d *fixtureFabric) TopologyCommonAncestor(_ nvml.Device, p *nvml.GpuTopologyLevel) error {
	*p = nvml.TopologySingle
	return d.topologyErr
}
func TestFabricPublicRoundTrip(t *testing.T) {
	for _, kind := range []string{"success", "unsupported", "denied", "lost", "identity", "peer identity", "expiry", "unit"} {
		t.Run(kind, func(t *testing.T) {
			d := &fixtureFabric{uuid: "GPU-0"}
			peer := &fixtureFabric{uuid: "GPU-1"}
			clock := NewFakeClock(1, 1)
			r, err := newNVMLFabricReader(d, []fabricPeer{{device: peer, handle: 1}}, clock, "node", "epoch", MIGScopeUnknown, "fixture-v3", 100)
			if err != nil {
				t.Fatal(err)
			}
			switch kind {
			case "unsupported":
				d.queryErr = nvml.ErrNotSupported
				d.topologyErr = nvml.ErrNotSupported
			case "denied":
				d.queryErr = nvml.ErrNoPermission
				d.topologyErr = nvml.ErrNoPermission
			case "lost":
				d.queryErr = nvml.ErrGpuIsLost
				d.topologyErr = nvml.ErrGpuIsLost
			case "identity":
				d.mutate = true
			case "peer identity":
				peer.uuid = "other"
			}
			s, err := r.Read(context.Background())
			if kind == "identity" || kind == "peer identity" {
				if err == nil {
					t.Fatal("identity mutation admitted")
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			if kind == "expiry" {
				s.AtNanos = 102
			}
			if kind == "unit" {
				s.Observations[0].Reading.Unit = "percent"
			}
			limits := HealthLimits{MaxBytes: 65536, MaxDevices: 2, MaxSources: 1, MaxObservations: 5, MaxEdges: 1}
			var b bytes.Buffer
			err = WriteHealthSnapshot(&b, *s, limits)
			if kind == "expiry" || kind == "unit" {
				if err == nil {
					t.Fatal("mutated join admitted")
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			out, err := ReadHealthSnapshot(&b, limits)
			if err != nil {
				t.Fatal(err)
			}
			if out.Observations[4].Class != "vendor_judgment" || out.Observations[0].Reading.HasSourceTimestamp {
				t.Fatal("fabric provenance lost")
			}
			if kind != "success" && out.Complete {
				t.Fatal("source outage claimed complete")
			}
		})
	}
	_, err := newNVMLFabricReader(&fixtureFabric{uuid: "GPU-0"}, []fabricPeer{{device: &fixtureFabric{uuid: "GPU-0"}}}, NewFakeClock(1, 1), "node", "epoch", MIGScopeUnknown, "v3", 1)
	if err == nil {
		t.Fatal("duplicate peer")
	}
}
