package nvtelemetry

import (
	"context"
	"fmt"
	"math"

	"github.com/tmc/lib/nvidia/nvml"
)

type fabricDevice interface {
	UUID() (string, error)
	GpuFabricInfoV(*nvml.GpuFabricInfoV) error
	TopologyCommonAncestor(nvml.Device, *nvml.GpuTopologyLevel) error
}
type fabricPeer struct {
	handle nvml.Device
	device fabricDevice
	uuid   string
}

// NVMLFabricReader reads the existing version 3 fabric query and common-ancestor
// relationship for explicitly selected peers. It never discovers peers or
// modifies fabric/IMEX state. The caller owns NVML and all device handles.
// The zero value is unusable; reads must not run concurrently.
type NVMLFabricReader struct {
	device                              fabricDevice
	uuid, nodeID, epoch, scope, version string
	clock                               Clock
	peers                               []fabricPeer
	validFor                            int64
}

// NewNVMLFabricReader verifies the selected device and at most 32 peer UUIDs.
// validForNanos bounds the snapshot's nominal topology validity; it is a caller
// policy, not a native guarantee. scope follows NewNVMLHealthReader conventions.
func NewNVMLFabricReader(device nvml.Device, peers []nvml.Device, clock Clock, nodeID, epoch, scope string, validForNanos int64) (*NVMLFabricReader, error) {
	if device == 0 {
		return nil, fmt.Errorf("fabric reader requires a device")
	}
	if len(peers) > 32 {
		return nil, fmt.Errorf("fabric peer limit exceeded")
	}
	version, err := nvml.SystemGetNVMLVersion()
	if err != nil {
		return nil, err
	}
	p := make([]fabricPeer, 0, len(peers))
	for _, peer := range peers {
		if peer == 0 {
			return nil, fmt.Errorf("fabric reader requires peer handles")
		}
		p = append(p, fabricPeer{handle: peer, device: peer})
	}
	return newNVMLFabricReader(device, p, clock, nodeID, epoch, scope, version, validForNanos)
}
func newNVMLFabricReader(device fabricDevice, peers []fabricPeer, clock Clock, nodeID, epoch, scope, version string, validFor int64) (*NVMLFabricReader, error) {
	if device == nil || clock == nil || nodeID == "" || epoch == "" || version == "" || len(peers) > 32 || validFor <= 0 {
		return nil, fmt.Errorf("invalid fabric reader configuration")
	}
	if scope != MIGScopeUnknown && scope != MIGScopeDisabled && scope != MIGScopePhysicalParent && scope != MIGScopeMIGInstance {
		return nil, fmt.Errorf("invalid fabric scope")
	}
	uuid, err := device.UUID()
	if err != nil || uuid == "" {
		return nil, fmt.Errorf("fabric device identity: %v", err)
	}
	peers = append([]fabricPeer(nil), peers...)
	seen := map[string]bool{uuid: true}
	for i := range peers {
		if peers[i].device == nil {
			return nil, fmt.Errorf("missing fabric peer")
		}
		id, err := peers[i].device.UUID()
		if err != nil || id == "" || seen[id] {
			return nil, fmt.Errorf("invalid fabric peer identity")
		}
		seen[id] = true
		peers[i].uuid = id
	}
	return &NVMLFabricReader{device: device, peers: peers, clock: clock, uuid: uuid, nodeID: nodeID, epoch: epoch, scope: scope, version: version, validFor: validFor}, nil
}

// Read returns raw fabric fields, a separately labeled vendor summary, and
// directed reported common-ancestor edges. Query success never establishes
// sensor freshness, measured traffic, full fabric membership or IMEX health.
func (r *NVMLFabricReader) Read(ctx context.Context) (*HealthSnapshot, error) {
	if r == nil || r.device == nil || ctx == nil {
		return nil, fmt.Errorf("invalid fabric reader")
	}
	meta, ok := r.clock.(ClockMetadata)
	if !ok || meta.MonotonicDomain() == "" || meta.MonotonicEpochID() == "" {
		return nil, fmt.Errorf("fabric clock metadata is required")
	}
	verify := func() error {
		if err := ctx.Err(); err != nil {
			return err
		}
		uuid, err := r.device.UUID()
		if err != nil {
			return err
		}
		if uuid != r.uuid {
			return fmt.Errorf("fabric device identity changed")
		}
		for _, peer := range r.peers {
			uuid, err := peer.device.UUID()
			if err != nil {
				return err
			}
			if uuid != peer.uuid {
				return fmt.Errorf("fabric peer identity changed")
			}
		}
		return nil
	}
	if err := verify(); err != nil {
		return nil, err
	}
	s := &HealthSnapshot{Version: 1, NodeID: r.nodeID, TopologyEpoch: r.epoch, ClockDomain: meta.MonotonicDomain(), ClockEpoch: meta.MonotonicEpochID(), DeviceStates: map[string]string{r.uuid: "present"}, Complete: true}
	source := HealthSource{ID: "nvml", Kind: "native", Version: r.version, State: "available"}
	start := r.clock.NowMonotonic()
	info := nvml.GpuFabricInfoV{Version: nvml.GpuFabricInfoV3Version}
	err := r.device.GpuFabricInfoV(&info)
	end := r.clock.NowMonotonic()
	if ctx.Err() != nil {
		return nil, ctx.Err()
	}
	if start < 0 || end < start || end > math.MaxInt64-r.validFor {
		return nil, fmt.Errorf("invalid fabric query clock bounds")
	}
	values := []struct {
		name, unit string
		value      uint64
		class      string
	}{{"fabric_state", "code", uint64(info.State), "raw"}, {"fabric_status", "code", uint64(info.Status), "raw"}, {"fabric_clique_id", "code", uint64(info.CliqueId), "raw"}, {"fabric_health_mask", UnitBitmask, uint64(info.HealthMask), "raw"}, {"fabric_health_summary", "code", uint64(info.HealthSummary), "vendor_judgment"}}
	if err == nil && info.Version != nvml.GpuFabricInfoV3Version {
		return nil, fmt.Errorf("fabric structure version changed")
	}
	if err == nil && (info.State < 0 || info.Status < 0) {
		return nil, fmt.Errorf("invalid signed fabric code")
	}
	for _, v := range values {
		o := HealthObservation{SourceID: "nvml", Class: v.class, Scope: r.scope + "/nvmlGpuFabricInfo_v3", Reading: Reading{Name: v.name, Unit: v.unit, Backend: "nvml", DeviceUUID: r.uuid, Evidence: EvidenceMeasured, IsInt: true, IntValue: v.value, Value: float64(v.value), TimestampMonotonic: end, TimestampUnix: r.clock.NowUnix(), HasQueryBounds: true, QueryStartNanos: start, QueryEndNanos: end}}
		if v.class == "vendor_judgment" {
			o.Judgment = fmt.Sprintf("nvml_v3_health_summary_%d", v.value)
		}
		if err != nil {
			o.Reading.Evidence = EvidenceUnavailable
			o.Reading.UnavailableReason = ClassifyNVMLError(err)
			o.Reading.IntValue = 0
			o.Reading.Value = 0
			o.NativeError = err.Error()
			if v.class == "vendor_judgment" {
				o.Judgment = "nvml_v3_health_summary_unavailable"
			}
		}
		s.Observations = append(s.Observations, o)
	}
	if err == nil {
		source.HasLastSuccess = true
		source.LastSuccessNanos = end
	} else {
		source.State = fabricFailureState(err)
		s.Complete = false
		if ClassifyNVMLError(err) == ReasonGPULost {
			s.DeviceStates[r.uuid] = "lost"
		}
	}
	for _, peer := range r.peers {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		s.DeviceStates[peer.uuid] = "present"
		var level nvml.GpuTopologyLevel
		start = r.clock.NowMonotonic()
		err := r.device.TopologyCommonAncestor(peer.handle, &level)
		end = r.clock.NowMonotonic()
		if start < 0 || end < start || end > math.MaxInt64-r.validFor {
			return nil, fmt.Errorf("invalid topology query clock bounds")
		}
		edge := TopologyEdge{SourceID: "nvml", From: r.uuid, To: peer.uuid, Kind: "common_ancestor", Scope: r.scope, State: "available", ValidFromNanos: end, ValidUntilNanos: end + r.validFor}
		if err == nil {
			edge.Capability = level.String()
			source.State = "available"
			source.HasLastSuccess = true
			source.LastSuccessNanos = end
		} else {
			edge.State = fabricFailureState(err)
			s.Complete = false
			if ClassifyNVMLError(err) == ReasonGPULost {
				s.DeviceStates[r.uuid] = "lost"
			}
		}
		s.Edges = append(s.Edges, edge)
	}
	if s.DeviceStates[r.uuid] == "lost" {
		for i := range s.Observations {
			o := &s.Observations[i]
			o.Reading.Evidence = EvidenceUnavailable
			o.Reading.UnavailableReason = ReasonGPULost
			o.Reading.Value = 0
			o.Reading.IntValue = 0
			if o.Class == "vendor_judgment" {
				o.Judgment = "nvml_v3_health_summary_unavailable"
			}
		}
		for i := range s.Edges {
			s.Edges[i].State = "failed"
			s.Edges[i].Capability = ""
		}
	}
	if err := verify(); err != nil {
		return nil, err
	}
	s.AtNanos = r.clock.NowMonotonic()
	if s.AtNanos < end || meta.MonotonicDomain() != s.ClockDomain || meta.MonotonicEpochID() != s.ClockEpoch {
		return nil, fmt.Errorf("fabric clock changed during query")
	}
	for _, edge := range s.Edges {
		if edge.State == "available" && s.AtNanos >= edge.ValidUntilNanos {
			return nil, fmt.Errorf("topology policy expired during query")
		}
	}
	s.Sources = []HealthSource{source}
	return s, nil
}
func fabricFailureState(err error) string {
	switch ClassifyNVMLError(err) {
	case ReasonUnsupported:
		return "unsupported"
	case ReasonPermissionDenied:
		return "denied"
	default:
		return "failed"
	}
}
