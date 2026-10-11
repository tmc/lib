package nvtelemetry

import (
	"fmt"
	"strings"

	"github.com/tmc/lib/ebpfhost/hostmetrics"
)

// JoinNodeHealth adds explicitly associated node evidence. The caller's join
// does not establish device consumption or causation. Sensor times stay unknown.
// The original snapshot is not modified.
func JoinNodeHealth(s HealthSnapshot, n hostmetrics.Snapshot, uuid string, l HealthLimits) (*HealthSnapshot, error) {
	if e := s.Validate(l); e != nil {
		return nil, e
	}
	if e := n.Validate(); e != nil {
		return nil, e
	}
	c := n.Context
	if c.NodeID != s.NodeID || c.ClockDomain != s.ClockDomain || c.ClockEpoch != s.ClockEpoch || c.QueryEndNanos > s.AtNanos || s.DeviceStates[uuid] != "present" {
		return nil, fmt.Errorf("node health identity or clock mismatch")
	}
	for _, x := range s.Sources {
		if x.ID == c.SourceID {
			return nil, fmt.Errorf("duplicate node health source")
		}
	}
	out := s
	out.DeviceStates = make(map[string]string, len(s.DeviceStates))
	for id, state := range s.DeviceStates {
		out.DeviceStates[id] = state
	}
	out.Edges = append([]TopologyEdge(nil), s.Edges...)
	out.Sources = append(append([]HealthSource(nil), s.Sources...), HealthSource{ID: c.SourceID, Kind: "linux_node", Version: c.Version, State: "available", HasLastSuccess: true, LastSuccessNanos: c.QueryEndNanos})
	out.Observations = append([]HealthObservation(nil), s.Observations...)
	for i := range out.Observations {
		out.Observations[i].Reading = cloneReadings([]Reading{out.Observations[i].Reading})[0]
	}
	for _, m := range n.Metrics {
		out.Observations = append(out.Observations, HealthObservation{SourceID: c.SourceID, Class: "raw", Scope: "node/" + c.NodeID + "/" + m.Resource, Reading: Reading{Name: m.Name, Unit: m.Unit, Backend: c.SourceID, DeviceUUID: uuid, Evidence: EvidenceMeasured, Value: m.Value, IsInt: m.Integer, IntValue: m.UintValue, TimestampMonotonic: c.QueryEndNanos, HasQueryBounds: true, QueryStartNanos: c.QueryStartNanos, QueryEndNanos: c.QueryEndNanos}})
	}
	if e := out.Validate(l); e != nil {
		return nil, e
	}
	return &out, nil
}

// JoinRedfishHealth associates a selected node sensor with a device explicitly.
// It does not convert chassis power to GPU power. Vendor health and source
// ReadingTime remain in the supplied sensor artifact; no clock mapping is
// inferred. Inactive or unknown sensor states remain unavailable even when
// the artifact retains a numeric reading. Synthesized enabled readings remain
// separately labeled inferences.
func JoinRedfishHealth(s HealthSnapshot, sensor hostmetrics.RedfishSensor, uuid string, l HealthLimits) (*HealthSnapshot, error) {
	if e := sensor.Validate(); e != nil {
		return nil, e
	}
	name := "node.bmc.power"
	if sensor.ReadingType == "Temperature" {
		name = "node.bmc.temperature"
	}
	value := 0.0
	if sensor.Reading != nil && sensor.State == "Enabled" {
		value = *sensor.Reading
	}
	n := hostmetrics.Snapshot{Version: 1, Context: sensor.Context, Metrics: []hostmetrics.Metric{{Name: name, Unit: sensor.Unit, Resource: sensor.ID, Value: value}}}
	out, e := JoinNodeHealth(s, n, uuid, l)
	if e != nil {
		return nil, e
	}
	out.Sources[len(out.Sources)-1].Kind = "redfish_sensor_v1_9_0"
	o := &out.Observations[len(out.Observations)-1]
	if sensor.State != "Enabled" {
		o.Reading.Evidence = EvidenceUnavailable
		if sensor.State == "" {
			o.Reading.UnavailableReason = "redfish_state_unknown"
		} else {
			o.Reading.UnavailableReason = "redfish_" + strings.ToLower(sensor.State)
		}
	} else if sensor.Reading == nil {
		o.Reading.Evidence = EvidenceUnavailable
		o.Reading.UnavailableReason = "redfish_reading_unavailable"
	} else if sensor.ImplementationType == "Synthesized" {
		o.Reading.Evidence = EvidenceDerived
		o.Class = "inference"
		o.Judgment = "redfish_synthesized"
	}
	if e = out.Validate(l); e != nil {
		return nil, e
	}
	return out, nil
}
